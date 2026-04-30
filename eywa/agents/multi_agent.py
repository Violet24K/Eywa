"""Multi-agent collaboration patterns: debate, refine, and mixture-of-agents."""

import asyncio
import time
from typing import Any

from eywa.agents.base_agent import SingleAgent, traced_ainvoke
from eywa.agents.eywa import EywaAgent

AgentLike = SingleAgent | EywaAgent
TokenInfo = dict[str, int]


def _agent_display_name(agent: AgentLike) -> str:
    """Return a stable identifier for an agent, including its FM if applicable."""
    if isinstance(agent, EywaAgent):
        return f"{agent.model}:{agent.fm_name}"
    return agent.model


class MultiAgent:
    """Base class for multi-agent strategies."""

    def __init__(self, models: list[AgentLike]):
        self.models = models
        # Each EywaAgent in the ensemble must use a distinct foundation model.
        fm_names: set[str] = set()
        for model in models:
            if isinstance(model, EywaAgent):
                if model.fm_name in fm_names:
                    raise ValueError(
                        f"Duplicate fm_name {model.fm_name!r} found across EywaAgents."
                    )
                fm_names.add(model.fm_name)
        self.fm_names = fm_names

    @staticmethod
    def _merge_token_info(token_infos: list[TokenInfo]) -> TokenInfo:
        merged = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        for info in token_infos:
            for key in merged:
                merged[key] += int(info.get(key, 0))
        return merged

    async def update_data(self, task: str, new_data: dict) -> None:
        """Push task data into every Eywa-backed member of the ensemble."""
        for model in self.models:
            if isinstance(model, EywaAgent):
                print(f"Updating data for {model.fm_name}")
                await model.update_data(task, new_data)


class MultiAgentDebate(MultiAgent):
    """A two-agent debate: agent1 proposes, agent2 critiques, agent1 refines."""

    def __init__(self, models: list[AgentLike], use_final_pass: bool = False):
        super().__init__(models)
        if len(models) != 2:
            raise ValueError("MultiAgentDebate requires exactly 2 models.")
        self.agent1, self.agent2 = models
        self.debate_rounds = 1
        self.use_final_pass = use_final_pass

    async def invoke(self, prompt: str):
        traces: list[dict[str, Any]] = []
        token_infos: list[TokenInfo] = []
        total_elapsed = 0.0

        # Step 1: agent1 proposes an initial answer.
        a1_result, a1_tokens, a1_elapsed, a1_raw = await traced_ainvoke(
            self.agent1, prompt, self.agent1.model
        )
        traces.append(
            {
                "stage": "initial_agent1",
                "result_text": a1_result,
                "token_info": a1_tokens,
                "elapsed": a1_elapsed,
                "raw_response_dict": a1_raw,
            }
        )
        token_infos.append(a1_tokens)
        total_elapsed += a1_elapsed

        current_answer = a1_result
        latest_feedback = ""
        for round_idx in range(self.debate_rounds):
            # agent2 critiques the current candidate.
            agent2_prompt = (
                f"{prompt}\n\n"
                "Another expert proposed the following candidate answer:\n"
                f"{current_answer}\n\n"
                "Please critique this candidate and suggest a better answer if needed.\n"
                "Return only your improved candidate answer in the exact required output format."
            )
            latest_feedback, r2_tokens, r2_elapsed, r2_raw = await traced_ainvoke(
                self.agent2, agent2_prompt, self.agent2.model
            )
            traces.append(
                {
                    "stage": f"debate_round{round_idx + 1}_agent2_feedback",
                    "result_text": latest_feedback,
                    "token_info": r2_tokens,
                    "elapsed": r2_elapsed,
                    "raw_response_dict": r2_raw,
                }
            )
            token_infos.append(r2_tokens)
            total_elapsed += r2_elapsed

            # agent1 refines its answer using agent2's feedback.
            agent1_prompt = (
                f"{prompt}\n\n"
                "Your previous candidate answer:\n"
                f"{current_answer}\n\n"
                "Peer feedback / alternative candidate:\n"
                f"{latest_feedback}\n\n"
                "Please produce your best final candidate.\n"
                "Return only the final answer in the exact required output format."
            )
            current_answer, r1_tokens, r1_elapsed, r1_raw = await traced_ainvoke(
                self.agent1, agent1_prompt, self.agent1.model
            )
            traces.append(
                {
                    "stage": f"debate_round{round_idx + 1}_agent1_refine",
                    "result_text": current_answer,
                    "token_info": r1_tokens,
                    "elapsed": r1_elapsed,
                    "raw_response_dict": r1_raw,
                }
            )
            token_infos.append(r1_tokens)
            total_elapsed += r1_elapsed

        final_result = current_answer
        if self.use_final_pass:
            final_prompt = (
                f"{prompt}\n\n"
                "Current best candidate answer:\n"
                f"{current_answer}\n\n"
                "Peer feedback / alternative candidate:\n"
                f"{latest_feedback}\n\n"
                "Produce the final answer.\n"
                "Return only the final answer in the exact required output format."
            )
            final_result, final_tokens, final_elapsed, final_raw = await traced_ainvoke(
                self.agent1, final_prompt, self.agent1.model
            )
            traces.append(
                {
                    "stage": "final_arbitration",
                    "result_text": final_result,
                    "token_info": final_tokens,
                    "elapsed": final_elapsed,
                    "raw_response_dict": final_raw,
                }
            )
            token_infos.append(final_tokens)
            total_elapsed += final_elapsed

        return (
            final_result,
            self._merge_token_info(token_infos),
            total_elapsed,
            {"debate_trace": traces},
        )


class MultiAgentRefine(MultiAgent):
    """A two-agent pipeline: agent1 drafts, agent2 refines."""

    def __init__(self, models: list[AgentLike]):
        super().__init__(models)
        if len(models) != 2:
            raise ValueError("MultiAgentRefine requires exactly 2 models.")
        self.agent1, self.agent2 = models

    async def invoke(self, prompt: str):
        traces: list[dict[str, Any]] = []
        token_infos: list[TokenInfo] = []
        total_elapsed = 0.0

        # Step 1: agent1 produces a draft.
        draft, draft_tokens, draft_elapsed, draft_raw = await traced_ainvoke(
            self.agent1, prompt, self.agent1.model
        )
        traces.append(
            {
                "stage": "initial_agent1",
                "result_text": draft,
                "token_info": draft_tokens,
                "elapsed": draft_elapsed,
                "raw_response_dict": draft_raw,
            }
        )
        token_infos.append(draft_tokens)
        total_elapsed += draft_elapsed

        # Step 2: agent2 refines.
        refine_prompt = (
            f"{prompt}\n\n"
            "Draft answer from another expert:\n"
            f"{draft}\n\n"
            "Please refine this draft and return a better final answer.\n"
            "Return only the final answer in the exact required output format."
        )
        refined, refine_tokens, refine_elapsed, refine_raw = await traced_ainvoke(
            self.agent2, refine_prompt, self.agent2.model
        )
        traces.append(
            {
                "stage": "refine_agent2",
                "result_text": refined,
                "token_info": refine_tokens,
                "elapsed": refine_elapsed,
                "raw_response_dict": refine_raw,
            }
        )
        token_infos.append(refine_tokens)
        total_elapsed += refine_elapsed

        return (
            refined,
            self._merge_token_info(token_infos),
            total_elapsed,
            {"refine_trace": traces},
        )


class MixtureOfAgents(MultiAgent):
    """N proposers run in parallel, then a single aggregator synthesizes the answer."""

    def __init__(self, models: list[AgentLike]):
        if len(models) < 2:
            raise ValueError(
                "MixtureOfAgents requires at least 2 models (1 aggregator + >=1 proposer)."
            )
        super().__init__(models)
        self.aggregator = self.models[0]
        self.proposers = self.models[1:]
        print(
            "MixtureOfAgents configured: "
            f"aggregator={self.aggregator.model}, "
            f"proposers={[p.model for p in self.proposers]}"
        )

        # Proposers must be uniquely identifiable (model + optional FM).
        seen: set[str] = set()
        for proposer in self.proposers:
            name = _agent_display_name(proposer)
            if name in seen:
                raise ValueError(f"Duplicate proposer name {name!r} found in proposers.")
            seen.add(name)

    async def invoke(self, prompt: str):
        flow_start = time.perf_counter()
        traces: list[dict[str, Any]] = []
        token_infos: list[TokenInfo] = []

        proposer_prompt = (
            f"{prompt}\n\n"
            "You are one expert in a Mixture-of-Agents system.\n"
            "Provide your best standalone candidate answer.\n"
            "Return only the final answer in the exact required output format."
        )

        proposer_results = await asyncio.gather(
            *[
                traced_ainvoke(proposer, proposer_prompt, proposer.model)
                for proposer in self.proposers
            ],
            return_exceptions=True,
        )

        candidates: list[str] = []
        for idx, result in enumerate(proposer_results):
            proposer = self.proposers[idx]
            proposer_name = _agent_display_name(proposer)

            if isinstance(result, Exception):
                traces.append(
                    {
                        "stage": "proposer",
                        "proposer_index": idx + 1,
                        "proposer_name": proposer_name,
                        "error": str(result),
                        "error_type": type(result).__name__,
                    }
                )
                continue

            result_text, token_info, elapsed, raw_response_dict = result
            candidates.append(result_text)
            token_infos.append(token_info)
            traces.append(
                {
                    "stage": "proposer",
                    "proposer_index": idx + 1,
                    "proposer_name": proposer_name,
                    "result_text": result_text,
                    "token_info": token_info,
                    "elapsed": elapsed,
                    "raw_response_dict": raw_response_dict,
                }
            )

        if not candidates:
            raise RuntimeError("All proposer agents failed in MixtureOfAgents.")

        candidate_block = "\n\n".join(
            f"Candidate #{i + 1}:\n{c}" for i, c in enumerate(candidates)
        )
        aggregation_prompt = (
            f"{prompt}\n\n"
            "Below are candidate answers from multiple experts:\n"
            f"{candidate_block}\n\n"
            "Synthesize the best final answer.\n"
            "Resolve conflicts, keep only content well supported by the prompt, "
            "and return one final answer in the exact required output format."
        )

        final_result, agg_tokens, agg_elapsed, agg_raw = await traced_ainvoke(
            self.aggregator, aggregation_prompt, self.aggregator.model
        )
        token_infos.append(agg_tokens)
        traces.append(
            {
                "stage": "aggregator",
                "aggregator_name": _agent_display_name(self.aggregator),
                "result_text": final_result,
                "token_info": agg_tokens,
                "elapsed": agg_elapsed,
                "raw_response_dict": agg_raw,
            }
        )

        return (
            final_result,
            self._merge_token_info(token_infos),
            time.perf_counter() - flow_start,
            {"mixture_trace": traces},
        )
