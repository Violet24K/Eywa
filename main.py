"""Entry point for running Eywa, baseline, and orchestration experiments.

Each task in the benchmark is dispatched to one of three execution modes:

* ``single-agent``   - a single LLM (optionally backed by an Eywa foundation
                       model through MCP).
* ``multi-agent``    - debate / refine / mixture-of-agents over an explicit
                       list of agents.
* ``orchestration``  - an LLM planner first chooses a setting (single vs.
                       multi-agent, which models, whether to use Eywa), then
                       runs the task through the chosen path.

Results are streamed to ``experiments/<output_folder>/<save_name>/`` as
``results.jsonl``, ``raw_responses.jsonl``, ``result_texts.jsonl``, and (for
orchestration runs) ``orchestration.jsonl``. Runs are resumable: tasks whose
descriptions already appear in ``results.jsonl`` are skipped.
"""

import argparse
import asyncio
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from eywa.agents.base_agent import SingleAgent, traced_ainvoke
from eywa.agents.eywa import EywaAgent
from eywa.agents.multi_agent import MixtureOfAgents, MultiAgentDebate, MultiAgentRefine
from eywa.utils.choose_prompt import choose_prompt
from eywa.utils.path import benchmark_dir, config_dir, exp_dir, log_dir, prompt_dir
from eywabench.eval.deep_principle import eval_deep_principle
from eywabench.eval.tabular import eval_tabular_string_format
from eywabench.eval.timeseries import eval_timeseries_string_format
from eywabench.reorder_experiment_outputs import reorder_experiment_outputs


load_dotenv()


SUPPORTED_MODELS = (
    "gpt-5-nano",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
)


class Tee:
    """Mirror stdout writes into a log file."""

    def __init__(self, filename: str):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


@dataclass
class TaskContext:
    """Resolved view of a benchmark row, ready to be turned into a prompt."""

    domain: str
    task: str
    description: str
    output_size: int
    input_data: Any
    label: Any
    fm: str | None
    output_format: str
    input_tag: str
    additional_instructions: str


def prepare_task_context(task_sample: pd.Series) -> TaskContext:
    """Resolve the task-specific FM, prompt tag, and output format."""
    task = task_sample["task"]
    if task == "time-series-forecasting":
        fm = "chronos"
        output_format = open(
            osp.join(prompt_dir, "format", "time-series.txt"), "r", encoding="utf-8"
        ).read()
        input_tag = "input_time_series"
        additional_instructions = ""
    elif task.startswith("tabular-"):
        fm = "tabpfn"
        output_format = open(
            osp.join(prompt_dir, "format", "tabular.txt"), "r", encoding="utf-8"
        ).read()
        input_tag = "input_tabular"
        additional_instructions = (
            "In the provided table, the target column (last column) contains some "
            "entries masked with 0. Your task is to predict the actual values for "
            "these masked entries (where the value is 0)."
        )
    elif task in ("deep_principle_physics", "mmlu"):
        fm = None
        output_format = "There is no strict output format."
        input_tag = "input_question"
        additional_instructions = (
            "output_size is the number of characters in the correct answer."
        )
    else:
        # Unknown task type: fall back to a permissive prompt.
        fm = None
        output_format = "There is no strict output format."
        input_tag = "input_data"
        additional_instructions = ""

    return TaskContext(
        domain=task_sample["domain"],
        task=task,
        description=task_sample["description"],
        output_size=task_sample["output_size"],
        input_data=task_sample["input"],
        label=task_sample["label"],
        fm=fm,
        output_format=output_format,
        input_tag=input_tag,
        additional_instructions=additional_instructions,
    )


def evaluate(task: str, label: Any, result_text: str) -> float:
    """Score a single prediction against the gold label, returning utility in [0, 1]."""
    if task == "time-series-forecasting":
        return eval_timeseries_string_format(label, result_text)
    if task.startswith("tabular-"):
        return eval_tabular_string_format(label, result_text, task)
    if task in ("deep_principle_physics", "mmlu"):
        return eval_deep_principle(label, result_text)
    return 0.0


def render_prompt(template_path: str, ctx: TaskContext, mcp_description: str) -> str:
    """Render the user-facing prompt template with task-specific fields."""
    template = open(template_path, "r", encoding="utf-8").read()
    return template.format(
        task=ctx.task,
        mcp_server_description=mcp_description,
        additional_instructions=ctx.additional_instructions,
        input_tag=ctx.input_tag,
        input_data=ctx.input_data,
        output_size=ctx.output_size,
        output_format=ctx.output_format,
    )


async def _build_agent_from_spec(agent_spec: str, fm: str | None, worker_id: int = 0):
    """Build an agent from a ``<type>:<llm>`` spec, falling back gracefully."""
    agent_type, agent_llm = agent_spec.split(":", 1)
    if agent_type == "base":
        return SingleAgent(model=agent_llm)
    if agent_type == "eywa":
        if fm is None:
            print(
                f"Warning: task has no foundation model; "
                f"falling back '{agent_spec}' to 'base:{agent_llm}'."
            )
            return SingleAgent(model=agent_llm)
        return await EywaAgent.create(model=agent_llm, fm=fm, worker_id=worker_id)
    raise ValueError(f"Invalid agent type: {agent_type!r}.")


async def run_single_agent(task_sample: pd.Series, args, worker_id: int = 0):
    ctx = prepare_task_context(task_sample)
    fm = ctx.fm or getattr(args, "foundation_model", None)
    template = choose_prompt(ctx.task, args)

    if args.eywa and fm is not None:
        agent = await EywaAgent.create(model=args.model, fm=fm, worker_id=worker_id)
        await agent.update_data(ctx.task, task_sample.to_dict())
        mcp_description = (
            "You can interact with a server that contains the data as a pandas dataframe."
        )
    else:
        agent = SingleAgent(model=args.model)
        mcp_description = ""

    prompt = render_prompt(template, ctx, mcp_description)
    return await traced_ainvoke(agent, prompt, args.model)


async def run_multi_agent(task_sample: pd.Series, args, worker_id: int = 0):
    ctx = prepare_task_context(task_sample)
    fm = ctx.fm or getattr(args, "foundation_model", None)
    template = choose_prompt(ctx.task, args)

    agents = [
        await _build_agent_from_spec(spec, fm, worker_id=worker_id)
        for spec in args.agents
    ]
    if args.multi_agent_type == "debate":
        ensemble = MultiAgentDebate(agents)
    elif args.multi_agent_type == "refine":
        ensemble = MultiAgentRefine(agents)
    elif args.multi_agent_type == "moa":
        ensemble = MixtureOfAgents(agents)
    else:
        raise ValueError(f"Invalid multi-agent type: {args.multi_agent_type!r}.")

    await ensemble.update_data(ctx.task, task_sample.to_dict())
    mcp_description = (
        "You might be able to interact with a server that contains the data as a pandas dataframe."
    )
    prompt = render_prompt(template, ctx, mcp_description)
    return await ensemble.invoke(prompt)


def _normalize_orchestration(payload: Any) -> dict:
    """Coerce an orchestration payload (dict or JSON-ish string) into a flat dict."""
    if isinstance(payload, dict):
        if "orchestration" in payload and isinstance(payload["orchestration"], dict):
            return payload["orchestration"]
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
    raise ValueError(f"Unsupported orchestration payload type: {type(payload)}.")


_REQUIRED_ORCH_KEYS = (
    "eywa",
    "setting",
    "model",
    "multi_agent_type",
    "foundation_model",
    "agents",
)


async def run_orchestration(
    task_sample: pd.Series,
    args,
    existing_orchestration: pd.DataFrame | None,
    worker_id: int = 0,
):
    """Plan a setting with an LLM (or reuse a cached plan) and execute the task.

    The planner emits JSON with the schema::

        {
          "eywa": true|false,
          "setting": "single-agent" | "multi-agent",
          "model": "<llm>" | null,
          "multi_agent_type": "debate" | "refine" | "moa" | null,
          "foundation_model": "chronos" | "tabpfn" | null,
          "agents": ["<type>:<llm>", ...]
        }
    """
    ctx = prepare_task_context(task_sample)
    planner_elapsed = 0.0
    planner_tokens = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }

    orchestration: dict | None = None
    if (
        existing_orchestration is not None
        and "description" in existing_orchestration.columns
        and ctx.description in existing_orchestration["description"].values
    ):
        cached_row = (
            existing_orchestration[
                existing_orchestration["description"] == ctx.description
            ]
            .iloc[-1]
            .to_dict()
        )
        orchestration = _normalize_orchestration(cached_row)

    if orchestration is None:
        planner = SingleAgent(model=args.model)
        planner_template = open(
            osp.join(prompt_dir, "orchestration_prompt.txt"), "r", encoding="utf-8"
        ).read()
        planner_prompt = planner_template.format(
            task_description=ctx.description,
            domain=ctx.domain,
            task_type=ctx.task,
            input_data=ctx.input_data,
        )
        planner_result, planner_tokens, planner_elapsed, _ = await traced_ainvoke(
            planner, planner_prompt, args.model
        )
        orchestration = _normalize_orchestration(planner_result)

    # Backward compatibility for older cached orchestrations.
    orchestration.setdefault("foundation_model", None)

    missing = [k for k in _REQUIRED_ORCH_KEYS if k not in orchestration]
    if missing:
        print(f"Error: invalid orchestration format (missing keys {missing}): {orchestration}")
        return None, None, None, None, None

    print(f"Orchestration: {orchestration}")

    sub_args = argparse.Namespace(**vars(args))
    sub_args.eywa = orchestration["eywa"]
    sub_args.foundation_model = orchestration["foundation_model"]

    if orchestration["setting"] == "single-agent":
        sub_args.setting = "single-agent"
        sub_args.model = orchestration["model"]
        result_text, token_info, elapsed, raw_response_dict = await run_single_agent(
            task_sample, sub_args, worker_id=worker_id
        )
    elif orchestration["setting"] == "multi-agent":
        sub_args.setting = "multi-agent"
        sub_args.multi_agent_type = orchestration["multi_agent_type"]
        sub_args.agents = orchestration["agents"]
        result_text, token_info, elapsed, raw_response_dict = await run_multi_agent(
            task_sample, sub_args, worker_id=worker_id
        )
    else:
        print(f"Error: unsupported orchestration setting: {orchestration['setting']!r}.")
        return None, None, None, None, None

    merged_tokens = planner_tokens.copy()
    if token_info is not None:
        for key in merged_tokens:
            merged_tokens[key] += int(token_info.get(key, 0))
    total_elapsed = planner_elapsed + (elapsed or 0.0)

    orchestration["description"] = ctx.description
    return result_text, merged_tokens, total_elapsed, raw_response_dict, orchestration


async def main_async(args, save_name: str) -> None:
    output_folder = (args.output_folder or "").strip()
    save_leaf = save_name + ("_eywa" if args.eywa else "")
    if output_folder == "" or output_folder.lower() == "default":
        save_dir = osp.join(exp_dir, save_leaf)
    else:
        save_dir = osp.join(exp_dir, output_folder, save_leaf)
    os.makedirs(save_dir, exist_ok=True)

    results_jsonl = osp.join(save_dir, "results.jsonl")
    raw_responses_jsonl = osp.join(save_dir, "raw_responses.jsonl")
    result_texts_jsonl = osp.join(save_dir, "result_texts.jsonl")
    orchestration_jsonl = osp.join(save_dir, "orchestration.jsonl")

    existing_orchestration: pd.DataFrame | None = None
    if (
        args.setting == "orchestration"
        and os.path.exists(orchestration_jsonl)
        and os.path.getsize(orchestration_jsonl) > 0
    ):
        existing_orchestration = pd.read_json(orchestration_jsonl, lines=True)

    completed_descriptions: set[str] = set()
    if os.path.exists(results_jsonl):
        completed_descriptions = set(
            pd.read_json(results_jsonl, lines=True)["description"].dropna().astype(str)
        )

    eywabench = pd.read_parquet(osp.join(benchmark_dir, f"{args.eywabench_name}.parquet"))
    end_idx = args.end_idx if args.end_idx != -1 else len(eywabench)
    eywabench = eywabench.iloc[args.start_idx - 1 : end_idx]
    num_tasks = len(eywabench)

    pending = [
        i
        for i in range(num_tasks)
        if str(eywabench.iloc[i]["description"]) not in completed_descriptions
    ]
    if completed_descriptions:
        print(
            f"Resume mode: skipping {num_tasks - len(pending)} completed tasks "
            f"(matched by description)."
        )

    if args.num_workers < 1:
        raise ValueError(f"--num_workers must be >= 1, got {args.num_workers}.")

    worker_queue: asyncio.Queue[int] = asyncio.Queue()
    for wid in range(args.num_workers):
        worker_queue.put_nowait(wid)

    async def process_task(i: int, task_sample: pd.Series):
        worker_id = await worker_queue.get()
        try:
            domain = task_sample.get("domain", "unknown")
            task = task_sample.get("task", "unknown")
            description = task_sample.get("description", f"task-{i + 1}")
            label = task_sample.get("label", None)
            task_name = f"{task}-{description}"

            print(f"\n{'=' * 60}")
            print(f"Task {i + 1}/{num_tasks} [worker {worker_id}]: {task_name}")
            print(f"Domain: {domain}, Task Type: {task}")
            print(f"{'=' * 60}\n")

            result_text, token_info, elapsed, raw_response_dict = None, None, None, None
            orchestration = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    if args.setting == "single-agent":
                        (
                            result_text,
                            token_info,
                            elapsed,
                            raw_response_dict,
                        ) = await run_single_agent(task_sample, args, worker_id=worker_id)
                    elif args.setting == "multi-agent":
                        (
                            result_text,
                            token_info,
                            elapsed,
                            raw_response_dict,
                        ) = await run_multi_agent(task_sample, args, worker_id=worker_id)
                    elif args.setting == "orchestration":
                        (
                            result_text,
                            token_info,
                            elapsed,
                            raw_response_dict,
                            orchestration,
                        ) = await run_orchestration(
                            task_sample, args, existing_orchestration, worker_id=worker_id
                        )
                    else:
                        raise ValueError(f"Setting {args.setting!r} not supported.")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"Got error: {e}")
                    retryable = (
                        "BadRequestError" in type(e).__name__
                        and "We could not parse the JSON body of your request." in str(e)
                    )
                    if retryable and attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raw_response_dict = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                    }
                    break

            utility = 0.0
            if result_text is None or token_info is None or elapsed is None:
                print("Inference failed after retries; recording utility as 0.")
            else:
                try:
                    utility = evaluate(task, label, result_text)
                    print(
                        f"Utility: {utility}, Elapsed: {elapsed:.2f}s, "
                        f"Total Tokens: {token_info['total_tokens']}"
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"Evaluation error: {e}")
                    utility = 0.0
                    token_info = None
                    elapsed = None

            record = {
                "task_name": task_name,
                "domain": domain,
                "task": task,
                "description": description,
                "utility": utility,
                "elapsed": elapsed,
                "token_info": token_info,
            }
            payload = (
                record,
                {"raw_response_dict": raw_response_dict},
                {"result_text": result_text},
            )
            if args.setting == "orchestration":
                return (*payload, orchestration)
            return payload
        except Exception as e:  # noqa: BLE001 - never let one task crash the run
            print(f"Fatal error in task {i + 1}: {e}")
            description = task_sample.get("description", f"task-{i + 1}")
            record = {
                "task_name": f"{task_sample.get('task', 'unknown')}-{description}",
                "domain": task_sample.get("domain", "unknown"),
                "task": task_sample.get("task", "unknown"),
                "description": description,
                "utility": 0.0,
                "elapsed": None,
                "token_info": None,
            }
            payload = (
                record,
                {
                    "raw_response_dict": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": 0,
                    }
                },
                {"result_text": None},
            )
            if args.setting == "orchestration":
                return (
                    *payload,
                    {
                        "description": description,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
            return payload
        finally:
            worker_queue.put_nowait(worker_id)

    futures = [
        asyncio.create_task(process_task(i, eywabench.iloc[i])) for i in pending
    ]

    def _append(path: str, row: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for future in asyncio.as_completed(futures):
        if args.setting == "orchestration":
            record, raw_row, text_row, orchestration_row = await future
            _append(orchestration_jsonl, orchestration_row)
        else:
            record, raw_row, text_row = await future
        _append(results_jsonl, record)
        _append(raw_responses_jsonl, raw_row)
        _append(result_texts_jsonl, text_row)

    reordered = reorder_experiment_outputs(
        benchmark_path=osp.join(benchmark_dir, f"{args.eywabench_name}.parquet"),
        results_path=results_jsonl,
        raw_responses_path=raw_responses_jsonl,
        result_texts_path=result_texts_jsonl,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    print(f"Reordered outputs by benchmark order: {reordered} records")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Eywa, baseline, or orchestration experiments.")
    parser.add_argument("--eywabench_name", type=str, default="eywabench",
                        help="Name of the benchmark parquet (under eywabench/).")
    parser.add_argument("--eywa", action="store_true",
                        help="Enable Eywa: route the task to the relevant FM via MCP.")
    parser.add_argument("--setting", type=str, default="single-agent",
                        choices=["single-agent", "multi-agent", "orchestration"])
    parser.add_argument("--model", type=str, default="gpt-5-nano",
                        choices=list(SUPPORTED_MODELS))
    parser.add_argument("--exp_name", type=str, default="test",
                        help="Free-form experiment name; used in the output folder name.")
    parser.add_argument("--output_folder", type=str, default="default",
                        help="Subfolder under experiments/ used to save this run.")
    parser.add_argument("--start_idx", type=int, default=1,
                        help="1-based index of the first task to run.")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Inclusive 1-based index of the last task (-1 means the last row).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of concurrent task workers.")
    parser.add_argument("--log_file_name", type=str, default="log.log",
                        help="Filename used inside logs/ for this run.")

    parser.add_argument("--multi_agent_type", type=str, default="debate",
                        choices=["debate", "refine", "moa"])
    parser.add_argument("--agents", nargs="+", type=str,
                        default=["base:gpt-5-nano", "base:gpt-5-nano"],
                        help='List of agent specs, e.g. "base:gpt-5-nano" or "eywa:gemini-2.5-flash".')

    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON config under eywa/configs/ that overrides CLI flags.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.config is not None:
        with open(osp.join(config_dir, args.config), "r", encoding="utf-8") as f:
            for key, value in json.load(f).items():
                setattr(args, key, value)

    if not getattr(args, "output_folder", None):
        args.output_folder = "default"

    save_name = f"{args.exp_name}_{args.model}_{args.setting}"
    if args.setting == "multi-agent":
        save_name += f"_{args.multi_agent_type}"
    args.log_file_name = save_name + ("_eywa" if args.eywa else "") + ".log"

    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(osp.join(log_dir, args.log_file_name))
    sys.stderr = sys.stdout

    asyncio.run(main_async(args, save_name))


if __name__ == "__main__":
    main()
