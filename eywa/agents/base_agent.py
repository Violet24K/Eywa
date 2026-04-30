"""Single-LLM agent and shared invocation helpers."""

import time
from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from eywa.utils.model_provider import is_gemini_model, is_openai_model
from eywa.utils.parse import parse_agent_response, serialize_langchain_response


def initialize_model(model: str):
    """Return a LangChain-compatible model handle for ``model``.

    OpenAI models are referenced by name; Gemini models go through
    ``ChatGoogleGenerativeAI``.
    """
    if is_openai_model(model):
        return model
    if is_gemini_model(model):
        return ChatGoogleGenerativeAI(model=model)
    raise ValueError(f"Model {model!r} is not supported.")


async def traced_ainvoke(
    agent: Any,
    prompt: str,
    model: str,
) -> tuple[str, dict[str, Any], float, dict[str, Any]]:
    """Invoke an agent and capture latency, token usage, and a serialized trace."""
    start = time.perf_counter()
    raw_response = await agent.invoke(prompt)
    elapsed = time.perf_counter() - start

    result_text, completion_tokens, prompt_tokens, total_tokens, reasoning_tokens = (
        parse_agent_response(model, raw_response)
    )
    token_info = {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    raw_response_dict = serialize_langchain_response(raw_response)
    return result_text, token_info, elapsed, raw_response_dict


class SingleAgent:
    """A plain LLM agent with no foundation-model tools attached."""

    def __init__(self, model: str):
        self.model = model
        self.agent = create_agent(model=initialize_model(model))

    async def invoke(self, prompt: str):
        return await self.agent.ainvoke({"messages": prompt})
