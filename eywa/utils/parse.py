"""LangChain response parsing and serialization helpers."""

from typing import Any

from eywa.utils.model_provider import is_gemini_model, is_openai_model


def serialize_langchain_response(raw_response: Any) -> dict:
    """Convert a LangChain agent response into a JSON-serializable dict.

    LangChain message objects are first turned into plain dicts using
    ``.dict()`` / ``.model_dump()`` when available, with a string fallback.
    """
    try:
        if isinstance(raw_response, dict) and "messages" in raw_response:
            messages: list[dict] = []
            for msg in raw_response["messages"]:
                if hasattr(msg, "model_dump"):
                    messages.append(msg.model_dump())
                elif hasattr(msg, "dict"):
                    messages.append(msg.dict())
                elif isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({"content": str(msg), "type": type(msg).__name__})
            return {"messages": messages}

        if hasattr(raw_response, "model_dump"):
            return raw_response.model_dump()
        if hasattr(raw_response, "dict"):
            return raw_response.dict()
        if isinstance(raw_response, dict):
            return dict(raw_response)
        return {"raw": str(raw_response)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Serialization failed: {e}", "raw": str(raw_response)}


def parse_agent_response(
    model: str, response: dict[str, Any]
) -> tuple[str, int, int, int, int]:
    """Extract ``(text, completion, prompt, total, reasoning)`` token counts.

    Both OpenAI and Gemini messages are supported.
    """
    messages = response["messages"]
    last = messages[-1]
    text = last.content

    if is_openai_model(model):
        usage = last.response_metadata["token_usage"]
        return (
            text,
            usage["completion_tokens"],
            usage["prompt_tokens"],
            usage["total_tokens"],
            usage["completion_tokens_details"]["reasoning_tokens"],
        )
    if is_gemini_model(model):
        usage = last.usage_metadata
        return (
            text,
            usage["output_tokens"],
            usage["input_tokens"],
            usage["total_tokens"],
            usage["output_token_details"]["reasoning"],
        )
    raise ValueError(f"Model {model!r} not supported by parser.")
