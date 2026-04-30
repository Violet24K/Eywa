"""Tiny helpers to identify the provider behind a model id."""


def is_openai_model(model: str) -> bool:
    return model.startswith("gpt-")


def is_gemini_model(model: str) -> bool:
    return model.startswith("gemini-")
