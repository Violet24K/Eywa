"""Prompt-template selection rules."""

import os.path as osp
from argparse import Namespace

from eywa.utils.model_provider import is_gemini_model
from eywa.utils.path import prompt_dir


def _uses_gemini(args: Namespace) -> bool:
    """Return ``True`` if any model that will run on this task is a Gemini model."""
    if args.setting == "single-agent":
        return is_gemini_model(args.model)
    if args.setting == "multi-agent":
        return any(is_gemini_model(spec.split(":", 1)[1]) for spec in args.agents)
    return False


def choose_prompt(task: str, args: Namespace) -> str:
    """Pick a prompt-template path for ``task`` under the current run config.

    Gemini models occasionally need a slightly different prompt phrasing;
    we route them to ``general_with_gemini.txt`` when relevant.
    """
    name = "general_with_gemini.txt" if _uses_gemini(args) else "general.txt"
    return osp.join(prompt_dir, name)
