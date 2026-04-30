"""Soft-match scoring for short free-form answers (deep principle / MMLU).

The scorer normalizes whitespace and common unicode look-alikes, then tries
three matching strategies in order:

1. **Exact normalized match** -> score 1.
2. **Numeric relative error** when both sides parse as a single number ->
   ``exp(-rel_error)``, smoothly decaying in [0, 1].
3. **Text fallback** -> a weighted combination of token F1 and character
   similarity, capped at ``text_fallback_cap`` so a soft text match can
   never tie an exact one.
"""

import difflib
import math
import re
from collections import Counter
from typing import Any


SOFT_SCORE_CONFIG: dict[str, float] = {
    "text_fallback_cap": 0.8,
    "token_weight": 0.6,
    "char_weight": 0.4,
}


def normalize_text(s: str) -> str:
    """Strip wrappers, collapse whitespace, and normalize a couple of unicode variants."""
    if not s:
        return ""
    text = s.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip("`\"' \n\t")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("−", "-").replace("’", "'")
    return text


def tokenize_for_similarity(text: str) -> list[str]:
    """Tokenizer that keeps math-ish symbols so formula structure survives."""
    return re.findall(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z0-9]", text)


def token_f1(pred: str, gold: str) -> tuple[float, dict[str, float | int]]:
    """Bag-of-tokens F1 with overlap statistics."""
    pred_tokens = tokenize_for_similarity(pred)
    gold_tokens = tokenize_for_similarity(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0, {
            "pred_tokens": 0,
            "gold_tokens": 0,
            "overlap_tokens": 0,
            "precision": 1.0,
            "recall": 1.0,
        }
    if not pred_tokens or not gold_tokens:
        return 0.0, {
            "pred_tokens": len(pred_tokens),
            "gold_tokens": len(gold_tokens),
            "overlap_tokens": 0,
            "precision": 0.0,
            "recall": 0.0,
        }

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum(min(pred_counter[t], gold_counter[t]) for t in pred_counter)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0, {
            "pred_tokens": len(pred_tokens),
            "gold_tokens": len(gold_tokens),
            "overlap_tokens": overlap,
            "precision": 0.0,
            "recall": 0.0,
        }
    f1 = 2 * precision * recall / (precision + recall)
    return f1, {
        "pred_tokens": len(pred_tokens),
        "gold_tokens": len(gold_tokens),
        "overlap_tokens": overlap,
        "precision": precision,
        "recall": recall,
    }


def parse_single_number(text: str) -> float | None:
    """Parse ``text`` as a single floating-point number, or return ``None``."""
    s = text.strip()
    if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def compute_soft_score(
    pred: str,
    gold: str,
    text_fallback_cap: float,
    token_weight: float,
    char_weight: float,
) -> tuple[float, dict[str, Any]]:
    """Score a prediction against a gold answer, returning ``(score, detail)``."""
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    detail: dict[str, Any] = {
        "pred_normalized": pred_norm,
        "gold_normalized": gold_norm,
        "strategy": "",
        "exact_match": False,
    }

    if pred_norm == gold_norm:
        detail.update(
            strategy="exact_normalized",
            exact_match=True,
            raw_similarity=1.0,
            final_score=1.0,
        )
        return 1.0, detail

    pred_num = parse_single_number(pred_norm)
    gold_num = parse_single_number(gold_norm)
    if pred_num is not None and gold_num is not None:
        denom = max(abs(gold_num), 1e-12)
        rel_err = abs(pred_num - gold_num) / denom
        numeric_score = math.exp(-rel_err)
        detail.update(
            strategy="numeric_relative_error",
            numeric_pred=pred_num,
            numeric_gold=gold_num,
            numeric_rel_error=rel_err,
            raw_similarity=numeric_score,
            final_score=numeric_score,
        )
        return numeric_score, detail

    tok_f1, tok_stats = token_f1(pred_norm, gold_norm)
    char_sim = difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()
    weighted_raw = token_weight * tok_f1 + char_weight * char_sim
    final_score = min(text_fallback_cap, weighted_raw)
    detail.update(
        strategy="text_fallback",
        token_f1=tok_f1,
        char_similarity=char_sim,
        text_fallback_cap=text_fallback_cap,
        weighted_raw_similarity=weighted_raw,
        final_score=final_score,
        token_stats=tok_stats,
    )
    return final_score, detail


def eval_deep_principle(label: str, result_text: str) -> float:
    """Public entry point used by the experiment runner."""
    score, _ = compute_soft_score(
        pred=result_text,
        gold=label,
        text_fallback_cap=SOFT_SCORE_CONFIG["text_fallback_cap"],
        token_weight=SOFT_SCORE_CONFIG["token_weight"],
        char_weight=SOFT_SCORE_CONFIG["char_weight"],
    )
    return score
