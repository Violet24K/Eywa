"""Aggregate per-task results into per-domain and overall summaries."""

import argparse
import os.path as osp
import sys
from os import path as _path

import pandas as pd

# When invoked directly as a script (not via ``python -m``), make sure the
# project root is importable so ``from eywa.utils.path ...`` resolves.
_root_dir = _path.dirname(_path.dirname(_path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from eywa.utils.path import benchmark_dir, exp_dir  # noqa: E402


# Canonical display order for benchmark domains.
EYWABENCH_DOMAINS = [
    "material",
    "energy",
    "space",
    "biology",
    "clinic",
    "drug",
    "economy",
    "business",
    "infrastructure",
]

_TOKEN_COLS = ("completion_tokens", "prompt_tokens", "total_tokens", "reasoning_tokens")


def _load_results(results_path: str) -> pd.DataFrame:
    df = pd.read_json(results_path, lines=True)
    if df.empty:
        return df

    token_info_df = pd.json_normalize(df["token_info"])
    for col in _TOKEN_COLS:
        df[col] = (
            pd.to_numeric(token_info_df[col], errors="coerce")
            if col in token_info_df.columns
            else 0
        )
    df["utility"] = pd.to_numeric(df["utility"], errors="coerce")
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    return df


def _summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_domain = (
        df.groupby("domain", dropna=False)
        .agg(
            n_tasks=("task_name", "count"),
            utility_mean=("utility", "mean"),
            utility_std=("utility", "std"),
            elapsed_mean_sec=("elapsed", "mean"),
            elapsed_sum_sec=("elapsed", "sum"),
            total_tokens_mean=("total_tokens", "mean"),
            total_tokens_sum=("total_tokens", "sum"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
            reasoning_tokens_mean=("reasoning_tokens", "mean"),
        )
        .reset_index()
    )
    domain_order = {d: i for i, d in enumerate(EYWABENCH_DOMAINS)}
    by_domain["_order"] = by_domain["domain"].map(domain_order).fillna(10_000)
    by_domain = by_domain.sort_values(by=["_order", "domain"]).drop(columns=["_order"])

    overall = pd.DataFrame(
        {
            "domain": ["ALL"],
            "n_tasks": [len(df)],
            "utility_mean": [df["utility"].mean()],
            "utility_std": [df["utility"].std()],
            "elapsed_mean_sec": [df["elapsed"].mean()],
            "elapsed_sum_sec": [df["elapsed"].sum()],
            "total_tokens_mean": [df["total_tokens"].mean()],
            "total_tokens_sum": [df["total_tokens"].sum()],
            "prompt_tokens_mean": [df["prompt_tokens"].mean()],
            "completion_tokens_mean": [df["completion_tokens"].mean()],
            "reasoning_tokens_mean": [df["reasoning_tokens"].mean()],
        }
    )
    return by_domain, overall


def main(args: argparse.Namespace) -> None:
    eval_root = osp.join(args.eval_folder, args.exp_name)
    # ``benchmark_dir`` is unused at runtime but kept around so callers can
    # double-check the benchmark path against ``--eywabench_name`` if needed.
    _ = osp.join(benchmark_dir, f"{args.eywabench_name}.parquet")
    results = _load_results(osp.join(eval_root, "results.jsonl"))
    if results.empty:
        print("No rows found in results.jsonl")
        return

    by_domain, overall = _summarize(results)
    print("\nPer-domain evaluation")
    print(by_domain.to_string(index=False))
    print("\nOverall")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eywabench_name", type=str, default="eywabench")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--eval_folder", type=str, default=exp_dir)
    main(parser.parse_args())