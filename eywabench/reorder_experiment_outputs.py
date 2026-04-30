import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reorder_experiment_outputs(
    benchmark_path: str,
    results_path: str,
    raw_responses_path: str,
    result_texts_path: str,
    start_idx: int = 1,
    end_idx: int = -1,
):
    benchmark = pd.read_parquet(benchmark_path)
    end = end_idx if end_idx != -1 else len(benchmark)
    benchmark = benchmark.iloc[start_idx - 1 : end]
    expected_task_names = [
        f"{row['task']}-{row['description']}" for _, row in benchmark.iterrows()
    ]

    results = _read_jsonl(Path(results_path))
    raw_responses = _read_jsonl(Path(raw_responses_path))
    result_texts = _read_jsonl(Path(result_texts_path))

    n = min(len(results), len(raw_responses), len(result_texts))
    results = results[:n]
    raw_responses = raw_responses[:n]
    result_texts = result_texts[:n]

    index_by_task = defaultdict(deque)
    for i, row in enumerate(results):
        task_name = row.get("task_name")
        if task_name is not None:
            index_by_task[task_name].append(i)

    ordered_results = []
    ordered_raw = []
    ordered_texts = []

    for task_name in expected_task_names:
        if not index_by_task[task_name]:
            continue
        idx = index_by_task[task_name].popleft()
        ordered_results.append(results[idx])
        ordered_raw.append(raw_responses[idx])
        ordered_texts.append(result_texts[idx])

    _write_jsonl(Path(results_path), ordered_results)
    _write_jsonl(Path(raw_responses_path), ordered_raw)
    _write_jsonl(Path(result_texts_path), ordered_texts)

    return len(ordered_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--raw_responses_path", type=str, required=True)
    parser.add_argument("--result_texts_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()

    count = reorder_experiment_outputs(
        benchmark_path=args.benchmark_path,
        results_path=args.results_path,
        raw_responses_path=args.raw_responses_path,
        result_texts_path=args.result_texts_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    print(f"Reordered {count} records.")
