"""Merge result rows from multiple experiment folders into a single set.

Each source folder is expected to contain three aligned jsonl files:
``results.jsonl``, ``result_texts.jsonl``, and ``raw_responses.jsonl``.
Rows are matched against the benchmark by ``(domain, task, description)``.

Example::

    python eywabench/merge_result.py \\
        --eywabench_name eywabench \\
        --result_paths experiments/run_a experiments/run_b \\
        --output_path experiments/run_merged
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_result_files(result_path: str) -> tuple[Path, Path, Path]:
    p = Path(result_path)
    base = p if p.is_dir() else p.parent
    return (
        base / "results.jsonl",
        base / "result_texts.jsonl",
        base / "raw_responses.jsonl",
    )


def merge_results(eywabench_path: str, result_paths: list[str], output_path: str) -> None:
    benchmark_df = pd.read_parquet(eywabench_path)
    index: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for source_path in result_paths:
        results_file, texts_file, raws_file = _resolve_result_files(source_path)
        for f in (results_file, texts_file, raws_file):
            if not f.exists():
                raise FileNotFoundError(
                    f"Missing required file {f}. "
                    "Expected results.jsonl, result_texts.jsonl, raw_responses.jsonl."
                )

        results_rows = _read_jsonl(results_file)
        texts_rows = _read_jsonl(texts_file)
        raws_rows = _read_jsonl(raws_file)

        n = min(len(results_rows), len(texts_rows), len(raws_rows))
        if n < max(len(results_rows), len(texts_rows), len(raws_rows)):
            print(
                f"Warning: length mismatch in {source_path}; "
                f"using first {n} aligned rows only."
            )

        for i in range(n):
            row = results_rows[i]
            domain = row.get("domain")
            task = row.get("task")
            description = row.get("description")
            if domain is None or task is None or description is None:
                print(
                    f"Warning: skipped row {i} in {source_path} "
                    "(missing domain/task/description)."
                )
                continue
            key = (str(domain), str(task), str(description))
            index[key].append(
                {
                    "source_path": str(source_path),
                    "result": row,
                    "result_text": texts_rows[i],
                    "raw_response": raws_rows[i],
                }
            )

    merged_results: list[dict] = []
    merged_texts: list[dict] = []
    merged_raws: list[dict] = []

    for i, row in benchmark_df.iterrows():
        key = (str(row["domain"]), str(row["task"]), str(row["description"]))
        matches = index.get(key, [])
        if not matches:
            print(
                f"Warning: no matched record for benchmark row {i}: "
                f"domain={key[0]}, task={key[1]}, description={key[2]}"
            )
            continue
        if len(matches) > 1:
            print(
                f"Warning: {len(matches)} matches for benchmark row {i}; "
                f"using {matches[0]['source_path']}."
            )
        chosen = matches[0]
        merged_results.append(chosen["result"])
        merged_texts.append(chosen["result_text"])
        merged_raws.append(chosen["raw_response"])

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "results.jsonl", merged_results)
    _write_jsonl(output_dir / "result_texts.jsonl", merged_texts)
    _write_jsonl(output_dir / "raw_responses.jsonl", merged_raws)

    print(
        f"Merge complete. benchmark_rows={len(benchmark_df)}, "
        f"merged_rows={len(merged_results)}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eywabench_name", type=str, required=True)
    parser.add_argument("--result_paths", nargs="+", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder where merged jsonl files will be written.")
    args = parser.parse_args()
    eywabench_path = os.path.join(_CURRENT_DIR, f"{args.eywabench_name}.parquet")
    merge_results(
        eywabench_path=eywabench_path,
        result_paths=args.result_paths,
        output_path=args.output_path,
    )
