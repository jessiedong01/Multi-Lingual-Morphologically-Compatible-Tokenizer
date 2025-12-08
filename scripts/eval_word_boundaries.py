#!/usr/bin/env python3
"""
Evaluate boundary-level segmentation metrics for a CSV of tokenizer outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


def _normalize(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii").lower()


def _load_csv(path: Path, limit: int | None = None) -> List[Mapping[str, str]]:
    rows: List[Mapping[str, str]] = []
    with path.open(encoding="cp1252", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rows.append(row)
            if limit is not None and idx + 1 >= limit:
                break
    return rows


def _load_gold_entries(root: Path, targets: Sequence[str]) -> Dict[str, Mapping[str, object]]:
    target_keys = {_normalize(word): word for word in targets}
    gold: Dict[str, Mapping[str, object]] = {}
    for json_path in sorted(root.rglob("*.jsonl")):
        with json_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = _normalize(record["word"])
                if key in target_keys and key not in gold:
                    gold[key] = record
            if len(gold) == len(target_keys):
                break
    return gold


def _token_boundaries(word: str, tokens: Sequence[str]) -> set[int]:
    bounds: set[int] = set()
    cursor = 0
    for tok in tokens[:-1]:
        cursor += len(tok)
        bounds.add(cursor)
    return bounds


def evaluate(csv_path: Path, uniseg_root: Path, limit: int | None) -> List[Tuple[str, str, float | None, float | None, float | None]]:
    rows = _load_csv(csv_path, limit=limit)
    gold = _load_gold_entries(uniseg_root, [row["word"] for row in rows])
    results: List[Tuple[str, str, float | None, float | None, float | None]] = []
    for row in rows:
        word = row["word"]
        tokens = json.loads(row["token_pieces"])
        cand_bounds = _token_boundaries(word, tokens)
        entry = gold.get(_normalize(word))
        if not entry:
            results.append((word, row["model"], None, None, None))
            continue
        gold_bounds = set(int(b) for b in entry.get("boundaries", []))
        tp = len(cand_bounds & gold_bounds)
        precision = tp / len(cand_bounds) if cand_bounds else 1.0
        recall = tp / len(gold_bounds) if gold_bounds else 1.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        results.append((word, row["model"], precision, recall, f1))
    return results


def save_results(results: Sequence[Tuple[str, str, float | None, float | None, float | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", "model", "precision", "recall", "f1"])
        for word, model, precision, recall, f1 in results:
            if precision is None:
                writer.writerow([word, model, "MISSING", "MISSING", "MISSING"])
            else:
                writer.writerow([word, model, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer outputs against UniSegments gold boundaries.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV file with columns word, model, token_pieces.")
    parser.add_argument("--uniseg-root", type=Path, required=True, help="Root directory containing normalized UniSeg JSONL files.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save evaluation CSV results.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of CSV rows to evaluate.")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file {args.csv} does not exist.")
    if not args.uniseg_root.exists():
        raise FileNotFoundError(f"UniSeg root {args.uniseg_root} does not exist.")

    results = evaluate(args.csv, args.uniseg_root, args.limit)
    save_results(results, args.output)
    print(f"Wrote evaluation results to {args.output}")


if __name__ == "__main__":
    main()
