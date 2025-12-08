#!/usr/bin/env python3
"""
Normalize UniSegments .useg files into JSONL trees for downstream evaluation.

This script walks the UniSegments release tree, parses each .useg file using
the same loader that backs tokenizer_core.segmentation_eval, and emits a
language-organized directory of lightweight JSONL files:

    {output_root}/{lang}/{dataset}.jsonl

Each line contains:
    {
      "word": "...",
      "boundaries": [ ... ],
      "segments": [{"start": i, "end": j, "type": "..."}],
      "morpheme_types": [...],
      "dataset": "...",
      "source": "relative/path/to/.useg"
    }

An index.json summary is also produced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import json
import re


def _parse_file_metadata(path: Path) -> Tuple[str, str]:
    """Infer language code and dataset name from UniSegments filename."""
    name = path.name
    parts = name.split("-")
    if len(parts) >= 4:
        lang = parts[2]
        dataset = "-".join(parts[3:]).rsplit(".", 1)[0]
    else:
        lang = "unknown"
        dataset = name.rsplit(".", 1)[0]
    return lang, dataset


def _serialize_entry(entry, dataset: str, source: Path) -> Dict[str, object]:
    return {
        "word": entry["word"],
        "boundaries": entry["boundaries"],
        "morpheme_types": entry["morpheme_types"],
        "segments": entry["segments"],
        "dataset": dataset,
        "source": source.as_posix(),
    }


def _normalize_type(raw) -> str:
    if isinstance(raw, str):
        norm = raw.strip()
        return norm or "morpheme"
    if isinstance(raw, (list, tuple)):
        pieces = [str(part).strip() for part in raw if str(part).strip()]
        return " ".join(pieces) if pieces else "morpheme"
    return "morpheme"


def _segments_from_meta(word: str, segments_meta: list) -> list[Tuple[int, int, str]]:
    segments: list[Tuple[int, int, str]] = []
    for segment_meta in segments_meta or []:
        span = segment_meta.get("span")
        start = end = None
        if isinstance(span, list) and span:
            indices = sorted(int(i) for i in span)
            start = indices[0]
            end = indices[-1] + 1
        elif segment_meta.get("morpheme"):
            text = str(segment_meta["morpheme"])
            idx = word.find(text)
            if idx != -1:
                start = idx
                end = idx + len(text)
        if start is None or end is None:
            continue
        morph_type = _normalize_type(segment_meta.get("type"))
        segments.append((start, end, morph_type))
    segments.sort(key=lambda triple: triple[0])
    return segments


def _covers_word(segments: list[Tuple[int, int, str]], word_len: int) -> bool:
    if not segments:
        return False
    prev = 0
    for start, end, _ in segments:
        if start != prev:
            return False
        prev = end
    return prev == word_len


def _segments_from_morph_string(word: str, morph_str: str) -> list[Tuple[int, int, str]]:
    parts = [piece.strip() for piece in re.split(r"\+", morph_str or "") if piece.strip()]
    if not parts:
        return []
    segments: list[Tuple[int, int]] = []
    cursor = 0
    for piece in parts:
        idx = word.find(piece, cursor)
        if idx == -1:
            idx = word.find(piece)
            if idx == -1:
                idx = cursor
        start = idx
        end = idx + len(piece)
        segments.append((start, end))
        cursor = end
    types: list[str] = []
    n = len(parts)
    for i in range(n):
        if n == 1:
            types.append("stem")
        elif i == 0:
            types.append("prefix")
        elif i == n - 1:
            types.append("suffix")
        else:
            types.append("stem")
    return [(start, end, morph_type) for (start, end), morph_type in zip(segments, types)]


def _iter_uniseg_entries(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            word = parts[0]
            morph_str = parts[3] if len(parts) >= 4 else ""
            try:
                meta = json.loads(parts[-1])
            except json.JSONDecodeError:
                continue
            segments = _segments_from_meta(word, meta.get("segmentation") or [])
            if not _covers_word(segments, len(word)):
                fallback = _segments_from_morph_string(word, morph_str)
                if _covers_word(fallback, len(word)):
                    segments = fallback
            if not segments:
                continue
            boundaries = sorted(end for _, end, _ in segments[:-1])
            yield {
                "word": word,
                "boundaries": boundaries,
                "morpheme_types": [seg_type for _, _, seg_type in segments],
                "segments": [
                    {"start": start, "end": end, "type": seg_type}
                    for start, end, seg_type in segments
                ],
            }


def build_tree(uniseg_root: Path, output_root: Path) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"entries": 0, "datasets": {}})
    files = sorted(uniseg_root.rglob("*.useg"))
    if not files:
        raise FileNotFoundError(f"No .useg files found under {uniseg_root}")

    for path in files:
        lang, dataset = _parse_file_metadata(path)
        dest_dir = output_root / lang
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{dataset}.jsonl"
        count = 0
        rel_source = path.relative_to(uniseg_root)
        with dest_file.open("w", encoding="utf-8") as handle:
            for entry in _iter_uniseg_entries(path):
                record = _serialize_entry(entry, dataset, rel_source)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        summary[lang]["entries"] += count
        summary[lang]["datasets"][dataset] = count
        print(f"Wrote {count:>8} entries -> {dest_file}")
    return {
        lang: {"entries": data["entries"], "datasets": data["datasets"]}
        for lang, data in summary.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize UniSegments .useg files for evaluation.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dictionary_data_bases/UniSegments-1.0-public/data"),
        help="Root folder containing UniSegments .useg files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/uniseg_extracted"),
        help="Destination directory for normalized JSONL files.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"UniSegments root {args.root} does not exist.")

    summary = build_tree(args.root.resolve(), args.output.resolve())
    index_path = args.output / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary written to {index_path}")


if __name__ == "__main__":
    main()
