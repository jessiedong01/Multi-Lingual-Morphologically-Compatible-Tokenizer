"""
Train and evaluate the MorphologyEncoder on an arbitrary corpus, without
running the full tokenizer/DP lattice. Useful for benchmarking embeddings
directly on WikiANN (or any other dataset) with UniSeg-derived affix cues.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.linguistic_features import MorphologyEncoder
from tokenizer_core.morphology_eval import (
    evaluate_morphology_encoder,
    summarize_morphology_eval,
)
from tokenizer_core.utils import ParagraphInfo

DEFAULT_CLASSES: Sequence[str] = ("PROG", "PL", "NEG", "PAST", "COMP")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def read_lines(path: Path, limit: int | None = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")
    lines = [line.strip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip()]
    if limit is not None:
        return lines[:limit]
    return lines


def read_langs(path: Path | None, n: int) -> List[str | None]:
    if path is None:
        return [None] * n
    langs = read_lines(path)
    if len(langs) != n:
        raise ValueError(f"Language file length ({len(langs)}) != corpus length ({n}).")
    return langs


def build_occurrences(paragraphs: Iterable[str]) -> dict[str, list[tuple[int, int]]]:
    occurrences: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for idx, text in enumerate(paragraphs):
        for token in TOKEN_RE.findall(text.lower()):
            if token:
                occurrences[token].append((idx, 0))
    return occurrences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone MorphologyEncoder trainer + evaluator."
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Text file with one paragraph per line.")
    parser.add_argument(
        "--langs",
        type=Path,
        help="Optional text file with one language tag per paragraph (ISO codes).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of paragraphs to load (for quick tests).",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="Morphological classes (UniSeg cross-equivalence keys) to evaluate.",
    )
    parser.add_argument(
        "--morph-only",
        action="store_true",
        help="Skip intrinsic metrics; just train the encoder and report stats.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save JSON metrics or encoder metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paragraphs = read_lines(args.corpus, args.limit)
    if not paragraphs:
        raise SystemExit("Corpus is empty after filtering blank lines.")
    langs = read_langs(args.langs, len(paragraphs))
    paragraph_infos = [ParagraphInfo(text, lang) for text, lang in zip(paragraphs, langs)]
    occurrences = build_occurrences(paragraphs)

    encoder = MorphologyEncoder()
    encoder.fit(paragraph_infos, occurrences, lambda i: paragraph_infos[i].lang)
    if not encoder.token_vec:
        raise SystemExit("MorphologyEncoder training produced no embeddings.")

    if args.morph_only:
        payload = {
            "token_count": len(encoder.token_vec),
            "feature_count": len(encoder.feat2id),
            "languages": sorted(encoder.lang_proto.keys()),
        }
        print(json.dumps(payload, indent=2))
        if args.json_out:
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    class_keys = tuple(args.classes) if args.classes else DEFAULT_CLASSES
    result = evaluate_morphology_encoder(encoder, class_keys=class_keys)
    if not result.classes:
        raise SystemExit("No morphology classes had enough members for metrics.")

    payload = {
        "random_similarity": result.random_similarity,
        "length_correlation": result.length_correlation,
        "classes": {k: asdict(v) for k, v in result.classes.items()},
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + summarize_morphology_eval(result))


if __name__ == "__main__":
    main()


