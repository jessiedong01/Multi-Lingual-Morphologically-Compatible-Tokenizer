"""
Quick regression test for the standalone morphology-encoder evaluation metrics.

This script builds a tiny multilingual corpus, trains the MorphologyEncoder
directly (without running the full tokenizer loop), and then reports the
intrinsic metrics defined in `tokenizer_core.morphology_eval`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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


def build_morphology_encoder() -> MorphologyEncoder:
    """Train a MorphologyEncoder on a miniature multilingual corpus."""
    paragraphs = [
        "Workers are running and walking through the old buildings.",
        "She talked about unhappiness and impossibility of reforms.",
        "Teachers teaching students remain hardworking professionals.",
        "Die Arbeiter laufen und gehen durch die alten Gebaeude.",
        "Sie sprach ueber Ungluecklichkeit und Unmoeglichkeit.",
        "Die Lehrer unterrichten fleissig die Studenten.",
        "Isciler kosuyorlar ve yuruyorlar binalarin icinden.",
        "Mutluluk ve mutsuzluk hakkinda uzun uzun konustu.",
        "Ogretmenler ogrencilere ders veriyorlar.",
    ]
    langs = ["en", "en", "en", "de", "de", "de", "tr", "tr", "tr"]
    paragraph_infos = [ParagraphInfo(text, lang) for text, lang in zip(paragraphs, langs)]

    # Build token occurrences with a simple whitespace/word tokenizer.
    occurrences = defaultdict(list)
    for idx, text in enumerate(paragraphs):
        for token in TOKEN_RE.findall(text.lower()):
            if token:
                occurrences[token].append((idx, 0))

    encoder = MorphologyEncoder()
    encoder.fit(paragraph_infos, occurrences, lambda i: paragraph_infos[i].lang)
    return encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone smoke test for morphology-encoder metrics."
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="Morphological classes (cross-equivalence keys) to evaluate.",
    )
    parser.add_argument(
        "--morph-only",
        action="store_true",
        help="Skip the evaluation metrics and only train the MorphologyEncoder.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save the JSON metrics payload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = build_morphology_encoder()
    if not encoder.token_vec:
        raise SystemExit("MorphologyEncoder training failed (no embeddings).")

    if args.morph_only:
        meta = {
            "token_count": len(encoder.token_vec),
            "feature_count": len(encoder.feat2id),
            "languages": sorted(encoder.lang_proto.keys()),
        }
        print(json.dumps(meta, indent=2))
        if args.json_out:
            args.json_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return

    class_keys = tuple(args.classes) if args.classes else DEFAULT_CLASSES
    result = evaluate_morphology_encoder(encoder, class_keys=class_keys)
    if not result.classes:
        raise SystemExit("No morphology classes had enough tokens for evaluation.")

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

