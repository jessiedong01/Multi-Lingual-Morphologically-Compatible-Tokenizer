"""
End-to-end WikiANN -> MorphologyEncoder benchmarking without the DP lattice.

Usage example (PowerShell):
    python scripts/run_wikiann_morph_eval.py --langs en de tr --samples-per-lang 1000 ^
        --json-out runs/wikiann_morph_eval.json --save-prefix runs/wikiann_corpus
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_wikiann_corpus  # noqa: E402
from tokenizer_core.linguistic_features import MorphologyEncoder  # noqa: E402
from tokenizer_core.morphology_eval import (  # noqa: E402
    evaluate_morphology_encoder,
    summarize_morphology_eval,
)
from tokenizer_core.utils import ParagraphInfo  # noqa: E402

DEFAULT_LANGS: Sequence[str] = ("en", "de", "tr")
DEFAULT_CLASSES: Sequence[str] = ("PROG", "PL", "NEG", "PAST", "COMP")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def build_occurrences(paragraphs: Iterable[str]) -> dict[str, list[tuple[int, int]]]:
    occurrences: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for idx, text in enumerate(paragraphs):
        for token in TOKEN_RE.findall(text.lower()):
            if token:
                occurrences[token].append((idx, 0))
    return occurrences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch WikiANN samples, train MorphologyEncoder, run intrinsic metrics.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(DEFAULT_LANGS),
        help="ISO codes for WikiANN languages (default: en de tr).",
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=1000,
        help="Number of WikiANN paragraphs to stream per language (default: 1000).",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="Morphology classes (cross-equivalence keys) to evaluate.",
    )
    parser.add_argument(
        "--morph-only",
        action="store_true",
        help="Skip intrinsic metrics; just train and report encoder stats.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save JSON metrics/metadata.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        help="If set, dump the fetched corpus to PREFIX.txt and PREFIX.langs for reuse.",
    )
    return parser.parse_args()


def maybe_save_corpus(prefix: Path, texts: Sequence[str], langs: Sequence[str]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = prefix.with_suffix(".txt")
    lang_path = prefix.with_suffix(".langs")
    txt_path.write_text("\n".join(texts), encoding="utf-8")
    lang_path.write_text("\n".join(langs), encoding="utf-8")
    print(f"[save] wrote {len(texts)} paragraphs to {txt_path}")
    print(f"[save] wrote {len(langs)} language tags to {lang_path}")


def main():
    args = parse_args()
    codes = {code: code for code in args.langs}
    texts, lang_tags = load_wikiann_corpus(codes, per_lang=args.samples_per_lang)
    if not texts:
        raise SystemExit("WikiANN corpus retrieval failed; no paragraphs loaded.")
    if args.save_prefix:
        maybe_save_corpus(args.save_prefix, texts, lang_tags)

    paragraph_infos = [ParagraphInfo(text, lang) for text, lang in zip(texts, lang_tags)]
    occurrences = build_occurrences(texts)

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


