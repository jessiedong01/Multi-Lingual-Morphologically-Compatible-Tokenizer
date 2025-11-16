#!/usr/bin/env python
"""Extract affix and cross-equivalence features from UniSeg datasets.

The script scans the UniSeg directory and builds two JSON files:

* data/uniseg_affixes.json        -> {lang: {"pre": [...], "suf": [...]}}
* data/uniseg_cross_equiv.json    -> {morph_type: {lang: [...]}}

Usage:
    python scripts/extract_uniseg_features.py \
        --output-dir data \
        --min-count 5 \
        --uniseg-root "path/to/uniseg"

If --uniseg-root is omitted we fall back to segmentation_eval.DEFAULT_UNISEG_ROOT.
"""

from __future__ import annotations

import argparse
import json
import sys
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer_core.segmentation_eval import DEFAULT_UNISEG_LANG_MAP, _resolve_uniseg_root

# Types recognised as affixes in UniSeg metadata
PREFIX_KEYS = {"prefix", "pref", "pre"}
SUFFIX_KEYS = {"suffix", "suf", "sfx"}
IGNORE_TYPES = {"root", "stem", "base", "word"}


def _canonical_lang(dataset_name: str) -> str | None:
    """Return the two-letter language code for a UniSeg dataset folder."""
    for lang, datasets in DEFAULT_UNISEG_LANG_MAP.items():
        if dataset_name in datasets:
            return lang
    # Fall back to heuristic (first two letters) if not in map
    short = dataset_name.split("-")[0]
    if len(short) >= 2:
        return short[:2]
    return None


def _extract_segments(word: str, segment_meta: dict) -> Tuple[str, int, int, str] | None:
    morph_type = (segment_meta.get("type") or "").strip()
    span = segment_meta.get("span")
    text = ""
    start = end = None
    if isinstance(span, list) and span:
        indices = sorted(int(i) for i in span)
        start = indices[0]
        end = indices[-1] + 1
        try:
            text = word[start:end]
        except IndexError:
            text = ""
    else:
        morpheme = segment_meta.get("morpheme") or ""
        if morpheme:
            idx = word.find(morpheme)
            if idx != -1:
                start = idx
                end = idx + len(morpheme)
                text = morpheme
    if not text or start is None or end is None:
        return None
    return text, start, end, morph_type


def gather_features(
    root: Path,
    min_count: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    affix_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: {"pre": Counter(), "suf": Counter()})
    cross_counts: DefaultDict[str, DefaultDict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        lang = _canonical_lang(dataset_dir.name)
        if not lang:
            continue
        files = list(dataset_dir.glob("*.useg")) + list(dataset_dir.glob("*.useg.gz"))
        for file in files:
            if not file.is_file():
                continue
            opener = gzip.open if file.suffix == ".gz" else open
            with opener(file, "rt", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    word = parts[0]
                    try:
                        meta = json.loads(parts[-1])
                    except json.JSONDecodeError:
                        continue
                    segments_meta = meta.get("segmentation") or []
                    for segment_meta in segments_meta:
                        extracted = _extract_segments(word, segment_meta)
                        if not extracted:
                            continue
                        text, start, end, morph_type = extracted
                        morph_type_lower = morph_type.lower()
                        if morph_type_lower in PREFIX_KEYS or start == 0:
                            affix_counts[lang]["pre"][text] += 1
                        elif morph_type_lower in SUFFIX_KEYS or end == len(word):
                            affix_counts[lang]["suf"][text] += 1
                        elif morph_type_lower in IGNORE_TYPES:
                            continue
                        else:
                            key = morph_type.strip().upper().replace(" ", "_") or "MISC"
                            cross_counts[key][lang][text] += 1

    affixes: Dict[str, Dict[str, List[str]]] = {}
    for lang, buckets in affix_counts.items():
        affixes[lang] = {}
        for bucket, counter in buckets.items():
            filtered = [tok for tok, count in counter.most_common() if count >= min_count]
            affixes[lang][bucket] = filtered

    cross_equiv: Dict[str, Dict[str, List[str]]] = {}
    for morph_type, lang_map in cross_counts.items():
        cross_equiv[morph_type] = {}
        for lang, counter in lang_map.items():
            filtered = [tok for tok, count in counter.most_common() if count >= min_count]
            if filtered:
                cross_equiv[morph_type][lang] = filtered
    return affixes, cross_equiv


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UniSeg affixes and cross-equivalence mappings.")
    parser.add_argument("--uniseg-root", type=str, default=None, help="Path to UniSegments root directory")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for JSON files")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum frequency threshold for keeping morphemes")
    args = parser.parse_args()

    root = _resolve_uniseg_root(args.uniseg_root)
    affixes, cross_equiv = gather_features(root, args.min_count)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    affix_path = output_dir / "uniseg_affixes.json"
    cross_path = output_dir / "uniseg_cross_equiv.json"

    affix_path.write_text(json.dumps(affixes, ensure_ascii=False, indent=2), encoding="utf-8")
    cross_path.write_text(json.dumps(cross_equiv, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {affix_path}")
    print(f"Wrote {cross_path}")


if __name__ == "__main__":
    main()
