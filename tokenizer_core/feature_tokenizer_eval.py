"""
Linguistic feature benchmark utilities for ScalableTokenizer.

This module replaces the old translation-centric evaluation with a probe-driven
workflow that checks how well a tokenizer keeps linguistically meaningful spans
intact (lexicon entries, morphology cues, affixes, etc.).  It can be used as:

1. A standalone CLI that loads a saved tokenizer JSON and a JSONL list of
   feature probes, then writes an evaluation report.
2. A library module imported from tests or notebooks to score multiple
   tokenizer variants against the same probe set.

Each probe describes:
    - `text`: raw string containing the span of interest.
    - `language`: language code used when tokenizing.
    - `span`: substring that should map cleanly to tokens.
    - `feature`: label (e.g., "lexicon", "suffix", "gazetteer").
    - `expectation`: optional behaviour hint (currently "single_token" or
      "multi_token").

Metrics focus on span preservation (tokens per span, single-token accuracy,
feature-wise hit rates) and link back to the morphology metadata defined in
`constants.CROSS_EQUIV`.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constants import CROSS_EQUIV
from .tokenizer import ScalableTokenizer

SEGMENT_RE = re.compile(r"\S+|\s+")


@dataclass(frozen=True)
class FeatureProbe:
    """A single linguistic feature expectation over a text span."""

    text: str
    language: str
    span: str
    feature: str = "lexicon"
    expectation: str = "single_token"
    case_sensitive: bool = False
    metadata: Optional[Dict[str, object]] = None

    def normalized_expectation(self) -> str:
        return (self.expectation or "single_token").lower()


def load_feature_probes(
    path: str | Path,
    *,
    limit: Optional[int] = None,
    default_language: Optional[str] = None,
) -> List[FeatureProbe]:
    """
    Load feature probes from a JSONL file.

    Each line should contain at least: `text`, `span`. Optional fields:
        - `language`
        - `feature`
        - `expectation`
        - `case_sensitive`
        - `metadata` (any JSON object)
    """

    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return False

    probes: List[FeatureProbe] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if limit is not None and len(probes) >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            data = json.loads(raw)
            text = data.get("text")
            span = data.get("span") or data.get("substring")
            if not text or not span:
                continue
            language = data.get("language") or default_language
            if not language:
                continue
            probe = FeatureProbe(
                text=text,
                language=language,
                span=span,
                feature=data.get("feature", "lexicon"),
                expectation=data.get("expectation", "single_token"),
                case_sensitive=_coerce_bool(data.get("case_sensitive", False)),
                metadata=data.get("metadata"),
            )
            probes.append(probe)
    return probes


def _align_token_spans(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Map each token to a (start, end) character span in `text`."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        if not tok:
            spans.append((cursor, cursor))
            continue
        start = text.find(tok, cursor)
        if start == -1:
            # fall back to first occurrence
            start = text.find(tok)
            if start == -1:
                spans.append((cursor, cursor))
                continue
        end = start + len(tok)
        spans.append((start, end))
        cursor = max(end, cursor)
    return spans


def _locate_span(text: str, span: str, *, case_sensitive: bool) -> Optional[Tuple[int, int]]:
    """Return the first occurrence of `span` in `text`."""
    if case_sensitive:
        idx = text.find(span)
    else:
        idx = text.lower().find(span.lower())
    if idx == -1:
        return None
    return idx, idx + len(span)


def _tokens_covering_span(
    spans: Sequence[Tuple[int, int]], tokens: Sequence[str], span_range: Tuple[int, int]
) -> Tuple[List[str], int]:
    """Return the tokens and count overlapping the requested span."""
    start, end = span_range
    collected: List[str] = []
    for (tok_start, tok_end), tok in zip(spans, tokens):
        if tok_end <= start:
            continue
        if tok_start >= end:
            break
        collected.append(tok)
    return collected, len(collected)


def _morph_classes(tokens: Sequence[str], lang: str) -> List[str]:
    classes: List[str] = []
    for class_key, lang_map in CROSS_EQUIV.items():
        suffixes = lang_map.get(lang)
        if not suffixes:
            continue
        if any(tok.endswith(suf) for tok in tokens for suf in suffixes):
            classes.append(class_key)
    return classes


def evaluate_tokenizer_features(
    tokenizer: ScalableTokenizer,
    probes: Sequence[FeatureProbe],
) -> Dict[str, object]:
    """Evaluate a tokenizer on the requested set of feature probes."""
    per_probe: List[Dict[str, object]] = []
    feature_stats: Dict[str, Dict[str, float]] = {}

    total_tokens_in_spans = 0
    total_matches = 0
    single_hits = 0

    for probe in probes:
        tokens = tokenizer.tokenize(probe.text, lang=probe.language)
        spans = _align_token_spans(probe.text, tokens)
        span_range = _locate_span(probe.text, probe.span, case_sensitive=probe.case_sensitive)
        matched = span_range is not None
        tokens_in_span: List[str] = []
        token_count = 0
        if matched:
            tokens_in_span, token_count = _tokens_covering_span(spans, tokens, span_range)  # type: ignore[arg-type]
        kept_single = matched and token_count == 1
        morph_classes = _morph_classes(tokens_in_span or tokens, probe.language)

        expectation = probe.normalized_expectation()
        expectation_met = False
        if expectation == "single_token":
            expectation_met = kept_single
        elif expectation == "multi_token":
            expectation_met = matched and token_count >= 2
        else:
            expectation_met = matched

        per_probe.append(
            {
                "feature": probe.feature,
                "language": probe.language,
                "span": probe.span,
                "matched": matched,
                "tokens_in_span": tokens_in_span,
                "token_count": token_count,
                "kept_single": kept_single,
                "expectation": expectation,
                "expectation_met": expectation_met,
                "morph_classes": morph_classes,
                "metadata": probe.metadata,
            }
        )

        stats = feature_stats.setdefault(
            probe.feature,
            {"total": 0.0, "matched": 0.0, "single_hits": 0.0, "token_sum": 0.0},
        )
        stats["total"] += 1
        stats["matched"] += float(matched)
        stats["token_sum"] += token_count if matched else 0.0
        if expectation == "single_token" and kept_single:
            stats["single_hits"] += 1

        if matched:
            total_matches += 1
            total_tokens_in_spans += token_count
        if expectation == "single_token" and kept_single:
            single_hits += 1

    aggregate = {
        "probes": len(probes),
        "matched_spans": total_matches,
        "match_rate": (total_matches / len(probes)) if probes else 0.0,
        "avg_tokens_per_span": (total_tokens_in_spans / total_matches) if total_matches else 0.0,
        "single_token_accuracy": (single_hits / len(probes)) if probes else 0.0,
        "feature_breakdown": {
            feat: {
                "total": stats["total"],
                "match_rate": stats["matched"] / stats["total"] if stats["total"] else 0.0,
                "avg_tokens_per_span": (stats["token_sum"] / stats["matched"]) if stats["matched"] else 0.0,
                "single_token_rate": (stats["single_hits"] / stats["total"]) if stats["total"] else 0.0,
            }
            for feat, stats in feature_stats.items()
        },
    }
    return {"aggregate": aggregate, "per_probe": per_probe}


def evaluate_tokenizers(
    tokenizers: Mapping[str, ScalableTokenizer],
    probes: Sequence[FeatureProbe],
) -> Dict[str, Dict[str, object]]:
    """Evaluate multiple tokenizers on the same probe set."""
    return {label: evaluate_tokenizer_features(tok, probes) for label, tok in tokenizers.items()}


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ScalableTokenizer on linguistic feature probes."
    )
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json produced by training.")
    parser.add_argument("--probes", required=True, help="JSONL file with feature probes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of probes to load.")
    parser.add_argument(
        "--default-language",
        default=None,
        help="Fallback language code when a probe omits one.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional output path for the JSON report (stdout otherwise).",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    tokenizer = ScalableTokenizer.load_from_file(args.tokenizer)
    probes = load_feature_probes(
        args.probes,
        limit=args.limit,
        default_language=args.default_language,
    )
    report = evaluate_tokenizer_features(tokenizer, probes)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(output, encoding="utf-8")
        print(f"Wrote evaluation report to {args.report}")
    else:
        print(output)


__all__ = [
    "FeatureProbe",
    "load_feature_probes",
    "evaluate_tokenizer_features",
    "evaluate_tokenizers",
]


if __name__ == "__main__":
    main()
