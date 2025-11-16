"""
Segmented evaluation utilities that prioritize UniSegments gold data and
fall back to FLORES translation pairs when UniSegments coverage is missing.

Typical usage:

    from tokenizer_core.tokenizer import ScalableTokenizer
    from tokenizer_core.segmentation_eval import (
        evaluate_languages_with_backoff,
        DEFAULT_UNISEG_LANG_MAP,
        DEFAULT_UNISEG_ROOT,
    )
    tokenizers = {"baseline": trained_tokenizer}
    flores_map = {"ja": {"path": "data/flores-ja-en.jsonl", "limit": 500}}
    results = evaluate_languages_with_backoff(
        tokenizers,
        ["en", "pl", "ja"],
        uniseg_root=DEFAULT_UNISEG_ROOT,  # auto-resolves to dictionary_data_bases if present
        flores_map=flores_map,
    )
"""

from __future__ import annotations

import inspect
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)


# Preferred UniSegments datasets per ISO-639-1 language code.
DEFAULT_UNISEG_LANG_MAP: Dict[str, List[str]] = {
    "en": ["eng-MorphoLex", "eng-MorphyNet"],
    "de": ["deu-MorphyNet", "deu-DerivBaseDE"],
    "pl": ["pol-MorphyNet"],
    "ru": ["rus-MorphyNet", "rus-DerivBaseRU"],
    "cs": ["ces-MorphyNet", "ces-DeriNet"],
    "fr": ["fra-MorphoLex", "fra-MorphyNet", "fra-Demonette"],
    "es": ["spa-MorphyNet"],
    "it": ["ita-MorphyNet", "ita-DerIvaTario"],
    "sv": ["swe-MorphyNet"],
    "fi": ["fin-MorphyNet"],
    "hu": ["hun-MorphyNet"],
    "pt": ["por-MorphyNet"],
    "hi": ["hin-KCIS"],
    "mr": ["mar-KCIS"],
    "ml": ["mal-KCIS"],
    "kn": ["kan-KCIS"],
    "bn": ["ben-KCIS"],
    "ta": ["tgk-Uniparser"],
    "tr": [],  # Not covered in UniSegments 1.0.
    "ja": [],
    "zh": [],
    "da": [],
    "tk": [],
}


@dataclass(frozen=True)
class UniSegEntry:
    word: str
    boundaries: Set[int]
    morpheme_types: List[str]
    segments: List[Tuple[int, int, str]]


@dataclass(frozen=True)
class ParallelPair:
    source: str
    target: str
    source_lang: str
    target_lang: str
    metadata: Optional[Dict[str, Any]] = None


class HasTokenize(Protocol):
    def tokenize(self, text: str, lang: Optional[str] = None) -> Sequence[str]:
        ...


TokenizerLike = Union[HasTokenize, Callable[..., Sequence[str]]]


PROJECT_ROOT = Path(__file__).resolve().parent
# Local cache where large linguistic resources live when synced alongside the repo.
DATA_ROOT = PROJECT_ROOT / "dictionary_data_bases"
DEFAULT_UNISEG_ROOT = (
    DATA_ROOT
    / "Universal Segmentations 1.0 (UniSegments 1.0)"
    / "UniSegments-1.0-public"
    / "UniSegments-1.0-public"
    / "data"
)
DEFAULT_FLORES_ROOT = DATA_ROOT / "flores"


def _extract_first(data: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _split_lang_pair(value: str) -> Optional[Tuple[str, str]]:
    if not value:
        return None
    for sep in ("->", "=>", ":", "|", "/", "-", ","):
        if sep in value:
            left, right = value.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    return None


def load_parallel_jsonl(
    path: str | Path,
    *,
    limit: Optional[int] = None,
    default_source_lang: Optional[str] = None,
    default_target_lang: Optional[str] = None,
) -> List[ParallelPair]:
    """
    Load translation pairs from a JSONL file.

    The loader tries to accommodate multiple field conventions used in FLORES-style
    exports. It looks for common aliases such as `source`/`target`, `src`/`tgt`,
    `sentence1`/`sentence2`, and optional language metadata:
        - `source_lang`, `src_lang`, `source_language`, `lang1`, `language1`
        - `target_lang`, `tgt_lang`, `target_language`, `lang2`, `language2`
        - `lang_pair`, `language_pair` (split on ->, :, /, -, |)
    """

    def _coerce_langs(record: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        src_lang = _extract_first(
            record,
            (
                "source_lang",
                "src_lang",
                "src_language",
                "source_language",
                "language_source",
                "lang_source",
                "lang1",
                "language1",
            ),
        )
        tgt_lang = _extract_first(
            record,
            (
                "target_lang",
                "tgt_lang",
                "tgt_language",
                "target_language",
                "language_target",
                "lang_target",
                "lang2",
                "language2",
            ),
        )
        if src_lang and tgt_lang:
            return src_lang, tgt_lang
        pair_field = _extract_first(record, ("lang_pair", "language_pair", "langs"))
        if isinstance(pair_field, str):
            split = _split_lang_pair(pair_field)
            if split:
                return split
        langs_field = record.get("langs")
        if isinstance(langs_field, (list, tuple)) and len(langs_field) >= 2:
            first = langs_field[0]
            second = langs_field[1]
            if isinstance(first, str) and isinstance(second, str):
                return first.strip(), second.strip()
        return src_lang, tgt_lang

    def _coerce_pair(record: Mapping[str, Any]) -> Optional[ParallelPair]:
        source_text = _extract_first(
            record,
            (
                "source",
                "src",
                "text_source",
                "sentence1",
                "source_sentence",
                "sourceText",
                "original",
            ),
        )
        target_text = _extract_first(
            record,
            (
                "target",
                "tgt",
                "text_target",
                "sentence2",
                "target_sentence",
                "translation",
                "translated",
            ),
        )
        if not source_text or not target_text:
            return None
        src_lang, tgt_lang = _coerce_langs(record)
        src_lang = src_lang or default_source_lang
        tgt_lang = tgt_lang or default_target_lang
        if not src_lang or not tgt_lang:
            return None
        metadata = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "source",
                "src",
                "text_source",
                "sentence1",
                "source_sentence",
                "sourceText",
                "original",
                "target",
                "tgt",
                "text_target",
                "sentence2",
                "target_sentence",
                "translation",
                "translated",
                "source_lang",
                "src_lang",
                "src_language",
                "source_language",
                "language_source",
                "lang_source",
                "target_lang",
                "tgt_lang",
                "tgt_language",
                "target_language",
                "language_target",
                "lang_target",
                "lang_pair",
                "language_pair",
                "langs",
                "lang1",
                "lang2",
                "language1",
                "language2",
            }
        }
        return ParallelPair(
            source=source_text,
            target=target_text,
            source_lang=src_lang,
            target_lang=tgt_lang,
            metadata=metadata or None,
        )

    pairs: List[ParallelPair] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(pairs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            pair = _coerce_pair(record)
            if pair:
                pairs.append(pair)
    return pairs


def _safe_signature(func: Callable[..., Any]) -> Optional[inspect.Signature]:
    try:
        return inspect.signature(func)
    except (TypeError, ValueError):
        return None


def _accepts_keyword(signature_obj: Optional[inspect.Signature], name: str) -> bool:
    if signature_obj is None:
        return False
    for parameter in signature_obj.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return True
    return False


def _accepts_positional(signature_obj: Optional[inspect.Signature], position_index: int) -> bool:
    if signature_obj is None:
        return False
    positional_count = 0
    for parameter in signature_obj.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional_count += 1
            if positional_count > position_index:
                return True
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
    return False


def _resolve_callable(tokenizer: TokenizerLike) -> Callable[..., Sequence[str]]:
    if hasattr(tokenizer, "tokenize") and callable(getattr(tokenizer, "tokenize")):
        return getattr(tokenizer, "tokenize")
    if callable(tokenizer):
        return tokenizer  # type: ignore[return-value]
    raise TypeError(f"Tokenizer {tokenizer!r} must provide a tokenize method or be callable.")


def _normalize_tokens(tokens: Sequence[Any] | Iterable[Any] | None) -> List[str]:
    if tokens is None:
        return []
    if isinstance(tokens, list):
        return [str(tok) for tok in tokens]
    if isinstance(tokens, tuple):
        return [str(tok) for tok in tokens]
    return [str(tok) for tok in list(tokens)]


def _tokenize_text(tokenizer: TokenizerLike, text: str, lang: Optional[str]) -> List[str]:
    fn = _resolve_callable(tokenizer)
    signature_obj = _safe_signature(fn)
    if lang is not None:
        if _accepts_keyword(signature_obj, "lang"):
            return _normalize_tokens(fn(text, lang=lang))
        if _accepts_positional(signature_obj, 1):
            return _normalize_tokens(fn(text, lang))
        # If we cannot inspect the signature, conservatively try both variants.
        for caller in (
            lambda: fn(text, lang=lang),
            lambda: fn(text, lang),
        ):
            try:
                return _normalize_tokens(caller())
            except TypeError as exc:
                if "argument" in str(exc):
                    continue
                raise
    return _normalize_tokens(fn(text))


def _resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    try:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    except OSError:
        pass
    return path


def _resolve_relative_path(raw: str | Path, search_roots: Sequence[Path]) -> Path:
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for root in search_roots:
        candidates.append(root / path)
    for candidate in candidates:
        candidate = _resolve_existing_path(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_uniseg_root(root: Optional[str | Path]) -> Path:
    if root is not None:
        return _resolve_existing_path(Path(root).expanduser())
    if DEFAULT_UNISEG_ROOT.exists():
        return DEFAULT_UNISEG_ROOT
    raise FileNotFoundError(
        "UniSegments root not provided and default path "
        f"{DEFAULT_UNISEG_ROOT} does not exist. "
        "Either download UniSegments or pass uniseg_root explicitly."
    )


def _resolve_flores_path(raw: str | Path) -> Path:
    search_roots = [DEFAULT_FLORES_ROOT, DATA_ROOT]
    return _resolve_relative_path(raw, search_roots)


def evaluate_tokenizers_on_pairs(
    tokenizers: Mapping[str, TokenizerLike],
    pairs: Sequence[ParallelPair],
) -> Dict[str, Dict[str, object]]:
    """Compute simple translation consistency stats for one or more tokenizers."""
    results: Dict[str, Dict[str, object]] = {}
    for label, tokenizer in tokenizers.items():
        per_pair: List[Dict[str, object]] = []
        src_counts: List[int] = []
        tgt_counts: List[int] = []
        abs_diffs: List[int] = []
        ratios: List[float] = []

        for pair in pairs:
            src_tokens = _tokenize_text(tokenizer, pair.source, pair.source_lang)
            tgt_tokens = _tokenize_text(tokenizer, pair.target, pair.target_lang)
            src_len = len(src_tokens)
            tgt_len = len(tgt_tokens)
            diff = abs(src_len - tgt_len)
            ratio = (src_len / tgt_len) if tgt_len else float("inf")

            per_pair.append(
                {
                    "source_lang": pair.source_lang,
                    "target_lang": pair.target_lang,
                    "source_tokens": src_tokens,
                    "target_tokens": tgt_tokens,
                    "token_diff": diff,
                    "length_ratio": ratio,
                    "metadata": pair.metadata,
                }
            )
            src_counts.append(src_len)
            tgt_counts.append(tgt_len)
            abs_diffs.append(diff)
            if tgt_len:
                ratios.append(ratio)

        total = len(per_pair)
        aggregate = {
            "pairs_evaluated": total,
            "avg_source_tokens": (sum(src_counts) / total) if total else 0.0,
            "avg_target_tokens": (sum(tgt_counts) / total) if total else 0.0,
            "avg_abs_token_diff": (sum(abs_diffs) / total) if total else 0.0,
            "median_abs_token_diff": statistics.median(abs_diffs) if abs_diffs else 0.0,
            "max_abs_token_diff": max(abs_diffs) if abs_diffs else 0,
            "near_match_rate": (sum(1 for diff in abs_diffs if diff <= 1) / total) if total else 0.0,
            "perfect_match_rate": (sum(1 for diff in abs_diffs if diff == 0) / total) if total else 0.0,
            "length_ratio_mean": (statistics.mean(ratios) if ratios else 0.0),
            "length_ratio_stdev": (statistics.pstdev(ratios) if len(ratios) > 1 else 0.0),
        }
        results[label] = {"aggregate": aggregate, "per_pair": per_pair}
    return results


def _uniseg_file_from_lang(
    root: Path,
    lang: str,
    lang_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> Optional[Path]:
    """Return the first matching UniSegments file for the language, if available."""
    lang_map = lang_map or DEFAULT_UNISEG_LANG_MAP
    candidates = lang_map.get(lang, [])
    for dataset in candidates:
        filename = root / dataset / f"UniSegments-1.0-{dataset}.useg"
        if filename.exists():
            return filename
    return None


def _iter_uniseg_entries(path: Path, limit: Optional[int] = None) -> Iterator[UniSegEntry]:
    """Yield UniSegments entries with precomputed gold boundaries."""
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            word = parts[0]
            try:
                meta = json.loads(parts[-1])
            except json.JSONDecodeError:
                continue
            segments = meta.get("segmentation") or []
            if not segments:
                continue
            lengths = []
            morpheme_types = []
            valid = True
            for seg in segments:
                span = seg.get("span")
                morpheme_types.append(seg.get("type") or "other")
                if isinstance(span, list) and span:
                    lengths.append(len(span))
                else:
                    morpheme = seg.get("morpheme") or ""
                    if not morpheme:
                        valid = False
                        break
                    lengths.append(len(morpheme))
            if not valid or not lengths:
                continue
            boundaries = set()
            cursor = 0
            segments = []
            for length, morph_type in zip(lengths, morpheme_types):
                start = cursor
                end = cursor + length
                segments.append((start, end, morph_type))
                cursor = end
            for start, end, _ in segments[:-1]:
                boundaries.add(end)
            yield UniSegEntry(
                word=word,
                boundaries=boundaries,
                morpheme_types=morpheme_types,
                segments=segments,
            )


def _token_boundaries(text: str, tokens: Sequence[str]) -> Optional[Set[int]]:
    """Compute character boundary positions implied by tokenizer output."""
    if not tokens:
        return None
    boundaries: Set[int] = set()
    cursor = 0
    text_len = len(text)
    for token in tokens[:-1]:
        if not token:
            continue
        # Attempt to align the token within the original text starting at cursor.
        pos = text.find(token, cursor)
        if pos == -1:
            pos = cursor
        cursor = min(pos + len(token), text_len)
        boundaries.add(cursor)
    return boundaries


def _align_token_spans(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        if not tok:
            spans.append((cursor, cursor))
            continue
        start = text.find(tok, cursor)
        if start == -1:
            start = text.find(tok)
            if start == -1:
                spans.append((cursor, cursor))
                continue
        end = start + len(tok)
        spans.append((start, end))
        cursor = max(end, cursor)
    return spans


def _boundary_metrics(
    gold: Iterable[Set[int]],
    preds: Iterable[Optional[Set[int]]],
) -> Dict[str, float]:
    """Aggregate precision/recall/F1 across all words."""
    tp = fp = fn = total = 0
    for gold_set, pred_set in zip(gold, preds):
        if pred_set is None:
            fn += len(gold_set)
            total += 1
            continue
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        total += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "words_evaluated": total,
    }


class MorphologyStats:
    def __init__(self) -> None:
        self._type_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total": 0.0, "covered": 0.0, "single": 0.0, "token_sum": 0.0}
        )

    @staticmethod
    def _tokens_covering_segment(
        spans: Sequence[Tuple[int, int]],
        start: int,
        end: int,
    ) -> Tuple[int, bool, bool]:
        indices = [idx for idx, (s, e) in enumerate(spans) if not (e <= start or s >= end)]
        if not indices:
            return 0, False, False
        first = spans[indices[0]]
        last = spans[indices[-1]]
        covers_exact = first[0] == start and last[1] == end
        single = covers_exact and len(indices) == 1
        return len(indices), covers_exact, single

    def observe(self, entry: UniSegEntry, tokens: Sequence[str]) -> None:
        spans = _align_token_spans(entry.word, tokens)
        for start, end, morph_type in entry.segments:
            stats = self._type_stats[morph_type]
            stats["total"] += 1
            token_count, covers_exact, single = self._tokens_covering_segment(spans, start, end)
            stats["token_sum"] += token_count
            if covers_exact:
                stats["covered"] += 1
            if single:
                stats["single"] += 1

    def report(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        aggregate = {"total": 0.0, "covered": 0.0, "single": 0.0, "token_sum": 0.0}
        for morph_type, stats in self._type_stats.items():
            total = stats["total"]
            if total <= 0:
                continue
            report[morph_type] = {
                "total_segments": total,
                "coverage_rate": stats["covered"] / total,
                "single_token_rate": stats["single"] / total,
                "avg_tokens_per_segment": stats["token_sum"] / total,
            }
            for key in aggregate:
                aggregate[key] += stats[key]
        if aggregate["total"]:
            report["_aggregate"] = {
                "total_segments": aggregate["total"],
                "coverage_rate": aggregate["covered"] / aggregate["total"],
                "single_token_rate": aggregate["single"] / aggregate["total"],
                "avg_tokens_per_segment": aggregate["token_sum"] / aggregate["total"],
            }
        return report


def evaluate_with_uniseg(
    tokenizer: TokenizerLike,
    lang: str,
    entries: Sequence[UniSegEntry],
) -> Dict[str, float]:
    """Evaluate boundary fidelity against UniSegments entries."""
    gold_sets = []
    pred_sets = []
    morph_stats = MorphologyStats()
    for entry in entries:
        tokens = _tokenize_text(tokenizer, entry.word, lang)
        gold_sets.append(entry.boundaries)
        pred_sets.append(_token_boundaries(entry.word, tokens))
        morph_stats.observe(entry, tokens)
    metrics = _boundary_metrics(gold_sets, pred_sets)
    metrics["mode"] = "uniseg"
    morphology_report = morph_stats.report()
    if morphology_report:
        metrics["morphology"] = morphology_report
    return metrics


def evaluate_with_flores(
    tokenizer: TokenizerLike,
    label: str,
    flores_jsonl: Path,
    limit: Optional[int] = None,
    *,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Dict[str, float]:
    """Fallback evaluation using translation consistency metrics."""
    flores_jsonl = _resolve_flores_path(flores_jsonl)
    pairs = load_parallel_jsonl(
        flores_jsonl,
        limit=limit,
        default_source_lang=source_lang or "en",
        default_target_lang=target_lang or label,
    )
    results = evaluate_tokenizers_on_pairs({label: tokenizer}, pairs)
    aggregate = results[label]["aggregate"]
    aggregate = dict(aggregate)  # copy
    aggregate["mode"] = "flores"
    aggregate["pairs"] = len(pairs)
    return aggregate


def evaluate_language_with_backoff(
    tokenizer: TokenizerLike,
    lang: str,
    *,
    uniseg_root: str | Path | None = None,
    flores_map: Optional[Mapping[str, Mapping[str, object]]] = None,
    max_uniseg_words: Optional[int] = 2000,
    lang_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, object]:
    """Evaluate a single language, preferring UniSegments and falling back to FLORES."""
    root = _resolve_uniseg_root(uniseg_root)
    uniseg_file = _uniseg_file_from_lang(root, lang, lang_map=lang_map)
    if uniseg_file:
        entries = list(_iter_uniseg_entries(uniseg_file, limit=max_uniseg_words))
        if entries:
            return evaluate_with_uniseg(tokenizer, lang, entries)
    flores_conf = (flores_map or {}).get(lang)
    if flores_conf:
        raw_path = flores_conf["path"]
        path = _resolve_flores_path(Path(str(raw_path)))
        raw_limit = flores_conf.get("limit")
        limit = int(raw_limit) if raw_limit is not None else None
        source_lang = _extract_first(
            flores_conf,
            ("source_lang", "src_lang", "pivot_lang", "reference_lang"),
        ) or "en"
        target_lang = _extract_first(
            flores_conf,
            ("target_lang", "tgt_lang", "language", "lang"),
        ) or lang
        return evaluate_with_flores(
            tokenizer,
            lang,
            path,
            limit=limit,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    return {
        "mode": "unavailable",
        "error": f"No UniSegments data or FLORES mapping for language '{lang}'.",
    }


def evaluate_sentences_with_uniseg(
    tokenizer: TokenizerLike,
    sentences: Sequence[str],
    languages: Sequence[str],
    *,
    uniseg_root: str | Path | None = None,
    lang_map: Optional[Mapping[str, Sequence[str]]] = None,
    word_tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, float]:
    """
    Evaluate tokenizer on full sentences by comparing with UniSeg word-level segmentations.
    
    This function:
    1. Tokenizes each sentence with the tokenizer
    2. Extracts words from each sentence
    3. For each word, looks up UniSeg segmentation
    4. Compares tokenizer boundaries with UniSeg boundaries
    5. Computes a composite similarity score accounting for morphological performance
    
    Args:
        tokenizer: Tokenizer to evaluate
        sentences: List of test sentences
        languages: List of language codes (one per sentence, or single code for all)
        uniseg_root: Root directory for UniSegments data
        lang_map: Language mapping for UniSeg datasets
        word_tokenizer: Optional function to extract words from sentences (default: simple whitespace split)
    
    Returns:
        Dictionary with similarity metrics including:
        - sentence_similarity: Overall similarity score (0-1, higher is better)
        - boundary_f1: F1 score for boundary alignment
        - morphological_score: Score based on morphological correctness
        - words_evaluated: Number of words successfully matched with UniSeg
        - sentences_evaluated: Number of sentences processed
    """
    import re
    
    if word_tokenizer is None:
        # Simple word tokenizer: split on whitespace and punctuation boundaries
        def _simple_word_tokenizer(text: str) -> List[str]:
            # Split on whitespace, keep punctuation attached
            words = re.findall(r'\b\w+\b', text)
            return [w for w in words if w]
        word_tokenizer = _simple_word_tokenizer
    
    # Normalize languages: if single language provided, use for all sentences
    if len(languages) == 1 and len(sentences) > 1:
        languages = [languages[0]] * len(sentences)
    elif len(languages) != len(sentences):
        # Pad with last language or default to first
        last_lang = languages[-1] if languages else "en"
        languages = list(languages) + [last_lang] * (len(sentences) - len(languages))
    
    # Load UniSeg data for all languages we'll need
    root = _resolve_uniseg_root(uniseg_root)
    lang_map = lang_map or DEFAULT_UNISEG_LANG_MAP
    
    uniseg_cache: Dict[str, Dict[str, UniSegEntry]] = {}
    requested_langs = set(languages)
    for lang in requested_langs:
        uniseg_file = _uniseg_file_from_lang(root, lang, lang_map=lang_map)
        if uniseg_file:
            entries = list(_iter_uniseg_entries(uniseg_file, limit=None))
            uniseg_cache[lang] = {entry.word.lower(): entry for entry in entries}
    missing_languages = sorted(requested_langs - set(uniseg_cache.keys()))
    
    # Evaluate sentences
    all_gold_sets: List[Set[int]] = []
    all_pred_sets: List[Optional[Set[int]]] = []
    morph_stats = MorphologyStats()
    words_evaluated = 0
    sentences_evaluated = 0
    lang_sentence_counts: Dict[str, int] = defaultdict(int)
    lang_word_counts: Dict[str, int] = defaultdict(int)
    
    for sentence, lang in zip(sentences, languages):
        if lang not in uniseg_cache:
            continue  # Skip if no UniSeg data for this language
        
        # Tokenize sentence with tokenizer
        sentence_tokens = _tokenize_text(tokenizer, sentence, lang)
        token_spans = _align_token_spans(sentence, sentence_tokens)
        
        # Extract words from sentence
        words = word_tokenizer(sentence)
        if not words:
            continue
        
        sentences_evaluated += 1
        lang_sentence_counts[lang] += 1
        sentence_word_boundaries: List[Set[int]] = []
        sentence_pred_boundaries: List[Optional[Set[int]]] = []
        
        # Find word positions in sentence
        cursor = 0
        for word in words:
            word_lower = word.lower()
            if word_lower not in uniseg_cache[lang]:
                # Try to find word at current position
                pos = sentence.find(word, cursor)
                if pos == -1:
                    cursor += len(word) + 1  # Approximate
                    continue
                cursor = pos + len(word)
                continue
            
            # Get UniSeg entry for this word
            uniseg_entry = uniseg_cache[lang][word_lower]
            
            # Find word position in sentence
            word_pos = sentence.find(word, cursor)
            if word_pos == -1:
                # Try case-insensitive search
                word_pos = sentence.lower().find(word_lower, cursor)
                if word_pos == -1:
                    cursor += len(word) + 1
                    continue
            
            # Adjust UniSeg boundaries to sentence coordinates
            word_start = word_pos
            word_end = word_pos + len(word)
            gold_boundaries = {word_start + b for b in uniseg_entry.boundaries if word_start + b <= word_end}
            # Extract tokenizer boundaries for this word
            word_tokens = []
            for tok, (tok_start, tok_end) in zip(sentence_tokens, token_spans):
                if not (tok_end <= word_start or tok_start >= word_end):
                    word_tokens.append(tok)
            
            if word_tokens:
                # Compute boundaries relative to word start
                word_text = sentence[word_start:word_end]
                word_pred_boundaries = _token_boundaries(word_text, word_tokens)
                if word_pred_boundaries:
                    # Adjust to sentence coordinates
                    word_pred_boundaries = {word_start + b for b in word_pred_boundaries if word_start + b <= word_end}
                else:
                    word_pred_boundaries = None
            else:
                word_pred_boundaries = None
            
            sentence_word_boundaries.append(gold_boundaries)
            sentence_pred_boundaries.append(word_pred_boundaries)
            
            # Update morphology stats
            if word_tokens:
                morph_stats.observe(uniseg_entry, word_tokens)
            
            words_evaluated += 1
            lang_word_counts[lang] += 1
            cursor = word_end
        
        # Aggregate boundaries for this sentence
        if sentence_word_boundaries:
            # Merge all word boundaries
            sentence_gold = set()
            for boundaries in sentence_word_boundaries:
                sentence_gold.update(boundaries)
            all_gold_sets.append(sentence_gold)
            
            # Merge predicted boundaries
            sentence_pred = set()
            for boundaries in sentence_pred_boundaries:
                if boundaries:
                    sentence_pred.update(boundaries)
            all_pred_sets.append(sentence_pred if sentence_pred else None)
    
    # Compute metrics
    boundary_metrics = _boundary_metrics(all_gold_sets, all_pred_sets)
    morphology_report = morph_stats.report()
    
    # Compute composite similarity score
    # Weight: 50% boundary F1, 50% morphological correctness
    boundary_f1 = boundary_metrics.get("f1", 0.0)
    
    # Morphological score: average coverage rate across all morpheme types
    morphological_score = 0.0
    if morphology_report:
        aggregate = morphology_report.get("_aggregate", {})
        if aggregate:
            morphological_score = aggregate.get("coverage_rate", 0.0)
        else:
            # Average across all types
            coverage_rates = [
                stats.get("coverage_rate", 0.0)
                for stats in morphology_report.values()
                if isinstance(stats, dict)
            ]
            if coverage_rates:
                morphological_score = sum(coverage_rates) / len(coverage_rates)
    
    # Composite similarity score
    sentence_similarity = 0.5 * boundary_f1 + 0.5 * morphological_score
    
    return {
        "sentence_similarity": sentence_similarity,
        "boundary_f1": boundary_f1,
        "boundary_precision": boundary_metrics.get("precision", 0.0),
        "boundary_recall": boundary_metrics.get("recall", 0.0),
        "morphological_score": morphological_score,
        "words_evaluated": words_evaluated,
        "sentences_evaluated": sentences_evaluated,
        "mode": "sentence_uniseg",
        "morphology": morphology_report,
        "languages_evaluated": sorted(lang_sentence_counts.keys()),
        "languages_skipped": missing_languages,
        "missing_languages": missing_languages,
        "words_per_language": {lang: int(count) for lang, count in lang_word_counts.items()},
        "sentences_per_language": {lang: int(count) for lang, count in lang_sentence_counts.items()},
    }


def evaluate_languages_with_backoff(
    tokenizers: Mapping[str, TokenizerLike],
    languages: Sequence[str],
    *,
    uniseg_root: str | Path | None = None,
    flores_map: Optional[Mapping[str, Mapping[str, object]]] = None,
    max_uniseg_words: Optional[int] = 2000,
    lang_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Evaluate multiple tokenizer variants across languages with UniSeg/FLORES backoff."""
    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    for label, tokenizer in tokenizers.items():
        lang_results: Dict[str, Dict[str, object]] = {}
        for lang in languages:
            lang_results[lang] = evaluate_language_with_backoff(
                tokenizer,
                lang,
                uniseg_root=uniseg_root,
                flores_map=flores_map,
                max_uniseg_words=max_uniseg_words,
                lang_map=lang_map,
            )
        results[label] = lang_results
    return results


__all__ = [
    "DEFAULT_UNISEG_LANG_MAP",
    "DEFAULT_UNISEG_ROOT",
    "DEFAULT_FLORES_ROOT",
    "ParallelPair",
    "load_parallel_jsonl",
    "evaluate_tokenizers_on_pairs",
    "evaluate_with_uniseg",
    "evaluate_with_flores",
    "evaluate_language_with_backoff",
    "evaluate_languages_with_backoff",
    "evaluate_sentences_with_uniseg",
]
