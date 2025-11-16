from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .linguistic_features import MorphologyEncoder
from .utils import ParagraphInfo
from .torch_utils import default_device

DEVICE = default_device()


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        return vec
    return vec / norm


@dataclass
class TokenEmbeddingAdapter:
    """Wraps a tokenizer + token-level embeddings to produce word vectors."""

    label: str
    tokenize: Callable[[str, Optional[str]], Sequence[str]]
    token_vectors: Mapping[str, torch.Tensor]

    def word_vector(self, word: str, lang: Optional[str] = None) -> Optional[torch.Tensor]:
        tokens = self.tokenize(word, lang)
        if not tokens:
            return None
        vecs = [self.token_vectors.get(tok) for tok in tokens if tok in self.token_vectors]
        if not vecs:
            return None
        mat = torch.stack(vecs, dim=0)
        return _normalize(mat.mean(dim=0))


def _slice_corpus(texts: Sequence[str], langs: Sequence[str], limit: Optional[int]) -> Tuple[List[str], List[str]]:
    if limit is None or limit <= 0 or limit >= len(texts):
        return list(texts), list(langs)
    return list(texts[:limit]), list(langs[:limit])


def build_reference_adapter(
    label: str,
    tokenize_fn: Callable[[str], Sequence[str]],
    texts: Sequence[str],
    langs: Sequence[str],
    morph_kwargs: Optional[Mapping[str, object]] = None,
) -> Optional[TokenEmbeddingAdapter]:
    texts_use, langs_use = _slice_corpus(texts, langs, morph_kwargs.get("max_corpus_paragraphs") if morph_kwargs else None)
    paragraphs = [ParagraphInfo(text, lang) for text, lang in zip(texts_use, langs_use)]
    occurrences: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for idx, text in enumerate(texts_use):
        try:
            tokens = tokenize_fn(text)
        except Exception:
            return None
        for tok in tokens:
            if tok:
                occurrences[tok].append((idx, 0))
    if not occurrences:
        return None
    morph_config = dict(morph_kwargs or {})
    morph_config.pop("max_corpus_paragraphs", None)
    encoder = MorphologyEncoder(**morph_config)
    encoder.fit(paragraphs, occurrences, lambda i: paragraphs[i].lang)
    if not encoder.token_vec:
        return None

    def wrapped_tokenize(text: str, lang: Optional[str] = None) -> Sequence[str]:
        return tokenize_fn(text)

    return TokenEmbeddingAdapter(label, wrapped_tokenize, encoder.token_vec)


def build_scalable_adapter(label: str, tokenizer, tokenize_fn: Callable[[str, Optional[str]], Sequence[str]]):
    morph = getattr(getattr(tokenizer, "_ling", None), "morph_encoder", None)
    if morph is None or not getattr(morph, "token_vec", None):
        return None
    return TokenEmbeddingAdapter(label, tokenize_fn, morph.token_vec)


def _load_muse_dictionary(path: Path, max_pairs: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def _load_similarity_file(path: Path, delimiter: Optional[str] = None, has_header: bool = True) -> List[Tuple[str, str, float]]:
    entries: List[Tuple[str, str, float]] = []
    opener = path.open("r", encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.load(opener)
            for item in data:
                entries.append((item["word1"], item["word2"], float(item["score"])))
            return entries
    finally:
        opener.close()
    with path.open("r", encoding="utf-8") as f:
        reader: Iterable[List[str]]
        if delimiter:
            reader = csv.reader(f, delimiter=delimiter)
        else:
            reader = (line.strip().split() for line in f)
        for idx, parts in enumerate(reader):
            if not parts or parts[0].startswith("#"):
                continue
            if has_header and idx == 0 and not parts[0].replace(".", "", 1).isdigit():
                continue
            if len(parts) < 3:
                continue
            try:
                score = float(parts[-1])
            except ValueError:
                continue
            w1, w2 = parts[0], parts[1]
            entries.append((w1, w2, score))
    return entries


def _cosine_matrix(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    src_norm = src / torch.clamp(torch.linalg.norm(src, dim=1, keepdim=True), min=1e-12)
    tgt_norm = tgt / torch.clamp(torch.linalg.norm(tgt, dim=1, keepdim=True), min=1e-12)
    return torch.clamp(src_norm @ tgt_norm.T, -1.0, 1.0)


def _csls(sim: torch.Tensor, k: int = 10) -> torch.Tensor:
    k = max(1, min(k, sim.shape[1], sim.shape[0]))
    src_part = torch.topk(sim, k, dim=1).values
    tgt_part = torch.topk(sim, k, dim=0).values
    r_src = src_part.mean(dim=1, keepdim=True)
    r_tgt = tgt_part.mean(dim=0, keepdim=True)
    return 2 * sim - r_src - r_tgt


def evaluate_muse(
    adapters: Mapping[str, TokenEmbeddingAdapter],
    dataset_conf: Mapping[str, object],
) -> Dict[str, Dict[str, float]]:
    path = Path(dataset_conf["path"])
    if not path.exists():
        return {}
    src_lang = dataset_conf.get("source_lang")
    tgt_lang = dataset_conf.get("target_lang")
    max_pairs = dataset_conf.get("max_pairs")
    csls_k = dataset_conf.get("csls_k", 10)
    pairs = _load_muse_dictionary(path, max_pairs=max_pairs)
    results: Dict[str, Dict[str, float]] = {}
    for label, adapter in adapters.items():
        src_words: Dict[str, torch.Tensor] = {}
        tgt_words: Dict[str, torch.Tensor] = {}
        for src, tgt in pairs:
            if src not in src_words:
                vec = adapter.word_vector(src, src_lang)
                if vec is not None:
                    src_words[src] = vec
            if tgt not in tgt_words:
                vec = adapter.word_vector(tgt, tgt_lang)
                if vec is not None:
                    tgt_words[tgt] = vec
        if not src_words or not tgt_words:
            continue
        src_list = sorted(src_words.keys())
        tgt_list = sorted(tgt_words.keys())
        src_mat = torch.stack([src_words[w] for w in src_list], dim=0)
        tgt_mat = torch.stack([tgt_words[w] for w in tgt_list], dim=0)
        cos = _cosine_matrix(src_mat, tgt_mat)
        gold_pairs = [
            (src_list.index(src), tgt_list.index(tgt))
            for src, tgt in pairs
            if src in src_words and tgt in tgt_words
        ]
        if not gold_pairs:
            continue
        cos_csls = _csls(cos.clone(), k=csls_k)

        def _p1(sim: torch.Tensor) -> float:
            correct = 0
            for src_idx, tgt_idx in gold_pairs:
                pred = int(torch.argmax(sim[src_idx]).item())
                if pred == tgt_idx:
                    correct += 1
            return correct / len(gold_pairs)

        results[label] = {
            "pairs_used": float(len(gold_pairs)),
            "p_at_1": _p1(cos),
            "csls_p1": _p1(cos_csls),
        }
    return results


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = torch.as_tensor(xs, dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(ys, dtype=torch.float64, device=DEVICE)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denom.item() == 0.0:
        return None
    return float(torch.dot(x, y) / denom)


def _rankdata(values: Sequence[float]) -> torch.Tensor:
    arr = torch.as_tensor(values, dtype=torch.float64, device=DEVICE)
    sorter = torch.argsort(arr)
    ranks = torch.empty_like(arr)
    ranks[sorter] = torch.arange(len(arr), dtype=torch.float64, device=DEVICE)
    unique_vals, inverse, counts = torch.unique(arr, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts.tolist()):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx, ry)


def evaluate_similarity(
    adapters: Mapping[str, TokenEmbeddingAdapter],
    dataset_conf: Mapping[str, object],
) -> Dict[str, Dict[str, float]]:
    path = Path(dataset_conf["path"])
    if not path.exists():
        return {}
    delimiter = dataset_conf.get("delimiter")
    has_header = dataset_conf.get("has_header", True)
    entries = _load_similarity_file(path, delimiter=delimiter, has_header=has_header)
    lang1 = dataset_conf.get("source_lang") or dataset_conf.get("language")
    lang2 = dataset_conf.get("target_lang") or lang1
    results: Dict[str, Dict[str, float]] = {}
    for label, adapter in adapters.items():
        gold_scores: List[float] = []
        pred_scores: List[float] = []
        for w1, w2, score in entries:
            v1 = adapter.word_vector(w1, lang1)
            v2 = adapter.word_vector(w2, lang2)
            if v1 is None or v2 is None:
                continue
            pred_scores.append(float(torch.dot(v1, v2).item()))
            gold_scores.append(score)
        if len(pred_scores) < 2:
            continue
        pearson = _pearson(pred_scores, gold_scores)
        spearman = _spearman(pred_scores, gold_scores)
        if pearson is None and spearman is None:
            continue
        results[label] = {
            "pairs_used": float(len(pred_scores)),
            "pearson": pearson if pearson is not None else float("nan"),
            "spearman": spearman if spearman is not None else float("nan"),
        }
    return results


def maybe_run_embedding_eval(
    tokenizer,
    references: Mapping[str, Callable[[str], Sequence[str]]],
    texts: Sequence[str],
    langs: Sequence[str],
    feat_args: Mapping[str, object],
    eval_cfg: Optional[Mapping[str, object]],
) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
    if not eval_cfg:
        return None
    adapters: Dict[str, TokenEmbeddingAdapter] = {}
    ours_adapter = build_scalable_adapter("ours", tokenizer, tokenizer.tokenize)
    if ours_adapter:
        adapters["ours"] = ours_adapter
    morph_kwargs = dict(feat_args.get("morphology_kwargs") or {})
    extra_kwargs = eval_cfg.get("morphology_kwargs")
    if isinstance(extra_kwargs, dict):
        morph_kwargs.update(extra_kwargs)
    max_ref_paras = eval_cfg.get("max_reference_paragraphs")
    if max_ref_paras is not None:
        morph_kwargs.setdefault("max_corpus_paragraphs", max_ref_paras)
    if not adapters:
        return None
    for label, ref_tokenize in references.items():
        adapter = build_reference_adapter(label, ref_tokenize, texts, langs, morph_kwargs)
        if adapter:
            adapters[label] = adapter
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    missing: Dict[str, List[str]] = defaultdict(list)
    empty: Dict[str, List[str]] = defaultdict(list)
    muse_cfg = eval_cfg.get("muse")
    if muse_cfg:
        muse_res = {}
        for entry in muse_cfg:
            dataset_name = entry.get("name") or Path(entry["path"]).stem
            path = Path(entry["path"])
            if not path.exists():
                missing["muse"].append(dataset_name)
                continue
            res = evaluate_muse(adapters, entry)
            if res:
                muse_res[dataset_name] = res
            else:
                empty["muse"].append(dataset_name)
        if muse_res:
            results["muse"] = muse_res
    sim_cfg = eval_cfg.get("similarity")
    if sim_cfg:
        sim_res = {}
        for entry in sim_cfg:
            dataset_name = entry.get("name") or Path(entry["path"]).stem
            path = Path(entry["path"])
            if not path.exists():
                missing["similarity"].append(dataset_name)
                continue
            res = evaluate_similarity(adapters, entry)
            if res:
                sim_res[dataset_name] = res
            else:
                empty["similarity"].append(dataset_name)
        if sim_res:
            results["similarity"] = sim_res
    xling_cfg = eval_cfg.get("crosslingual_similarity")
    if xling_cfg:
        xling_res = {}
        for entry in xling_cfg:
            dataset_name = entry.get("name") or Path(entry["path"]).stem
            path = Path(entry["path"])
            if not path.exists():
                missing["crosslingual_similarity"].append(dataset_name)
                continue
            res = evaluate_similarity(adapters, entry)
            if res:
                xling_res[dataset_name] = res
            else:
                empty["crosslingual_similarity"].append(dataset_name)
        if xling_res:
            results["crosslingual_similarity"] = xling_res
    meta_payload: Dict[str, Dict[str, List[str]]] = {}
    if missing:
        meta_payload["missing"] = {k: sorted(v) for k, v in missing.items()}
    if empty:
        meta_payload["empty"] = {k: sorted(v) for k, v in empty.items()}
    if meta_payload:
        results["__meta__"] = meta_payload  # type: ignore[assignment]
    if results:
        return results
    if meta_payload:
        return {"__meta__": meta_payload}  # type: ignore[return-value]
    return None


def write_embedding_report(folder: Path, results: Mapping[str, Dict[str, Dict[str, float]]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["# Embedding Benchmark Summary\n"]
    meta_section = results.get("__meta__") if isinstance(results, Mapping) else None
    if isinstance(meta_section, Mapping):
        missing = meta_section.get("missing") or {}
        empty = meta_section.get("empty") or {}
        if missing or empty:
            lines.append("## Dataset Availability\n")
            if missing:
                lines.append("**Missing data sources:**")
                for category, names in sorted(missing.items()):
                    if names:
                        lines.append(f"- {category}: {', '.join(names)}")
            if empty:
                lines.append("\n**Datasets with no usable pairs:**")
                for category, names in sorted(empty.items()):
                    if names:
                        lines.append(f"- {category}: {', '.join(names)}")
            lines.append("")
    for bench_type, datasets in results.items():
        if bench_type == "__meta__":
            continue
        if not datasets:
            continue
        lines.append(f"## {bench_type.replace('_', ' ').title()}")
        for dataset, scores in datasets.items():
            if not scores:
                continue
            metric_keys = sorted({metric for metrics in scores.values() for metric in metrics.keys()})
            lines.append(f"### {dataset}")
            header = "| tokenizer | " + " | ".join(metric_keys) + " |"
            lines.append(header)
            lines.append("|---|" + "|".join(["---"] * len(metric_keys)) + "|")
            for label, metrics in sorted(scores.items()):
                row = [label]
                for key in metric_keys:
                    val = metrics.get(key)
                    if val is None or (isinstance(val, float) and not math.isfinite(val)):
                        row.append("—")
                    else:
                        row.append(f"{val:.4f}")
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")  # blank line between tables
    (folder / "embedding_benchmarks.md").write_text("\n".join(lines), encoding="utf-8")


def write_segmentation_report(folder: Path, segmentation_results: Mapping[str, Dict[str, object]]) -> None:
    if not isinstance(segmentation_results, Mapping) or not segmentation_results:
        return
    report_path = folder / "segmentation_report.md"
    if "error" in segmentation_results:
        report_path.write_text(
            f"# UniSeg Segmentation Comparison\n\nSegmentation evaluation failed: {segmentation_results['error']}",
            encoding="utf-8",
        )
        return
    lines = ["# UniSeg Segmentation Comparison\n"]
    exclude_metrics = {
        "mode",
        "error",
        "morphology",
        "words_per_language",
        "sentences_per_language",
        "languages_evaluated",
        "languages_skipped",
        "missing_languages",
    }

    def _append_meta(meta: Optional[Mapping[str, object]]) -> None:
        if not isinstance(meta, Mapping):
            return
        evaluated = meta.get("evaluated_languages")
        missing = meta.get("missing_languages")
        bullet_lines: List[str] = []
        if evaluated:
            bullet_lines.append(f"- Evaluated languages: {', '.join(evaluated)}")
        if missing:
            bullet_lines.append(f"- Missing UniSeg coverage: {', '.join(missing)}")
        if bullet_lines:
            lines.append("**Coverage summary**")
            lines.extend(bullet_lines)
            lines.append("")

    def _append_per_language_tables(
        title: str,
        per_language: Mapping[str, Mapping[str, Mapping[str, object]]],
    ) -> None:
        if not isinstance(per_language, Mapping) or not per_language:
            return
        tokenizers_local = sorted(
            {
                label
                for lang_stats in per_language.values()
                for label in lang_stats.keys()
            }
        )
        languages = sorted(per_language.keys())
        if not tokenizers_local or not languages:
            return
        metric_names = sorted(
            {
                metric
                for lang_stats in per_language.values()
                for stats in lang_stats.values()
                if isinstance(stats, Mapping)
                for metric, value in stats.items()
                if metric not in exclude_metrics and isinstance(value, (int, float)) and math.isfinite(value)
            }
        )
        if not metric_names:
            return
        for metric in metric_names:
            lines.append(f"### {title}: {metric.replace('_', ' ').title()}")
            header = "| language | " + " | ".join(tokenizers_local) + " |"
            lines.append(header)
            lines.append("|---|" + "|".join(["---"] * len(tokenizers_local)) + "|")
            for lang in languages:
                row = [lang]
                for tok_label in tokenizers_local:
                    value = per_language.get(lang, {}).get(tok_label, {}).get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        row.append(f"{value:.4f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    def _append_morphology_tables(
        title: str,
        per_language: Mapping[str, Mapping[str, Mapping[str, object]]],
    ) -> None:
        if not isinstance(per_language, Mapping):
            return
        morph_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for lang, label_map in per_language.items():
            for label, stats in label_map.items():
                if not isinstance(stats, Mapping):
                    continue
                morphology = stats.get("morphology")
                if not isinstance(morphology, Mapping):
                    continue
                for morph_type, morph_stats in morphology.items():
                    if not isinstance(morph_stats, Mapping) or morph_type.startswith("_"):
                        continue
                    coverage = morph_stats.get("coverage_rate")
                    if isinstance(coverage, (int, float)) and math.isfinite(coverage):
                        morph_map[morph_type][lang][label] = float(coverage)
        if not morph_map:
            return
        lines.append(f"### {title} Morphology Coverage\n")
        for morph_type, lang_map in sorted(morph_map.items()):
            langs = sorted(lang_map.keys())
            tokenizers_local = sorted({label for stats in lang_map.values() for label in stats.keys()})
            if not langs or not tokenizers_local:
                continue
            lines.append(f"#### {morph_type}")
            header = "| language | " + " | ".join(tokenizers_local) + " |"
            lines.append(header)
            lines.append("|---|" + "|".join(["---"] * len(tokenizers_local)) + "|")
            for lang in langs:
                row = [lang]
                for tok_label in tokenizers_local:
                    value = lang_map.get(lang, {}).get(tok_label)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        row.append(f"{value:.4f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    def _append_aggregate_table(
        title: str,
        aggregate: Mapping[str, Mapping[str, object]],
    ) -> None:
        if not isinstance(aggregate, Mapping) or not aggregate:
            return
        tokenizers_local = sorted(aggregate.keys())
        metric_names = sorted(
            {
                metric
                for stats in aggregate.values()
                if isinstance(stats, Mapping)
                for metric, value in stats.items()
                if metric not in exclude_metrics and isinstance(value, (int, float)) and math.isfinite(value)
            }
        )
        if not tokenizers_local or not metric_names:
            return
        lines.append(f"### {title}")
        header = "| metric | " + " | ".join(tokenizers_local) + " |"
        lines.append(header)
        lines.append("|---|" + "|".join(["---"] * len(tokenizers_local)) + "|")
        for metric in metric_names:
            row = [metric.replace("_", " ").title()]
            for tok_label in tokenizers_local:
                value = aggregate.get(tok_label, {}).get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    row.append(f"{value:.4f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Word-level evaluation
    word_level = segmentation_results.get("word_level")
    if isinstance(word_level, Mapping) and not word_level.get("error"):
        lines.append("## Word-Level Evaluation\n")
        _append_meta(word_level.get("meta"))
        word_per_language = word_level.get("per_language", {})
        _append_per_language_tables("Word-Level", word_per_language)
        _append_morphology_tables("Word-Level", word_per_language)
        _append_aggregate_table("Word-Level Aggregate Metrics", word_level.get("aggregate", {}))

    # Sentence-level evaluation
    sentence_level = segmentation_results.get("sentence_level")
    if isinstance(sentence_level, Mapping) and not sentence_level.get("error"):
        lines.append("## Sentence-Level Evaluation\n")
        lines.append("The **sentence_similarity** score combines boundary F1 (50%) and morphological coverage (50%).\n")
        _append_meta(sentence_level.get("meta"))
        details = sentence_level.get("details", {})
        if isinstance(details, Mapping):
            note_lines: List[str] = []
            for label, payload in sorted(details.items()):
                if not isinstance(payload, Mapping):
                    continue
                skipped = payload.get("languages_skipped") or payload.get("missing_languages")
                if skipped:
                    note_lines.append(f"- {label}: skipped languages without UniSeg coverage: {', '.join(sorted(set(skipped)))}")
            if note_lines:
                lines.append("**Tokenizer notes**")
                lines.extend(note_lines)
                lines.append("")
        sentence_per_language = sentence_level.get("per_language", {})
        _append_per_language_tables("Sentence-Level", sentence_per_language)
        _append_morphology_tables("Sentence-Level", sentence_per_language)
        _append_aggregate_table("Sentence-Level Aggregate Metrics", sentence_level.get("aggregate", {}))

    if len(lines) == 1:  # Only header
        lines.append("No evaluation results available.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
