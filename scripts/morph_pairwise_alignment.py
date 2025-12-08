"""
Pairwise morphology alignment diagnostics.

This script trains a MorphologyEncoder (either from a provided corpus or by
streaming WikiANN via Hugging Face) and then, for each UniSeg cross-equivalence
class, compares:

  1. Same-class / same-language cosine similarity vs. different-class baseline
  2. Same-class / cross-language cosine similarity vs. different-class baseline

This makes it easy to verify that the Laplacian regularizer is actually pulling
tokens of the same morphological function closer together both within and
across languages.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_wikiann_corpus  # noqa: E402
from tokenizer_core.constants import CROSS_EQUIV  # noqa: E402
from tokenizer_core.linguistic_features import MorphologyEncoder  # noqa: E402
from tokenizer_core.utils import ParagraphInfo  # noqa: E402

DEFAULT_CLASSES: Sequence[str] = ("PROG", "PL", "NEG", "PAST", "COMP")
DEFAULT_WIKIANN_LANGS: Sequence[str] = ("en", "de", "tr")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_PCA_POINTS = 200
PAIR_SAMPLE_LIMIT = 25


def read_lines(path: Path, limit: Optional[int] = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip()]
    if limit is not None:
        return lines[:limit]
    return lines


def read_langs(path: Optional[Path], n: int, limit: Optional[int] = None) -> List[str | None]:
    if path is None:
        return [None] * (limit if limit is not None else n)
    langs = read_lines(path, limit)
    if len(langs) != (limit if limit is not None else n):
        raise ValueError(f"Language file length ({len(langs)}) does not match corpus length.")
    return langs


def build_occurrences(paragraphs: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    occ: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for idx, text in enumerate(paragraphs):
        for token in TOKEN_RE.findall(text.lower()):
            if token:
                occ[token].append((idx, 0))
    return occ


def assign_token_langs(
    occurrences: Mapping[str, Sequence[Tuple[int, int]]],
    paragraph_infos: Sequence[ParagraphInfo],
) -> Dict[str, Optional[str]]:
    token_lang: Dict[str, Optional[str]] = {}
    for tok, occs in occurrences.items():
        counts = Counter(paragraph_infos[pi].lang or "other" for (pi, _) in occs)
        if counts:
            token_lang[tok] = counts.most_common(1)[0][0]
    return token_lang


def normalize(vecs: Sequence[torch.Tensor]) -> torch.Tensor:
    mat = torch.stack(vecs, dim=0)
    norms = torch.linalg.norm(mat, dim=1, keepdim=True).clamp(min=1e-9)
    return mat / norms


def avg_same_language(items: Sequence[Tuple[str, Optional[str], torch.Tensor]]) -> Optional[float]:
    if len(items) < 2:
        return None
    mat = normalize([vec for _, _, vec in items])
    sim = mat @ mat.T
    idx = torch.triu_indices(sim.size(0), sim.size(1), offset=1)
    values = sim[idx[0], idx[1]]
    return float(values.mean().item()) if values.numel() > 0 else None


def avg_cross_language(
    items_a: Sequence[Tuple[str, Optional[str], torch.Tensor]],
    items_b: Sequence[Tuple[str, Optional[str], torch.Tensor]],
) -> Optional[float]:
    if not items_a or not items_b:
        return None
    Va = normalize([vec for _, _, vec in items_a])
    Vb = normalize([vec for _, _, vec in items_b])
    sims = Va @ Vb.T
    return float(sims.mean().item()) if sims.numel() > 0 else None


def token_matches_class(token: str, lang: Optional[str], class_key: str) -> bool:
    if lang is None:
        return False
    suffix_map = CROSS_EQUIV.get(class_key, {})
    suffixes = suffix_map.get(lang, set())
    return any(token.endswith(suf) for suf in suffixes)


def sample_within_pairs(
    items: Sequence[Tuple[str, Optional[str], torch.Tensor]],
    max_pairs: int = PAIR_SAMPLE_LIMIT,
) -> List[Dict[str, object]]:
    n = len(items)
    if n < 2:
        return []
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 10:
        i, j = random.sample(range(n), 2)
        key = tuple(sorted((i, j)))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        tok_i, lang_i, vec_i = items[i]
        tok_j, lang_j, vec_j = items[j]
        cos = float(torch.dot(vec_i, vec_j).item())
        pairs.append(
            {"token_a": tok_i, "token_b": tok_j, "lang": lang_i, "cosine": cos}
            if lang_i == lang_j
            else {"token_a": tok_i, "lang_a": lang_i, "token_b": tok_j, "lang_b": lang_j, "cosine": cos}
        )
        attempts += 1
    return pairs


def sample_cross_pairs(
    items_a: Sequence[Tuple[str, Optional[str], torch.Tensor]],
    items_b: Sequence[Tuple[str, Optional[str], torch.Tensor]],
    max_pairs: int = PAIR_SAMPLE_LIMIT,
) -> List[Dict[str, object]]:
    if not items_a or not items_b:
        return []
    pairs = []
    for _ in range(min(max_pairs, len(items_a) * len(items_b))):
        tok_a, lang_a, vec_a = random.choice(items_a)
        tok_b, lang_b, vec_b = random.choice(items_b)
        cos = float(torch.dot(vec_a, vec_b).item())
        pairs.append(
            {"token_a": tok_a, "lang_a": lang_a, "token_b": tok_b, "lang_b": lang_b, "cosine": cos}
        )
    return pairs


def compute_pca_projection(
    class_items: Sequence[Tuple[str, Optional[str], torch.Tensor]],
    baseline_items: Sequence[Tuple[str, Optional[str], torch.Tensor]],
    max_points: int = MAX_PCA_POINTS,
) -> Optional[Dict[str, object]]:
    if len(class_items) + len(baseline_items) < 2:
        return None
    class_limit = max_points // 2 if baseline_items else max_points
    base_limit = max_points - class_limit
    selected_class = (
        random.sample(list(class_items), min(len(class_items), class_limit))
        if len(class_items) > class_limit
        else list(class_items)
    )
    selected_base = (
        random.sample(list(baseline_items), min(len(baseline_items), base_limit))
        if len(baseline_items) > base_limit
        else list(baseline_items)
    )
    combined = [
        {"token": tok, "lang": lang, "is_class": True, "vec": vec}
        for tok, lang, vec in selected_class
    ] + [
        {"token": tok, "lang": lang, "is_class": False, "vec": vec}
        for tok, lang, vec in selected_base
    ]
    if len(combined) < 2:
        return None
    mat = torch.stack([entry["vec"] for entry in combined], dim=0)
    mean = mat.mean(0, keepdim=True)
    centered = mat - mean
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        return None
    comps = Vh[:2]
    coords = centered @ comps.T
    explained = (S[:2] ** 2) / max(len(combined) - 1, 1)
    return {
        "components": comps.cpu().tolist(),
        "explained_variance": explained.cpu().tolist(),
        "points": [
            {
                "token": entry["token"],
                "lang": entry["lang"],
                "is_class": entry["is_class"],
                "x": float(coords[i, 0].item()),
                "y": float(coords[i, 1].item()) if coords.shape[1] > 1 else 0.0,
            }
            for i, entry in enumerate(combined)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise morphology alignment diagnostics.")
    parser.add_argument("--corpus", type=Path, help="Optional path to text file (one paragraph per line).")
    parser.add_argument("--langs-file", type=Path, help="Optional path to language tags (one per paragraph).")
    parser.add_argument("--limit", type=int, help="Optional cap on number of paragraphs to load from --corpus.")
    parser.add_argument(
        "--wikiann-langs",
        nargs="+",
        default=list(DEFAULT_WIKIANN_LANGS),
        help="ISO codes for WikiANN sampling (used when --corpus is not provided).",
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=1000,
        help="WikiANN paragraphs per language (only when --corpus is omitted).",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="Cross-equivalence keys to evaluate.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save JSON summary.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        help="If set, dump the fetched corpus/lang tags to PREFIX.txt/.langs for reuse.",
    )
    parser.add_argument(
        "--plot-class",
        nargs="+",
        help="Optional one or more class keys to render PCA scatters for (deprecated alias).",
    )
    parser.add_argument(
        "--plot-classes",
        nargs="+",
        help="Optional list of class keys to render PCA scatters for (one PNG per class).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Path to save the PCA scatter when plotting a single class.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Directory to store PCA plots when multiple classes are requested.",
    )
    return parser.parse_args()


def maybe_save_corpus(prefix: Path, texts: Sequence[str], langs: Sequence[str | None]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = prefix.with_suffix(".txt")
    lang_path = prefix.with_suffix(".langs")
    txt_path.write_text("\n".join(texts), encoding="utf-8")
    lang_path.write_text("\n".join(lang or "" for lang in langs), encoding="utf-8")
    print(f"[save] wrote {len(texts)} paragraphs to {txt_path}")
    print(f"[save] wrote {len(langs)} language tags to {lang_path}")


def load_data(args: argparse.Namespace) -> Tuple[List[str], List[str | None]]:
    if args.corpus:
        paragraphs = read_lines(args.corpus, args.limit)
        langs = read_langs(args.langs_file, len(paragraphs), args.limit)
        return paragraphs, langs
    codes = {code: code for code in args.wikiann_langs}
    texts, lang_tags = load_wikiann_corpus(codes, per_lang=args.samples_per_lang)
    if not texts:
        raise SystemExit("WikiANN corpus retrieval failed (no paragraphs loaded).")
    if args.save_prefix:
        maybe_save_corpus(args.save_prefix, texts, lang_tags)
    return texts, lang_tags


def main():
    args = parse_args()
    paragraphs, langs = load_data(args)
    paragraph_infos = [ParagraphInfo(text, lang) for text, lang in zip(paragraphs, langs)]
    occurrences = build_occurrences(paragraphs)
    token_lang_map = assign_token_langs(occurrences, paragraph_infos)

    encoder = MorphologyEncoder()
    encoder.fit(paragraph_infos, occurrences, lambda i: paragraph_infos[i].lang)
    if not encoder.token_vec:
        raise SystemExit("MorphologyEncoder training produced no embeddings.")

    token_vectors: Dict[str, Tuple[str | None, torch.Tensor]] = {}
    for tok, vec in encoder.token_vec.items():
        norm = torch.linalg.norm(vec)
        if norm.item() == 0.0:
            continue
        lang = token_lang_map.get(tok)
        token_vectors[tok] = (lang, (vec / norm).detach().cpu())

    class_results = {}
    for class_key in args.classes or DEFAULT_CLASSES:
        suffix_map = CROSS_EQUIV.get(class_key, {})
        if not suffix_map:
            continue
        class_lang_items: Dict[str, List[Tuple[str, Optional[str], torch.Tensor]]] = defaultdict(list)
        other_lang_items: Dict[str, List[Tuple[str, Optional[str], torch.Tensor]]] = defaultdict(list)
        languages = set(suffix_map.keys())

        for tok, (lang, vec) in token_vectors.items():
            if lang not in languages:
                continue
            entry = (tok, lang, vec)
            if token_matches_class(tok, lang, class_key):
                class_lang_items[lang].append(entry)
            else:
                other_lang_items[lang].append(entry)

        same_lang_stats = {}
        for lang in sorted(languages):
            cls_items = class_lang_items.get(lang, [])
            base_items = other_lang_items.get(lang, [])
            cls_sim = avg_same_language(cls_items)
            base_sim = avg_same_language(base_items)
            delta = (
                None
                if cls_sim is None or base_sim is None
                else cls_sim - base_sim
            )
            same_lang_stats[lang] = {
                "class_sim": cls_sim,
                "baseline_sim": base_sim,
                "delta": delta,
                "class_token_count": len(cls_items),
                "baseline_token_count": len(base_items),
                "class_pair_samples": sample_within_pairs(cls_items),
                "baseline_pair_samples": sample_within_pairs(base_items),
            }

        cross_lang_stats = {}
        langs_list = sorted(languages)
        for i in range(len(langs_list)):
            for j in range(i + 1, len(langs_list)):
                lang_a, lang_b = langs_list[i], langs_list[j]
                cls_items_a = class_lang_items.get(lang_a, [])
                cls_items_b = class_lang_items.get(lang_b, [])
                base_items_a = other_lang_items.get(lang_a, [])
                base_items_b = other_lang_items.get(lang_b, [])
                cls_sim = avg_cross_language(cls_items_a, cls_items_b)
                base_sim = avg_cross_language(base_items_a, base_items_b)
                delta = (
                    None
                    if cls_sim is None or base_sim is None
                    else cls_sim - base_sim
                )
                cross_lang_stats[f"{lang_a}-{lang_b}"] = {
                    "class_sim": cls_sim,
                    "baseline_sim": base_sim,
                    "delta": delta,
                    "class_token_count_a": len(cls_items_a),
                    "class_token_count_b": len(cls_items_b),
                    "baseline_token_count_a": len(base_items_a),
                    "baseline_token_count_b": len(base_items_b),
                    "class_pair_samples": sample_cross_pairs(cls_items_a, cls_items_b),
                    "baseline_pair_samples": sample_cross_pairs(base_items_a, base_items_b),
                }

        all_class_items = [item for items in class_lang_items.values() for item in items]
        all_baseline_items = [item for items in other_lang_items.values() for item in items]
        pca_payload = compute_pca_projection(all_class_items, all_baseline_items)

        class_results[class_key] = {
            "same_language": same_lang_stats,
            "cross_language": cross_lang_stats,
            "pca": pca_payload,
        }

    payload = {
        "classes": class_results,
        "total_tokens": len(token_vectors),
    }
    maybe_plot_pcas(args, class_results)
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_plot_pcas(args: argparse.Namespace, class_results: Dict[str, Dict[str, object]]) -> None:
    requested: List[str] = []
    if args.plot_class:
        requested.extend(args.plot_class)
    if args.plot_classes:
        requested.extend(args.plot_classes)
    targets: List[str] = []
    seen = set()
    for key in requested:
        if key and key not in seen:
            targets.append(key)
            seen.add(key)
    if not targets:
        return
    if plt is None:
        print("[warn] matplotlib not available; skipping PCA plot generation.")
        return

    plot_paths: Dict[str, Path] = {}
    if len(targets) == 1:
        key = targets[0]
        if args.plot_output:
            plot_paths[key] = args.plot_output
        elif args.plot_dir:
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            plot_paths[key] = args.plot_dir / f"{key}.png"
        else:
            plot_paths[key] = Path(f"{key}_pca.png")
    else:
        if args.plot_dir:
            plot_dir = args.plot_dir
        elif args.plot_output and (args.plot_output.is_dir() or args.plot_output.suffix == ""):
            plot_dir = args.plot_output
        else:
            print("[warn] Multiple plot classes requested but no --plot-dir provided; skipping plots.")
            return
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {key: plot_dir / f"{key}.png" for key in targets}

    for key, dest in plot_paths.items():
        class_payload = class_results.get(key)
        if not class_payload:
            print(f"[warn] No data for class '{key}', skipping PCA plot.")
            continue
        pca_data = class_payload.get("pca")
        if not pca_data:
            print(f"[warn] Class '{key}' has no PCA data, skipping plot.")
            continue
        points = pca_data.get("points", [])
        if not points:
            print(f"[warn] Class '{key}' PCA has no points.")
            continue

        highlight_tokens: set[str] = set()
        same_lang = class_payload.get("same_language", {})
        for lang_stats in same_lang.values():
            for sample in lang_stats.get("class_pair_samples", []):
                token_a = sample.get("token_a")
                token_b = sample.get("token_b")
                if token_a and token_b:
                    highlight_tokens.update([token_a, token_b])
                    break
            if highlight_tokens:
                break

        fig, ax = plt.subplots(figsize=(6, 5))
        class_x, class_y = [], []
        base_x, base_y = [], []
        for pt in points:
            x, y = pt["x"], pt["y"]
            if pt.get("is_class"):
                class_x.append(x)
                class_y.append(y)
            else:
                base_x.append(x)
                base_y.append(y)
        ax.scatter(base_x, base_y, c="#bbbbbb", label="Baseline", s=30, alpha=0.7)
        ax.scatter(class_x, class_y, c="#1f77b4", label="Class", s=40, alpha=0.85)

        if highlight_tokens:
            hx, hy = [], []
            for pt in points:
                if pt["token"] in highlight_tokens:
                    hx.append(pt["x"])
                    hy.append(pt["y"])
            if hx:
                ax.scatter(hx, hy, c="#ff7f0e", s=120, marker="*", label="Example pair")

        ax.set_title(f"PCA projection for class {key}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        fig.tight_layout()

        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest)
        plt.close(fig)
        print(f"[info] Saved PCA scatter for '{key}' to {dest}")


if __name__ == "__main__":
    main()


