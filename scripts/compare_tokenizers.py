#!/usr/bin/env python3
"""
Train two tokenizer variants (with/without UniSeg reward) on WikiANN text,
evaluate the full benchmark suite, and produce ACL-style bar plots.
"""

from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:
    raise ImportError("The 'datasets' package is required. Install it via 'pip install datasets'.") from exc

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_pipeline import (
    compute_compactness,
    compute_distribution_metrics,
    compute_morph_metrics,
    tokenize_corpus,
    train_lm,
)
from tokenizer_core.segmentation_eval import _iter_uniseg_entries, evaluate_with_uniseg
from tokenizer_core.tokenizer import ScalableTokenizer
from tokenizer_core.morphology_eval import evaluate_morphology_encoder
def serialize_morph_eval(result) -> Dict[str, object]:
    summary = {
        "random_similarity": result.random_similarity,
        "length_correlation": result.length_correlation,
        "classes": {},
    }
    for key, stats in result.classes.items():
        summary["classes"][key] = {
            "token_count": stats.token_count,
            "languages": stats.languages,
            "avg_similarity": stats.avg_similarity,
            "gain_vs_random": stats.gain_vs_random,
            "cross_similarity": stats.cross_similarity,
            "cross_pairs": stats.cross_pairs,
            "cross_baseline": stats.cross_baseline,
            "cross_baseline_pairs": stats.cross_baseline_pairs,
            "cross_gain_vs_baseline": stats.cross_gain_vs_baseline,
        }
    return summary

LANG_TO_UNISEG = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "tr": "tur",
    "ru": "rus",
    "hu": "hun",
    "sv": "swe",
    "bn": "ben",
    "fa": "fas",
    "ca": "cat",
    "es": "spa",
    "pt": "por",
}


@dataclass
class LMArgs:
    lm_device: str
    lm_epochs: int
    lm_block_size: int
    lm_batch_size: int
    lm_embed_dim: int
    lm_hidden_dim: int
    lm_lr: float
    lm_clip: float
    lm_max_windows: int
    lm_val_ratio: float
    lm_seed: int
    lm_patience: int


def load_wikiann_corpus(langs: Sequence[str], samples_per_lang: int) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    lang_tags: List[str] = []
    for lang in langs:
        dataset = load_dataset("wikiann", lang, split="train", streaming=True)
        count = 0
        for entry in dataset:
            tokens = entry.get("tokens", [])
            if not tokens:
                continue
            text = " ".join(tokens).strip()
            if not text:
                continue
            texts.append(text)
            lang_tags.append(lang)
            count += 1
            if count >= samples_per_lang:
                break
        if count == 0:
            raise RuntimeError(f"Unable to load WikiANN data for language '{lang}'.")
    return texts, lang_tags


def compute_uniseg_f1(tokenizer: ScalableTokenizer, uniseg_root: Path, langs: Sequence[str], limit: int) -> Tuple[float, Dict[str, float]]:
    lang_scores: Dict[str, float] = {}
    for lang in langs:
        folder = LANG_TO_UNISEG.get(lang, lang)
        jsonl_path = uniseg_root / folder / "MorphoLex.jsonl"
        if not jsonl_path.exists():
            continue
        entries = list(_iter_uniseg_entries(jsonl_path, limit))
        if not entries:
            continue
        metrics = evaluate_with_uniseg(tokenizer, lang, entries)
        lang_scores[lang] = metrics.get("f1", 0.0)
    aggregate = float(np.mean(list(lang_scores.values()))) if lang_scores else float("nan")
    return aggregate, lang_scores


def train_and_evaluate(
    name: str,
    tokenizer_args: dict,
    corpus: List[str],
    langs: List[str],
    eval_corpus: List[str],
    eval_langs: List[str],
    morph_file: Path,
    morph_lang: str,
    morph_limit: int,
    uniseg_root: Path,
    uniseg_langs: Sequence[str],
    uniseg_limit: int,
    lm_args,
    max_iterations: int,
    output_dir: Path,
    quiet: bool,
) -> Dict[str, float]:
    subdir = output_dir / name
    subdir.mkdir(parents=True, exist_ok=True)

    tokenizer_kwargs = dict(tokenizer_args)
    disable_affix = tokenizer_kwargs.pop("disable_affix_reward", False)
    tokenizer = ScalableTokenizer(**tokenizer_kwargs)
    if disable_affix:
        tokenizer._ling.prefix_reward = 0.0
        tokenizer._ling.suffix_reward = 0.0
    tokenizer.train(corpus, langs, max_iterations=max_iterations, verbose=not quiet)
    tokenizer.save(subdir / "tokenizer_checkpoint.json", include_morphology=True)

    tokenized_eval = tokenize_corpus(tokenizer, eval_corpus, eval_langs)
    metrics: Dict[str, float] = {}
    metrics.update(compute_compactness(tokenized_eval, eval_corpus))
    metrics.update(compute_distribution_metrics(tokenized_eval))
    morph_metrics = compute_morph_metrics(tokenizer, morph_file, morph_lang, morph_limit)
    metrics.update({f"morph_{k}": v for k, v in morph_metrics.items()})
    encoder = getattr(getattr(tokenizer, "_ling", None), "morph_encoder", None)
    if encoder and getattr(encoder, "token_vec", None):
        morph_eval = evaluate_morphology_encoder(encoder)
        metrics["morph_intrinsic"] = serialize_morph_eval(morph_eval)
    uniseg_f1, per_lang_f1 = compute_uniseg_f1(tokenizer, uniseg_root, uniseg_langs, uniseg_limit)
    metrics["uniseg_boundary_f1"] = uniseg_f1
    metrics["per_lang_uniseg_f1"] = per_lang_f1
    metrics.update(train_lm(tokenized_eval, tokenizer.tok2id, lm_args, doc_langs=eval_langs))

    (subdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_training_task(payload):
    (
        name,
        tokenizer_args,
        corpus,
        langs,
        eval_corpus,
        eval_langs,
        morph_file,
        morph_lang,
        morph_limit,
        uniseg_root,
        uniseg_langs,
        uniseg_limit,
        lm_args,
        max_iterations,
        output_dir,
        quiet,
    ) = payload
    print(f"[compare_tokenizers] Starting '{name}' on device {tokenizer_args.get('device')} (pid={os.getpid()})")
    metrics = train_and_evaluate(
        name,
        tokenizer_args,
        corpus,
        langs,
        eval_corpus,
        eval_langs,
        morph_file,
        morph_lang,
        morph_limit,
        uniseg_root,
        uniseg_langs,
        uniseg_limit,
        lm_args,
        max_iterations,
        output_dir,
        quiet,
    )
    return name, metrics


def plot_metrics(results: Dict[str, Dict[str, float]], labels: Dict[str, str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    palette = list(plt.get_cmap("tab10").colors)
    metric_specs = [
        ("tpc", "Tokens per Char (↓)"),
        ("tpw", "Tokens per Word (↓)"),
        ("entropy", "Entropy (↑)"),
        ("gini", "Gini (↓)"),
        ("rare_type_fraction", "Rare Type Fraction (↓)"),
        ("morph_spm", "Subwords per Morpheme (↓)"),
        ("morph_boundary_f1", "Morph Boundary F1 (↑)"),
        ("uniseg_boundary_f1", "UniSeg Boundary F1 (↑)"),
        ("lm_perplexity", "LM Perplexity (↓)"),
    ]
    names = list(labels.keys())
    x = np.arange(len(names))
    for metric, label in metric_specs:
        values = [results[name].get(metric, float("nan")) for name in names]
        fig_width = max(3.2, 1.0 + 0.9 * len(values))
        fig, ax = plt.subplots(figsize=(fig_width, 2.2))
        colors = [palette[i % len(palette)] for i in range(len(values))]
        bars = ax.bar(x, values, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[name] for name in names], rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, pad=4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.pdf", bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare tokenizer variants on WikiANN.")
    parser.add_argument("--wikiann_langs", nargs="+", default=["en"], help="Languages to sample from WikiANN.")
    parser.add_argument("--samples_per_lang", type=int, default=200, help="Number of WikiANN sentences per language.")
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--uniseg_root", type=Path, default=Path("data/uniseg_word_segments"))
    parser.add_argument("--uniseg_limit", type=int, default=1000)
    parser.add_argument("--morph_file", type=Path, default=Path("data/uniseg_word_segments/eng/MorphoLex.jsonl"))
    parser.add_argument("--morph_lang", default="en")
    parser.add_argument("--morph_limit", type=int, default=1000)
    parser.add_argument("--device", default=None, help="Primary device (e.g., cuda:0). Auto-detected if omitted.")
    parser.add_argument("--secondary_device", default=None, help="Device for baseline tokenizer. Auto if omitted.")
    parser.add_argument("--top_k_add", type=int, default=32)
    parser.add_argument("--vocab_budget", type=int, default=4000)
    parser.add_argument("--pricing_device", default="cpu")
    parser.add_argument("--lm_device", default="cuda")
    parser.add_argument("--lm_epochs", type=int, default=8)
    parser.add_argument("--lm_block_size", type=int, default=128)
    parser.add_argument("--lm_batch_size", type=int, default=32)
    parser.add_argument("--lm_embed_dim", type=int, default=128)
    parser.add_argument("--lm_hidden_dim", type=int, default=256)
    parser.add_argument("--lm_lr", type=float, default=1e-3)
    parser.add_argument("--lm_clip", type=float, default=1.0)
    parser.add_argument("--lm_max_windows", type=int, default=20000)
    parser.add_argument("--lm_val_ratio", type=float, default=0.15)
    parser.add_argument("--lm_patience", type=int, default=3, help="Early-stop GRU training after N bad epochs (0 disables).")
    parser.add_argument("--lm_seed", type=int, default=13, help="Seed for GRU probe shuffling.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def select_devices(primary_arg: str | None, secondary_arg: str | None) -> Tuple[str, str]:
    import torch

    available = []
    if torch.cuda.is_available():
        available = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not available:
        available = ["cpu"]

    primary = primary_arg or available[0]
    if primary == "auto":
        primary = available[0]

    secondary = secondary_arg or None
    if secondary is None or secondary == "auto":
        if len(available) > 1:
            for dev in available:
                if dev != primary:
                    secondary = dev
                    break
        if secondary is None:
            secondary = primary
    return primary, secondary


def main():
    args = parse_args()
    primary_device, secondary_device = select_devices(args.device, args.secondary_device)
    args.device = primary_device
    args.secondary_device = secondary_device
    corpus, langs = load_wikiann_corpus(args.wikiann_langs, args.samples_per_lang)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lm_args = LMArgs(
        lm_device=args.lm_device,
        lm_epochs=args.lm_epochs,
        lm_block_size=args.lm_block_size,
        lm_batch_size=args.lm_batch_size,
        lm_embed_dim=args.lm_embed_dim,
        lm_hidden_dim=args.lm_hidden_dim,
        lm_lr=args.lm_lr,
        lm_clip=args.lm_clip,
        lm_max_windows=args.lm_max_windows,
        lm_val_ratio=args.lm_val_ratio,
        lm_seed=args.lm_seed,
        lm_patience=args.lm_patience,
    )

    base_kwargs = dict(
        max_token_len=12,
        min_freq=3,
        alpha=1.0,
        beta=0.5,
        tau=0.01,
        top_k_add=args.top_k_add,
        vocab_budget=args.vocab_budget,
        merge_reward=0.05,
        short_penalty=0.2,
        space_penalty=0.25,
        device=args.device,
        uniseg_root=args.uniseg_root,
        seed_uniseg_segments=True,
        pricing_device=args.pricing_device,
        use_morph_encoder=False,
    )

    configs = {
        "uniseg": dict(base_kwargs, uniseg_reward=0.3, force_seed_uniseg_tokens=False),
        "baseline": dict(
            base_kwargs,
            uniseg_reward=0.0,
            seed_uniseg_segments=False,
            force_seed_uniseg_tokens=False,
            disable_affix_reward=True,
            device=args.secondary_device or args.device,
        ),
    }
    labels = {"uniseg": "UniSeg Reward", "baseline": "No UniSeg"}

    payloads = [
        (
            name,
            params,
            corpus,
            langs,
            corpus,
            langs,
            args.morph_file,
            args.morph_lang,
            args.morph_limit,
            args.uniseg_root,
            args.wikiann_langs,
            args.uniseg_limit,
            lm_args,
            args.max_iterations,
            output_dir,
            args.quiet,
        )
        for name, params in configs.items()
    ]

    results: Dict[str, Dict[str, float]] = {}
    if len(payloads) == 1:
        name, metrics = _run_training_task(payloads[0])
        results[name] = metrics
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as executor:
            future_to_name = {executor.submit(_run_training_task, payload): payload[0] for payload in payloads}
            for future in as_completed(future_to_name):
                name, metrics = future.result()
                results[name] = metrics

    plot_metrics(results, labels, output_dir / "plots")
    summary_path = output_dir / "comparison_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
