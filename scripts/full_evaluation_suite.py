#!/usr/bin/env python3
"""
Unified training + evaluation driver.

This script trains the baseline + UniSeg (and optional morph-aware) tokenizers,
runs the standard benchmark suite, launches the morphology pairwise PCA plots,
and evaluates the morphology encoder's impact on the GRU probe — all from a
single command.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_tokenizers import (  # noqa: E402
    LMArgs,
    load_wikiann_corpus,
    plot_metrics,
    select_devices,
    train_and_evaluate,
)
from scripts.evaluate_morph_role import (  # noqa: E402
    LMEvalArgs,
    evaluate_checkpoint,
)


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
    dev = tokenizer_args.get("device")
    print(f"[full_evaluation_suite] Launching '{name}' on device={dev}")
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


def read_lines(path: Path, limit: int | None = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip()]
    return lines[:limit] if limit is not None else lines


def load_local_corpus(corpus: Path, langs_file: Path, limit: int | None) -> Tuple[List[str], List[str]]:
    texts = read_lines(corpus, limit)
    langs = read_lines(langs_file, limit)
    if len(texts) != len(langs):
        raise ValueError("Corpus and language files must contain the same number of lines.")
    return texts, langs


def dump_shared_corpus(txt_path: Path, lang_path: Path, docs: Sequence[str], langs: Sequence[str]):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(docs), encoding="utf-8")
    lang_path.write_text("\n".join(langs), encoding="utf-8")


def run_pairwise_eval(
    corpus_path: Path,
    langs_path: Path,
    classes: Sequence[str],
    plot_dir: Path,
    json_out: Path,
):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "morph_pairwise_alignment.py"),
        "--corpus",
        str(corpus_path),
        "--langs-file",
        str(langs_path),
        "--classes",
        *classes,
        "--plot-classes",
        *classes,
        "--plot-dir",
        str(plot_dir),
        "--json-out",
        str(json_out),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tokenizers and run all evaluations in one shot.")
    parser.add_argument("--corpus", type=Path, help="Optional local corpus (one paragraph per line).")
    parser.add_argument("--langs-file", type=Path, help="Optional language tags aligned with --corpus.")
    parser.add_argument("--limit", type=int, help="Optional limit when using --corpus.")
    parser.add_argument("--wikiann-langs", nargs="+", default=["en", "de", "tr"])
    parser.add_argument("--samples-per-lang", type=int, default=500)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uniseg-root", type=Path, default=Path("data/uniseg_word_segments"))
    parser.add_argument("--uniseg-limit", type=int, default=1000)
    parser.add_argument("--morph-file", type=Path, default=Path("data/uniseg_word_segments/eng/MorphoLex.jsonl"))
    parser.add_argument("--morph-lang", default="en")
    parser.add_argument("--morph-limit", type=int, default=1000)
    parser.add_argument("--classes", nargs="+", default=["PROG", "PL", "NEG", "PAST", "COMP"])
    parser.add_argument("--pairwise-plot-dir", type=Path, help="Directory for PCA plots (defaults to OUTPUT/pairwise_plots).")
    parser.add_argument("--pairwise-json", type=Path, help="Path for pairwise summary JSON (defaults to OUTPUT/pairwise_eval.json).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--secondary-device", default=None)
    parser.add_argument("--pricing-device", default="cpu")
    parser.add_argument("--morph-device", default=None, help="Device for the morphology-enabled tokenizer (defaults to primary).")
    parser.add_argument("--morph-high-device", default=None, help="Device for the high-reward morph tokenizer (defaults to primary).")
    parser.add_argument("--morph-low-device", default=None, help="Device for the low-reward morph tokenizer (defaults to secondary/primary).")
    parser.add_argument("--top-k-add", type=int, default=32)
    parser.add_argument("--vocab-budget", type=int, default=4000)
    parser.add_argument("--lm-device", default="cuda")
    parser.add_argument("--lm-epochs", type=int, default=8)
    parser.add_argument("--lm-block-size", type=int, default=128)
    parser.add_argument("--lm-batch-size", type=int, default=32)
    parser.add_argument("--lm-embed-dim", type=int, default=128)
    parser.add_argument("--lm-hidden-dim", type=int, default=256)
    parser.add_argument("--lm-lr", type=float, default=1e-3)
    parser.add_argument("--lm-clip", type=float, default=1.0)
    parser.add_argument("--lm-max-windows", type=int, default=20000)
    parser.add_argument("--lm-val-ratio", type=float, default=0.15)
    parser.add_argument("--lm-seed", type=int, default=13)
    parser.add_argument("--lm-patience", type=int, default=3, help="Early-stop GRU training after N non-improving epochs (0 disables).")
    parser.add_argument("--include-morph-variant", action="store_true", help="(Deprecated) Morph variant now runs by default.")
    parser.add_argument("--disable-morph-variant", action="store_true", help="Skip training the morphology-enabled tokenizer.")
    parser.add_argument("--parallel-tokenizers", action="store_true", help="Train tokenizer variants in parallel (one process per variant).")
    parser.add_argument("--morph-high-reward", type=float, default=0.45, help="UniSeg/morph reward weight for the high-bias morph tokenizer.")
    parser.add_argument("--morph-low-reward", type=float, default=0.15, help="UniSeg/morph reward weight for the low-bias morph tokenizer.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    primary_device, secondary_device = select_devices(args.device, args.secondary_device)
    include_morph_variant = True
    if args.disable_morph_variant:
        include_morph_variant = False
    elif args.include_morph_variant:
        include_morph_variant = True

    if args.corpus:
        if not args.langs_file:
            raise ValueError("--langs-file is required when --corpus is provided.")
        docs, langs = load_local_corpus(args.corpus, args.langs_file, args.limit)
        corpus_label = str(args.corpus)
    else:
        docs, langs = load_wikiann_corpus(args.wikiann_langs, args.samples_per_lang)
        corpus_label = f"wikiann[{','.join(args.wikiann_langs)}]"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_corpus = output_dir / "shared_corpus.txt"
    shared_langs = output_dir / "shared_corpus.langs"
    dump_shared_corpus(shared_corpus, shared_langs, docs, langs)

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
        device=primary_device,
        uniseg_root=args.uniseg_root,
        seed_uniseg_segments=True,
        pricing_device=args.pricing_device,
        use_morph_encoder=False,
    )

    configs = OrderedDict()
    configs["baseline"] = dict(
        base_kwargs,
        device=primary_device,
        uniseg_reward=0.0,
        seed_uniseg_segments=False,
        force_seed_uniseg_tokens=False,
        disable_affix_reward=True,
    )
    if include_morph_variant:
        # High and low reward, no encoder
        configs["morph_high_noenc"] = dict(
            base_kwargs,
            device=args.morph_high_device or args.morph_device or primary_device,
            uniseg_reward=args.morph_high_reward,
            use_morph_encoder=False,
            force_seed_uniseg_tokens=False,
        )
        configs["morph_low_noenc"] = dict(
            base_kwargs,
            device=args.morph_low_device or args.morph_device or secondary_device or primary_device,
            uniseg_reward=args.morph_low_reward,
            use_morph_encoder=False,
            force_seed_uniseg_tokens=False,
        )
        # Encoder-enabled variant uses the low reward setting
        configs["morph_mean_enc"] = dict(
            base_kwargs,
            device=args.morph_device or primary_device,
            uniseg_reward=args.morph_low_reward,
            use_morph_encoder=True,
            force_seed_uniseg_tokens=False,
        )

    wikiann_eval_langs = args.wikiann_langs if not args.corpus else sorted({lang for lang in langs if lang})
    payloads = []
    for name, tokenizer_args in configs.items():
        payloads.append(
            (
                name,
                tokenizer_args,
                docs,
                langs,
                docs,
                langs,
                args.morph_file,
                args.morph_lang,
                args.morph_limit,
                args.uniseg_root,
                wikiann_eval_langs,
                args.uniseg_limit,
                lm_args,
                args.max_iterations,
                output_dir,
                args.quiet,
            )
        )

    results = {}
    checkpoints = {}
    if args.parallel_tokenizers and len(payloads) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            for name, metrics in pool.map(_run_training_task, payloads):
                results[name] = metrics
                checkpoints[name] = output_dir / name / "tokenizer_checkpoint.json"
    else:
        for payload in payloads:
            name, metrics = _run_training_task(payload)
            results[name] = metrics
            checkpoints[name] = output_dir / name / "tokenizer_checkpoint.json"

    labels = {
        "baseline": "No UniSeg",
        "morph_high_noenc": "Morph High (no encoder)",
        "morph_low_noenc": "Morph Low (no encoder)",
        "morph_mean_enc": "Morph Mean (encoder)",
    }
    # Drop labels for configs that were not instantiated
    labels = {k: v for k, v in labels.items() if k in configs}
    plot_metrics(results, labels, output_dir / "plots")
    summary_path = output_dir / "comparison_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Morph encoder role evaluation (GRU + intrinsic metrics).
    lm_eval_args = LMEvalArgs(
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
    morph_role = {}
    for name, checkpoint in checkpoints.items():
        morph_role[name] = evaluate_checkpoint(
            name,
            checkpoint,
            docs,
            langs,
            lm_eval_args,
            args.classes,
        )
    morph_role_path = output_dir / "morph_role.json"
    morph_role_path.write_text(json.dumps({"dataset": corpus_label, "results": morph_role}, indent=2), encoding="utf-8")

    # Pairwise PCA evaluation (external script).
    pairwise_plot_dir = args.pairwise_plot_dir or (output_dir / "pairwise_plots")
    pairwise_json = args.pairwise_json or (output_dir / "pairwise_eval.json")
    run_pairwise_eval(shared_corpus, shared_langs, args.classes, pairwise_plot_dir, pairwise_json)

    print(f"[full_evaluation_suite] Completed end-to-end run. Metrics: {summary_path}")
    print(f"[full_evaluation_suite] Morph role report: {morph_role_path}")
    print(f"[full_evaluation_suite] Pairwise PCA JSON: {pairwise_json}")


if __name__ == "__main__":
    main()

