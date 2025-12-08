#!/usr/bin/env python3
"""
Evaluate how a tokenizer's morphology encoder affects downstream GRU perplexity.

Given one or more tokenizer checkpoints and a shared evaluation corpus, this tool:
  1. Tokenizes the corpus with each checkpoint
  2. Runs the intrinsic morphology encoder metrics (cosine deltas, cross-lingual gains)
  3. Trains the Tiny GRU probe with identical hyperparameters and reports overall / per-language LM metrics

The resulting JSON bundle makes it easy to compare intrinsic vs. extrinsic signals.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - informative error
    raise ImportError("The 'datasets' package is required. Install it via 'pip install datasets'.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_pipeline import tokenize_corpus, train_lm  # noqa: E402
from tokenizer_core.morphology_eval import evaluate_morphology_encoder  # noqa: E402
from tokenizer_core.tokenizer import ScalableTokenizer  # noqa: E402


DEFAULT_LANGS = ("en", "de", "tr")


def read_lines(path: Path, limit: Optional[int] = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[:limit] if limit is not None else lines


def load_local_corpus(corpus: Path, langs: Optional[Path], limit: Optional[int]) -> Tuple[List[str], List[Optional[str]]]:
    texts = read_lines(corpus, limit)
    if langs:
        lang_tags = read_lines(langs, limit)
        if len(lang_tags) != len(texts):
            raise ValueError("Language tags must have the same length as the corpus.")
    else:
        lang_tags = [None] * len(texts)
    return texts, lang_tags


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


def parse_checkpoints(raw: Sequence[str]) -> List[Tuple[str, Path]]:
    pairs = []
    for item in raw:
        if "=" in item:
            label, path_str = item.split("=", 1)
        else:
            label, path_str = Path(item).stem, item
        path = Path(path_str)
        checkpoint = path / "tokenizer_checkpoint.json" if path.is_dir() else path
        if not checkpoint.exists():
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {checkpoint}")
        pairs.append((label, checkpoint))
    return pairs


@dataclass
class LMEvalArgs:
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


def evaluate_checkpoint(
    label: str,
    checkpoint: Path,
    docs: List[str],
    langs: List[Optional[str]],
    lm_args: LMEvalArgs,
    class_keys: Sequence[str],
) -> Dict[str, object]:
    tokenizer = ScalableTokenizer.load_from_file(checkpoint)
    tokenized_eval = tokenize_corpus(tokenizer, docs, langs)
    lm_metrics = train_lm(tokenized_eval, tokenizer.tok2id, lm_args, doc_langs=langs)
    result: Dict[str, object] = {"lm": lm_metrics}
    encoder = getattr(getattr(tokenizer, "_ling", None), "morph_encoder", None)
    if encoder and getattr(encoder, "token_vec", None):
        morph_eval = evaluate_morphology_encoder(encoder, class_keys=class_keys)
        result["intrinsic"] = serialize_morph_eval(morph_eval)
    else:
        result["intrinsic"] = None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate morphology encoder role via intrinsic + GRU metrics.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths or label=path pairs. Directories are resolved to tokenizer_checkpoint.json.",
    )
    parser.add_argument("--corpus", type=Path, help="Optional plaintext corpus (one paragraph per line).")
    parser.add_argument("--langs-file", type=Path, help="Optional language tags aligned with --corpus.")
    parser.add_argument("--limit", type=int, help="Optional limit on number of paragraphs to read from --corpus.")
    parser.add_argument(
        "--wikiann-langs",
        nargs="+",
        default=list(DEFAULT_LANGS),
        help="Fallback WikiANN languages when --corpus is omitted.",
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=200,
        help="WikiANN paragraphs per language (only used when --corpus is omitted).",
    )
    parser.add_argument("--classes", nargs="*", default=["PROG", "PL", "NEG", "PAST", "COMP"])
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report.")
    parser.add_argument("--lm-device", default="cpu")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)

    if args.corpus:
        docs, langs = load_local_corpus(args.corpus, args.langs_file, args.limit)
    else:
        docs, lang_tags = load_wikiann_corpus(args.wikiann_langs, args.samples_per_lang)
        docs = docs[: args.limit] if args.limit else docs
        lang_tags = lang_tags[: args.limit] if args.limit else lang_tags
        langs = lang_tags

    lm_args = LMEvalArgs(
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

    results: Dict[str, object] = {}
    for label, checkpoint in checkpoints:
        print(f"[eval_morph_role] Evaluating '{label}' from {checkpoint}")
        results[label] = evaluate_checkpoint(label, checkpoint, docs, langs, lm_args, args.classes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": {
            "documents": len(docs),
            "languages": list(sorted(set(langs))),
            "source": str(args.corpus) if args.corpus else f"wikiann[{','.join(args.wikiann_langs)}]",
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval_morph_role] Wrote report to {args.output}")


if __name__ == "__main__":
    main()

