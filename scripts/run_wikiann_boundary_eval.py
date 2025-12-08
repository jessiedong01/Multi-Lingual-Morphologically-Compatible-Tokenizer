#!/usr/bin/env python3
"""
Evaluate three tokenizers on WikiANN paragraphs for multiple languages,
compare against UniSegments boundaries, and plot aggregate results vertically.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer

LANGS = ["en", "hu", "tr", "ru", "de", "sv", "bn", "fa", "ca"]
HF_MODEL_IDS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}
TIKTOKEN_ENCODINGS = {
    "gpt4o_o200k": "cl100k_base",
}
MODEL_ORDER = ["gpt4o_o200k", "qwen2.5-7b", "mistral-7b"]

def _normalize(text: str) -> str:
    """Case-fold + NFC normalize so Cyrillic/Hungarian words remain intact."""
    text = unicodedata.normalize("NFC", text)
    return text.casefold()

def _token_boundaries(text: str, tokens: Sequence[str]) -> set[int]:
    bounds = set(); cursor = 0
    for tok in tokens[:-1]:
        cursor += len(tok)
        bounds.add(cursor)
    return bounds

def _load_gold_entries(root: Path, targets: Sequence[str]) -> Dict[str, Mapping[str, object]]:
    target_keys = {_normalize(w): w for w in targets}
    gold: Dict[str, Mapping[str, object]] = {}
    for json_path in sorted(root.rglob('*.jsonl')):
        with json_path.open(encoding='utf-8') as fh:
            for line in fh:
                rec = json.loads(line)
                key = _normalize(rec['word'])
                if key in target_keys and key not in gold:
                    gold[key] = rec
            if len(gold) == len(target_keys):
                break
    return gold

def _collect_samples(lang: str, limit: int) -> List[str]:
    dataset = load_dataset('wikiann', lang, split='train', streaming=True)
    texts = []
    for row in dataset:
        tokens = row.get('tokens', [])
        text = ' '.join(tokens)
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts

def _tokenize(model_name: str, tokenizer, text: str) -> List[str]:
    if model_name in TIKTOKEN_ENCODINGS:
        token_ids = tokenizer.encode(text)
        pieces = [
            tokenizer.decode_single_token_bytes(tok).decode("utf-8", errors="replace")
            for tok in token_ids
        ]
        return pieces if pieces else list(text)
    tokens = tokenizer.tokenize(text)
    return tokens if tokens else list(text)

def evaluate_language(lang: str, limit: int, gold_root: Path, tokenizers: Mapping[str, object]) -> Dict[str, Tuple[float, float, float]]:
    texts = _collect_samples(lang, limit)
    words = [word for text in texts for word in text.split()]
    gold = _load_gold_entries(gold_root, words)
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # P, R, F1, count
    for word in words:
        entry = gold.get(_normalize(word))
        if not entry:
            continue
        gold_bounds = set(int(b) for b in entry.get('boundaries', []))
        for model_name, tokenizer in tokenizers.items():
            cand_tokens = _tokenize(model_name, tokenizer, word)
            cand_bounds = _token_boundaries(word, cand_tokens)
            tp = len(cand_bounds & gold_bounds)
            precision = tp / len(cand_bounds) if cand_bounds else 1.0
            recall = tp / len(gold_bounds) if gold_bounds else 1.0
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            sums[model_name][0] += precision
            sums[model_name][1] += recall
            sums[model_name][2] += f1
            sums[model_name][3] += 1
    return {
        model: (
            totals[0] / totals[3] if totals[3] else 0.0,
            totals[1] / totals[3] if totals[3] else 0.0,
            totals[2] / totals[3] if totals[3] else 0.0,
        )
        for model, totals in sums.items()
    }

def plot_f1(results: Mapping[str, Mapping[str, Tuple[float, float, float]]], output: Path) -> None:
    models = MODEL_ORDER
    x = np.arange(len(models))
    width = 0.12
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, lang in enumerate(LANGS):
        lang_metrics = results.get(lang, {})
        offsets = x + (idx - len(LANGS) / 2) * width
        f1_vals = [lang_metrics.get(model, (0, 0, 0))[2] for model in models]
        ax.bar(offsets, f1_vals, width=width, label=lang)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniseg-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('wikiann_word_metrics_vertical.png'))
    args = parser.parse_args()

    tokenizers: Dict[str, object] = {}
    for name, hf_id in HF_MODEL_IDS.items():
        tokenizers[name] = AutoTokenizer.from_pretrained(hf_id)
    for name, encoding in TIKTOKEN_ENCODINGS.items():
        tokenizers[name] = tiktoken.get_encoding(encoding)
    results = {lang: evaluate_language(lang, args.limit, args.uniseg_root, tokenizers) for lang in LANGS}
    output_csv = args.output.with_suffix('.csv')
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['lang', 'model', 'f1'])
        for lang, lang_metrics in results.items():
            for model, (_, _, f1) in lang_metrics.items():
                writer.writerow([lang, model, f'{f1:.4f}'])
    plot_f1(results, args.output)
    print(f'Saved metrics to {output_csv} and plot to {args.output}')

if __name__ == '__main__':
    main()
