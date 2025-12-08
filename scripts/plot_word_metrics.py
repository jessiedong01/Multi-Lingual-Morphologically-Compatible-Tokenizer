#!/usr/bin/env python3
"""Plot word-level boundary metrics vertically."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_metrics(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows

def plot(rows: List[Dict[str, str]], output: Path) -> None:
    words = sorted(set(row['word'] for row in rows))
    models = sorted(set(row['model'] for row in rows))
    width = 0.2
    fig, axes = plt.subplots(len(words), 1, figsize=(6, 3.5 * len(words)), sharex=True)
    if len(words) == 1:
        axes = [axes]
    for ax, word in zip(axes, words):
        word_rows = [row for row in rows if row['word'] == word]
        for j, metric in enumerate(['precision', 'recall', 'f1']):
            vals = []
            for model in models:
                match = next((row for row in word_rows if row['model'] == model), None)
                if not match or match[metric].upper() == 'MISSING':
                    vals.append(0.0)
                else:
                    vals.append(float(match[metric]))
            offsets = [i + (j - 1) * width for i in range(len(models))]
            ax.bar(offsets, vals, width=width, label=metric.title())
        ax.set_ylabel(word)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    axes[-1].set_xticks(range(len(models)))
    axes[-1].set_xticklabels(models, rotation=30, ha='right')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot boundary metrics vertically.')
    parser.add_argument('--metrics-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('word_level_metrics_vertical.png'))
    args = parser.parse_args()
    rows = load_metrics(args.metrics_csv)
    if not rows:
        raise ValueError('Metrics CSV is empty.')
    plot(rows, args.output)
    print(f'Saved plot to {args.output}')

if __name__ == '__main__':
    main()
