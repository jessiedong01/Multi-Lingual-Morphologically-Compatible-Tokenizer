"""
Analyze morphology encoder embeddings to verify cross-lingual alignment.

This script evaluates whether the morphology encoder learns representations where:
1. Tokens with shared morphological function cluster together (regardless of language)
2. Cross-lingual morpheme pairs are more similar than random token pairs
3. The alignment isn't just "prefer longer tokens"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.tokenizer import ScalableTokenizer
from tokenizer_core.morphology_eval import (
    evaluate_morphology_encoder,
    summarize_morphology_eval,
)

DEFAULT_CLASSES: Sequence[str] = ("PROG", "PL", "NEG", "PAST", "COMP")


def build_multilingual_tokenizer() -> ScalableTokenizer:
    """Build a tokenizer with diverse multilingual data."""
    paragraphs = [
        # English - various morphological forms
        "The workers are running and walking through the buildings.",
        "She talked about the unhappiness and impossibility of change.",
        "The teachers teaching students are hardworking professionals.",
        # German
        "Die Arbeiter laufen und gehen durch die Gebaeude.",
        "Sie sprach ueber die Ungluecklichkeit und Unmoeglichkeit.",
        "Die Lehrer unterrichten fleissig die Studenten.",
        # Turkish
        "Isciler kosuyorlar ve yuruyorlar binalarin icinden.",
        "Mutluluk ve mutsuzluk hakkinda konustu.",
        "Ogretmenler ogrencilere ders veriyorlar.",
    ]
    langs = ["en", "en", "en", "de", "de", "de", "tr", "tr", "tr"]

    tok = ScalableTokenizer(
        max_token_len=12,
        min_freq=1,
        vocab_budget=200,
        device="cpu",
    )
    tok._initialize_stats_and_vocab(paragraphs, langs)
    return tok


def show_nearest_neighbors(encoder, query_tok: str, k: int = 5):
    """Return k nearest neighbors for a query token."""
    if query_tok not in encoder.token_vec:
        return []
    q = encoder.token_vec[query_tok]
    q = q / (torch.linalg.norm(q) + 1e-9)
    neighbors = []
    for tok, vec in encoder.token_vec.items():
        if tok == query_tok:
            continue
        v = vec / (torch.linalg.norm(vec) + 1e-9)
        sim = torch.dot(q, v).item()
        neighbors.append((tok, sim))
    neighbors.sort(key=lambda x: -x[1])
    return neighbors[:k]


def run_analysis():
    print("=" * 70)
    print("MORPHOLOGY ENCODER ALIGNMENT ANALYSIS")
    print("=" * 70)

    print("\nBuilding tokenizer...")
    tokenizer = build_multilingual_tokenizer()
    encoder = tokenizer._ling.morph_encoder

    if encoder is None or not encoder.token_vec:
        print("ERROR: No morphology encoder or empty embeddings")
        return

    print(f"Tokens with embeddings: {len(encoder.token_vec)}")
    print(f"Languages with prototypes: {list(encoder.lang_proto.keys())}")

    # Intrinsic metrics
    print("\n" + "-" * 70)
    print("1. INTRINSIC EVALUATION (cohesion + length bias)")
    print("-" * 70)
    eval_result = evaluate_morphology_encoder(encoder, class_keys=DEFAULT_CLASSES)
    print(summarize_morphology_eval(eval_result))

    # Example nearest neighbors
    print("\n" + "-" * 70)
    print("2. NEAREST NEIGHBOR EXAMPLES")
    print("-" * 70)
    example_queries = ["running", "walking", "ing", "un", "ung", "ler", "lar"]
    for query in example_queries:
        if query.lower() in encoder.token_vec:
            query = query.lower()
        if query not in encoder.token_vec:
            continue
        neighbors = show_nearest_neighbors(encoder, query, k=5)
        if neighbors:
            nn_str = ", ".join([f"{t}({s:.2f})" for t, s in neighbors])
            print(f"   '{query}' -> {nn_str}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        """
If cross-lingual alignment is working:
  - Within-class similarity should be higher than the random baseline.
  - Cross-lingual similarity should be positive (morphemes align across languages).
  - Length correlation should be low (embeddings aren't just encoding length).
  - Nearest neighbors should show morphologically similar tokens.
"""
    )


if __name__ == "__main__":
    run_analysis()
