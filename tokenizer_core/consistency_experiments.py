"""
Experiment harness to compare tokenizer behavior with and without the
morphology consistency loss.
"""

import re
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import torch

from .tokenizer import ScalableTokenizer
from .constants import CROSS_EQUIV
from data import load_wikipedia_corpus
from .torch_utils import default_device

EVAL_SENTENCES: Sequence[Tuple[str, str]] = [
    ("The workers are running tests on the devices.", "en"),
    ("Die Fahrer reparieren Wagen und bauen Katamarane.", "de"),
    ("Ogrenciler projeler hazirliyor ve oyunlar kuruyor.", "tr"),
]

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def build_lexicon(samples: Sequence[Tuple[str, str]], boost: float = 30.0) -> Dict[str, float]:
    seen = {}
    for text, _ in samples:
        for word in WORD_RE.findall(text):
            seen[word] = boost
    return seen


def build_corpus_lexicon(texts: Sequence[str], langs: Sequence[str], sample_limit: int = 500) -> Dict[str, float]:
    sample_pairs = list(zip(texts[:sample_limit], langs[:sample_limit]))
    sample_pairs.extend(EVAL_SENTENCES)
    return build_lexicon(sample_pairs, boost=30.0)


DEVICE = default_device()


def _mean_direction(vectors: List[torch.Tensor]) -> torch.Tensor | None:
    if not vectors:
        return None
    mat = torch.stack(vectors, dim=0)
    avg = mat.mean(dim=0)
    norm = torch.linalg.norm(avg)
    if norm.item() == 0.0:
        return None
    return avg / norm


def _collect_class_vectors(encoder, class_key: str) -> Dict[str, List[torch.Tensor]]:
    lang_vectors: Dict[str, List[torch.Tensor]] = {}
    suffix_map = CROSS_EQUIV.get(class_key, {})
    for tok, lang in encoder.token_lang.items():
        suffixes = suffix_map.get(lang)
        if not suffixes:
            continue
        if any(tok.endswith(suf) for suf in suffixes):
            vec = encoder.token_vec.get(tok)
            if vec is None:
                continue
            lang_vectors.setdefault(lang, []).append(vec)
    return lang_vectors


def summarize_consistency(encoder) -> Dict[str, Dict]:
    summary = {}
    for class_key in CROSS_EQUIV.keys():
        lang_vectors = _collect_class_vectors(encoder, class_key)
        if len(lang_vectors) < 2:
            continue
        pairs = []
        for (lang_a, vecs_a), (lang_b, vecs_b) in combinations(lang_vectors.items(), 2):
            mean_a = _mean_direction(vecs_a)
            mean_b = _mean_direction(vecs_b)
            if mean_a is None or mean_b is None:
                continue
            cos = float(torch.dot(mean_a, mean_b).item())
            pairs.append(((lang_a, lang_b), cos))
        if pairs:
            avg = sum(score for _, score in pairs) / len(pairs)
            summary[class_key] = {"avg_cos": avg, "pairs": pairs}
    return summary


def format_summary(label: str, summary: Dict[str, Dict]) -> None:
    print(f"\n=== Consistency metrics: {label} ===")
    if not summary:
        print("No comparable classes were found.")
        return
    for class_key, stats in summary.items():
        print(f"{class_key}: avg cosine = {stats['avg_cos']:.4f}")
        for (la, lb), score in stats["pairs"]:
            print(f"  {la}-{lb}: {score:.4f}")


def train_variant(label: str, texts: Sequence[str], langs: Sequence[str], morph_cfg: Dict, lexicon: Dict[str, float]) -> ScalableTokenizer:
    print(f"\n--- Training tokenizer: {label} ---")
    tok = ScalableTokenizer(
        max_token_len=16,
        min_freq=1,
        top_k_add=8,
        vocab_budget=None,
        tau=0.02,
    )
    tok.set_feature_models(
        morphology_config=morph_cfg,
        lexicon=lexicon,
        space_penalty=5.0,
    )
    tok.train(texts, langs, max_iterations=40, verbose=False)
    tok.ensure_vocab_tokens(lexicon.keys())
    return tok


def compare_tokenizations(baseline: ScalableTokenizer, tuned: ScalableTokenizer) -> None:
    print("\n=== Tokenization comparison ===")
    for text, lang in EVAL_SENTENCES:
        base_tokens = baseline.tokenize(text, lang=lang)
        tuned_tokens = tuned.tokenize(text, lang=lang)
        print(f"[{lang}] {text}")
        print(f"  no-consistency : {base_tokens}")
        print(f"  consistency    : {tuned_tokens}")


def run_experiments(per_lang: int = 800, lexicon_sample_limit: int = 500) -> None:
    texts, langs = load_wikipedia_corpus(per_lang=per_lang)
    if not texts:
        print("No corpus data available; aborting experiment.")
        return

    lexicon_hints = build_corpus_lexicon(texts, langs, sample_limit=lexicon_sample_limit)

    baseline_cfg = {"lambda_morph": 0.0, "refine_steps": 0}
    tuned_cfg = {"lambda_morph": 0.2, "refine_steps": 50, "refine_lr": 0.05}

    baseline_tok = train_variant("no-consistency", texts, langs, baseline_cfg, lexicon_hints)
    tuned_tok = train_variant("consistency-loss", texts, langs, tuned_cfg, lexicon_hints)

    base_summary = summarize_consistency(baseline_tok._ling.morph_encoder)
    tuned_summary = summarize_consistency(tuned_tok._ling.morph_encoder)

    format_summary("no-consistency", base_summary)
    format_summary("consistency-loss", tuned_summary)

    compare_tokenizations(baseline_tok, tuned_tok)


if __name__ == "__main__":
    run_experiments()
