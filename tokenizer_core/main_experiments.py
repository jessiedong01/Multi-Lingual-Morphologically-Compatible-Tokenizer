# -*- coding: utf-8 -*-
"""
Advanced experiment harness for ScalableTokenizer.

Features
--------
* Manual or config-driven parameter sweeps (JSON config via --config).
* Automatic saving of trained models / debug info per experiment.
* Comprehensive evaluation suite (CPT/TPC, Zipf divergence, fragmentation curves,
  domain compression differential, identifier handling, script fracture rate,
  token allocation balance, perturbation stability, effective context gain, cosine
  similarity across languages).
* Baseline comparisons against reference tokenizers when available (GPT / Gemma /
  Qwen tokenizer backends).
* Result packaging: metrics + plots written to disk and zipped per experiment.
"""

from __future__ import annotations

import argparse
import re
import csv
import itertools
import json
import math
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import unicodedata as ud
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .tokenizer import ScalableTokenizer
from data import load_wikiann_corpus
from .constants import (
    URL_RE,
    EMAIL_RE,
    CROSS_EQUIV,
)
from . import utils
from .embedding_benchmarks import (
    maybe_run_embedding_eval,
    write_embedding_report,
    write_segmentation_report,
)
from .torch_utils import default_device

DEVICE = default_device()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from .segmentation_eval import DEFAULT_UNISEG_ROOT, evaluate_languages_with_backoff

    HAS_SEGMENTATION_EVAL = True
except Exception:
    DEFAULT_UNISEG_ROOT = None
    evaluate_languages_with_backoff = None
    HAS_SEGMENTATION_EVAL = False

# ---------------------------------------------------------------------------
# Default baseline configuration (can be overridden via config file)
# ---------------------------------------------------------------------------

DEFAULT_LANG_CODES = {
    "en": "English",
    "ja": "Japanese",
}
DEFAULT_PER_LANG = 120

DEFAULT_TOKENIZER_ARGS = dict(
    max_token_len=12,
    min_freq=7,
    top_k_add=8,
    vocab_budget=500,
    tau=0.001,
)

DEFAULT_LEXICON = {"New York": 2.0, "San Jose": 1.0, "'s": 0.5}
DEFAULT_NE = {"LOC": {"New York": 2.0, "Berlin": 1.4, "東京": 3.5}}
DEFAULT_TOKEN_BIGRAM = {
    ("<BOS>", "InitCap"): -0.2,
    ("InitCap", "InitCap"): -0.3,
    ("NUM", "NUM"): -0.15,
}

DEFAULT_FEATURE_ARGS = dict(
    gamma_boundary=0.06,
    mu_morph=0.25,
    prefix_reward=0.025,
    suffix_reward=0.01,
    space_penalty=0.1,
    morphology_kwargs=dict(
        embedding_mode="glove",
        lambda_morph=0.08,
        gamma=1e-3,
        refine_lr=0.05,
        refine_steps=20,
        glove_iters=20,
        glove_lr=0.05,
        glove_xmax=80.0,
        glove_alpha=0.75,
        glove_max_pairs=200_000,
        use_minibatch=True,
        batch_size_pairs=8192,
        batch_size_edges=2048,
        batch_size_semantic=512,
        optimizer="adagrad",
        adagrad_eps=1e-8,
        max_tokens=25000,
        ngram_orders=(2, 3, 4),
        use_semantic_consistency=True,
        semantic_lr=0.02,
        semantic_iters=5,
        use_dp_semantic=True,
        use_structure_mapping=False,
        use_cross_kl=False,
        use_dp_eig=True,
        use_iterative_eig=False,
    ),
)

DEFAULT_TRAIN_ARGS = dict(max_iterations=80)

DEFAULT_SEMANTIC_TOGGLES = dict(
    email_reward=-0.25,
    url_reward=-0.35,
    hashtag_reward=-0.05,
)

# ---------------------------------------------------------------------------
# Evaluation corpus configuration (fallback values used if corpus sampling fails).
# ---------------------------------------------------------------------------

EVAL_PARAGRAPHS_PER_LANG = 10

FALLBACK_EVAL_SAMPLES = [
    {
        "text": "R.H. Saunders ( St. Lawrence River ) ( 968 MW )",
        "language": "en",
        "domain": "news",
    },
    {
        "text": "Karl Ove Knausgård ( born 1968 )",
        "language": "en",
        "domain": "biography",
    },
    {
        "text": "Atlantic City , New Jersey",
        "language": "en",
        "domain": "news",
    },
    {
        "text": "Her daughter from the second marriage was Marie d'Agoult ( 1805-1876 ), who in turn gave birth to several children, among them—via her liaison to Franz Liszt—Cosima Wagner ( 1837-1930 ).",
        "language": "en",
        "domain": "biography",
    },
    {
        "text": "St. Mary 's Catholic Church ( Sandusky , Ohio )",
        "language": "en",
        "domain": "news",
    },
    {
        "text": "表現の最終テストです。",
        "language": "ja",
        "domain": "chat",
    },
    {
        "text": "メールは alice@example.com に送ってください。https://example.org/docs を参照。",
        "language": "ja",
        "domain": "web",
    },
    {
        "text": "def compute_average(values):\n    total = sum(values)\n    return total / len(values)",
        "language": "en",
        "domain": "code",
    },
]


def build_eval_samples_from_corpus(
    texts: List[str],
    langs: List[str],
    lang_codes: Dict[str, str],
    samples_per_lang: int = EVAL_PARAGRAPHS_PER_LANG,
) -> List[dict]:
    """Take the first N paragraphs per language from the loaded corpus."""
    if samples_per_lang <= 0 or not texts:
        return []
    counts = {code: 0 for code in lang_codes}
    samples: List[dict] = []
    for text, lang in zip(texts, langs):
        if lang not in counts:
            continue
        if counts[lang] >= samples_per_lang:
            continue
        samples.append({"text": text, "language": lang, "domain": "corpus"})
        counts[lang] += 1
        if all(count >= samples_per_lang for count in counts.values()):
            break
    return samples

# ---------------------------------------------------------------------------
# Reference tokenizers (optional)
# ---------------------------------------------------------------------------


def load_reference_tokenizers() -> Dict[str, Callable[[str], Sequence[str]]]:
    references = {}
    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding("cl100k_base")

        def gpt_encode(text: str) -> Sequence[str]:
            ids = enc.encode(text)
            # For consistent comparison we map back to strings (approx by decoding each token)
            return [enc.decode_single_token_bytes(tok).decode("utf-8", errors="replace") for tok in ids]

        references["gpt-3.5-tiktoken"] = gpt_encode
    if HAS_TRANSFORMERS:
        for model_name in ["google/gemma-2b", "qwen1.5-1.8b-chat"]:
            try:
                tok = AutoTokenizer.from_pretrained(model_name)

                def hf_encode(text: str, tokenizer=tok):
                    tokens = tokenizer.tokenize(text)
                    return tokens

                references[model_name] = hf_encode
            except Exception:
                continue
    return references


# ---------------------------------------------------------------------------
# Config parsing & experiment expansion
# ---------------------------------------------------------------------------


def load_config(path: Optional[str]):
    if not path:
        return (
            DEFAULT_LANG_CODES,
            DEFAULT_PER_LANG,
            DEFAULT_TOKENIZER_ARGS,
            DEFAULT_FEATURE_ARGS,
            DEFAULT_TRAIN_ARGS,
            DEFAULT_SEMANTIC_TOGGLES,
            None,
            None,
        )
    cfg = json.loads(Path(path).read_text())
    lang_codes = cfg.get("base_lang_codes", DEFAULT_LANG_CODES)
    per_lang = cfg.get("per_lang", DEFAULT_PER_LANG)
    tok_args = DEFAULT_TOKENIZER_ARGS.copy()
    tok_args.update(cfg.get("base_tokenizer_args", {}))
    feature_args = DEFAULT_FEATURE_ARGS.copy()
    feature_args.update(cfg.get("base_feature_args", {}))
    train_args = DEFAULT_TRAIN_ARGS.copy()
    train_args.update(cfg.get("base_train_args", {}))
    semantic_toggles = DEFAULT_SEMANTIC_TOGGLES.copy()
    semantic_toggles.update(cfg.get("semantic_toggles", {}))
    experiments = cfg.get("experiments", None)
    external_eval = cfg.get("external_eval")
    embedding_eval = cfg.get("embedding_eval")
    return (
        lang_codes,
        per_lang,
        tok_args,
        feature_args,
        train_args,
        semantic_toggles,
        experiments,
        external_eval,
        embedding_eval,
    )


def expand_experiment(defn: dict) -> List[dict]:
    """Expands grid specifications into individual experiments."""
    grid = defn.get("grid")
    if not grid:
        repeat = defn.get("repeat", 1)
        return [
            {
                "name": defn["name"] if repeat == 1 else f"{defn['name']}_run{idx+1}",
                "tokenizer_args": defn.get("tokenizer_args", {}),
                "feature_args": defn.get("feature_args", {}),
                "train_args": defn.get("train_args", {}),
            }
            for idx in range(repeat)
        ]
    # Flatten grid: expects dict with same keys as tokenizer_args/feature_args/train_args
    keys = []
    values = []
    for section, entries in grid.items():
        for param, options in entries.items():
            keys.append((section, param))
            values.append(options)
    combos = list(itertools.product(*values))
    expanded = []
    for combo_idx, combo in enumerate(combos):
        tok_args = deepcopy(defn.get("tokenizer_args", {}))
        feat_args = deepcopy(defn.get("feature_args", {}))
        train_args = deepcopy(defn.get("train_args", {}))
        for (section, param), value in zip(keys, combo):
            if section == "tokenizer_args":
                tok_args[param] = value
            elif section == "feature_args":
                feat_args[param] = value
            elif section == "train_args":
                train_args[param] = value
        repeat = defn.get("repeat", 1)
        for r in range(repeat):
            run_name = defn["name"]
            if len(combos) > 1:
                run_name += f"_combo{combo_idx+1}"
            if repeat > 1:
                run_name += f"_run{r+1}"
            expanded.append(
                {
                    "name": run_name,
                    "tokenizer_args": tok_args,
                    "feature_args": feat_args,
                    "train_args": train_args,
                }
            )
    return expanded


def prepare_experiments(user_experiments: Optional[List[dict]]) -> List[dict]:
    if not user_experiments:
        default_defs = [
            {"name": "baseline"},
            {"name": "medium_budget_morph", "tokenizer_args": {"vocab_budget": 1200, "tau": 7.5e-4}},
        ]
        result = []
        for defn in default_defs:
            result.extend(expand_experiment(defn))
        return result

    result = []
    for defn in user_experiments:
        result.extend(expand_experiment(defn))
    return result


# ---------------------------------------------------------------------------
# Tokenizer construction utilities
# ---------------------------------------------------------------------------


def build_tokenizer(
    base_tok_args: dict,
    base_feat_args: dict,
    base_semantic_toggles: dict,
    custom_tok_args: dict,
    custom_feat_args: dict,
) -> ScalableTokenizer:
    args = base_tok_args.copy()
    args.update(custom_tok_args)
    tokenizer = ScalableTokenizer(**args)
    feature_args = base_feat_args.copy()
    feature_args.update(base_semantic_toggles)
    custom_feat_args = deepcopy(custom_feat_args)
    if "morphology_kwargs" in feature_args and "morphology_kwargs" in custom_feat_args:
        merged = feature_args["morphology_kwargs"].copy()
        merged.update(custom_feat_args["morphology_kwargs"])
        custom_feat_args["morphology_kwargs"] = merged
    feature_args.update(custom_feat_args)
    lexicon = deepcopy(DEFAULT_LEXICON)
    if "lexicon" in feature_args:
        override = feature_args.pop("lexicon")
        if isinstance(override, dict):
            lexicon.update(override)
    ne_gaz = {tag: dict(vals) for tag, vals in DEFAULT_NE.items()}
    if "ne_gaz" in feature_args:
        override = feature_args.pop("ne_gaz")
        for tag, entries in override.items():
            ne_gaz.setdefault(tag, {})
            if isinstance(entries, dict):
                ne_gaz[tag].update(entries)
            elif isinstance(entries, (list, set, tuple)):
                for entry in entries:
                    ne_gaz[tag][entry] = ne_gaz[tag].get(entry, 1.0)
            else:
                ne_gaz[tag][entries] = ne_gaz[tag].get(entries, 1.0)
    token_bigram = DEFAULT_TOKEN_BIGRAM.copy()
    if "token_bigram" in feature_args:
        override = feature_args.pop("token_bigram")
        token_bigram.update(override)
    tokenizer.set_feature_models(
        lexicon=lexicon,
        ne_gaz=ne_gaz,
        token_bigram=token_bigram,
        **feature_args,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a, b) -> float:
    a_t = torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=DEVICE)
    denom = torch.linalg.norm(a_t) * torch.linalg.norm(b_t) + 1e-12
    if denom.item() == 0.0:
        return 0.0
    return float(torch.dot(a_t, b_t) / denom)


def compute_cpt_tpc(token_sequences: List[List[str]], texts: List[str]) -> Tuple[float, float]:
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(len(seq) for seq in token_sequences)
    if total_tokens == 0:
        return 0.0, 0.0
    cpt = total_chars / total_tokens
    tpc = 1.0 / cpt if cpt > 0 else 0.0
    return cpt, tpc


def compute_zipf_divergence(tokens: List[str], alpha_grid: Iterable[float]) -> Tuple[float, float]:
    freq = Counter(tokens)
    if not freq:
        return 0.0, 0.0
    sorted_items = freq.most_common()
    total = sum(freq.values())
    p = torch.as_tensor([count / total for _, count in sorted_items], dtype=torch.float64, device=DEVICE)
    best_alpha = None
    best_div = float("inf")
    ranks = torch.arange(1, len(sorted_items) + 1, dtype=torch.float64, device=DEVICE)
    for alpha in alpha_grid:
        q = torch.pow(ranks, -alpha)
        q = q / q.sum()
        div = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))
        if float(div) < best_div:
            best_div = float(div)
            best_alpha = alpha
    return best_div, best_alpha


def compute_fragmentation_curve(tokenizer: ScalableTokenizer, corpus: List[str], deciles: int = 10):
    word_counter = Counter()
    token_lengths = {}
    for doc in corpus:
        for word in doc.split():
            word_counter[word] += 1
            token_lengths.setdefault(word, len(tokenizer.tokenize(word, lang=None)))
    if not word_counter:
        return []
    words_sorted = [w for w, _ in word_counter.most_common()]
    bucket_size = max(1, len(words_sorted) // deciles)
    curve = []
    for i in range(deciles):
        bucket = words_sorted[i * bucket_size : (i + 1) * bucket_size]
        if not bucket:
            break
        avg_frag = statistics.mean(token_lengths[w] for w in bucket)
        curve.append((i + 1, avg_frag))
    return curve


def compute_domain_cpt(tokenizer: ScalableTokenizer, samples: List[dict]) -> Dict[str, float]:
    domain_totals = defaultdict(lambda: {"chars": 0, "tokens": 0})
    for sample in samples:
        text = sample["text"]
        tokens = tokenizer.tokenize(text, lang=sample.get("language"))
        domain = sample.get("domain", "unknown")
        domain_totals[domain]["chars"] += len(text)
        domain_totals[domain]["tokens"] += len(tokens)
    result = {}
    for domain, stats in domain_totals.items():
        if stats["tokens"] > 0:
            result[domain] = stats["chars"] / stats["tokens"]
        else:
            result[domain] = 0.0
    return result


def compute_identifier_fragmentation(tokenizer: ScalableTokenizer, samples: List[dict]) -> float:
    identifier_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    lengths = []
    for sample in samples:
        if sample.get("domain") != "code":
            continue
        text = sample["text"]
        for ident in identifier_pattern.findall(text):
            lengths.append(len(tokenizer.tokenize(ident, lang=sample.get("language"))))
    if not lengths:
        return 0.0
    return float(statistics.mean(lengths))


def _script_label(ch: str) -> str:
    try:
        name = ud.name(ch)
    except ValueError:
        return "other"
    for script_code, script_name in [
        ("CYRILLIC", "Cyrillic"),
        ("ARABIC", "Arabic"),
        ("GREEK", "Greek"),
        ("HIRAGANA", "Hiragana"),
        ("KATAKANA", "Katakana"),
        ("CJK UNIFIED", "CJK"),
    ]:
        if script_code in name:
            return script_name
    if ch.isascii():
        return "Latin"
    return "other"


def compute_script_fracture_rate(tokenizer: ScalableTokenizer, samples: List[dict]) -> Dict[str, float]:
    fracture_counts = defaultdict(lambda: {"chars": 0, "fractures": 0})
    for sample in samples:
        text = sample["text"]
        tokens = tokenizer.tokenize(text, lang=sample.get("language"))
        # align tokens to characters greedily
        pos = 0
        for token in tokens:
            span = text[pos : pos + len(token)]
            if not span:
                continue
            scripts = {_script_label(ch) for ch in span}
            script = scripts.pop() if len(scripts) == 1 else "mixed"
            fracture_counts[script]["chars"] += len(span)
            if len(token) < len(span):
                fracture_counts[script]["fractures"] += len(span) - len(token)
            pos += len(token)
    rates = {}
    for script, stats in fracture_counts.items():
        if stats["chars"] == 0:
            rates[script] = 0.0
        else:
            rates[script] = stats["fractures"] / stats["chars"]
    return rates


def compute_token_allocation_balance(tokenizer: ScalableTokenizer, samples: List[dict]) -> float:
    lang_char = defaultdict(int)
    lang_token = defaultdict(int)
    for sample in samples:
        lang = sample.get("language", "unknown")
        text = sample["text"]
        lang_char[lang] += len(text)
        lang_token[lang] += len(tokenizer.tokenize(text, lang=lang))
    total_char = sum(lang_char.values())
    total_token = sum(lang_token.values())
    if total_char == 0 or total_token == 0:
        return 0.0
    p = torch.as_tensor([lang_char[l] / total_char for l in lang_char], dtype=torch.float64, device=DEVICE)
    q = torch.as_tensor([lang_token[l] / total_token for l in lang_char], dtype=torch.float64, device=DEVICE)
    m = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * (torch.log(p + 1e-12) - torch.log(m + 1e-12)))
    js += 0.5 * torch.sum(q * (torch.log(q + 1e-12) - torch.log(m + 1e-12)))
    return float(js.item())


def apply_perturbations(text: str) -> List[str]:
    if not text:
        return [text]
    chars = list(text)
    perturbed = []
    # swap adjacent characters
    if len(chars) > 1:
        swap = chars[:]
        idx = random.randrange(len(chars) - 1)
        swap[idx], swap[idx + 1] = swap[idx + 1], swap[idx]
        perturbed.append("".join(swap))
    # delete a character
    if len(chars) > 1:
        delete = chars[:]
        delete.pop(random.randrange(len(chars)))
        perturbed.append("".join(delete))
    # replace with unicode confusable
    replacements = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "c": "ç"}
    replaced = "".join(replacements.get(ch, ch) for ch in chars)
    perturbed.append(replaced)
    return perturbed or [text]


def compute_perturbation_stability(tokenizer: ScalableTokenizer, samples: List[dict]) -> float:
    diffs = []
    for sample in samples:
        original_len = len(tokenizer.tokenize(sample["text"], lang=sample.get("language")))
        for perturbed in apply_perturbations(sample["text"]):
            pert_len = len(tokenizer.tokenize(perturbed, lang=sample.get("language")))
            diffs.append(abs(original_len - pert_len))
    return float(statistics.mean(diffs)) if diffs else 0.0


def compute_effective_context_gain(tpc_tokenizer: float, tpc_baseline: float) -> float:
    if tpc_tokenizer <= 0 or tpc_baseline <= 0:
        return 0.0
    return tpc_baseline / tpc_tokenizer


def compute_morph_cosine_summary(tokenizer: ScalableTokenizer) -> Dict[str, float]:
    morph = getattr(tokenizer._ling, "morph_encoder", None)
    if morph is None or not morph.token_vec:
        return {}
    vectors = morph.token_vec
    lang_vectors = defaultdict(list)
    for tok, vec in vectors.items():
        # attempt to guess language from stored paragraphs
        for lang in tokenizer._ling.morphology_kwargs.get("language_inventory", []):
            if tok in tokenizer._ling.lexicon.get(lang, {}):
                lang_vectors[lang].append(vec)
                break
    # compute average cosine for cross-language pairs using CROSS_EQUIV
    results = {}
    for key, mapping in CROSS_EQUIV.items():
        cosines = []
        langs = list(mapping.keys())
        for a in range(len(langs)):
            for b in range(a + 1, len(langs)):
                lang_a, lang_b = langs[a], langs[b]
                tokens_a = [tok for tok in mapping[lang_a] if tok in vectors]
                tokens_b = [tok for tok in mapping[lang_b] if tok in vectors]
                for ta in tokens_a:
                    for tb in tokens_b:
                        cosines.append(cosine_similarity(vectors[ta], vectors[tb]))
        if cosines:
            results[key] = float(statistics.mean(cosines))
    return results


def compute_reference_metrics(references: Dict[str, Callable[[str], Sequence[str]]], samples: List[dict]):
    metrics = {}
    if not references:
        return metrics
    texts = [sample["text"] for sample in samples]
    for name, tokenizer_fn in references.items():
        token_sequences = [list(tokenizer_fn(text)) for text in texts]
        cpt, tpc = compute_cpt_tpc(token_sequences, texts)
        metrics[name] = {"cpt": cpt, "tpc": tpc}
    return metrics


def compute_language_breakdowns(
    tokenizer: ScalableTokenizer,
    samples: List[dict],
    token_sequences: List[List[str]],
    references: Dict[str, Callable[[str], Sequence[str]]],
) -> Dict[str, dict]:
    grouped = defaultdict(lambda: {"samples": [], "texts": [], "tokens": []})
    for sample, tokens in zip(samples, token_sequences):
        lang = sample.get("language", "unknown")
        grouped[lang]["samples"].append(sample)
        grouped[lang]["texts"].append(sample["text"])
        grouped[lang]["tokens"].append(tokens)

    per_language = {}
    alpha_grid = torch.linspace(0.5, 2.0, 30).tolist()
    for lang, data in grouped.items():
        texts = data["texts"]
        seqs = data["tokens"]
        lang_samples = data["samples"]
        cpt, tpc = compute_cpt_tpc(seqs, texts)
        zipf_div, best_alpha = compute_zipf_divergence([tok for seq in seqs for tok in seq], alpha_grid)
        fragmentation_curve = compute_fragmentation_curve(tokenizer, texts)
        domain_cpt = compute_domain_cpt(tokenizer, lang_samples)
        identifier_fragment = compute_identifier_fragmentation(tokenizer, lang_samples)
        script_rates = compute_script_fracture_rate(tokenizer, lang_samples)
        token_balance = compute_token_allocation_balance(tokenizer, lang_samples)
        perturb_stability = compute_perturbation_stability(tokenizer, lang_samples)
        ref_metrics = compute_reference_metrics(references, lang_samples)
        baseline_tpc = ref_metrics[next(iter(ref_metrics))]["tpc"] if ref_metrics else 1.0
        effective_gain = compute_effective_context_gain(tpc, baseline_tpc)
        per_language[lang] = {
            "avg_token_length": float(cpt),
            "tokens_per_character": float(tpc),
            "zipf_divergence": float(zipf_div),
            "zipf_best_alpha": float(best_alpha) if best_alpha is not None else None,
            "fragmentation_curve": fragmentation_curve,
            "domain_cpt": domain_cpt,
            "identifier_fragmentation": float(identifier_fragment),
            "script_fracture_rates": script_rates,
            "token_allocation_js": float(token_balance),
            "perturbation_stability": float(perturb_stability),
            "effective_context_gain": float(effective_gain),
            "reference_metrics": ref_metrics,
        }
    return per_language


def maybe_run_segmentation_eval(
    tokenizer: ScalableTokenizer,
    lang_codes: Dict[str, str],
    external_cfg: Optional[dict],
    references: Dict[str, Callable[[str], Sequence[str]]],
    eval_samples: Optional[List[dict]] = None,
):
    if not (HAS_SEGMENTATION_EVAL and external_cfg):
        return None
    languages = external_cfg.get("languages") or list(lang_codes.keys())
    if not languages:
        return None
    exclude_numeric_keys = {
        "mode",
        "error",
        "morphology",
        "languages_evaluated",
        "languages_skipped",
        "missing_languages",
        "words_per_language",
        "sentences_per_language",
    }

    def _aggregate_by_label(per_language_map: Dict[str, Dict[str, Mapping[str, object]]]) -> Dict[str, Dict[str, float]]:
        collector: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for lang, label_stats in per_language_map.items():
            for label, metrics in label_stats.items():
                if not isinstance(metrics, Mapping):
                    continue
                for key, value in metrics.items():
                    if key in exclude_numeric_keys or not isinstance(value, (int, float)):
                        continue
                    collector[label][key].append(float(value))
        aggregates: Dict[str, Dict[str, float]] = {}
        for label, metric_map in collector.items():
            label_metrics: Dict[str, float] = {}
            for key, values in metric_map.items():
                if not values:
                    continue
                if key.endswith("_evaluated") or key.endswith("_count") or key in {"pairs", "pairs_evaluated"}:
                    label_metrics[key] = float(sum(values))
                else:
                    label_metrics[key] = float(sum(values) / len(values))
            if label_metrics:
                aggregates[label] = label_metrics
        return aggregates

    tokenizers: Dict[str, Callable[..., Sequence[str]] | ScalableTokenizer] = {
        external_cfg.get("label", "trained"): tokenizer
    }
    if external_cfg.get("compare_references", False) and references:
        for name, fn in references.items():
            tokenizers[f"ref::{name}"] = fn
    flores_map = external_cfg.get("flores_map")
    if flores_map:
        flores_map = {lang: dict(conf) for lang, conf in flores_map.items()}

    results: Dict[str, object] = {}
    overall_meta = {
        "requested_languages": sorted(languages),
        "tokenizers": sorted(tokenizers.keys()),
    }

    # Standard word-level evaluation
    try:
        word_results = evaluate_languages_with_backoff(
            tokenizers,
            languages,
            uniseg_root=external_cfg.get("uniseg_root") or DEFAULT_UNISEG_ROOT,
            flores_map=flores_map,
            max_uniseg_words=external_cfg.get("max_uniseg_words"),
            lang_map=external_cfg.get("lang_map"),
        )
        per_language_word: Dict[str, Dict[str, Mapping[str, object]]] = {}
        missing_langs: set[str] = set()
        covered_langs: set[str] = set()
        for label, lang_stats in word_results.items():
            if not isinstance(lang_stats, Mapping):
                continue
            for lang, stats in lang_stats.items():
                if not isinstance(stats, Mapping):
                    continue
                if stats.get("mode") == "unavailable":
                    missing_langs.add(lang)
                    continue
                per_language_word.setdefault(lang, {})[label] = stats
                covered_langs.add(lang)
        results["word_level"] = {
            "per_language": per_language_word,
            "aggregate": _aggregate_by_label(per_language_word),
            "meta": {
                "evaluated_languages": sorted(covered_langs),
                "missing_languages": sorted(missing_langs),
            },
            "raw": word_results,
        }
    except Exception as exc:
        results["word_level"] = {"error": str(exc)}

    # Sentence-level evaluation (if eval_samples provided)
    if eval_samples and external_cfg.get("evaluate_sentences", True):
        try:
            from .segmentation_eval import evaluate_sentences_with_uniseg

            per_language_sentence: Dict[str, Dict[str, Mapping[str, object]]] = {}
            aggregate_details: Dict[str, Mapping[str, object]] = {}
            meta_sentence = {
                "evaluated_languages": set(),
                "missing_languages": set(),
            }
            lang_samples: Dict[str, List[dict]] = defaultdict(list)
            for sample in eval_samples:
                lang = sample.get("language", list(lang_codes.keys())[0])
                lang_samples[lang].append(sample)

            aggregate_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

            for lang, samples_for_lang in lang_samples.items():
                sentences = [sample["text"] for sample in samples_for_lang]
                sample_langs = [sample.get("language", lang) for sample in samples_for_lang]
                for label, tok in tokenizers.items():
                    lang_result = evaluate_sentences_with_uniseg(
                        tok,
                        sentences,
                        sample_langs,
                        uniseg_root=external_cfg.get("uniseg_root") or DEFAULT_UNISEG_ROOT,
                        lang_map=external_cfg.get("lang_map"),
                    )
                    per_language_sentence.setdefault(lang, {})[label] = lang_result
                    meta_sentence["missing_languages"].update(lang_result.get("languages_skipped", []) or [])
                    meta_sentence["evaluated_languages"].update(lang_result.get("languages_evaluated", []) or [])
                    aggregate_details.setdefault(label, {"per_language": {}})["per_language"][lang] = lang_result

                    weight = float(lang_result.get("sentences_evaluated") or lang_result.get("words_evaluated") or 0.0)
                    aggregate_sums[label]["weight"] += weight
                    aggregate_sums[label]["sentence_similarity_sum"] += weight * float(lang_result.get("sentence_similarity", 0.0))
                    aggregate_sums[label]["boundary_f1_sum"] += weight * float(lang_result.get("boundary_f1", 0.0))
                    aggregate_sums[label]["boundary_precision_sum"] += weight * float(lang_result.get("boundary_precision", 0.0))
                    aggregate_sums[label]["boundary_recall_sum"] += weight * float(lang_result.get("boundary_recall", 0.0))
                    aggregate_sums[label]["morphological_score_sum"] += weight * float(lang_result.get("morphological_score", 0.0))
                    aggregate_sums[label]["words_evaluated"] += float(lang_result.get("words_evaluated", 0.0))
                    aggregate_sums[label]["sentences_evaluated"] += float(lang_result.get("sentences_evaluated", 0.0))

            aggregate_sentence: Dict[str, Dict[str, float]] = {}
            for label, sums in aggregate_sums.items():
                weight = sums.get("weight", 0.0)
                agg_entry = {}
                if weight > 0:
                    agg_entry["sentence_similarity"] = sums["sentence_similarity_sum"] / weight
                    agg_entry["boundary_f1"] = sums["boundary_f1_sum"] / weight
                    agg_entry["boundary_precision"] = sums["boundary_precision_sum"] / weight
                    agg_entry["boundary_recall"] = sums["boundary_recall_sum"] / weight
                    agg_entry["morphological_score"] = sums["morphological_score_sum"] / weight
                agg_entry["words_evaluated"] = sums.get("words_evaluated", 0.0)
                agg_entry["sentences_evaluated"] = sums.get("sentences_evaluated", 0.0)
                aggregate_sentence[label] = agg_entry

            results["sentence_level"] = {
                "per_language": per_language_sentence,
                "aggregate": aggregate_sentence,
                "details": aggregate_details,
                "meta": {
                    "evaluated_languages": sorted(meta_sentence["evaluated_languages"]),
                    "missing_languages": sorted(meta_sentence["missing_languages"]),
                },
            }
        except Exception as exc:
            results["sentence_level"] = {"error": str(exc)}

    word_meta = results.get("word_level", {}).get("meta", {}) if isinstance(results.get("word_level"), Mapping) else {}
    sentence_meta = results.get("sentence_level", {}).get("meta", {}) if isinstance(results.get("sentence_level"), Mapping) else {}
    evaluated_union = set(word_meta.get("evaluated_languages", []) or [])
    evaluated_union.update(sentence_meta.get("evaluated_languages", []) or [])
    missing_union = set(word_meta.get("missing_languages", []) or [])
    missing_union.update(sentence_meta.get("missing_languages", []) or [])
    if evaluated_union:
        overall_meta["evaluated_languages"] = sorted(evaluated_union)
    if missing_union:
        overall_meta["missing_languages"] = sorted(missing_union)
    if "evaluated_languages" in overall_meta or "missing_languages" in overall_meta:
        results["__meta__"] = overall_meta

    return results


def export_metrics(folder: Path, metrics: dict, samples: List[Tuple[str, List[str]]], extra: dict):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    samples_payload = [{"text": text, "tokens": tokens} for text, tokens in samples]
    (folder / "token_samples.json").write_text(json.dumps(samples_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if extra:
        for name, data in extra.items():
            (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_plots(folder: Path, metrics: dict):
    if not HAS_MATPLOTLIB:
        return
    global_metrics = metrics.get("global", {})
    per_language = metrics.get("per_language", {})
    lang_names = list(per_language.keys())
    if lang_names:
        cmap = plt.cm.get_cmap("tab10", len(lang_names))
        lang_colors = {lang: cmap(i) for i, lang in enumerate(lang_names)}
    else:
        lang_colors = {}

    def _maybe_plot_fragmentation():
        has_data = False
        plt.figure()
        curve = global_metrics.get("fragmentation_curve")
        if curve:
            xs, ys = zip(*curve)
            plt.plot(xs, ys, label="global", color="black", linewidth=2)
            has_data = True
        for lang, data in per_language.items():
            curve = data.get("fragmentation_curve")
            if curve:
                xs, ys = zip(*curve)
                plt.plot(xs, ys, label=lang, color=lang_colors.get(lang), linestyle="--")
                has_data = True
        if not has_data:
            plt.close()
            return
        plt.xlabel("Frequency decile")
        plt.ylabel("Avg tokens per word")
        plt.title("Fragmentation Curve by Language")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(folder / "fragmentation_curve.png", dpi=160)
        plt.close()

    def _stacked_bar(data_map, title, ylabel, filename):
        filtered = {label: data for label, data in data_map.items() if data}
        if not filtered:
            return
        keys = sorted({key for d in filtered.values() for key in d.keys()})
        if not keys:
            return
        series = list(filtered.keys())
        x = list(range(len(keys)))
        width = 0.8 / max(len(series), 1)
        plt.figure()
        for idx, label in enumerate(series):
            values = [filtered[label].get(k, math.nan) for k in keys]
            color = "black" if label == "global" else lang_colors.get(label)
            offsets = [xi + idx * width for xi in x]
            plt.bar(offsets, values, width=width, label=label, color=color)
        tick_positions = [xi + width * (len(series) - 1) / 2 for xi in x]
        plt.xticks(tick_positions, keys, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder / filename, dpi=160)
        plt.close()

    def _plot_identifier_fragment():
        values = {"global": global_metrics.get("identifier_fragmentation")}
        values.update({lang: data.get("identifier_fragmentation") for lang, data in per_language.items()})
        if all(v is None for v in values.values()):
            return
        labels = list(values.keys())
        heights = [(values[label] if values[label] is not None else 0.0) for label in labels]
        colors = ["black" if label == "global" else lang_colors.get(label) for label in labels]
        plt.figure()
        plt.bar(labels, heights, color=colors)
        plt.ylabel("Tokens")
        plt.title("Identifier Fragmentation by Language")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(folder / "identifier_fragmentation.png", dpi=160)
        plt.close()

    def _plot_script_fracture():
        data_map = {"global": global_metrics.get("script_fracture_rates", {})}
        for lang, data in per_language.items():
            rates = data.get("script_fracture_rates")
            if rates:
                data_map[lang] = rates
        if not any(data_map.values()):
            return
        _stacked_bar(data_map, "Script Fracture Rate by Language", "Fracture rate", "script_fracture.png")

    def _plot_domain_cpt():
        data_map = {"global": global_metrics.get("domain_cpt", {})}
        for lang, data in per_language.items():
            domains = data.get("domain_cpt")
            if domains:
                data_map[lang] = domains
        if not any(data_map.values()):
            return
        _stacked_bar(data_map, "Domain CPT by Language", "Characters per token", "domain_cpt.png")

    def _plot_reference_comparison():
        scopes = ["global"] + lang_names
        series_data = {}
        # our tokenizer
        series_data["ours"] = {}
        for scope in scopes:
            data = global_metrics if scope == "global" else per_language.get(scope, {})
            val = data.get("tokens_per_character")
            if val is not None:
                series_data["ours"][scope] = val
        # references
        for scope in scopes:
            data = global_metrics if scope == "global" else per_language.get(scope, {})
            refs = data.get("reference_metrics", {})
            for ref_name, ref_vals in refs.items():
                series_data.setdefault(ref_name, {})
                series_data[ref_name][scope] = ref_vals.get("tpc")
        filtered_series = {name: vals for name, vals in series_data.items() if vals}
        if not filtered_series:
            plt.figure()
            plt.text(
                0.5,
                0.5,
                "Reference tokenizers unavailable.\nInstall tiktoken and/or transformers for comparison.",
                ha="center",
                va="center",
            )
            plt.axis("off")
            plt.title("Reference Tokenizer Comparison (TPC)")
            plt.savefig(folder / "reference_tpc.png", dpi=160)
            plt.close()
            return
        x = list(range(len(scopes)))
        width = 0.8 / max(len(filtered_series), 1)
        plt.figure()
        for idx, (name, scope_map) in enumerate(filtered_series.items()):
            heights = [scope_map.get(scope, math.nan) for scope in scopes]
            offsets = [xi + idx * width for xi in x]
            plt.bar(offsets, heights, width=width, label=name)
        tick_positions = [xi + width * (len(filtered_series) - 1) / 2 for xi in x]
        plt.xticks(tick_positions, scopes, rotation=30, ha="right")
        plt.ylabel("Tokens per character (higher is better)")
        plt.title("Reference Tokenizer Comparison (TPC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder / "reference_tpc.png", dpi=160)
        plt.close()

    def _plot_external_eval():
        external = metrics.get("external_eval")
        if not isinstance(external, Mapping) or not external:
            return
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

        def _plot_language_matrix(
            per_language: Mapping[str, Mapping[str, Mapping[str, object]]],
            title_prefix: str,
            filename_prefix: str,
        ) -> None:
            if not isinstance(per_language, Mapping) or not per_language:
                return
            languages = sorted(per_language.keys())
            tokenizers_local = sorted(
                {
                    label
                    for lang_stats in per_language.values()
                    for label in lang_stats.keys()
                }
            )
            if not languages or not tokenizers_local:
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
            base_positions = list(range(len(languages)))
            width = 0.8 / max(len(tokenizers_local), 1)
            for metric in metric_names:
                plt.figure()
                plotted = False
                for idx, label in enumerate(tokenizers_local):
                    heights = []
                    for lang in languages:
                        val = per_language.get(lang, {}).get(label, {}).get(metric)
                        if isinstance(val, (int, float)) and math.isfinite(val):
                            heights.append(val)
                        else:
                            heights.append(math.nan)
                    if all(math.isnan(h) for h in heights):
                        continue
                    positions = [pos + idx * width for pos in base_positions]
                    plt.bar(positions, heights, width=width, label=label)
                    plotted = True
                if not plotted:
                    plt.close()
                    continue
                tick_positions = [pos + width * (len(tokenizers_local) - 1) / 2 for pos in base_positions]
                plt.xticks(tick_positions, languages, rotation=30, ha="right")
                plt.ylabel(metric.replace("_", " ").title())
                plt.title(f"{title_prefix}: {metric.replace('_', ' ').title()}")
                plt.legend()
                plt.tight_layout()
                safe_metric = metric.replace("/", "_").replace("::", "_")
                plt.savefig(folder / f"{filename_prefix}_{safe_metric}.png", dpi=160)
                plt.close()

        def _plot_aggregate_section(
            aggregate: Mapping[str, Mapping[str, object]],
            title_prefix: str,
            filename_prefix: str,
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
            for metric in metric_names:
                heights = []
                labels = []
                for label in tokenizers_local:
                    val = aggregate.get(label, {}).get(metric)
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        labels.append(label)
                        heights.append(val)
                if not heights:
                    continue
                plt.figure()
                plt.bar(range(len(labels)), heights, color=[plt.cm.tab10(i % 10) for i in range(len(labels))])
                plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
                plt.ylabel(metric.replace("_", " ").title())
                plt.title(f"{title_prefix}: {metric.replace('_', ' ').title()}")
                plt.tight_layout()
                safe_metric = metric.replace("/", "_").replace("::", "_")
                plt.savefig(folder / f"{filename_prefix}_{safe_metric}.png", dpi=160)
                plt.close()

        word_section = external.get("word_level")
        if isinstance(word_section, Mapping):
            word_per_language = word_section.get("per_language", {})
            _plot_language_matrix(
                word_per_language,
                "UniSeg Word-Level",
                "uniseg_word",
            )
            _plot_aggregate_section(
                word_section.get("aggregate", {}),
                "UniSeg Word-Level Aggregate",
                "uniseg_word_aggregate",
            )
            morph_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
            for lang, label_map in word_per_language.items():
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
            for morph_type, lang_map in morph_map.items():
                pseudo = {
                    lang: {label: {"coverage_rate": cov} for label, cov in label_stats.items()}
                    for lang, label_stats in lang_map.items()
                }
                _plot_language_matrix(
                    pseudo,
                    f"Morphology Coverage ({morph_type})",
                    f"uniseg_word_morph_{morph_type}",
                )

        sentence_section = external.get("sentence_level")
        if isinstance(sentence_section, Mapping):
            sentence_per_language = sentence_section.get("per_language", {})
            _plot_language_matrix(
                sentence_per_language,
                "UniSeg Sentence-Level",
                "uniseg_sentence",
            )
            _plot_aggregate_section(
                sentence_section.get("aggregate", {}),
                "UniSeg Sentence-Level Aggregate",
                "uniseg_sentence_aggregate",
            )
            morph_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
            for lang, label_map in sentence_per_language.items():
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
            for morph_type, lang_map in morph_map.items():
                pseudo = {
                    lang: {label: {"coverage_rate": cov} for label, cov in label_stats.items()}
                    for lang, label_stats in lang_map.items()
                }
                _plot_language_matrix(
                    pseudo,
                    f"Sentence Morphology Coverage ({morph_type})",
                    f"uniseg_sentence_morph_{morph_type}",
                )

    def _plot_embedding_benchmarks():
        embedding = metrics.get("embedding_benchmarks")
        if not embedding:
            return

        def _plot_section(section: Dict[str, Dict[str, Dict[str, float]]], metric: str, title: str, filename: str):
            if not section:
                return
            datasets = sorted(k for k in section.keys() if not str(k).startswith("__"))
            tokenizers = sorted({label for stats in section.values() for label in stats.keys()})
            if not datasets or not tokenizers:
                return
            base_positions = list(range(len(datasets)))
            width = 0.8 / max(len(tokenizers), 1)
            plt.figure()
            plotted = False
            for idx, label in enumerate(tokenizers):
                heights = []
                for dataset in datasets:
                    val = section[dataset].get(label, {}).get(metric)
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        heights.append(val)
                    else:
                        heights.append(math.nan)
                if all(math.isnan(h) for h in heights):
                    continue
                bar_positions = [pos + idx * width for pos in base_positions]
                plt.bar(bar_positions, heights, width=width, label=label)
                plotted = True
            if not plotted:
                plt.close()
                return
            tick_positions = [pos + width * (len(tokenizers) - 1) / 2 for pos in base_positions]
            plt.xticks(tick_positions, datasets, rotation=30, ha="right")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder / filename, dpi=160)
            plt.close()

        muse_section = embedding.get("muse")
        if muse_section:
            _plot_section(muse_section, "p_at_1", "MUSE Word Translation (P@1)", "embedding_muse_p1.png")
            _plot_section(muse_section, "csls_p1", "MUSE Word Translation (CSLS P@1)", "embedding_muse_csls.png")
        sim_section = embedding.get("similarity")
        if sim_section:
            _plot_section(sim_section, "spearman", "Semantic Similarity (Spearman)", "embedding_similarity_spearman.png")
            _plot_section(sim_section, "pearson", "Semantic Similarity (Pearson)", "embedding_similarity_pearson.png")
        xling_section = embedding.get("crosslingual_similarity")
        if xling_section:
            _plot_section(
                xling_section,
                "spearman",
                "Cross-lingual Semantic Similarity (Spearman)",
                "embedding_xling_spearman.png",
            )
            _plot_section(
                xling_section,
                "pearson",
                "Cross-lingual Semantic Similarity (Pearson)",
                "embedding_xling_pearson.png",
            )

    _maybe_plot_fragmentation()
    _plot_domain_cpt()
    _plot_identifier_fragment()
    _plot_script_fracture()
    _plot_reference_comparison()
    _plot_embedding_benchmarks()
    _plot_external_eval()


def zip_results(folder: Path):
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            zf.write(path, path.relative_to(folder.parent))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    exp_def: dict,
    lang_codes: dict,
    per_lang: int,
    base_tok_args: dict,
    base_feat_args: dict,
    base_train_args: dict,
    base_semantic_toggles: dict,
    references: dict,
    output_dir: Path,
    external_eval_cfg: Optional[dict] = None,
    embedding_eval_cfg: Optional[dict] = None,
):
    print(f"\n=== Experiment: {exp_def['name']} ===")
    tok_args = deepcopy(base_tok_args)
    tok_args.update(exp_def.get("tokenizer_args", {}))
    feat_args = deepcopy(base_feat_args)
    custom_feature_args = deepcopy(exp_def.get("feature_args", {}))
    if "morphology_kwargs" in feat_args and "morphology_kwargs" in custom_feature_args:
        merged = feat_args["morphology_kwargs"].copy()
        merged.update(custom_feature_args["morphology_kwargs"])
        custom_feature_args["morphology_kwargs"] = merged
    feat_args.update(custom_feature_args)
    sem_toggles = deepcopy(base_semantic_toggles)
    sem_toggles.update(exp_def.get("semantic_toggles", {}))

    tokenizer = build_tokenizer(
        base_tok_args,
        base_feat_args,
        base_semantic_toggles,
        exp_def.get("tokenizer_args", {}),
        exp_def.get("feature_args", {}),
    )
    train_args = base_train_args.copy()
    train_args.update(exp_def.get("train_args", {}))

    texts, langs = load_wikiann_corpus(lang_codes, per_lang=per_lang)
    if not texts:
        print("No training data available.")
        return
    eval_samples = build_eval_samples_from_corpus(texts, langs, lang_codes)
    if not eval_samples:
        eval_samples = FALLBACK_EVAL_SAMPLES
    initial_token_sequences = [
        tokenizer.tokenize(sample["text"], lang=sample.get("language"))
        for sample in eval_samples
    ]
    tokenizer.train(texts, langs, **train_args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_folder = output_dir / f"{exp_def['name']}_{timestamp}"
    exp_folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save(exp_folder / "tokenizer.json")
    tokenizer.dump_debug_info(exp_folder / "debug.json", include_filtered=True)
    manifest = {
        "experiment": exp_def["name"],
        "timestamp": timestamp,
        "lang_codes": lang_codes,
        "per_lang": per_lang,
        "tokenizer_args": tok_args,
        "feature_args": feat_args,
        "semantic_toggles": sem_toggles,
        "train_args": train_args,
        "eval_samples": {
            "source": "corpus_first_k" if eval_samples is not FALLBACK_EVAL_SAMPLES else "fallback_static",
            "per_lang": EVAL_PARAGRAPHS_PER_LANG if eval_samples is not FALLBACK_EVAL_SAMPLES else None,
        },
    }
    (exp_folder / "experiment_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Evaluation
    token_sequences = [tokenizer.tokenize(sample["text"], lang=sample.get("language")) for sample in eval_samples]
    texts_eval = [sample["text"] for sample in eval_samples]
    cpt, tpc = compute_cpt_tpc(token_sequences, texts_eval)
    zipf_div, best_alpha = compute_zipf_divergence(
        [tok for seq in token_sequences for tok in seq], torch.linspace(0.5, 2.0, 30).tolist()
    )
    fragmentation_curve = compute_fragmentation_curve(tokenizer, texts_eval)
    domain_cpt = compute_domain_cpt(tokenizer, eval_samples)
    identifier_fragment = compute_identifier_fragmentation(tokenizer, eval_samples)
    script_rates = compute_script_fracture_rate(tokenizer, eval_samples)
    token_balance = compute_token_allocation_balance(tokenizer, eval_samples)
    perturb_stability = compute_perturbation_stability(tokenizer, eval_samples)

    references_metrics = compute_reference_metrics(references, eval_samples)
    baseline_tpc = references_metrics[next(iter(references_metrics))]["tpc"] if references_metrics else 1.0
    effective_gain = compute_effective_context_gain(tpc, baseline_tpc)
    morph_cosine = compute_morph_cosine_summary(tokenizer)
    per_language_metrics = compute_language_breakdowns(tokenizer, eval_samples, token_sequences, references)

    global_metrics = {
        "avg_token_length": float(cpt),
        "tokens_per_character": float(tpc),
        "zipf_divergence": float(zipf_div),
        "zipf_best_alpha": float(best_alpha) if best_alpha is not None else None,
        "fragmentation_curve": fragmentation_curve,
        "domain_cpt": domain_cpt,
        "identifier_fragmentation": float(identifier_fragment),
        "script_fracture_rates": script_rates,
        "token_allocation_js": float(token_balance),
        "perturbation_stability": float(perturb_stability),
        "effective_context_gain": float(effective_gain),
        "reference_metrics": references_metrics,
    }
    metrics = {
        "global": global_metrics,
        "per_language": per_language_metrics,
    }
    external_eval_results = maybe_run_segmentation_eval(tokenizer, lang_codes, external_eval_cfg, references, eval_samples=eval_samples)
    if external_eval_results:
        metrics["external_eval"] = external_eval_results
        write_segmentation_report(exp_folder, external_eval_results)
    embedding_eval_results = maybe_run_embedding_eval(
        tokenizer,
        references,
        texts,
        langs,
        feat_args,
        embedding_eval_cfg,
    )
    if embedding_eval_results:
        metrics["embedding_benchmarks"] = embedding_eval_results
        write_embedding_report(exp_folder, embedding_eval_results)
    extra_outputs = {"morph_cosine": morph_cosine}
    export_metrics(exp_folder, metrics, list(zip(texts_eval, token_sequences)), extra_outputs)
    initial_samples_payload = [
        {"text": sample["text"], "tokens": tokens}
        for sample, tokens in zip(eval_samples, initial_token_sequences)
    ]
    (exp_folder / "token_samples_initial.json").write_text(
        json.dumps(initial_samples_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_plots(exp_folder, metrics)
    zip_results(exp_folder)

    avg_cpt = global_metrics["avg_token_length"]
    single_char_ratio = sum(len(seq) == 1 for seq in token_sequences) / max(
        sum(len(seq) for seq in token_sequences), 1
    )
    print(
        f"Vocab size: {len(tokenizer.vocab)}, "
        f"lambda*: {tokenizer._lambda_global:.4f}, "
        f"CPT: {avg_cpt:.2f}, "
        f"single-char ratio: {single_char_ratio:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(description="Tokenizer experiment sweeper")
    parser.add_argument("--config", help="JSON config describing experiments")
    parser.add_argument("--select", nargs="*", help="Subset of experiments to run")
    parser.add_argument("--output-dir", default="experiment_outputs")
    args = parser.parse_args()

    (
        lang_codes,
        per_lang,
        base_tok_args,
        base_feat_args,
        base_train_args,
        base_semantic_toggles,
        user_experiments,
        external_eval_cfg,
        embedding_eval_cfg,
    ) = load_config(args.config)

    experiments = prepare_experiments(user_experiments)
    if args.select:
        selected = set(args.select)
        experiments = [exp for exp in experiments if exp["name"] in selected]
    if not experiments:
        print("No experiments to run.")
        return

    references = load_reference_tokenizers()
    output_dir = Path(args.output_dir)
    for exp in experiments:
        run_experiment(
            exp,
            lang_codes,
            per_lang,
            base_tok_args,
            base_feat_args,
            base_train_args,
            base_semantic_toggles,
            references,
            output_dir,
            external_eval_cfg,
            embedding_eval_cfg,
        )


if __name__ == "__main__":
    main()
