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
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tokenizer import ScalableTokenizer
from data import load_wikiann_corpus
from constants import (
    URL_RE,
    EMAIL_RE,
    CROSS_EQUIV,
)
import utils

try:
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
except ImportError:
    HAS_TRANSFORMERS = False

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
)

DEFAULT_TRAIN_ARGS = dict(max_iterations=80)

DEFAULT_SEMANTIC_TOGGLES = dict(
    email_reward=-0.25,
    url_reward=-0.35,
    hashtag_reward=-0.05,
)

# ---------------------------------------------------------------------------
# Evaluation corpus (lightweight demo). Users can override via config.
# ---------------------------------------------------------------------------

EVAL_SAMPLES = [
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
    return (
        lang_codes,
        per_lang,
        tok_args,
        feature_args,
        train_args,
        semantic_toggles,
        experiments,
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
    feature_args.update(custom_feat_args)
    tokenizer.set_feature_models(
        lexicon=deepcopy(DEFAULT_LEXICON),
        ne_gaz={k: set(v) for k, v in DEFAULT_NE.items()},
        token_bigram=DEFAULT_TOKEN_BIGRAM.copy(),
        **feature_args,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


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
    p = np.array([count / total for _, count in sorted_items], dtype=np.float64)
    best_alpha = None
    best_div = float("inf")
    ranks = np.arange(1, len(sorted_items) + 1, dtype=np.float64)
    for alpha in alpha_grid:
        q = 1.0 / np.power(ranks, alpha)
        q /= q.sum()
        div = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))
        if div < best_div:
            best_div = div
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
    p = np.array([lang_char[l] / total_char for l in lang_char], dtype=np.float64)
    q = np.array([lang_token[l] / total_token for l in lang_char], dtype=np.float64)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12))) + 0.5 * np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)))
    return float(js)


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
                        cosines.append(cosine_similarity(np.array(vectors[ta]), np.array(vectors[tb])))
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


def export_metrics(folder: Path, metrics: dict, samples: List[Tuple[str, List[str]]], extra: dict):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    samples_payload = [{"text": text, "tokens": tokens} for text, tokens in samples]
    (folder / "token_samples.json").write_text(json.dumps(samples_payload, ensure_ascii=False, indent=2))
    if extra:
        for name, data in extra.items():
            (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


def export_plots(folder: Path, fragmentation_curve, domain_cpt, identifier_fragment, script_rates):
    if not HAS_MATPLOTLIB:
        return
    if fragmentation_curve:
        xs, ys = zip(*fragmentation_curve)
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Frequency decile")
        plt.ylabel("Avg tokens per word")
        plt.title("Fragmentation Curve")
        plt.grid(True)
        plt.savefig(folder / "fragmentation_curve.png", dpi=160)
        plt.close()
    if domain_cpt:
        plt.figure()
        domains = list(domain_cpt.keys())
        values = [domain_cpt[d] for d in domains]
        plt.bar(domains, values)
        plt.ylabel("Characters per token")
        plt.title("Domain CPT")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(folder / "domain_cpt.png", dpi=160)
        plt.close()
    plt.figure()
    plt.bar(["Avg identifier length"], [identifier_fragment])
    plt.ylabel("Tokens")
    plt.title("Identifier Fragmentation")
    plt.savefig(folder / "identifier_fragmentation.png", dpi=160)
    plt.close()
    if script_rates:
        plt.figure()
        scripts = list(script_rates.keys())
        vals = [script_rates[s] for s in scripts]
        plt.bar(scripts, vals)
        plt.ylabel("Fracture rate")
        plt.title("Script Fracture Rate")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(folder / "script_fracture.png", dpi=160)
        plt.close()


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
):
    print(f"\n=== Experiment: {exp_def['name']} ===")
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
    tokenizer.train(texts, langs, **train_args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_folder = output_dir / f"{exp_def['name']}_{timestamp}"
    exp_folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save(exp_folder / "tokenizer.json")
    tokenizer.dump_debug_info(exp_folder / "debug.json", include_filtered=True)

    # Evaluation
    token_sequences = [tokenizer.tokenize(sample["text"], lang=sample.get("language")) for sample in EVAL_SAMPLES]
    texts_eval = [sample["text"] for sample in EVAL_SAMPLES]
    cpt, tpc = compute_cpt_tpc(token_sequences, texts_eval)
    zipf_div, best_alpha = compute_zipf_divergence([tok for seq in token_sequences for tok in seq], np.linspace(0.5, 2.0, 30))
    fragmentation_curve = compute_fragmentation_curve(tokenizer, texts_eval)
    domain_cpt = compute_domain_cpt(tokenizer, EVAL_SAMPLES)
    identifier_fragment = compute_identifier_fragmentation(tokenizer, EVAL_SAMPLES)
    script_rates = compute_script_fracture_rate(tokenizer, EVAL_SAMPLES)
    token_balance = compute_token_allocation_balance(tokenizer, EVAL_SAMPLES)
    perturb_stability = compute_perturbation_stability(tokenizer, EVAL_SAMPLES)

    references_metrics = compute_reference_metrics(references, EVAL_SAMPLES)
    baseline_tpc = references_metrics[next(iter(references_metrics))]["tpc"] if references_metrics else 1.0
    effective_gain = compute_effective_context_gain(tpc, baseline_tpc)
    morph_cosine = compute_morph_cosine_summary(tokenizer)

    metrics = {
        "avg_token_length": float(cpt),
        "tokens_per_character": float(tpc),
        "zipf_divergence": float(zipf_div),
        "zipf_best_alpha": float(best_alpha) if best_alpha is not None else None,
        "domain_cpt": domain_cpt,
        "identifier_fragmentation": float(identifier_fragment),
        "script_fracture_rates": script_rates,
        "token_allocation_js": float(token_balance),
        "perturbation_stability": float(perturb_stability),
        "effective_context_gain": float(effective_gain),
    }
    extra_outputs = {"reference_metrics": references_metrics, "morph_cosine": morph_cosine}
    export_metrics(exp_folder, metrics, list(zip(texts_eval, token_sequences)), extra_outputs)
    export_plots(exp_folder, fragmentation_curve, domain_cpt, identifier_fragment, script_rates)
    zip_results(exp_folder)

    print(
        f"Vocab size: {len(tokenizer.vocab)}, "
        f"lambda*: {tokenizer._lambda_global:.4f}, "
        f"CPT: {cpt:.2f}, "
        f"single-char ratio: {sum(len(seq)==1 for seq in token_sequences)/sum(len(seq) for seq in token_sequences):.2%}"
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
        )


if __name__ == "__main__":
    main()
