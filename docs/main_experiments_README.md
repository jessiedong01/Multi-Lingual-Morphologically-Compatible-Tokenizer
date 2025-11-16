## `main_experiments.py` — Advanced Experiment Harness

`main_experiments.py` orchestrates large-scale tokenizer explorations. It can:

1. Train the tokenizer across user-defined parameter grids (or simple presets).
2. Automatically save models, debug data, and evaluation artefacts.
3. Produce quantitative metrics and plots assessing tokenization quality.
4. Compare against reference tokenizers (GPT, Gemma, Qwen) when available.
5. Bundle results (metrics + plots) into zipped experiment folders for easy sharing.

This document explains configuration, usage, outputs, and the metric suite.

---

## 1. Quick Start

### Simple Baseline Run
```bash
python main_experiments.py
```
If you do not supply a configuration, the script runs two preset experiments:

* `baseline`: original hyperparameters.
* `medium_budget_morph`: larger vocabulary budget with stronger morphology.

Models, debug info, and evaluation outputs are written under `experiment_outputs/`.

### Custom Selection
```bash
python main_experiments.py --config configs/my_sweep.json --select exp1 exp2
```
This runs only `exp1` and `exp2` from the provided config.

### Output Directory
```bash
python main_experiments.py --config cfg.json --output-dir runs/august_sweep
```
Results (including zipped artefacts) are stored under `runs/august_sweep`.

---

## 2. Configuration Files

Supply a JSON config via `--config`. Structure:
```json5
{
  "base_lang_codes": {"en": "English", "ja": "Japanese"},
  "per_lang": 120,
  "base_tokenizer_args": {
    "max_token_len": 12,
    "min_freq": 5
  },
  "base_feature_args": {
    "gamma_boundary": 0.05,
    "mu_morph": 0.30,
    "prefix_reward": 0.04,
    "suffix_reward": 0.08,
    "email_reward": -0.25,
    "url_reward": -0.35,
    "hashtag_reward": -0.05,
    "morphology_kwargs": {
      "embedding_mode": "glove",
      "glove_lr": 0.001,
      "glove_iters": 5
    }
  },
  "base_train_args": {
    "max_iterations": 120
  },
  "experiments": [
    { "name": "baseline" },
    {
      "name": "merge_sweep",
      "grid": {
        "tokenizer_args": {
          "merge_reward": [0.08, 0.10, 0.12],
          "short_penalty": [0.04, 0.06]
        },
        "feature_args": {
          "mu_morph": [0.25, 0.35]
        }
      },
      "repeat": 2
    }
  ]
}
```
Key sections:

| Section | Purpose |
|---------|---------|
| `base_lang_codes` | Languages for training corpus (defaults to English + Japanese). |
| `per_lang` | Paragraphs per language during training (controls dataset size). |
| `base_tokenizer_args` | Default `ScalableTokenizer` kwargs. |
| `base_feature_args` | Default `LinguisticModels` kwargs (including email/url/hashtag rewards and morphology settings). |
| `base_train_args` | Default `tokenizer.train()` kwargs. |
| `experiments` | List of experiment configs. Each may include direct args, or a `grid` to sweep over values. `repeat` re-runs same setting with different seeds. |

If `experiments` is omitted, the script falls back to the built-in pairs (`baseline`, `medium_budget_morph`).

---

## 3. Main Workflow

### 3.1 Build Tokenizer
`build_tokenizer(...)` merges base args and experiment overrides, instantiates `ScalableTokenizer`, and configures `LinguisticModels` with lexicon, gazetteer, bigrams, semantic bonuses, and morphology kwargs.

### 3.2 Training
`run_experiment(...)` loads training corpus via `load_wikiann_corpus`, executes `tokenizer.train(...)`, then saves:

* `tokenizer.json` – serialized model (vocab, parameters, morphology embeddings).
* `debug.json` – counts of filtered tokens, vocab stats, etc.

### 3.3 Evaluation Suite
For each experiment, tokenizes a fixed evaluation set (`EVAL_SAMPLES`) and computes:

| Metric | Meaning |
|--------|---------|
| CPT / TPC | Characters-per-token and tokens-per-character (efficiency). |
| Zipf divergence | KL divergence from ideal Zipf distribution. |
| Fragmentation curve | Tokens per word across frequency deciles. |
| Domain-specific CPT | Compression per domain/class in eval set. |
| Identifier fragmentation | Tokens per code identifier (snake, camel, etc.). |
| Script fracture rate | Fraction of characters split across multiple tokens by script. |
| Token allocation balance | Jensen–Shannon divergence between language coverage and token usage. |
| Perturbation stability | Sensitivity to typos/unicode variants. |
| Effective context gain | TPC improvement vs reference tokenizer. |
| Morph cosine summary | Average cosine alignment for cross-lingual suffix groups. |
| Sample tuples | Stored for manual inspection. |

If reference tokenizers are available (GPT via `tiktoken`, Gemma/Qwen via `transformers`), their CPT/TPC is recorded too.

### 3.4 Artefacts and Packaging
Outputs per experiment:

* `metrics.json` – all numeric results.
* `token_samples.json` – raw text with token breakdown.
* `reference_metrics.json` – baseline CPT/TPC comparisons (if computed).
* `morph_cosine.json` – cross-language cosine summaries.
* Plots (`fragmentation_curve.png`, etc.) if matplotlib is installed.
* Zipped bundle `<experiment_name>_<timestamp>.zip` collecting all files.

---

## 4. Extending `main_experiments.py`

You can customize further by modifying `EVAL_SAMPLES`, supplying domain-specific corpora, or adding new metrics inside `run_experiment`. The script is designed so that `ScalableTokenizer` internals (email/url rewards, morphology configuration, cross-lingual penalties) can be toggled through the config without touching source code.

---

## 5. Dependencies & Reference Tokenizers

* Core dependencies match the main project (`numpy`, `matplotlib`, etc.).
* Optional extras for comparison:
  * `tiktoken` (OpenAI GPT tokenizer).
  * `transformers` (for Gemma/Qwen). If unavailable, comparison silently skips.

Install them using:
```bash
pip install tiktoken transformers
```

---

## 6. Tips

* Large sweeps can be compute-intensive; adjust `per_lang`, `max_iterations`, or prune the grid for quick tests.
* To avoid OOM errors in the morphology encoder, lower `per_lang`, increase `min_freq`, or switch to the `glove` mini-batch path with smaller `glove_max_pairs`.
* The zipped artefacts provide a self-contained record of each run—handy for sharing results or tracking regression over time.

Enjoy exploring the tokenizer landscape!
