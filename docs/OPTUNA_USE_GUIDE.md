# Optuna Hyperparameter Optimization Guide

## Quick Start

### Basic Usage

```bash
python optuna_main.py --config full_eval_losses.json --n-trials 50
```

All example commands assume you invoke the launcher from the repository root. The study uses `tokenizer_core.main_experiments` under the hood, so any configuration you can run with `python -m tokenizer_core.main_experiments` can also be optimized through Optuna.

### With Custom Objective Weights

```bash
python optuna_main.py \
    --config full_eval_losses.json \
    --n-trials 100 \
    --objective-weights '{"tpc": 0.4, "zipf": 0.3, "fragmentation": 0.3}'
```

### With Database Persistence

```bash
python optuna_main.py \
    --config full_eval_losses.json \
    --n-trials 100 \
    --storage "sqlite:///optuna.db" \
    --study-name "my_optimization"
```

---

## Parameters Being Optimized

### Tokenizer Arguments
- `max_token_len`: [8, 32] – Maximum token length
- `min_freq`: [1, 10] – Minimum frequency threshold
- `top_k_add`: [4, 50] – Tokens to add per iteration
- `vocab_budget`: [200, 2000] – Target vocabulary size
- `tau`: [1e-5, 1e-2] (log scale) – Length penalty
- `alpha`: [0.5, 2.0] – NLL weight
- `beta`: [0.1, 1.0] – PMI weight
- `merge_reward`: [0.0, 1.0] – Merge reward
- `short_penalty`: [0.0, 2.0] – Short token penalty
- `space_penalty`: [0.0, 1.0] – Space penalty

### Feature Arguments
- `gamma_boundary`: [0.01, 0.2] – Boundary transition penalty
- `mu_morph`: [0.1, 0.5] – Morphology weight
- `prefix_reward`: [0.0, 0.1] – Prefix reward
- `suffix_reward`: [0.0, 0.15] – Suffix reward
- `space_penalty`: [0.0, 1.0] – Space penalty
- `email_reward`: [-0.5, 0.0] – Email reward
- `url_reward`: [-0.5, 0.0] – URL reward
- `hashtag_reward`: [-0.2, 0.0] – Hashtag reward

### Morphology Kwargs
By default the base configuration already enables GloVe embeddings, minibatching, DP eigendecomposition, and DP semantic alignment. Optuna explores alternatives around those defaults:

- `embedding_mode`: ["ppmi", "glove"] – Embedding method
- `k`: [32, 128] – Embedding dimension
- `lambda_morph`: [0.01, 0.2] – Morphological regulariser
- `gamma`: [1e-5, 1e-2] (log scale) – Weight decay

**PPMI-specific** (when `embedding_mode == "ppmi"`):
- `refine_lr`: [0.01, 0.1] – Refinement learning rate
- `refine_steps`: [5, 50] – Refinement iterations

**GloVe-specific** (when `embedding_mode == "glove"`):
- `glove_iters`: [5, 30] – GloVe iterations
- `glove_lr`: [0.001, 0.1] (log scale) – GloVe learning rate
- `glove_xmax`: [10.0, 200.0] – GloVe xmax
- `glove_alpha`: [0.5, 1.0] – GloVe alpha
- `glove_max_pairs`: [50000, 500000] – Max co-occurrence pairs

**Optimisation**:
- `optimizer`: ["sgd", "adagrad"] – Optimiser type
- `use_minibatch`: [True, False] – Toggle minibatching
- `batch_size_pairs`: [512, 16384] – Pair batch size (only when minibatching)
- `batch_size_edges`: [128, batch_size_pairs] – Edge batch size (only when minibatching)
- `batch_size_semantic`: [256, 2048] – Semantic batch size

**Cross-lingual Alignment**:
- `use_semantic_consistency`: [True, False] – Enable semantic consistency term
- `use_dp_semantic`: [True, False] – Use DP alignment updates
- `semantic_lr`: [0.005, 0.05] – Semantic learning rate
- `semantic_iters`: [1, 10] – Semantic iterations
- `use_structure_mapping`: [True, False] – Enable structure mapping
- `structure_lr`: [0.005, 0.05] – Structure learning rate
- `structure_iters`: [1, 10] – Structure iterations
- `use_cross_kl`: [True, False] – Enable KL regularisation
- `kl_weight`: [0.0, 0.2] – KL weight
- `kl_lr`: [0.001, 0.02] – KL learning rate

**Advanced**:
- `use_dp_eig`: [True, False] – Use DP eigendecomposition
- `use_iterative_eig`: [True, False] – Use iterative eigendecomposition
- `ngram_orders`: [[2,3], [2,3,4], [2,4]] – Character n-gram orders
- `max_tokens`: [10000, 50000] – Max tokens limit

### Training Arguments
- `max_iterations`: [50, 160] – Maximum training iterations

---

## Objective Function

Only the metrics that are available for a given trial contribute to the score. Missing MEN/MUSE benchmarks (for example, due to absent files) are recorded in the output metadata and simply removed from the weighted average rather than forcing the trial to fail.

```python
available = {}

# Compression metrics (lower is better)
available["tpc"] = clamp(tokens_per_character / 1.0)
available["zipf"] = clamp(zipf_divergence / 0.1)
available["fragmentation"] = clamp(identifier_fragmentation / 0.5)

# Embedding metrics (higher is better) – subtract from 1
if muse_scores:
    available["muse"] = 1.0 - average(muse_scores)
if men_scores:
    available["men"] = 1.0 - average(men_scores)

objective = weighted_average(available, objective_weights)
```

Weights default to `{"tpc": 0.3, "zipf": 0.2, "fragmentation": 0.2, "men": 0.15, "muse": 0.15}`. Any metric with a zero weight is ignored. The helper transparently rescales by the sum of active weights so that removing a metric does not bias the result.

You can still override the weights via:

```bash
--objective-weights '{"tpc": 0.5, "zipf": 0.3, "fragmentation": 0.2}'
```

---

## Result Files & Metadata

Optuna writes its artefacts to `optuna_results/`:

- `study_summary.json` – Summary statistics and the best trial.
- `trials.csv` – Full trial history.
- `best_params.json` – Best parameter set in JSON format compatible with `main_experiments`.
- `*.png` – Plotly snapshots such as optimisation history and parameter importances.

Each trial stores additional context:

- Embedding benchmark outputs now include a `__meta__` section listing missing datasets (`missing`) and datasets that produced no matching pairs (`empty`).
- Segmentation metrics (written by `main_experiments`) contain both per-language results and aggregated language-level scores, making it easier to compare tokenisers on UniSeg sentence-level evaluations.

---

## Practical Tips

- Start with the smoke configuration (`full_eval_losses_smoke.json`) to validate infrastructure before launching longer runs.
- Use `--skip-slow-eval` when you only care about compression metrics; embedding benchmarks and UniSeg evaluations can dominate runtime on large corpora.
- Store large-scale studies in a database (`--storage sqlite:///optuna.db`) so you can resume or analyse them interactively with Optuna’s dashboard utilities.
- Combine Optuna trials with the plotting suite in `tokenizer_core/main_experiments.py` to inspect segmentation and embedding trends for the top-performing configurations.

