# ScalableTokenizer

ScalableTokenizer is a research tokenizer that combines statistical optimisation with explicit linguistic knowledge.  It learns a vocabulary with column generation and decodes text with dynamic programming while consulting morphology-aware features, named-entity gazetteers, and regex guards.  The repository contains the core implementation (`tokenizer_core`), evaluation pipeline, Optuna integration, and documentation for extending the system.

## Repository Layout

| Path | Description |
|------|-------------|
| `tokenizer_core/` | Core library: tokenizer, linguistic feature engines, segmentation utilities, Optuna objective helpers. |
| `scripts/extract_uniseg_features.py` | Generates affix and cross-equivalence inventories directly from UniSeg corpora. |
| `configs/*.json` (example) | Experiment configurations consumed by the runner. |
| `docs/` | User guides (experiment template, Optuna usage, background notes). |
| `tests/` | Lightweight unit tests that exercise the UniSeg sentence-level evaluation stack. |
| `experiment_outputs/` | Default location for metrics, plots, and zipped artefacts. |
| `optuna_results/` | Optuna study outputs (created on demand). |

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Train a smoke experiment**
   ```bash
   python -m tokenizer_core.main_experiments --config full_eval_losses_smoke.json
   ```
   The smoke configuration trains two small experiments and writes results under `experiment_outputs/`.

3. **Inspect artefacts**
   Each experiment directory contains:
   - `metrics.json`: global metrics, per-language breakdowns, and `external_eval` segmentation results with both per-language and aggregate views.
   - `*.png`: plots for fragmentation, domain CPT, reference comparisons, embedding benchmarks, and UniSeg word/sentence charts (including morphology coverage).
   - `segmentation_report.md` / `embedding_benchmarks.md`: Markdown summaries for quick review.

## Data Requirements

- **Training corpora**: The runner samples from Hugging Face `wikiann`. Provide your own loader if you need a different dataset.
- **UniSeg**: Place UniSegments 1.0 under `dictionary_data_bases/` (or supply `external_eval.uniseg_root`). Use `scripts/extract_uniseg_features.py` to derive updated affix and cross-equivalence resources before training.
- **MUSE / MEN / other embedding benchmarks**: Configure paths in the experiment JSON. Missing files are reported in the output metadata but do not abort the run.

## Running Experiments

Use the main harness to execute one or more experiments defined in a JSON configuration:

```bash
python -m tokenizer_core.main_experiments --config full_eval_losses.json
```

Key features of the runner:
- Supports `--select` to run a subset of experiment definitions.
- Writes consolidated metrics, plots, and zipped bundles per experiment.
- Segmentation results now include per-language metrics (`per_language`) and tokenizer-level aggregates (`aggregate`) for both UniSeg word-level and sentence-level evaluations.
- Reference tokenizers can be compared by enabling `external_eval.compare_references`.

Refer to `docs/EXPERIMENT_CONFIG_TEMPLATE.md` for a detailed schema and recommended defaults. The base configuration enables GloVe embeddings, minibatching, DP eigendecomposition, and DP semantic alignment by default; override individual flags inside `experiments[i].feature_args` when you need to fall back to classical solvers.

## Hyperparameter Optimisation with Optuna

Launch Optuna studies via:
```bash
python optuna_main.py --config full_eval_losses_smoke.json --n-trials 20 --storage sqlite:///optuna.db
```

The objective function automatically normalises available metrics and averages them with the weights you provide (missing MEN/MUSE files are logged and ignored).  See `docs/OPTUNA_USE_GUIDE.md` for the full list of search-space parameters, database options, and plotting utilities.

## Updating Linguistic Resources

Run the UniSeg extractor whenever you refresh or extend the UniSeg data:
```bash
python scripts/extract_uniseg_features.py --uniseg-root path/to/UniSegments-1.0 --output-dir data
```
The generated JSON files override the built-in affix and cross-equivalence tables at import time, ensuring the morphology encoder always sees the latest statistics.

## Testing

A minimal smoke suite validates the UniSeg sentence-level evaluator and the aggregation logic used by `main_experiments`:
```bash
python -m unittest discover tests
```
Tests rely solely on synthetic UniSeg fixtures and do not require large external assets.

## Further Reading

- `docs/EXPERIMENT_CONFIG_TEMPLATE.md` – Configuration cheatsheet.
- `docs/OPTUNA_USE_GUIDE.md` – Advanced Optuna usage and metadata interpretation.
- `tokenizer_core/main_experiments.py` – Entrypoint for full runs (includes CLI help via `--help`).
- `tokenizer_core/embedding_benchmarks.py` – Details on MEN/MUSE evaluation and reporting structure.

Feel free to adapt the pipeline, swap corpora, or plug in additional evaluation routines.  Issues and improvements are welcome via pull requests.
