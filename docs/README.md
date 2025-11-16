# Documentation Overview

This folder gathers companion guides for the ScalableTokenizer codebase.  Each document expands on a specific part of the pipeline—configuration, optimisation, or background notes—and complements the high-level introduction in the project root `README.md`.

## Guides

| Document | Description |
|----------|-------------|
| `EXPERIMENT_CONFIG_TEMPLATE.md` | JSON schema walkthrough with recommended defaults (GloVe + DP enabled, UniSeg/embedding knobs, segmentation settings). |
| `OPTUNA_USE_GUIDE.md` | How to launch Optuna studies, interpret the weighted objective, and understand the metadata emitted for missing benchmarks. |
| `2_Related_works.pdf` | Survey of tokenisation literature and references for further reading. |

## Additional Resources

- The main README describes repository layout, quick-start commands, and where experiments write their artefacts.
- `scripts/extract_uniseg_features.py` explains how to refresh affix and cross-equivalence data from UniSeg corpora.
- Inline docstrings within `tokenizer_core` provide deeper implementation details (e.g. DP eigendecomposition helpers, segmentation backoff logic).

If you add new workflows or utilities, append pointers here so future contributors can find the relevant material quickly.
