# Experiment Configuration Template

This template shows the structure of a configuration JSON for `main_experiments.py`. Adjust the values to suit your corpus, languages, model budget, and evaluation scope.

```json
{
  "base_lang_codes": {
    "en": "English",
    "de": "German"
  },
  "per_lang": 1000,
  "base_tokenizer_args": {
    "max_token_len": 24,
    "min_freq": 2,
    "top_k_add": 200,
    "vocab_budget": 2200,
    "tau": 0.001,
    "merge_reward": 0.85,
    "short_penalty": 1.0,
    "space_penalty": 0.60,
    "device": "cuda"
  },
  "base_feature_args": {
    "gamma_boundary": 0.05,
    "mu_morph": 0.32,
    "prefix_reward": 0.045,
    "suffix_reward": 0.08,
    "space_penalty": 0.60,
    "morphology_kwargs": {
      "embedding_mode": "glove",
      "k": 64,
      "lambda_morph": 0.08,
      "gamma": 0.00075,
      "refine_lr": 0.04,
      "refine_steps": 15,
      "glove_iters": 18,
      "glove_lr": 0.035,
      "glove_xmax": 120.0,
      "glove_alpha": 0.75,
      "glove_max_pairs": 250000,
      "use_minibatch": true,
      "batch_size_pairs": 8000,
      "batch_size_edges": 2000,
      "batch_size_semantic": 512,
      "use_semantic_consistency": true,
      "semantic_lr": 0.02,
      "semantic_iters": 6,
      "use_dp_semantic": true,
      "use_structure_mapping": false,
      "use_cross_kl": false,
      "use_dp_eig": true,
      "use_iterative_eig": false,
      "optimizer": "adagrad",
      "adagrad_eps": 1e-8,
      "max_tokens": 25000,
      "ngram_orders": [2, 3, 4]
    },
    "token_bigram": {
      "<BOS>|||InitCap": -0.25,
      "InitCap|||InitCap": -0.32,
      "NUM|||NUM": -0.30
    }
  },
  "base_train_args": {
    "max_iterations": 2000,
    "rc_stop_tol": -1e-6
  },
  "semantic_toggles": {
    "email_reward": -0.25,
    "url_reward": -0.32,
    "hashtag_reward": -0.08
  },
  "experiments": [
    {
      "name": "baseline",
      "tokenizer_args": {},
      "feature_args": {},
      "train_args": {}
    },
    {
      "name": "semantic_consistency",
      "feature_args": {
        "morphology_kwargs": {
          "use_semantic_consistency": true,
          "semantic_lr": 0.02,
          "semantic_iters": 5,
          "use_dp_semantic": true
        }
      }
    }
  ],
  "external_eval": {
    "languages": ["en", "de"],
    "max_uniseg_words": 1000,
    "compare_references": true,
    "flores_map": {
      "de": {
        "path": "benchmarks/flores/de-en.jsonl",
        "limit": 200,
        "source_lang": "en",
        "target_lang": "de"
      }
    }
  },
  "embedding_eval": {
    "max_reference_paragraphs": 400,
    "morphology_kwargs": {
      "max_tokens": 15000,
      "ngram_orders": [2, 3]
    },
    "muse": [
      {
        "name": "muse-en-de",
        "path": "benchmarks/muse/en-de.txt",
        "source_lang": "en",
        "target_lang": "de",
        "max_pairs": 1500,
        "csls_k": 10
      }
    ],
    "similarity": [
      {
        "name": "men-3k",
        "path": "benchmarks/MEN/MEN_dataset_natural_form_full",
        "language": "en",
        "has_header": false
      }
    ]
  }
}
```

## Parameter Guidance

- **`device`**: Set to `"cuda"` (or `"cuda:0"`, etc.) to keep training on GPU; the default falls back to CUDA automatically when available.
- **DP options**: Dynamic-programming eigendecomposition (`use_dp_eig`) and semantic alignment (`use_dp_semantic`) are enabled by default alongside GloVe + minibatching. Set them to `false` (or enable `use_iterative_eig`) if you need to fall back to the classical solvers for debugging.
- **Per experiment overrides**: Place overrides inside `experiments[].tokenizer_args` or `feature_args`. Values merge with the base dictionaries, so specify only the fields you want to change.
- **Evaluation scopes**: Trim `max_uniseg_words`, `max_pairs`, and `max_reference_paragraphs` for smoke tests; raise them for full benchmarks.
- **Segmentation outputs**: `main_experiments` now records UniSeg results as `{"word_level": {"per_language": ..., "aggregate": ...}, "sentence_level": {…}}`. The accompanying plots mirror that structure, so add or remove languages here to control which bars appear in `external_eval` figures.

Use this template as a starting point and tailor the numeric budgets (iterations, vocabulary size, batch sizes) to your hardware constraints.
