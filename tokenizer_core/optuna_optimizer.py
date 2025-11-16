"""
Optuna hyperparameter optimization for ScalableTokenizer.

This module integrates Optuna with the existing experiment framework to automatically
search for optimal hyperparameters.
"""

from __future__ import annotations

import json
import math
import optuna
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List, Mapping
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna.visualization as vis

from .main_experiments import (
    build_eval_samples_from_corpus,
    compute_cpt_tpc,
    compute_zipf_divergence,
    compute_identifier_fragmentation,
    maybe_run_segmentation_eval,
    DEFAULT_LANG_CODES,
    DEFAULT_PER_LANG,
    DEFAULT_TOKENIZER_ARGS,
    DEFAULT_FEATURE_ARGS,
    DEFAULT_TRAIN_ARGS,
    DEFAULT_SEMANTIC_TOGGLES,
)
from data import load_wikiann_corpus
from .torch_utils import default_device
import torch


def define_search_space(trial: optuna.Trial, base_config: dict) -> dict:
    """
    Define Optuna search space for all tunable parameters.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration dict (from JSON config file)
    
    Returns:
        Experiment definition dict with sampled parameters
    """
    # Extract base configs
    base_tok_args = base_config.get("base_tokenizer_args", DEFAULT_TOKENIZER_ARGS.copy())
    base_feat_args = base_config.get("base_feature_args", DEFAULT_FEATURE_ARGS.copy())
    base_train_args = base_config.get("base_train_args", DEFAULT_TRAIN_ARGS.copy())
    base_semantic_toggles = base_config.get("semantic_toggles", DEFAULT_SEMANTIC_TOGGLES.copy())
    
    # ========================================================================
    # Tokenizer Arguments
    # ========================================================================
    tokenizer_args = {}
    
    # Core tokenizer parameters
    tokenizer_args["max_token_len"] = trial.suggest_int(
        "max_token_len",
        base_tok_args.get("max_token_len", 12) - 4,
        base_tok_args.get("max_token_len", 12) + 8,
    )
    
    tokenizer_args["min_freq"] = trial.suggest_int(
        "min_freq",
        max(1, base_tok_args.get("min_freq", 5) - 3),
        base_tok_args.get("min_freq", 5) + 5,
    )
    
    tokenizer_args["top_k_add"] = trial.suggest_int(
        "top_k_add",
        20,
        200,
    )
    
    tokenizer_args["vocab_budget"] = trial.suggest_int(
        "vocab_budget",
        1000,
        20000,
    )
    
    tokenizer_args["tau"] = trial.suggest_float(
        "tau",
        1e-5,
        1e-2,
        log=True,
    )
    
    tokenizer_args["alpha"] = trial.suggest_float(
        "alpha",
        0.5,
        2.0,
    )
    
    tokenizer_args["beta"] = trial.suggest_float(
        "beta",
        0.1,
        1.0,
    )
    
    tokenizer_args["merge_reward"] = trial.suggest_float(
        "merge_reward",
        0.0,
        1.0,
    )
    
    tokenizer_args["short_penalty"] = trial.suggest_float(
        "short_penalty",
        0.0,
        2.0,
    )
    
    tokenizer_args["space_penalty"] = trial.suggest_float(
        "tok_space_penalty",
        0.0,
        1.0,
    )
    
    # ========================================================================
    # Feature Arguments (LinguisticModels)
    # ========================================================================
    feature_args = {}
    
    feature_args["gamma_boundary"] = trial.suggest_float(
        "gamma_boundary",
        0.01,
        0.2,
    )
    
    feature_args["mu_morph"] = trial.suggest_float(
        "mu_morph",
        0.1,
        0.5,
    )
    
    feature_args["prefix_reward"] = trial.suggest_float(
        "prefix_reward",
        0.0,
        0.1,
    )
    
    feature_args["suffix_reward"] = trial.suggest_float(
        "suffix_reward",
        0.0,
        0.15,
    )
    
    feature_args["space_penalty"] = trial.suggest_float(
        "feat_space_penalty",
        0.0,
        1.0,
    )
    
    # Semantic toggles
    if "email_reward" in base_semantic_toggles:
        feature_args["email_reward"] = trial.suggest_float(
            "email_reward",
            -0.5,
            0.0,
        )
    
    if "url_reward" in base_semantic_toggles:
        feature_args["url_reward"] = trial.suggest_float(
            "url_reward",
            -0.5,
            0.0,
        )
    
    if "hashtag_reward" in base_semantic_toggles:
        feature_args["hashtag_reward"] = trial.suggest_float(
            "hashtag_reward",
            -0.2,
            0.0,
        )
    
    # ========================================================================
    # Morphology Kwargs (Nested in feature_args)
    # ========================================================================
    base_morph_kwargs = base_feat_args.get("morphology_kwargs", {})
    morphology_kwargs = {}
    
    # Embedding mode (categorical)
    morphology_kwargs["embedding_mode"] = trial.suggest_categorical(
        "embedding_mode",
        ["ppmi", "glove"],
    )
    
    # Common morphology parameters
    morphology_kwargs["k"] = trial.suggest_int(
        "morph_k",
        32,
        128,
    )
    
    morphology_kwargs["lambda_morph"] = trial.suggest_float(
        "lambda_morph",
        0.01,
        0.2,
    )
    
    morphology_kwargs["gamma"] = trial.suggest_float(
        "morph_gamma",
        1e-5,
        1e-2,
        log=True,
    )
    
    # Conditional: PPMI-specific parameters
    if morphology_kwargs["embedding_mode"] == "ppmi":
        morphology_kwargs["refine_lr"] = trial.suggest_float(
            "refine_lr",
            0.01,
            0.1,
        )
        
        morphology_kwargs["refine_steps"] = trial.suggest_int(
            "refine_steps",
            5,
            50,
        )
    
    # Conditional: GloVe-specific parameters
    if morphology_kwargs["embedding_mode"] == "glove":
        morphology_kwargs["glove_iters"] = trial.suggest_int(
            "glove_iters",
            5,
            30,
        )
        
        morphology_kwargs["glove_lr"] = trial.suggest_float(
            "glove_lr",
            0.001,
            0.1,
            log=True,
        )
        
        morphology_kwargs["glove_xmax"] = trial.suggest_float(
            "glove_xmax",
            10.0,
            200.0,
        )
        
        morphology_kwargs["glove_alpha"] = trial.suggest_float(
            "glove_alpha",
            0.5,
            1.0,
        )
        
        morphology_kwargs["glove_max_pairs"] = trial.suggest_int(
            "glove_max_pairs",
            50000,
            500000,
        )
    
    # Optimization parameters
    morphology_kwargs["optimizer"] = trial.suggest_categorical(
        "optimizer",
        ["sgd", "adagrad"],
    )
    
    # Always batched; only semantic batching is relevant for coverage
    
    # Batching for semantic consistency
    morphology_kwargs["batch_size_semantic"] = trial.suggest_int(
        "batch_size_semantic",
        256,
        2048,
    )
    
    # Morphology loss mode: exactly one of none / semantic / structure / kl
    morph_loss_mode = trial.suggest_categorical(
        "morph_loss_mode",
        ["none", "semantic", "structure", "kl"],
    )
    # Initialize all as False
    morphology_kwargs["use_semantic_consistency"] = False
    morphology_kwargs["use_structure_mapping"] = False
    morphology_kwargs["use_cross_kl"] = False
    if morph_loss_mode == "semantic":
        morphology_kwargs["use_semantic_consistency"] = True
        morphology_kwargs["semantic_lr"] = trial.suggest_float(
            "semantic_lr",
            0.005,
            0.05,
        )
        morphology_kwargs["semantic_iters"] = trial.suggest_int(
            "semantic_iters",
            1,
            10,
        )
    elif morph_loss_mode == "structure":
        morphology_kwargs["use_structure_mapping"] = True
        morphology_kwargs["structure_lr"] = trial.suggest_float(
            "structure_lr",
            0.005,
            0.05,
        )
        morphology_kwargs["structure_iters"] = trial.suggest_int(
            "structure_iters",
            1,
            10,
        )
    elif morph_loss_mode == "kl":
        morphology_kwargs["use_cross_kl"] = True
        morphology_kwargs["kl_weight"] = trial.suggest_float(
            "kl_weight",
            0.0,
            0.2,
        )
        morphology_kwargs["kl_lr"] = trial.suggest_float(
            "kl_lr",
            0.001,
            0.02,
        )
    
    # DP paths removed from search space
    
    # N-gram orders (categorical choice)
    ngram_choices = {
        "2_3": (2, 3),
        "2_3_4": (2, 3, 4),
        "2_4": (2, 4),
    }
    selected_ngram_key = trial.suggest_categorical(
        "ngram_orders",
        list(ngram_choices.keys()),
    )
    morphology_kwargs["ngram_orders"] = list(ngram_choices[selected_ngram_key])
    
    # Max tokens limit - much higher
    morphology_kwargs["max_tokens"] = trial.suggest_int("max_tokens", 25000, 200000)
    
    feature_args["morphology_kwargs"] = morphology_kwargs
    
    # ========================================================================
    # Training Arguments
    # ========================================================================
    train_args = {}
    
    train_args["max_iterations"] = trial.suggest_int("max_iterations", 100, 1000)
    
    # ========================================================================
    # Return experiment definition
    # ========================================================================
    return {
        "name": f"optuna_trial_{trial.number}",
        "tokenizer_args": tokenizer_args,
        "feature_args": feature_args,
        "train_args": train_args,
    }


def compute_objective(
    global_metrics: dict,
    objective_weights: dict,
) -> float:
    """
    Objective: Only UniSeg sentence-level similarity (minimize 1 - score).
    """
    uniseg_sim = global_metrics.get("uniseg_sentence_similarity")
    if isinstance(uniseg_sim, (int, float)) and math.isfinite(uniseg_sim):
        return 1.0 - max(0.0, min(1.0, float(uniseg_sim)))
    return 1.0


def create_objective_function(
    base_config: dict,
    lang_codes: dict,
    per_lang: int,
    references: dict,
    output_dir: Path,
    external_eval_cfg: Optional[dict],
    objective_weights: dict,
    skip_slow_eval: bool = False,
) -> Callable[[optuna.Trial], float]:
    """
    Create Optuna objective function.
    
    Args:
        base_config: Base configuration dict
        lang_codes: Language codes dict
        per_lang: Paragraphs per language
        references: Reference tokenizers dict
        output_dir: Output directory for experiments
        external_eval_cfg: External evaluation config
        embedding_eval_cfg: Embedding evaluation config
        objective_weights: Weights for objective components
        skip_slow_eval: If True, skip embedding eval for faster trials
    
    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        # Sample parameters
        exp_def = define_search_space(trial, base_config)
        
        # Extract base configs
        base_tok_args = base_config.get("base_tokenizer_args", DEFAULT_TOKENIZER_ARGS.copy())
        base_feat_args = base_config.get("base_feature_args", DEFAULT_FEATURE_ARGS.copy())
        base_train_args = base_config.get("base_train_args", DEFAULT_TRAIN_ARGS.copy())
        base_semantic_toggles = base_config.get("semantic_toggles", DEFAULT_SEMANTIC_TOGGLES.copy())
        
        # Run experiment (modified to return metrics instead of saving)
        try:
            # Build tokenizer
            from .main_experiments import build_tokenizer
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
            
            tokenizer = build_tokenizer(
                base_tok_args,
                base_feat_args,
                base_semantic_toggles,
                exp_def.get("tokenizer_args", {}),
                exp_def.get("feature_args", {}),
            )
            train_args = base_train_args.copy()
            train_args.update(exp_def.get("train_args", {}))
            
            # Load data
            texts, langs = load_wikiann_corpus(lang_codes, per_lang=per_lang)
            if not texts:
                return float('inf')
            
            eval_samples = build_eval_samples_from_corpus(texts, langs, lang_codes)
            if not eval_samples:
                from .main_experiments import FALLBACK_EVAL_SAMPLES
                eval_samples = FALLBACK_EVAL_SAMPLES
            
            # Train
            tokenizer.train(texts, langs, **train_args)
            
            # Early pruning: Check basic metrics first
            token_sequences = [
                tokenizer.tokenize(sample["text"], lang=sample.get("language"))
                for sample in eval_samples
            ]
            texts_eval = [sample["text"] for sample in eval_samples]
            cpt, tpc = compute_cpt_tpc(token_sequences, texts_eval)
            
            # Report intermediate value for pruning
            trial.report(tpc, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Compute more metrics
            zipf_div, best_alpha = compute_zipf_divergence(
                [tok for seq in token_sequences for tok in seq],
                torch.linspace(0.5, 2.0, 30).tolist()
            )
            identifier_fragment = compute_identifier_fragmentation(tokenizer, eval_samples)
            
            # UniSeg segmentation evaluation (sentence-level aggregate)
            uniseg_sentence_sim = None
            if external_eval_cfg:
                try:
                    seg_results = maybe_run_segmentation_eval(
                        tokenizer,
                        lang_codes,
                        external_eval_cfg,
                        references,
                        eval_samples=eval_samples,
                    )
                    if isinstance(seg_results, dict):
                        sl = seg_results.get("sentence_level") or {}
                        agg = sl.get("aggregate") or {}
                        uniseg_sentence_sim = agg.get("sentence_similarity")
                except Exception as _e:
                    # Keep objective robust; just skip if UniSeg data unavailable
                    uniseg_sentence_sim = None
            
            # Report for pruning
            trial.report(zipf_div, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            global_metrics = {
                "tokens_per_character": float(tpc),
                "zipf_divergence": float(zipf_div),
                "identifier_fragmentation": float(identifier_fragment),
            }
            if isinstance(uniseg_sentence_sim, (int, float)):
                global_metrics["uniseg_sentence_similarity"] = float(uniseg_sentence_sim)
            
            # Compute objective
            objective_value = compute_objective(
                global_metrics,
                objective_weights,
            )
            
            # Store metrics as user attributes for analysis
            trial.set_user_attr("tpc", float(tpc))
            trial.set_user_attr("zipf_div", float(zipf_div))
            trial.set_user_attr("identifier_frag", float(identifier_fragment))
            # No MEN/MUSE user attrs
            
            return objective_value
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Handle CUDA OOM as a prunable trial instead of INF
            msg = str(e).lower()
            try:
                import torch
                if "out of memory" in msg or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", tuple())):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    raise optuna.TrialPruned()
            except Exception:
                pass
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    return objective


def load_config_dict(path: str) -> dict:
    """Load config as dict (for Optuna use)."""
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_optuna_study(
    base_config_path: str,
    study_name: str = "tokenizer_optimization",
    n_trials: int = 100,
    n_jobs: int = 1,
    timeout: Optional[float] = None,
    direction: str = "minimize",
    objective_weights: Optional[dict] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    skip_slow_eval: bool = False,
    output_dir: Optional[Path] = None,
) -> optuna.Study:
    """
    Run Optuna optimization study.
    
    Args:
        base_config_path: Path to base JSON config file
        study_name: Name for the Optuna study
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (1 = sequential)
        timeout: Maximum time in seconds (None = no limit)
        direction: "minimize" or "maximize"
        objective_weights: Weights for objective components
        storage: Database URL for study persistence (e.g., "sqlite:///optuna.db")
        load_if_exists: If True, load existing study
        skip_slow_eval: If True, skip embedding evaluation for faster trials
        output_dir: Directory to save study results
    
    Returns:
        Optuna study object
    """
    # Load base config
    base_config = load_config_dict(base_config_path)
    if not base_config:
        raise ValueError(f"Failed to load config from {base_config_path}")
    
    lang_codes = base_config.get("base_lang_codes", DEFAULT_LANG_CODES)
    per_lang = base_config.get("per_lang", DEFAULT_PER_LANG)
    external_eval_cfg = base_config.get("external_eval")
    
    # Load references
    from .main_experiments import load_reference_tokenizers
    references = load_reference_tokenizers()
    
    # Default objective weights
    if objective_weights is None:
        # Only UniSeg drives the objective
        objective_weights = {"uniseg": 1.0}
    
    # Create objective function
    if output_dir is None:
        output_dir = Path("optuna_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    objective = create_objective_function(
        base_config,
        lang_codes,
        per_lang,
        references,
        output_dir,
        external_eval_cfg,
        objective_weights,
        skip_slow_eval=skip_slow_eval,
    )
    
    # Create study
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
        interval_steps=1,
    )
    
    sampler = TPESampler(seed=42)
    
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
        )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=True,
    )
    
    return study


def export_best_params(study: optuna.Study, base_config_path: str, output_path: Path):
    """
    Export best parameters to JSON config format.
    
    Args:
        study: Optuna study
        base_config_path: Path to base config
        output_path: Path to save best config
    """
    base_config = load_config_dict(base_config_path)
    if not base_config:
        return
    
    best_params = study.best_params
    
    # Extract parameters by category
    tokenizer_args = {}
    feature_args = {}
    train_args = {}
    
    # Tokenizer args
    for key in ["max_token_len", "min_freq", "top_k_add", "vocab_budget", "tau",
                "alpha", "beta", "merge_reward", "short_penalty", "tok_space_penalty"]:
        if key in best_params:
            param_key = key.replace("tok_space_penalty", "space_penalty")
            tokenizer_args[param_key] = best_params[key]
    
    # Feature args (non-morphology)
    for key in ["gamma_boundary", "mu_morph", "prefix_reward", "suffix_reward",
                "feat_space_penalty", "email_reward", "url_reward", "hashtag_reward"]:
        if key in best_params:
            param_key = key.replace("feat_space_penalty", "space_penalty")
            feature_args[param_key] = best_params[key]
    
    # Morphology kwargs
    morphology_kwargs = {}
    morph_keys = [
        "embedding_mode", "morph_k", "lambda_morph", "morph_gamma",
        "refine_lr", "refine_steps", "glove_iters", "glove_lr", "glove_xmax",
        "glove_alpha", "glove_max_pairs", "optimizer",
        "batch_size_semantic",
        "use_semantic_consistency", "semantic_lr", "semantic_iters",
        "use_structure_mapping", "structure_lr", "structure_iters",
        "use_cross_kl", "kl_weight", "kl_lr",
        "ngram_orders", "max_tokens",
    ]
    
    for key in morph_keys:
        if key in best_params:
            param_key = key.replace("morph_k", "k").replace("morph_gamma", "gamma")
            value = best_params[key]
            if key == "ngram_orders" and isinstance(value, str):
                mapping = {
                    "2_3": [2, 3],
                    "2_3_4": [2, 3, 4],
                    "2_4": [2, 4],
                }
                value = mapping.get(value, value)
            morphology_kwargs[param_key] = value
    
    if morphology_kwargs:
        feature_args["morphology_kwargs"] = morphology_kwargs
    
    # Train args
    if "max_iterations" in best_params:
        train_args["max_iterations"] = best_params["max_iterations"]
    
    # Create output config
    output_config = {
        "base_lang_codes": base_config.get("base_lang_codes", DEFAULT_LANG_CODES),
        "per_lang": base_config.get("per_lang", DEFAULT_PER_LANG),
        "base_tokenizer_args": tokenizer_args,
        "base_feature_args": feature_args,
        "base_train_args": train_args,
        "semantic_toggles": base_config.get("semantic_toggles", DEFAULT_SEMANTIC_TOGGLES),
        "experiments": [
            {
                "name": "optuna_best",
                "tokenizer_args": {},
                "feature_args": {},
                "train_args": {},
            }
        ],
    }
    
    # Copy external_eval if present
    if "external_eval" in base_config:
        output_config["external_eval"] = base_config["external_eval"]
    
    output_path.write_text(
        json.dumps(output_config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Best parameters saved to {output_path}")


def export_study_results(study: optuna.Study, output_dir: Path):
    """
    Export study results, visualizations, and analysis.
    
    Args:
        study: Optuna study
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study as JSON
    study_dict = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
    }
    
    (output_dir / "study_summary.json").write_text(
        json.dumps(study_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # Export trials dataframe
    try:
        df = study.trials_dataframe()
        df.to_csv(output_dir / "trials.csv", index=False)
    except Exception as e:
        print(f"Warning: Could not export trials dataframe: {e}")
    
    # Generate visualizations
    try:
        fig = vis.plot_optimization_history(study)
        fig.write_image(output_dir / "optimization_history.png")
        
        fig = vis.plot_param_importances(study)
        fig.write_image(output_dir / "param_importances.png")
        
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(output_dir / "parallel_coordinate.png")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")



