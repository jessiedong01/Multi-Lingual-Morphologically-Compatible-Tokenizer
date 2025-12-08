"""
CLI entry point for Optuna hyperparameter optimization.
"""

import argparse
from pathlib import Path
from tokenizer_core.optuna_optimizer import (
    run_optuna_study,
    export_best_params,
    export_study_results,
)


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for ScalableTokenizer")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to base JSON config file",
    )
    parser.add_argument(
        "--study-name",
        default="tokenizer_optimization",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum time in seconds (None = no limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optuna_results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for study persistence (e.g., 'sqlite:///optuna.db')",
    )
    parser.add_argument(
        "--skip-slow-eval",
        action="store_true",
        help="Skip embedding evaluation for faster trials",
    )
    parser.add_argument(
        "--objective-weights",
        type=str,
        default=None,
        help="JSON string with objective weights: '{\"tpc\": 0.3, \"zipf\": 0.2, ...}'",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Device list for training (e.g., cuda:0 cuda:1 or 'auto' to use all GPUs).",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        help="Distributed backend to use with multi-GPU Optuna trials.",
    )
    
    args = parser.parse_args()
    
    # Parse objective weights
    objective_weights = None
    if args.objective_weights:
        import json
        objective_weights = json.loads(args.objective_weights)
    
    # Run optimization
    study = run_optuna_study(
        base_config_path=args.config,
        study_name=args.study_name,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        objective_weights=objective_weights,
        storage=args.storage,
        skip_slow_eval=args.skip_slow_eval,
        output_dir=args.output_dir,
        devices=args.devices,
        dist_backend=args.dist_backend,
    )
    
    # Export results
    export_study_results(study, args.output_dir)
    export_best_params(study, args.config, args.output_dir / "best_params.json")
    
    print(f"\n=== Optimization Complete ===")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

