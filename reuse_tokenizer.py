import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from tokenizer_core.tokenizer import ScalableTokenizer


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_transition(key: Any) -> Tuple[str, str]:
    if isinstance(key, str) and "|||" in key:
        prev, curr = key.split("|||", 1)
        return prev, curr
    if isinstance(key, (list, tuple)) and len(key) == 2:
        return key[0], key[1]
    raise ValueError(f"Unsupported bigram key format: {key!r}")


def _apply_tokenizer_overrides(tokenizer: ScalableTokenizer, overrides: Dict[str, Any]) -> None:
    for attr, value in overrides.items():
        if not hasattr(tokenizer, attr):
            raise AttributeError(f"Tokenizer has no attribute '{attr}' to override.")
        setattr(tokenizer, attr, value)
    tokenizer._cost_cache.clear()


def _apply_feature_overrides(tokenizer: ScalableTokenizer, overrides: Dict[str, Any]) -> None:
    ling = tokenizer._ling
    for key, value in overrides.items():
        if key == "lexicon":
            ling.lexicon.update(value)
        elif key == "mwe":
            if isinstance(value, (list, tuple, set)):
                ling.mwe.update(value)
            else:
                ling.mwe.add(value)
        elif key == "ne_gaz":
            for tag, entries in value.items():
                if isinstance(entries, (list, tuple, set)):
                    ling.ne_gaz.setdefault(tag, set()).update(entries)
                else:
                    ling.ne_gaz.setdefault(tag, set()).add(entries)
        elif key == "token_bigram":
            for trans_key, cost in value.items():
                ling.token_bigram[_normalize_transition(trans_key)] = cost
        elif key in {
            "gamma_boundary",
            "mu_morph",
            "prefix_reward",
            "suffix_reward",
            "space_penalty",
            "email_reward",
            "url_reward",
            "hashtag_reward",
        }:
            setattr(ling, key, value)
        elif key == "morphology_kwargs":
            ling.morphology_kwargs.update(value)
        else:
            raise AttributeError(f"Unknown feature override '{key}'.")


def _snapshot_config(tokenizer: ScalableTokenizer) -> Dict[str, Any]:
    ling = tokenizer._ling
    token_bigram = {f"{a}|||{b}": cost for (a, b), cost in ling.token_bigram.items()}
    return {
        "tokenizer_args": {
            "max_token_len": tokenizer.max_token_len,
            "min_freq": tokenizer.min_freq,
            "alpha": tokenizer.alpha,
            "beta": tokenizer.beta,
            "tau": tokenizer.tau,
            "top_k_add": tokenizer.top_k_add,
            "vocab_budget": tokenizer.vocab_budget,
            "lambda_lo": tokenizer.lambda_lo,
            "lambda_hi": tokenizer.lambda_hi,
            "merge_reward": tokenizer.merge_reward,
            "short_penalty": tokenizer.short_penalty,
            "space_penalty": tokenizer.space_penalty,
        },
        "feature_args": {
            "gamma_boundary": ling.gamma_boundary,
            "mu_morph": ling.mu_morph,
            "prefix_reward": ling.prefix_reward,
            "suffix_reward": ling.suffix_reward,
            "space_penalty": ling.space_penalty,
            "email_reward": ling.email_reward,
            "url_reward": ling.url_reward,
            "hashtag_reward": ling.hashtag_reward,
            "lexicon_size": len(ling.lexicon),
            "mwe_size": len(ling.mwe),
            "ne_tags": list(ling.ne_gaz.keys()),
            "token_bigram": token_bigram,
        },
    }


def _collect_samples(args) -> Iterable[Tuple[str, str]]:
    samples = []
    if args.text:
        samples.append(("<inline>", args.text))
    if args.text_file:
        path = Path(args.text_file)
        samples.append((str(path), path.read_text(encoding="utf-8")))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Load a saved ScalableTokenizer, optionally override settings, and tokenize new text."
    )
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json produced by training.")
    parser.add_argument(
        "--manifest",
        help="Optional path to experiment_config.json. Defaults to alongside the tokenizer file.",
    )
    parser.add_argument(
        "--override",
        help="JSON file with tokenizer_args/feature_args overrides to apply before tokenizing.",
    )
    parser.add_argument("--text", help="Inline text to tokenize.")
    parser.add_argument("--text-file", help="Path to a UTF-8 text file to tokenize.")
    parser.add_argument("--lang", help="Language code to pass to tokenizer.tokenize.", default=None)
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the stored config snapshot (and manifest if present) before tokenizing.",
    )
    args = parser.parse_args()

    tok = ScalableTokenizer.load_from_file(args.tokenizer)

    manifest_path = Path(args.manifest) if args.manifest else Path(args.tokenizer).parent / "experiment_config.json"
    manifest = None
    if manifest_path.exists():
        manifest = _load_json(manifest_path)

    if args.override:
        overrides = _load_json(Path(args.override))
        if "tokenizer_args" in overrides:
            _apply_tokenizer_overrides(tok, overrides["tokenizer_args"])
        if "feature_args" in overrides:
            _apply_feature_overrides(tok, overrides["feature_args"])

    if args.show_config:
        snapshot = _snapshot_config(tok)
        print("---- Tokenizer Snapshot ----")
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        if manifest:
            print("\n---- Experiment Manifest ----")
            print(json.dumps(manifest, ensure_ascii=False, indent=2))

    samples = _collect_samples(args)
    if not samples:
        return

    for label, text in samples:
        tokens = tok.tokenize(text, lang=args.lang)
        print(f"\n[{label}]")
        print(tokens)


if __name__ == "__main__":
    main()
