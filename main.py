import argparse
from pathlib import Path
from datetime import datetime

from tokenizer import ScalableTokenizer
from data import load_wikiann_corpus

def main(test_corpus=False, corpus_eval_samples=25):
    # --- 1. Define and Load the Corpus ---
    # Define the languages we want to train on.
    # We map short codes (e.g., 'en') to their full names for clarity.
    lang_codes = {
        'en': 'English', 
        #'da': 'Danish',
        #'de': 'German', 
        #'fr': 'French',
        #'tr': 'Turkish', 
        'ru': 'Russian', 
        'ja': 'Japanese',
        #'ar': 'Arabic', 'ta': 'Tamil', 'xh': 'Xhosa',
        #'zu': 'Zulu', 
        #'tk': 'Turkmen'
    }

    # Load the training data from the Hugging Face 'wikiann' dataset.
    num_paragraphs = 100000
    corpus_texts, corpus_langs = load_wikiann_corpus(lang_codes, per_lang=num_paragraphs)
    # fail-safe
    if not corpus_texts:
        return

    
    '''# --- 2. Initialize and Configure the Tokenizer ---
    # Create an instance of our ScalableTokenizer.
    # We configure its core behavior with these parameters:
    tokenizer = ScalableTokenizer(
        max_token_len=12, # The longest possible token (in characters)
        min_freq=7, # A word must appear at least 7 times in the corpus
        top_k_add=8, # In each training step, add the 8 "best" new tokens to the vocabulary
        vocab_budget=500, # The target size for our final vocabulary (number unique tokens)
        tau=0.001 # reward(-) penalization(+) for length
    )'''

    # Allow space_penalty to be re-used when wiring linguistic hints so the DP path
    # and feature models share the same whitespace behaviour.
    #  # increase to discourage tokens that span whitespace

    tokenizer = ScalableTokenizer(
        max_token_len=15,
        min_freq=3,
        top_k_add=500,
        #vocab_budget=1200,
        tau=2.5e-4,
        merge_reward=0.5,
        short_penalty=0.5,
        space_penalty=1.0,
        prefix_reward=0.04,
        suffix_reward=0.08,
    )
    # --- 3. Set up Linguistic "Hints" ---
    # We can give the tokenizer extra knowledge to improve its accuracy.

    # A 'lexicon' of known multi-word phrases. The numbers are scores; higher is better.
    # This encourages "New York" to be one token instead of two ("New", "York").
    lex = {"New York": 2.0, "San Jose": 1.0, "’s": 0.5, "'s": 0.5}

    # A 'named entity' (ne) gazetteer. This lists known entities.
    # We're telling it that "New York", "Berlin", and "東京" are locations (LOC).   
    ne  = {"LOC": {"New York": 2.0, "Berlin": 1.4, "東京": 3.5}}

    # 'Token bigrams' (tb) define costs for sequences of token types.
    # A negative cost is a "reward". This encourages sentences to start with a
    # capitalized word and for capitalized words to follow each other (like in a name).
    tb  = {
        ("<BOS>", "InitCap"): -0.2,
        ("InitCap", "InitCap"): -0.3,
        ("NUM", "NUM"): -0.15,
    }

    '''# Now, we apply these linguistic models and tune some internal algorithm weights.
    tokenizer.set_feature_models(
        lexicon=lex,
        ne_gaz=ne,
        token_bigram=tb,
        gamma_boundary=0.06,
        mu_morph=0.25,
    )
    '''



    # ------------------------------------------------------------------
    # Feature / optimisation catalogue (toggle via `morphology_kwargs`)
    #
    # Core embeddings:
    #   * "embedding_mode" = "ppmi" -> spectral PPMI eigendecomposition.
    #   * "embedding_mode" = "glove" -> weighted GloVe factorisation.
    #
    # Morphological regularisers:
    #   * lambda_morph > 0: local Laplacian smoothing within cross-lingual affix sets.
    #   * use_weighted_cross=True: weight Laplacian edges by language similarity scores.
    #
    # Cross-lingual alignment extras (can be combined):
    #   * use_semantic_consistency=True: learn φ per language to align embeddings semantically.
    #   * use_structure_mapping=True: learn M_{ℓ1→ℓ2} matrices for asymmetric morphology.
    #   * use_cross_kl=True with kl_weight>0: KL regularisation over softmaxed embeddings.
    #
    # Optimisers & training loops:
    #   * optimizer="sgd": plain SGD updates (stable on small corpora).
    #   * optimizer="adagrad": per-parameter AdaGrad (better coverage of rare pairs).
    #   * use_minibatch=True with batch_size_pairs/edges: enable stochastic GloVe loop.
    #       Example snippet (remember dict uses colons, not equals):
    #           "optimizer": "sgd",
    #           "use_minibatch": True,
    #           "batch_size_pairs": 2048,
    #   * glove_max_pairs limits sampled co-occurrence pairs; trim if memory is tight.
    #
    # Refinement & stabilisers:
    #   * refine_steps / refine_lr: post-factorisation smoothing of token vectors.
    #   * gamma: L2 weight decay on embeddings during optimisation.
    #   * adagrad_reset (int): optional accumulator reset period to avoid step collapse.
    #
    # Structured rewards (outside morphology_kwargs):
    #   * email_reward/url_reward/hashtag_reward in set_feature_models encourage intact spans.
    #   * prefix_reward/suffix_reward nudge known affix segments to stay unsplit.
    #
    # Adjust these switches in the dict below to activate specific behaviours.
    # ------------------------------------------------------------------
    tokenizer.set_feature_models(
        lexicon=lex,
        ne_gaz=ne,
        token_bigram=tb,
        gamma_boundary=0.05,
        mu_morph=0.30,
        # Penalise gratuitous whitespace splits; raise to discourage tokens that
        # introduce internal spaces, lower to allow more aggressive splitting.
        morphology_kwargs={
            "embedding_mode": "glove",
            # Reduce glove_lr below 1e-3 if training explodes; raise it (up to ~0.01)
            # for tiny corpora that underfit. Iteration count should grow with corpus size.
            "glove_iters": 1,      # optional overrides
            "glove_lr": 0.001,
            # xmax soft-caps frequency weighting; lower values emphasise rare morphs,
            # higher values keep common affixes influential.
            "glove_xmax": 100,
            # alpha controls curvature of the weighting function; 0.75 is GloVe default,
            # decrease towards 0.5 for steadier gradients on noisy multilingual data.
            "glove_alpha": 0.5,
            "lambda_morph": 0.05,
            "optimizer": "sgd",
            "use_minibatch": True,
            # Keep edge batches roughly 10–25% of pair batches so morphology nudges
            # each update without dominating the co-occurrence gradients.
            "batch_size_pairs": 10000,
            "batch_size_edges": 2500,
            "refine_steps": 0,
        },
    )


    # --- 4. Train the Tokenizer ---
    # This is the main training step. The tokenizer will analyze the corpus,
    # learn the best vocabulary according to our rules, and prepare for use.
    # It will run for a maximum of 300 iterations.
    tokenizer.train(corpus_texts, corpus_langs, max_iterations=200)

    # --- Persist model and debugging artefacts ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = artifacts_dir / f"tokenizer_{timestamp}.json"
    debug_path = artifacts_dir / f"debug_{timestamp}.json"
    tokenizer.save(tokenizer_path)
    tokenizer.dump_debug_info(debug_path, include_filtered=True)
    # also keep a latest pointer for convenience
    tokenizer.save(artifacts_dir / "tokenizer_latest.json")
    tokenizer.dump_debug_info(artifacts_dir / "debug_latest.json", include_filtered=True)

    # --- 5. Test the Trained Tokenizer ---

    print("\n--- Tokenization Examples ---")

    # A list of test sentences in various languages to see how well it works.
    tests = [
        ("This is a final test of the representations.", "en"),
        ("Die endgültige Prüfung der Darstellungen.", "de"),
        #("Temsilleriň soňky synagy.", "tk"),
        ("表現の最終テストです。", "ja"),
        ("Email me at alice@example.com or visit https://example.org/docs.", "en"),
        ("The price was 12,345.67 dollars on 2024-09-04.", "en"),
        ("#REDIRECT United States", "en"),
        ("# Weiterleitung Berlin", "de"),
        ("# Yönlendirme Türkiye", "tr"),
    ]

    for sentence, lang in tests:
        tokens = tokenizer.tokenize(sentence, lang=lang)
        print(f"   '{sentence}'\n   -> {tokens}\n")

    # Optional corpus-level smoke test: re-tokenize a slice of the training data
    # to confirm the learned vocabulary behaves sensibly on in-domain text.
    if test_corpus:
        print("\n--- Training Corpus Sample Tokenizations ---")
        if corpus_eval_samples <= 0:
            print("Skipped: corpus_eval_samples set to 0.")
        else:
            # Preserve the configured language ordering first, then any extras encountered.
            seen_langs = set()
            present_langs = []
            corpus_lang_set = set(corpus_langs)
            for code in lang_codes.keys():
                if code in seen_langs:
                    continue
                # Only include languages that actually appear in the corpus labels.
                if code in corpus_lang_set:
                    present_langs.append(code)
                    seen_langs.add(code)
            for lang in corpus_langs:
                if lang not in seen_langs:
                    present_langs.append(lang)
                    seen_langs.add(lang)

            if not present_langs:
                print("No labelled languages found in corpus; skipping sample inspection.")
            else:
                remaining_per_lang = {lang: corpus_eval_samples for lang in present_langs}
                emitted_per_lang = {lang: 0 for lang in present_langs}
                total_needed = sum(remaining_per_lang.values())

                for corpus_idx, (text, lang) in enumerate(zip(corpus_texts, corpus_langs)):
                    quota = remaining_per_lang.get(lang)
                    if quota is None or quota <= 0:
                        continue

                    tokens = tokenizer.tokenize(text, lang=lang)
                    emitted_per_lang[lang] += 1
                    remaining_per_lang[lang] -= 1
                    total_needed -= 1

                    print(f"[{lang} | sample {emitted_per_lang[lang]:02d}/{corpus_eval_samples} | corpus_idx={corpus_idx:07d}] {tokens}")

                    if total_needed <= 0:
                        break

                unmet = {lang: quota for lang, quota in remaining_per_lang.items() if quota > 0}
                if unmet:
                    print("Warning: insufficient corpus examples to meet requested samples per language:")
                    for lang, shortfall in unmet.items():
                        obtained = emitted_per_lang.get(lang, 0)
                        print(f"  - {lang}: requested {corpus_eval_samples}, obtained {obtained}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ScalableTokenizer and optionally inspect corpus tokenizations.")
    parser.add_argument(
        "--test-corpus",
        action="store_true",
        help="After training, tokenize a sample of the training corpus for quick inspection.",
    )
    parser.add_argument(
        "--test-corpus-samples",
        type=int,
        default=25,
        help="How many training paragraphs per language to tokenize when --test-corpus is set.",
    )
    args = parser.parse_args()
    main(test_corpus=args.test_corpus, corpus_eval_samples=args.test_corpus_samples)
