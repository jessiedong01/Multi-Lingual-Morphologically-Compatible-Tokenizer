"""
Test tokenizer WITHOUT morphological encoder, WITH UniSeg alignment.

This script:
1. Loads WikiANN English corpus (or fallback)
2. Trains tokenizer with morph_encoder disabled
3. Uses UniSeg boundary alignment only
4. Shows tokenization of first 5 paragraphs

Usage:
    python scripts/test_uniseg_no_morph.py [options]

Options (edit CONFIG below or pass as arguments in future):
    --n_samples: Number of corpus samples (default: 10000)
    --vocab_budget: Target vocabulary size (default: 4000)
    --iterations: Training iterations (default: 10)
    --top_k_add: Tokens to add per iteration (default: 8)
    --uniseg_reward: UniSeg boundary alignment reward (default: 0.15)
    --affix_reward: Affix-based token reward (default: 0.01)
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================================
# CONFIGURATION - Edit these to experiment
# ============================================================================
CONFIG = {
    "n_samples": 10000,        # Number of corpus samples to load
    "vocab_budget": 4000,      # Target vocabulary size
    "iterations": 10,          # Number of training iterations
    "top_k_add": 8,            # Tokens to add per iteration
    "uniseg_reward": 0.15,     # UniSeg boundary alignment reward
    "prefix_reward": 0.025,    # Reward for tokens with known prefixes
    "suffix_reward": 0.01,     # Reward for tokens with known suffixes
    "max_token_len": 12,       # Maximum token length
    "min_freq": 2,             # Minimum token frequency
    "alpha": 1.0,              # NLL weight
    "beta": 0.5,               # PMI penalty weight
    "tau": 0.01,               # Length penalty weight
}


def parse_args():
    """Parse command line arguments to override CONFIG."""
    parser = argparse.ArgumentParser(description="Test tokenizer without morph encoder")
    for key, default in CONFIG.items():
        parser.add_argument(f"--{key}", type=type(default), default=default)
    return parser.parse_args()


def load_wikiann_english(n_samples=10000):
    """Load WikiANN English corpus."""
    print(f"Loading {n_samples} samples from WikiANN English...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikiann", "en", split='train', streaming=True)
        
        texts = []
        for ex in dataset:
            if len(texts) >= n_samples:
                break
            toks = ex.get('tokens', [])
            txt = " ".join(toks) if isinstance(toks, list) else str(toks)
            if txt.strip():
                texts.append(txt)
        
        print(f"Loaded {len(texts)} samples from WikiANN.")
        return texts, ["en"] * len(texts)
    
    except Exception as e:
        print(f"Could not load WikiANN: {e}")
        print("Using fallback synthetic corpus...")
        
        # Fallback: synthetic English-like corpus with diverse morphology
        base_sentences = [
            "The workers are walking through the buildings.",
            "Teachers teaching students are hardworking professionals.",
            "Running and walking are excellent exercises for health.",
            "The builders are building new structures downtown.",
            "Unhappiness and impossibility are abstract concepts.",
            "Speaking clearly helps in communication skills.",
            "The scientists are researching new discoveries.",
            "Learning requires patience and dedication.",
            "The writers are writing interesting stories.",
            "Swimming and cycling are popular activities.",
            "The managers are managing multiple projects efficiently.",
            "Wonderful performers are performing beautifully tonight.",
            "The teachers are teaching the children patiently.",
            "Understanding complex problems requires careful thinking.",
            "The developers are developing new applications.",
        ]
        
        import random
        random.seed(42)
        texts = [random.choice(base_sentences) for _ in range(n_samples)]
        
        print(f"Generated {len(texts)} synthetic samples.")
        return texts, ["en"] * len(texts)


def find_uniseg_root():
    """Find UniSeg data directory."""
    candidates = [
        ROOT / "data" / "uniseg_word_segments",
        ROOT / "tokenizer_core" / "data" / "uniseg_word_segments",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def test_uniseg_reward_working(config):
    """Verify UniSeg reward is actually being applied."""
    print("=" * 70)
    print("STAGE 0: Verify UniSeg Reward is Working")
    print("=" * 70)
    
    from tokenizer_core.linguistic_features import LinguisticModels
    
    uniseg_root = find_uniseg_root()
    if not uniseg_root:
        print("ERROR: UniSeg data not found!")
        return False
    
    ling = LinguisticModels(
        uniseg_root=uniseg_root, 
        uniseg_reward=config["uniseg_reward"],
        prefix_reward=config["prefix_reward"],
        suffix_reward=config["suffix_reward"],
    )
    
    # Load English
    loaded = ling.load_uniseg_for_lang("en")
    if not loaded:
        print("ERROR: Could not load English UniSeg data!")
        return False
    
    n_words = len(ling._uniseg_boundaries.get('en', {}))
    print(f"UniSeg data loaded: {n_words:,} English words with morpheme boundaries")
    
    # Test boundary extraction
    test_text = "The teachers are walking unhappily to the buildings."
    gold_bounds = ling.precompute_paragraph_boundaries(test_text, "en")
    
    print(f"\nTest text: '{test_text}'")
    print(f"Gold boundaries found: {sorted(gold_bounds)}")
    
    if gold_bounds:
        print("\nBoundary positions (where morpheme splits occur):")
        for pos in sorted(gold_bounds):
            before = test_text[max(0, pos-4):pos]
            after = test_text[pos:min(len(test_text), pos+4)]
            print(f"  Position {pos:2d}: ...'{before}|{after}'...")
    
    # Also show affix reward (separate system)
    print(f"\nAffix reward settings:")
    print(f"  prefix_reward = {config['prefix_reward']}")
    print(f"  suffix_reward = {config['suffix_reward']}")
    print(f"  uniseg_reward = {config['uniseg_reward']}")
    
    # Test affix detection
    test_tokens = ["walking", "unhappy", "teachers", "buildings"]
    print(f"\nAffix detection (from hardcoded AFFIXES dict):")
    for tok in test_tokens:
        affix_bias = ling._affix_bias(tok, "en")
        print(f"  '{tok}': affix_reward = {affix_bias:.3f}")
    
    print("\n[OK] UniSeg reward is working correctly!")
    return True


def main():
    # Parse arguments
    args = parse_args()
    config = {k: getattr(args, k) for k in CONFIG}
    
    print("=" * 70)
    print("TOKENIZER TEST: No Morph Encoder, With UniSeg Alignment")
    print("=" * 70)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Stage 0: Verify UniSeg is working
    print()
    if not test_uniseg_reward_working(config):
        print("Aborting: UniSeg reward not working.")
        return
    
    # Stage 1: Load corpus
    print("\n" + "=" * 70)
    print("STAGE 1: Load Corpus")
    print("=" * 70)
    
    texts, langs = load_wikiann_english(n_samples=config["n_samples"])
    
    # Show first 5 paragraphs
    print("\nFirst 5 paragraphs to be tokenized:")
    for i, text in enumerate(texts[:5]):
        print(f"  [{i+1}] {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Stage 2: Create tokenizer
    print("\n" + "=" * 70)
    print("STAGE 2: Initialize Tokenizer")
    print("=" * 70)
    
    from tokenizer_core.tokenizer import ScalableTokenizer
    
    uniseg_root = find_uniseg_root()
    
    t0 = time.time()
    tokenizer = ScalableTokenizer(
        max_token_len=config["max_token_len"],
        min_freq=config["min_freq"],
        alpha=config["alpha"],
        beta=config["beta"],
        tau=config["tau"],
        top_k_add=config["top_k_add"],
        vocab_budget=config["vocab_budget"],
        device="cpu",
        uniseg_root=uniseg_root,
        uniseg_reward=config["uniseg_reward"],
        use_morph_encoder=False,  # DISABLE morphology encoder
    )
    
    # Update affix rewards in the linguistic models
    tokenizer._ling.prefix_reward = config["prefix_reward"]
    tokenizer._ling.suffix_reward = config["suffix_reward"]
    
    # Initialize with corpus
    print("\nAnalyzing corpus...")
    tokenizer._initialize_stats_and_vocab(texts, langs)
    
    init_time = time.time() - t0
    print(f"Initialization complete in {init_time:.1f}s")
    print(f"Initial vocabulary size: {len(tokenizer.vocab)}")
    print(f"Potential tokens discovered: {len(tokenizer._potential_tokens):,}")
    
    # Stage 3: Train
    print("\n" + "=" * 70)
    print(f"STAGE 3: Training ({config['iterations']} iterations)")
    print("=" * 70)
    
    train_start = time.time()
    for i in range(config["iterations"]):
        iter_start = time.time()
        tokenizer.train(iterations=1)
        iter_time = time.time() - iter_start
        
        # Progress update
        print(f"  Iteration {i+1:3d}/{config['iterations']}: "
              f"vocab={len(tokenizer.vocab):5d}, "
              f"lambda={tokenizer._lambda_global:.4f}, "
              f"time={iter_time:.2f}s")
    
    train_time = time.time() - train_start
    total_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.1f}s (total: {total_time:.1f}s)")
    print(f"Final vocabulary size: {len(tokenizer.vocab)}")
    
    # Stage 4: Vocabulary analysis
    print("\n" + "=" * 70)
    print("STAGE 4: Vocabulary Analysis")
    print("=" * 70)
    
    # Multi-character tokens
    multi_char = [t for t in tokenizer.vocab if len(t) > 1 and not t.startswith(' ')]
    multi_char_sorted = sorted(multi_char, key=len, reverse=True)
    
    print(f"\nMulti-character tokens: {len(multi_char)}")
    print(f"  Longest tokens: {multi_char_sorted[:20]}")
    print(f"  Sample tokens: {multi_char[:30]}")
    
    # Morpheme-like patterns
    morpheme_patterns = ["ing", "ed", "er", "ers", "tion", "ness", "ly", "un", "re", "ment"]
    print("\nMorpheme-like tokens found:")
    for pattern in morpheme_patterns:
        matching = [t for t in tokenizer.vocab if pattern in t.lower() and len(t) > len(pattern)]
        if matching:
            print(f"  *{pattern}*: {matching[:8]}{'...' if len(matching) > 8 else ''} ({len(matching)} total)")
    
    # Stage 5: Tokenize first 5 paragraphs
    print("\n" + "=" * 70)
    print("STAGE 5: Tokenize First 5 Paragraphs")
    print("=" * 70)
    
    for i, text in enumerate(texts[:5]):
        tokens = tokenizer.tokenize(text, "en")
        
        print(f"\n[{i+1}] Input: {text}")
        print(f"    Output ({len(tokens)} tokens): {tokens}")
        
        # Check boundary alignment
        gold_bounds = tokenizer._ling.precompute_paragraph_boundaries(text, "en")
        if gold_bounds:
            pos = 0
            aligned = []
            for tok in tokens[:-1]:
                pos += len(tok)
                if pos in gold_bounds:
                    aligned.append(pos)
            if aligned:
                print(f"    Aligned with UniSeg at: {aligned}")
    
    # Stage 6: Statistics
    print("\n" + "=" * 70)
    print("STAGE 6: Summary Statistics (on first 100 samples)")
    print("=" * 70)
    
    total_tokens = 0
    total_chars = 0
    aligned_count = 0
    total_boundaries = 0
    
    for text in texts[:100]:
        tokens = tokenizer.tokenize(text, "en")
        total_tokens += len(tokens)
        total_chars += len(text)
        
        # Count aligned boundaries
        gold_bounds = tokenizer._ling.precompute_paragraph_boundaries(text, "en")
        pos = 0
        for tok in tokens[:-1]:
            pos += len(tok)
            total_boundaries += 1
            if pos in gold_bounds:
                aligned_count += 1
    
    print(f"  Avg tokens per sample: {total_tokens / 100:.1f}")
    print(f"  Avg chars per token: {total_chars / total_tokens:.2f}")
    print(f"  Compression ratio: {total_chars / total_tokens:.2f}x")
    
    if total_boundaries > 0:
        alignment_rate = aligned_count / total_boundaries * 100
        print(f"  UniSeg alignment rate: {aligned_count}/{total_boundaries} ({alignment_rate:.1f}%)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"\nTo re-run with different settings:")
    print(f"  python scripts/test_uniseg_no_morph.py --uniseg_reward 0.2 --top_k_add 16 --iterations 20")


if __name__ == "__main__":
    main()
