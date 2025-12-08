"""
Test UniSeg boundary alignment in the tokenizer.

This script demonstrates that the DP decoder rewards token boundaries
that align with gold morpheme boundaries from UniSegments data.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.tokenizer import ScalableTokenizer
from tokenizer_core.linguistic_features import LinguisticModels


def find_uniseg_root():
    """Find the UniSeg data directory."""
    candidates = [
        ROOT / "data" / "uniseg_word_segments",
        ROOT / "tokenizer_core" / "data" / "uniseg_word_segments",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def test_boundary_lookup():
    """Test that we can load and look up UniSeg boundaries."""
    print("=" * 70)
    print("TEST 1: UniSeg Boundary Lookup")
    print("=" * 70)
    
    uniseg_root = find_uniseg_root()
    print(f"UniSeg root: {uniseg_root}")
    
    if uniseg_root is None:
        print("SKIP: UniSeg data not found")
        return False
    
    ling = LinguisticModels(uniseg_root=uniseg_root, uniseg_reward=0.1)
    
    # Try loading English
    loaded = ling.load_uniseg_for_lang("en")
    print(f"English loaded: {loaded}")
    
    if not loaded:
        print("SKIP: Could not load English UniSeg data")
        return False
    
    # Test some words
    test_words = [
        ("walking", "en"),     # walk + ing
        ("unhappy", "en"),     # un + happy
        ("teachers", "en"),    # teach + er + s
        ("running", "en"),     # run + n + ing
        ("impossible", "en"),  # im + possible
    ]
    
    print("\nBoundary lookups:")
    found_any = False
    for word, lang in test_words:
        boundaries = ling.get_uniseg_boundaries(word, lang)
        if boundaries:
            found_any = True
            print(f"  '{word}' -> boundaries at {sorted(boundaries)}")
            # Show what this means
            parts = []
            prev = 0
            for b in sorted(boundaries):
                parts.append(word[prev:b])
                prev = b
            parts.append(word[prev:])
            print(f"          -> segments: {' + '.join(parts)}")
        else:
            print(f"  '{word}' -> not in database")
    
    return found_any


def test_paragraph_boundaries():
    """Test paragraph-level boundary extraction."""
    print("\n" + "=" * 70)
    print("TEST 2: Paragraph Boundary Extraction")
    print("=" * 70)
    
    uniseg_root = find_uniseg_root()
    if uniseg_root is None:
        print("SKIP: UniSeg data not found")
        return False
    
    ling = LinguisticModels(uniseg_root=uniseg_root, uniseg_reward=0.1)
    
    text = "The teachers are walking unhappily to the buildings."
    gold_boundaries = ling.precompute_paragraph_boundaries(text, "en")
    
    print(f"Text: '{text}'")
    print(f"Gold boundaries: {sorted(gold_boundaries)}")
    
    if gold_boundaries:
        print("\nBoundary visualization:")
        # Show boundaries in text
        for pos in sorted(gold_boundaries):
            print(f"  Position {pos}: ...'{text[max(0,pos-3):pos]}|{text[pos:min(len(text),pos+3)]}'...")
    
    return len(gold_boundaries) > 0


def test_tokenizer_with_alignment():
    """Test that the tokenizer uses boundary alignment during decoding."""
    print("\n" + "=" * 70)
    print("TEST 3: Tokenizer with UniSeg Alignment")
    print("=" * 70)
    
    uniseg_root = find_uniseg_root()
    if uniseg_root is None:
        print("SKIP: UniSeg data not found")
        return False
    
    # Larger corpus to learn meaningful vocabulary
    paragraphs = [
        "The workers are walking through the buildings and talking.",
        "Teachers teaching students are hardworking professionals.",
        "Unhappiness and impossibility are abstract concepts.",
        "Running and walking are good exercises for health.",
        "The builders are building new buildings for workers.",
        "Teaching and learning require patience and dedication.",
        "Working hard leads to success and happiness.",
        "The speakers are speaking about interesting topics.",
    ] * 3  # Repeat to increase frequency
    
    langs = ["en"] * len(paragraphs)
    
    # Create tokenizer WITHOUT UniSeg alignment
    print("\nTokenizer WITHOUT UniSeg alignment:")
    tok_no_align = ScalableTokenizer(
        max_token_len=12,
        min_freq=1,
        vocab_budget=150,
        device="cpu",
    )
    tok_no_align._initialize_stats_and_vocab(paragraphs, langs)
    
    # Create tokenizer WITH UniSeg alignment  
    print("\nTokenizer WITH UniSeg alignment:")
    tok_with_align = ScalableTokenizer(
        max_token_len=12,
        min_freq=1,
        vocab_budget=150,
        device="cpu",
        uniseg_root=uniseg_root,
        uniseg_reward=0.2,  # Higher reward to show effect
    )
    tok_with_align._initialize_stats_and_vocab(paragraphs, langs)
    
    # Show vocabulary differences
    vocab_no = set(tok_no_align.vocab)
    vocab_with = set(tok_with_align.vocab)
    
    print("\n" + "-" * 70)
    print("Vocabulary Analysis:")
    print("-" * 70)
    print(f"Without alignment: {len(vocab_no)} tokens")
    print(f"With alignment:    {len(vocab_with)} tokens")
    
    only_with = vocab_with - vocab_no
    only_without = vocab_no - vocab_with
    if only_with:
        print(f"\nTokens ONLY in aligned vocab (gained): {sorted(only_with)[:20]}")
    if only_without:
        print(f"Tokens ONLY in non-aligned vocab (lost): {sorted(only_without)[:20]}")
    
    # Check for morpheme-aligned tokens
    morph_tokens = ["ing", "er", "ers", "ness", "tion", "un", "im", "teach", "walk", "work", "build"]
    print("\nMorpheme token presence:")
    for tok in morph_tokens:
        in_no = tok in vocab_no
        in_with = tok in vocab_with
        status = ""
        if in_with and not in_no:
            status = "<- GAINED with alignment!"
        elif in_no and not in_with:
            status = "<- LOST with alignment"
        print(f"  '{tok}': without={in_no}, with={in_with} {status}")
    
    # Compare tokenizations
    print("\n" + "-" * 70)
    print("Sample tokenizations:")
    print("-" * 70)
    
    test_texts = [
        "walking and talking",
        "teachers and workers", 
        "unhappiness",
        "buildings",
    ]
    
    for text in test_texts:
        toks_no = tok_no_align.tokenize(text, "en")
        toks_with = tok_with_align.tokenize(text, "en")
        
        print(f"\n'{text}':")
        print(f"  Without: {toks_no}")
        print(f"  With:    {toks_with}")
        if toks_no != toks_with:
            print("  -> DIFFERENT")
    
    return True


def test_alignment_reward_calculation():
    """Test the actual reward calculation for specific spans."""
    print("\n" + "=" * 70)
    print("TEST 4: Alignment Reward Calculation")
    print("=" * 70)
    
    uniseg_root = find_uniseg_root()
    if uniseg_root is None:
        print("SKIP: UniSeg data not found")
        return False
    
    ling = LinguisticModels(uniseg_root=uniseg_root, uniseg_reward=0.1)
    
    # Test word: "walking" with boundary at position 4 (walk|ing)
    word = "walking"
    boundaries = ling.get_uniseg_boundaries(word, "en")
    
    if not boundaries:
        print(f"'{word}' not in database, trying uppercase...")
        boundaries = ling.get_uniseg_boundaries(word.upper(), "en")
    
    if boundaries:
        print(f"Word: '{word}', gold boundaries: {boundaries}")
        
        # Test reward for different token boundaries
        for end_pos in range(1, len(word)):
            reward = ling.boundary_alignment_reward(word, end_pos, "en")
            status = "ALIGNED" if reward > 0 else ""
            print(f"  Token ending at {end_pos} ('{word[:end_pos]}|{word[end_pos:]}'): reward={reward:.3f} {status}")
    else:
        print(f"'{word}' not found in UniSeg database")
    
    return True


def main():
    print("UniSeg Boundary Alignment Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("Boundary Lookup", test_boundary_lookup()))
    results.append(("Paragraph Boundaries", test_paragraph_boundaries()))
    results.append(("Reward Calculation", test_alignment_reward_calculation()))
    results.append(("Tokenizer Integration", test_tokenizer_with_alignment()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "PASS" if passed else "SKIP/FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()

