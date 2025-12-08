"""Debug why 'unhappy' gets char-level tokenization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.tokenizer import ScalableTokenizer
from tokenizer_core.linguistic_features import LinguisticModels


def main():
    uniseg_root = ROOT / "data" / "uniseg_word_segments"
    
    # Same corpus as smoke test
    corpus = [
        "walking talking running jumping",
        "walked talked jumped",
        "walker talker runner jumper", 
        "teachers workers builders painters",
        "teaching working building painting",
        "unhappy unkind unfair unclear",  # <- "unhappy" is here
        "happiness kindness fairness clearness",
    ] * 20
    
    langs = ["en"] * len(corpus)
    
    print("=" * 60)
    print("DEBUG: Why 'unhappy' gets char-level tokenization")
    print("=" * 60)
    
    # Check UniSeg boundaries
    ling = LinguisticModels(uniseg_root=uniseg_root, uniseg_reward=0.2)
    ling.load_uniseg_for_lang("en")
    
    bounds = ling.get_uniseg_boundaries("unhappy", "en")
    print(f"\nUniSeg says 'unhappy' should split at: {bounds}")
    print(f"  -> 'un' + 'happy'")
    
    # Count occurrences in corpus
    full_text = " ".join(corpus)
    
    print(f"\nCorpus statistics:")
    print(f"  'unhappy' occurrences: {full_text.lower().count('unhappy')}")
    print(f"  'un' as standalone: {full_text.lower().split().count('un')}")
    print(f"  'happy' as standalone: {full_text.lower().split().count('happy')}")
    print(f"  'walking' occurrences: {full_text.lower().count('walking')}")
    print(f"  'walk' as substring: {full_text.lower().count('walk')}")
    print(f"  'ing' as substring: {full_text.lower().count('ing')}")
    
    # Train tokenizer
    tok = ScalableTokenizer(
        max_token_len=10,
        min_freq=2,
        vocab_budget=100,
        top_k_add=8,
        device="cpu",
        uniseg_root=uniseg_root,
        uniseg_reward=0.2,
        use_morph_encoder=False,
    )
    
    tok._initialize_stats_and_vocab(corpus, langs)
    tok.train(corpus, langs, max_iterations=5, verbose=False)
    
    # Check what's in vocabulary
    print(f"\n" + "-" * 60)
    print("VOCABULARY CHECK")
    print("-" * 60)
    
    check_tokens = ["un", "happy", "unhappy", "walk", "ing", "walking"]
    for t in check_tokens:
        in_vocab = t in tok.vocab
        print(f"  '{t}' in vocabulary: {in_vocab}")
    
    # Show multi-char tokens that contain 'un' or 'happy'
    print(f"\nTokens containing 'un' or 'happy':")
    for t in tok.vocab:
        if len(t) > 1 and ('un' in t.lower() or 'happy' in t.lower()):
            print(f"  '{t}'")
    
    # Explain why
    print(f"\n" + "=" * 60)
    print("EXPLANATION")
    print("=" * 60)
    print("""
    The tokenizer can only use tokens that are IN THE VOCABULARY.
    
    For a token to be added to vocabulary, it needs:
    1. Frequency >= min_freq (default 2)
    2. Positive reduced cost (would reduce total segmentation cost)
    3. Space in the budget (vocab_budget=100)
    
    Why 'walk' + 'ing' works:
    - 'walking', 'talking', 'running', 'jumping' all end in 'ing'
    - 'walked', 'walker' share 'walk'
    - High frequency + high reduced cost -> added to vocab
    
    Why 'un' + 'happy' doesn't work:
    - 'unhappy' appears only in ONE sentence pattern
    - 'un' prefix appears in: unhappy, unkind, unfair, unclear
    - But these don't share the SAME suffix/root combination
    - Lower frequency signal -> may not reach vocab
    
    The UniSeg REWARD helps, but can't force tokens into vocab!
    """)
    
    # Try with more 'happy' occurrences
    print("\n" + "-" * 60)
    print("EXPERIMENT: Add more 'happy' variations")
    print("-" * 60)
    
    corpus2 = corpus + [
        "happy happier happiest happily",
        "unhappy very unhappy so unhappy",
    ] * 20
    langs2 = ["en"] * len(corpus2)
    
    tok2 = ScalableTokenizer(
        max_token_len=10,
        min_freq=2,
        vocab_budget=150,  # slightly bigger
        top_k_add=8,
        device="cpu",
        uniseg_root=uniseg_root,
        uniseg_reward=0.2,
        use_morph_encoder=False,
    )
    
    tok2._initialize_stats_and_vocab(corpus2, langs2)
    tok2.train(corpus2, langs2, max_iterations=5, verbose=False)
    
    print(f"With more 'happy' variations:")
    for t in ["un", "happy", "happi", "unhappy"]:
        in_vocab = t in tok2.vocab
        print(f"  '{t}' in vocabulary: {in_vocab}")
    
    tokens = tok2.tokenize("unhappy", "en")
    print(f"\n  'unhappy' -> {tokens}")


if __name__ == "__main__":
    main()


