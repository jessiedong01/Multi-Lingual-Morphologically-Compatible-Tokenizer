"""
Smoke test: Does the tokenizer learn morpheme-aligned tokens from UniSeg data?

This test verifies:
1. UniSeg word boundaries are loaded from JSONL files
2. Affixes (prefixes/suffixes) are extracted from UniSeg JSONL segments
3. The tokenizer learns morpheme-aligned tokens like "walk" + "ing"
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.tokenizer import ScalableTokenizer
from tokenizer_core.linguistic_features import LinguisticModels


def compute_coverage(texts, loader, lang="en"):
    import re

    if loader is None:
        return 0.0, 0, 0
    word_pattern = re.compile(r"\b\w+\b", re.UNICODE)
    vocab = set()
    for text in texts:
        for match in word_pattern.finditer(text):
            vocab.add(match.group().lower())
    hits = sum(1 for w in vocab if loader.get_boundaries(w, lang))
    percent = (hits / len(vocab) * 100) if vocab else 0.0
    return percent, hits, len(vocab)


def main():
    print("=" * 60)
    print("SMOKE TEST: UniSeg-Based Morpheme Alignment")
    print("=" * 60)
    
    uniseg_root = ROOT / "data" / "uniseg_word_segments"
    
    # ========================================
    # STEP 1: Verify UniSeg data loading
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 1: Verify UniSeg Data Loading")
    print("-" * 60)
    
    ling = LinguisticModels(
        uniseg_root=uniseg_root,
        uniseg_reward=0.2,
        prefix_reward=0.025,
        suffix_reward=0.01,
    )
    
    # Load English UniSeg data
    loaded = ling.load_uniseg_for_lang("en")
    print(f"UniSeg English loaded: {loaded}")
    
    # Get stats from loader
    stats = ling._uniseg_loader.get_stats("en") if ling._uniseg_loader else {}
    n_words = stats.get("words", 0)
    print(f"Words with boundaries: {n_words:,}")
    
    # ========================================
    # STEP 2: Verify affixes extracted from JSONL
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 2: Affixes Extracted from UniSeg JSONL")
    print("-" * 60)
    
    affixes = ling.get_uniseg_affixes("en")
    prefixes = affixes["prefixes"]
    suffixes = affixes["suffixes"]
    
    print(f"Prefixes extracted: {len(prefixes)}")
    print(f"  Sample: {sorted(list(prefixes))[:15]}")
    
    print(f"\nSuffixes extracted: {len(suffixes)}")
    print(f"  Sample: {sorted(list(suffixes))[:15]}")
    
    # Verify key morphemes are in the extracted sets
    key_suffixes = ["ing", "ed", "er", "ness", "tion", "ment", "ly"]
    key_prefixes = ["un", "re", "pre", "dis"]
    
    print(f"\nKey suffix verification:")
    for suf in key_suffixes:
        found = suf in suffixes
        print(f"  '{suf}': {'FOUND' if found else 'MISSING'}")
    
    print(f"\nKey prefix verification:")
    for pre in key_prefixes:
        found = pre in prefixes
        print(f"  '{pre}': {'FOUND' if found else 'MISSING'}")
    
    # ========================================
    # STEP 3: Verify boundary lookups from JSONL
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 3: Boundary Lookups from UniSeg JSONL")
    print("-" * 60)
    
    test_words = ["walking", "teachers", "unhappy", "happiness", "working"]
    for word in test_words:
        bounds = ling.get_uniseg_boundaries(word, "en")
        if bounds:
            parts = []
            prev = 0
            for b in sorted(bounds):
                parts.append(word[prev:b])
                prev = b
            parts.append(word[prev:])
            print(f"  '{word}' -> {bounds} -> {' + '.join(parts)}")
        else:
            print(f"  '{word}' -> NOT IN DATABASE")

    coverage_pct, coverage_hits, coverage_total = compute_coverage(test_words, ling._uniseg_loader, "en")
    print(f"\nBoundary sample coverage: {coverage_hits}/{coverage_total} words ({coverage_pct:.2f}%)")
    
    # ========================================
    # STEP 4: Verify affix reward uses UniSeg
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 4: Affix Rewards Using UniSeg Data")
    print("-" * 60)
    
    test_tokens = ["walking", "unhappy", "teaching", "runner", "kindness"]
    for tok in test_tokens:
        reward = ling._affix_bias(tok, "en")
        print(f"  '{tok}': affix_reward = {reward:.3f}")
    
    # ========================================
    # STEP 5: Train tokenizer and verify
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 5: Train Tokenizer with UniSeg Rewards")
    print("-" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        from datasets import load_dataset
        print("Loading 100 samples from WikiANN (English)...")
        dataset = load_dataset("wikiann", "en", split="train", streaming=True)
        corpus = []
        for ex in dataset:
            if len(corpus) >= 1000:
                break
            toks = ex.get("tokens", [])
            text = " ".join(toks) if isinstance(toks, list) else str(toks)
            if text.strip():
                corpus.append(text)
    except Exception as e:
        print(f"WikiANN load failed: {e}; using fallback sentences.")
        base_sentences = [
            "walking talking running jumping",
            "walked talked jumped quickly",
            "walker talker runner jumper",
            "teachers workers builders painters",
            "teaching working building painting",
            "unhappy unkind unfair unclear",
            "happiness kindness fairness clearness",
            "building bridges requires teamwork",
            "working overtime improves earnings",
            "painters are painting colorful murals",
        ]
        multiplier = -(-100 // len(base_sentences))  # ceil division
        corpus = (base_sentences * multiplier)[:100]

    langs = ["en"] * len(corpus)

    print(f"Corpus: {len(corpus)} sentences")
    cov_pct, cov_hits, cov_total = compute_coverage(corpus, ling._uniseg_loader, "en")
    print(f"UniSeg coverage on corpus vocab: {cov_hits}/{cov_total} words ({cov_pct:.2f}%)")
    
    tok = ScalableTokenizer(
        max_token_len=16,
        min_freq=2,
        vocab_budget=20000,
        top_k_add=40,
        device=device,
        uniseg_root=uniseg_root,
        uniseg_reward=0.3,
        use_morph_encoder=False,
        seed_uniseg_segments=True,
        force_seed_uniseg_tokens=False,
    )
    
    # train() calls _initialize_stats_and_vocab internally, no need to call twice!
    tok.train(corpus, langs, max_iterations=1000, verbose=True)
    
    print(f"Training complete: vocab={len(tok.vocab)}")
    
    # ========================================
    # STEP 6: Check morpheme-aligned tokens
    # ========================================
    print("\n" + "-" * 60)
    print("STEP 6: Morpheme-Aligned Tokenization Results")
    print("-" * 60)
    
    test_sentences = ["walking", "teachers", "unhappy", "walking and talking"]
    
    for sent in test_sentences:
        tokens = tok.tokenize(sent, "en")
        is_char = all(len(t) <= 1 for t in tokens if t.strip())
        status = "CHAR-LEVEL" if is_char else "MULTI-CHAR"
        print(f"\n  '{sent}'")
        print(f"  -> {tokens}")
        print(f"  -> {status}")
    
    print("\nFirst 5 training paragraphs tokenization:")
    for i, para in enumerate(corpus[:20]):
        tokens = tok.tokenize(para, "en")
        print(f"\n[{i+1}] {para}")
        print(f"Tokens ({len(tokens)}): {tokens}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY: All Morpheme Rewards Based on UniSeg JSONL")
    print("=" * 60)
    
    # Get final stats
    final_stats = ling._uniseg_loader.get_stats("en") if ling._uniseg_loader else {}
    print(f"""
    1. Word boundaries: Loaded from {uniseg_root}/eng/*.jsonl
       - {final_stats.get('words', 0):,} words with morpheme boundaries
    
    2. Affixes: Extracted from JSONL segment types
       - {len(prefixes)} prefixes (e.g., {list(prefixes)[:5]})
       - {len(suffixes)} suffixes (e.g., {list(suffixes)[:5]})
    
    3. Boundary alignment: BOTH start AND end must match
       - Rewards complete morpheme segments only
    
    4. Affix reward: Uses UniSeg-derived affixes from JSONL
       - NOT hardcoded lists anymore
    
    Pipeline: JSONL file -> UniSegLoader -> LinguisticModels -> Tokenizer
    """)


if __name__ == "__main__":
    main()
