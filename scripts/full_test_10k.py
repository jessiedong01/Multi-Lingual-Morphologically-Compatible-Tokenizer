"""
Full test: Debug + 10k WikiANN corpus training

This script:
1. Verifies UniSeg loading (fast)
2. Runs a small smoke test (fast)
3. Trains on 10k WikiANN English corpus
4. Shows results
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=" * 70)
print("FULL TEST: UniSeg + 10K WikiANN")
print("=" * 70)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Detected torch device: {device}")

# ============================================
# PHASE 1: Quick UniSeg verification
# ============================================
print("\n" + "=" * 70)
print("PHASE 1: UniSeg Loader Verification")
print("=" * 70)

t0 = time.time()

from tokenizer_core.uniseg_loader import UniSegLoader


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

uniseg_root = ROOT / "data" / "uniseg_word_segments"
print(f"UniSeg root: {uniseg_root}")
print(f"MorphoLex exists: {(uniseg_root / 'eng' / 'MorphoLex.jsonl').exists()}")

loader = UniSegLoader(uniseg_root)
loaded = loader.load_language("en")
stats = loader.get_stats("en")

print(f"Loaded: {loaded}")
print(f"Stats: {stats}")
print(f"Time: {time.time() - t0:.2f}s")

# Test lookups
test_words = ["walking", "unhappy", "teachers"]
print("\nBoundary lookups:")
for word in test_words:
    bounds = loader.get_boundaries(word, "en")
    morphs = loader.word_to_morphemes(word, "en")
    print(f"  '{word}' -> {bounds} -> {morphs}")

# Test affixes
print(f"\nPrefixes sample: {list(loader.get_prefixes('en'))[:10]}")
print(f"Suffixes sample: {list(loader.get_suffixes('en'))[:10]}")

print("\n[PHASE 1 COMPLETE]")

# ============================================
# PHASE 2: Quick smoke test
# ============================================
print("\n" + "=" * 70)
print("PHASE 2: Quick Smoke Test (small corpus)")
print("=" * 70)

t0 = time.time()

from tokenizer_core.tokenizer import ScalableTokenizer

small_corpus = [
    "walking talking running",
    "teachers workers builders",
    "unhappy unkind unfair",
] * 10  # 30 sentences

small_langs = ["en"] * len(small_corpus)

print(f"Small corpus: {len(small_corpus)} sentences")
cov_pct, cov_hits, cov_total = compute_coverage(small_corpus, loader, "en")
print(f"UniSeg coverage (small corpus vocab): {cov_hits}/{cov_total} words ({cov_pct:.2f}%)")

tok_small = ScalableTokenizer(
    max_token_len=10,
    min_freq=1,
    vocab_budget=50,
    top_k_add=4,
    device=device,
    uniseg_root=uniseg_root,
    uniseg_reward=0.2,
    use_morph_encoder=False,
    seed_uniseg_segments=True,
    force_seed_uniseg_tokens=True,
)

tok_small.train(small_corpus, small_langs, max_iterations=3, verbose=True)

print(f"Vocab size: {len(tok_small.vocab)}")
print(f"Time: {time.time() - t0:.2f}s")

# Test tokenization
print("\nTokenization:")
for text in ["walking", "teachers", "unhappy"]:
    tokens = tok_small.tokenize(text, "en")
    print(f"  '{text}' -> {tokens}")

print("\n[PHASE 2 COMPLETE]")

# ============================================
# PHASE 3: Load 10K WikiANN
# ============================================
print("\n" + "=" * 70)
print("PHASE 3: Load 10K WikiANN English Corpus")
print("=" * 70)

t0 = time.time()

def load_wikiann(n_samples=10000):
    """Load WikiANN English corpus."""
    try:
        from datasets import load_dataset
        print("Loading from HuggingFace datasets...")
        dataset = load_dataset("wikiann", "en", split='train', streaming=True)
        
        texts = []
        for ex in dataset:
            if len(texts) >= n_samples:
                break
            toks = ex.get('tokens', [])
            txt = " ".join(toks) if isinstance(toks, list) else str(toks)
            if txt.strip():
                texts.append(txt)
            
            if len(texts) % 1000 == 0:
                print(f"  Loaded {len(texts):,} samples...")
        
        return texts, ["en"] * len(texts)
    
    except Exception as e:
        print(f"HuggingFace failed: {e}")
        print("Using fallback corpus...")
        
        # Fallback with more morphological variety
        base = [
            "The workers are walking through the buildings",
            "Teachers teaching students are hardworking",
            "Running and walking are good exercises",
            "The builders are building new structures",
            "Unhappiness and impossibility are abstract",
            "Speaking clearly helps communication",
            "Scientists are researching discoveries",
            "Learning requires patience and dedication",
            "Writers are writing interesting stories",
            "Swimming and cycling are popular",
        ]
        
        import random
        random.seed(42)
        texts = [random.choice(base) for _ in range(n_samples)]
        return texts, ["en"] * len(texts)

corpus, langs = load_wikiann(10000)
print(f"Loaded {len(corpus):,} samples in {time.time() - t0:.2f}s")

# Show samples
print("\nSample paragraphs:")
for i in range(5):
    print(f"  [{i+1}] {corpus[i][:60]}...")

cov_pct_big, cov_hits_big, cov_total_big = compute_coverage(corpus, loader, "en")
print(f"\nUniSeg coverage (10K corpus vocab): {cov_hits_big}/{cov_total_big} words ({cov_pct_big:.2f}%)")

print("\n[PHASE 3 COMPLETE]")

# ============================================
# PHASE 4: Train on 10K corpus
# ============================================
print("\n" + "=" * 70)
print("PHASE 4: Train Tokenizer on 10K Corpus")
print("=" * 70)

t0 = time.time()

tok = ScalableTokenizer(
    max_token_len=12,
    min_freq=3,
    vocab_budget=4000,
    top_k_add=16,
    device=device,
    uniseg_root=uniseg_root,
    uniseg_reward=0.15,
    use_morph_encoder=False,
    seed_uniseg_segments=True,
    force_seed_uniseg_tokens=True,
)

print("Training (this may take a few minutes)...")
print("Config: vocab_budget=4000, uniseg_reward=0.15, use_morph_encoder=False")

tok.train(corpus, langs, max_iterations=20, verbose=True)

train_time = time.time() - t0
print(f"\nTraining complete in {train_time:.1f}s")
print(f"Final vocab size: {len(tok.vocab)}")

print("\n[PHASE 4 COMPLETE]")

# ============================================
# PHASE 5: Results
# ============================================
print("\n" + "=" * 70)
print("PHASE 5: Results")
print("=" * 70)

# Vocabulary analysis
multi_char = sorted([t for t in tok.vocab if len(t) > 1 and t.strip()], key=len, reverse=True)
print(f"\nMulti-character tokens: {len(multi_char)}")
print(f"Longest: {multi_char[:15]}")

# Morpheme patterns
print("\nMorpheme patterns in vocabulary:")
patterns = ["ing", "ed", "er", "ness", "tion", "ment", "un", "ly"]
for pat in patterns:
    matches = [t for t in tok.vocab if pat in t.lower() and len(t) > len(pat)]
    if matches:
        print(f"  *{pat}*: {matches[:8]}")

# Tokenize first 5 paragraphs
print("\n" + "-" * 70)
print("TOKENIZATION OF FIRST 5 PARAGRAPHS:")
print("-" * 70)

for i, text in enumerate(corpus[:5]):
    tokens = tok.tokenize(text, "en")
    print(f"\n[{i+1}] Input: {text}")
    print(f"    Tokens ({len(tokens)}): {tokens}")

# Summary statistics
print("\n" + "-" * 70)
print("STATISTICS (on first 100 samples):")
print("-" * 70)

total_tokens = 0
total_chars = 0
for text in corpus[:100]:
    tokens = tok.tokenize(text, "en")
    total_tokens += len(tokens)
    total_chars += len(text)

print(f"Avg tokens per sample: {total_tokens / 100:.1f}")
print(f"Avg chars per token: {total_chars / total_tokens:.2f}")
print(f"Compression ratio: {total_chars / total_tokens:.2f}x")

# Final summary
print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {time.time() - t0:.1f}s")
print(f"""
Summary:
- UniSeg MorphoLex loaded: {stats.get('words', 0):,} words
- Corpus: {len(corpus):,} WikiANN samples  
- Final vocab: {len(tok.vocab)} tokens
- Multi-char tokens: {len(multi_char)}
- Training time: {train_time:.1f}s

The tokenizer is ready for use!
""")

