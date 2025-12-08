"""
Trace through exactly how "walking" would be tokenized.
Shows all candidates, costs, and UniSeg rewards.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.linguistic_features import LinguisticModels
from tokenizer_core.constants import AFFIXES

def main():
    # Setup
    uniseg_root = ROOT / "data" / "uniseg_word_segments"
    
    ling = LinguisticModels(
        uniseg_root=uniseg_root,
        uniseg_reward=0.15,
        prefix_reward=0.025,
        suffix_reward=0.01,
    )
    
    # Load UniSeg for English
    ling.load_uniseg_for_lang("en")
    
    print("=" * 70)
    print("TRACING: How 'walking' gets tokenized")
    print("=" * 70)
    
    # First, check what UniSeg says about "walking"
    word = "walking"
    boundaries = ling.get_uniseg_boundaries(word, "en")
    print(f"\n1. UniSeg lookup for '{word}':")
    print(f"   Internal morpheme boundaries: {boundaries}")
    if boundaries:
        for b in sorted(boundaries):
            print(f"   -> Split at position {b}: '{word[:b]}' + '{word[b:]}'")
    
    # Now simulate a paragraph
    text = "The walking dog"
    print(f"\n2. Paragraph context: '{text}'")
    print(f"   Length T = {len(text)}")
    
    # Get gold boundaries at paragraph level
    gold_bounds = ling.precompute_paragraph_boundaries(text, "en")
    print(f"\n3. Gold boundaries in paragraph: {sorted(gold_bounds)}")
    for pos in sorted(gold_bounds):
        print(f"   Position {pos}: '{text[:pos]}|{text[pos:]}'")
    
    # Now enumerate all possible spans for "walking" (positions 4-11)
    print(f"\n4. All possible spans covering 'walking' (at positions 4-11):")
    print(f"   Text:  '{text}'")
    print(f"   Index:  0123456789...")
    
    word_start = 4  # "walking" starts at position 4
    word_end = 11   # "walking" ends at position 11
    T = len(text)
    
    # Enumerate all spans that could be part of tokenizing "walking"
    spans = []
    for start in range(word_start, word_end):
        for end in range(start + 1, min(word_end + 1, T + 1)):
            tok = text[start:end]
            spans.append((start, end, tok))
    
    print(f"\n5. Candidate spans and their UniSeg rewards:")
    print(f"   {'Span':<12} {'Token':<10} {'Start':<6} {'End':<6} {'Start Reward':<14} {'End Reward':<12} {'Total'}")
    print(f"   {'-'*12} {'-'*10} {'-'*6} {'-'*6} {'-'*14} {'-'*12} {'-'*8}")
    
    uniseg_reward = 0.15
    
    for start, end, tok in spans:
        # Check start alignment
        start_reward = 0.0
        start_note = ""
        if start > 0 and start in gold_bounds:
            start_reward = uniseg_reward
            start_note = f"+{uniseg_reward:.2f} (aligned!)"
        else:
            if start == 0:
                start_note = "0 (start of text)"
            elif start in gold_bounds:
                start_note = "0 (start=0 excluded)"
            else:
                start_note = f"0 ({start} not in {sorted(gold_bounds)})"
        
        # Check end alignment
        end_reward = 0.0
        end_note = ""
        if end < T and end in gold_bounds:
            end_reward = uniseg_reward
            end_note = f"+{uniseg_reward:.2f} (aligned!)"
        else:
            if end >= T:
                end_note = f"0 (end={end} >= T={T})"
            else:
                end_note = f"0 ({end} not in {sorted(gold_bounds)})"
        
        total = start_reward + end_reward
        total_str = f"+{total:.2f}" if total > 0 else "0"
        
        print(f"   ({start:2d},{end:2d})      '{tok}'     {start:<6} {end:<6} {start_note:<14} {end_note:<12} {total_str}")
    
    # Now show which segmentations would benefit
    print(f"\n6. Example segmentations of '{text}' and their UniSeg bonuses:")
    
    segmentations = [
        # (tokens, description)
        (["T", "h", "e", " ", "w", "a", "l", "k", "i", "n", "g", " ", "d", "o", "g"], "char-by-char"),
        (["The", " ", "walking", " ", "dog"], "word-level"),
        (["The", " ", "walk", "ing", " ", "dog"], "morpheme-aligned!"),
        (["The", " ", "wal", "king", " ", "dog"], "wrong split"),
        (["The", " ", "walki", "ng", " ", "dog"], "wrong split"),
    ]
    
    for tokens, desc in segmentations:
        # Calculate boundary positions
        pos = 0
        boundaries_made = []
        for tok in tokens[:-1]:
            pos += len(tok)
            boundaries_made.append(pos)
        
        # Check alignment
        aligned = [b for b in boundaries_made if b in gold_bounds]
        total_reward = len(aligned) * uniseg_reward
        
        print(f"\n   {desc}:")
        print(f"   Tokens: {tokens}")
        print(f"   Boundaries at: {boundaries_made}")
        print(f"   Aligned with gold: {aligned}")
        print(f"   UniSeg reward: +{total_reward:.2f}")
    
    # Show the affix rewards (separate system!)
    print(f"\n7. Affix rewards (SEPARATE from UniSeg):")
    print(f"   Affix dictionary for English:")
    print(f"     Prefixes: {AFFIXES.get('en', {}).get('pre', [])[:10]}...")
    print(f"     Suffixes: {AFFIXES.get('en', {}).get('suf', [])[:10]}...")
    
    test_tokens = ["walking", "walk", "ing", "wal", "king"]
    print(f"\n   Affix rewards for tokens:")
    for tok in test_tokens:
        bias = ling._affix_bias(tok, "en")
        suffix_match = any(tok.endswith(s) for s in AFFIXES.get('en', {}).get('suf', []))
        prefix_match = any(tok.startswith(p) for p in AFFIXES.get('en', {}).get('pre', []))
        print(f"     '{tok}': {bias:.3f} (suffix={suffix_match}, prefix={prefix_match})")
    
    print(f"\n" + "=" * 70)
    print("SUMMARY: How the algorithm chooses")
    print("=" * 70)
    print("""
    For "The walking dog":
    
    Total cost = SUM of cost(token)
    
    cost(token) = a*NLL + b*PMI_pen + t*length + lambda - affix_reward - uniseg_reward
    
    The segmentation "The walk ing dog" gets:
      - UniSeg reward at position 8 (walk|ing boundary aligned!) = +0.15
      - Affix reward for "ing" (ends with -ing suffix) = +0.01
    
    The segmentation "The walking dog" gets:
      - Affix reward for "walking" (ends with -ing suffix) = +0.01
      - NO UniSeg reward (no internal boundary at position 8)
    
    The segmentation "The wal king dog" gets:
      - NO UniSeg reward (boundary at position 7, not 8)
      - NO meaningful affix reward
    
    Winner depends on the balance of:
      - NLL: Is "walk" frequent enough? Is "ing" frequent enough?
      - UniSeg reward (0.15): Bonus for morpheme alignment
      - Affix reward (0.01): Small bonus for -ing suffix
      
    If "walk" and "ing" are both in vocabulary with decent NLL,
    AND uniseg_reward > NLL difference, then "walk"+"ing" wins!
    """)


if __name__ == "__main__":
    main()

