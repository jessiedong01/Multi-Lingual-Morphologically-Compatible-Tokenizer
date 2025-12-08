"""
Test UniSeg boundary alignment across multiple languages.

This script:
1. Loads UniSeg data for available languages
2. Tests the NEW logic: BOTH start AND end must align for reward
3. Shows examples from each language's actual data
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer_core.linguistic_features import LinguisticModels


def find_uniseg_root():
    """Find UniSeg data directory."""
    return ROOT / "data" / "uniseg_word_segments"


def load_sample_words(lang_dir: Path, n_samples=5):
    """Load sample words with morpheme boundaries from a language's JSONL."""
    samples = []
    
    for jsonl_file in lang_dir.glob("*.jsonl"):
        try:
            with jsonl_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if len(samples) >= n_samples:
                        break
                    try:
                        record = json.loads(line)
                        word = record.get("word", "")
                        boundaries = record.get("boundaries", [])
                        segments = record.get("segments", [])
                        
                        # Only include words with at least one internal boundary
                        if boundaries and word:
                            samples.append({
                                "word": word,
                                "boundaries": set(int(b) for b in boundaries),
                                "segments": segments,
                                "source": jsonl_file.name,
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            continue
    
    return samples


def safe_print(s):
    """Print with ASCII fallback for Windows console."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))


def test_single_word(ling: LinguisticModels, word: str, boundaries: set, lang: str):
    """Test all possible spans of a word and show which get rewarded."""
    safe_print(f"\n  Word: '{word}' (lang={lang})")
    safe_print(f"  Internal morpheme boundaries: {sorted(boundaries)}")
    
    # Show segmentation
    parts = []
    prev = 0
    for b in sorted(boundaries):
        parts.append(word[prev:b])
        prev = b
    parts.append(word[prev:])
    safe_print(f"  Morphemes: {' + '.join(parts)}")
    
    # Valid boundary positions (word edges + internal splits)
    valid_positions = {0, len(word)} | boundaries
    safe_print(f"  Valid positions: {sorted(valid_positions)}")
    
    # Test some key spans
    test_spans = []
    
    # Each morpheme segment
    prev = 0
    for b in sorted(boundaries):
        test_spans.append((prev, b, word[prev:b], "morpheme"))
        prev = b
    test_spans.append((prev, len(word), word[prev:], "morpheme"))
    
    # Full word
    test_spans.append((0, len(word), word, "full word"))
    
    # Some wrong spans
    if len(word) > 3:
        test_spans.append((1, len(word)-1, word[1:-1], "wrong span"))
        test_spans.append((0, 2, word[:2], "wrong span"))
    
    safe_print(f"\n  Testing spans:")
    for start, end, tok, desc in test_spans:
        start_valid = start in valid_positions
        end_valid = end in valid_positions
        reward = ling.boundary_alignment_reward(word, start, end, lang)
        
        status = "REWARD" if reward > 0 else "no reward"
        safe_print(f"    ({start:2d},{end:2d}) '{tok}' [{desc}]: start={start_valid}, end={end_valid} -> {status}")


def main():
    print("=" * 70)
    print("UniSeg Boundary Alignment Test - Multiple Languages")
    print("NEW LOGIC: BOTH start AND end must align for reward")
    print("=" * 70)
    
    uniseg_root = find_uniseg_root()
    if not uniseg_root.exists():
        print(f"ERROR: UniSeg data not found at {uniseg_root}")
        return
    
    # Setup LinguisticModels
    ling = LinguisticModels(
        uniseg_root=uniseg_root,
        uniseg_reward=0.15,
    )
    
    # Language mapping (3-letter to 2-letter codes)
    lang_map = {
        "eng": "en",
        "deu": "de", 
        "fra": "fr",
        "spa": "es",
        "ita": "it",
        "por": "pt",
        "rus": "ru",
        "pol": "pl",
        "ces": "cs",
        "hun": "hu",
        "fin": "fi",
        "swe": "sv",
        "hin": "hi",
        "ben": "bn",
    }
    
    # Find available language directories - prioritize Latin-script languages
    latin_langs = ["eng", "deu", "fra", "spa", "ita", "por", "pol", "ces", "hun", "fin", "swe", "cat"]
    lang_dirs = [uniseg_root / lang for lang in latin_langs if (uniseg_root / lang).exists()]
    
    print(f"\nFound {len(lang_dirs)} Latin-script language directories")
    
    results = {}
    
    for lang_dir in lang_dirs[:6]:  # Test first 6 languages
        lang_3 = lang_dir.name
        lang_2 = lang_map.get(lang_3, lang_3[:2])
        
        print(f"\n{'=' * 70}")
        print(f"LANGUAGE: {lang_3} ({lang_2})")
        print("=" * 70)
        
        # Load sample words
        samples = load_sample_words(lang_dir, n_samples=3)
        
        if not samples:
            print(f"  No samples with morpheme boundaries found")
            continue
        
        print(f"  Found {len(samples)} sample words from {samples[0]['source']}")
        
        # Load UniSeg for this language
        ling.load_uniseg_for_lang(lang_2)
        
        # Test each sample word
        for sample in samples:
            test_single_word(ling, sample["word"].lower(), sample["boundaries"], lang_2)
        
        results[lang_3] = len(samples)
    
    # Test paragraph-level alignment
    print(f"\n{'=' * 70}")
    print("PARAGRAPH-LEVEL TEST")
    print("=" * 70)
    
    test_paragraphs = [
        ("The teachers are walking unhappily.", "en"),
        ("Die Arbeiter sind fleissig.", "de"),
    ]
    
    for text, lang in test_paragraphs:
        safe_print(f"\n  Text: '{text}' (lang={lang})")
        
        gold = ling.precompute_paragraph_boundaries(text, lang)
        safe_print(f"  All valid boundaries: {sorted(gold)}")
        
        # Show what each boundary means
        for pos in sorted(gold):
            before = text[max(0, pos-3):pos]
            after = text[pos:min(len(text), pos+3)]
            safe_print(f"    Position {pos:2d}: ...'{before}|{after}'...")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: NEW LOGIC VERIFICATION")
    print("=" * 70)
    print("""
    The new logic requires BOTH start AND end to align:
    
    For word "walking" with boundary at {4}:
    - Valid positions: {0, 4, 7} (word-start, morpheme-split, word-end)
    
    Span (0, 4) "walk": start=0 valid, end=4 valid -> REWARD
    Span (4, 7) "ing":  start=4 valid, end=7 valid -> REWARD  
    Span (0, 7) "walking": start=0 valid, end=7 valid -> REWARD (full word)
    Span (0, 3) "wal":  start=0 valid, end=3 NOT valid -> NO REWARD
    Span (2, 7) "lking": start=2 NOT valid, end=7 valid -> NO REWARD
    
    This ensures only COMPLETE morpheme segments get rewarded.
    """)
    
    print(f"Languages tested: {list(results.keys())}")


if __name__ == "__main__":
    main()

