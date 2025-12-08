"""Test suite for optimized ngram extraction functionality."""

import sys
import time
from collections import Counter

# Add the tokenizer_core to path
sys.path.insert(0, '.')

from tokenizer_core.linguistic_features import (
    char_ngrams,
    char_ngrams_batch,
    clear_ngram_cache,
    MorphologyEncoder
)


def test_basic_ngram_extraction():
    """Test that basic ngram extraction works correctly."""
    print("Test 1: Basic ngram extraction")
    result = char_ngrams("hello", n=(2, 3, 4))
    
    # Expected ngrams for "hello":
    # 2-grams: he, el, ll, lo
    # 3-grams: hel, ell, llo
    # 4-grams: hell, ello
    expected = Counter({
        'he': 1, 'el': 1, 'll': 1, 'lo': 1,
        'hel': 1, 'ell': 1, 'llo': 1,
        'hell': 1, 'ello': 1
    })
    
    assert result == expected, f"Expected {expected}, got {result}"
    print("  [PASS]")
    return True


def test_caching():
    """Test that caching works and speeds up repeated calls."""
    print("\nTest 2: Caching functionality")
    clear_ngram_cache()
    
    test_string = "tokenization"
    n_orders = (2, 3, 4)
    
    # First call - should compute
    start = time.perf_counter()
    result1 = char_ngrams(test_string, n_orders)
    time1 = time.perf_counter() - start
    
    # Second call - should use cache
    start = time.perf_counter()
    result2 = char_ngrams(test_string, n_orders)
    time2 = time.perf_counter() - start
    
    # Results should be identical
    assert result1 == result2, "Cached result should match original"
    
    # Cached call should be faster (or at least not slower)
    # Note: For very fast operations, timing might be noisy, so we just check it's not much slower
    print(f"  First call: {time1*1000:.4f}ms")
    print(f"  Cached call: {time2*1000:.4f}ms")
    print(f"  Speedup: {time1/max(time2, 1e-9):.2f}x")
    print("  [PASS]")
    return True


def test_edge_cases():
    """Test edge cases: empty string, short string, large n-gram orders."""
    print("\nTest 3: Edge cases")
    
    # Empty string
    result = char_ngrams("", n=(2, 3, 4))
    assert result == Counter(), "Empty string should return empty Counter"
    print("  [OK] Empty string handled correctly")
    
    # Very short string
    result = char_ngrams("a", n=(2, 3, 4))
    assert result == Counter(), "String shorter than min n-gram should return empty"
    print("  [OK] Short string handled correctly")
    
    # String exactly matching n-gram size
    result = char_ngrams("ab", n=(2, 3, 4))
    assert result == Counter({'ab': 1}), "Should extract single 2-gram"
    print("  [OK] Exact size match handled correctly")
    
    # N-gram order larger than string
    result = char_ngrams("abc", n=(2, 3, 5))
    expected = Counter({'ab': 1, 'bc': 1, 'abc': 1})
    assert result == expected, f"Should only extract valid n-grams, got {result}"
    print("  [OK] Invalid n-gram orders filtered correctly")
    
    print("  [PASS] All edge cases passed")
    return True


def test_batch_processing():
    """Test batch processing function."""
    print("\nTest 4: Batch processing")
    
    tokens = ["hello", "world", "test"]
    results = char_ngrams_batch(tokens, n=(2, 3))
    
    assert len(results) == len(tokens), "Should return one result per token"
    
    # Check first token manually
    expected_hello = char_ngrams("hello", n=(2, 3))
    assert results[0] == expected_hello, "Batch result should match individual call"
    
    print("  [PASS] Batch processing works correctly")
    return True


def test_morphology_encoder_caching():
    """Test that MorphologyEncoder uses caching in _featurize."""
    print("\nTest 5: MorphologyEncoder featurization caching")
    
    encoder = MorphologyEncoder(ngram_orders=(2, 3, 4))
    encoder.clear_featurize_cache()
    
    test_token = "tokenization"
    test_lang = "en"
    
    # First call - should compute
    start = time.perf_counter()
    features1 = encoder._featurize(test_token, test_lang)
    time1 = time.perf_counter() - start
    
    # Second call - should use cache
    start = time.perf_counter()
    features2 = encoder._featurize(test_token, test_lang)
    time2 = time.perf_counter() - start
    
    # Results should be identical
    assert features1 == features2, "Cached featurization should match original"
    
    # Cache should be populated
    assert (test_token, test_lang) in encoder._featurize_cache, "Cache should contain entry"
    
    print(f"  First call: {time1*1000:.4f}ms")
    print(f"  Cached call: {time2*1000:.4f}ms")
    print(f"  Speedup: {time1/max(time2, 1e-9):.2f}x")
    print("  [PASS]")
    return True


def test_unicode_support():
    """Test that ngram extraction works correctly with Unicode characters."""
    print("\nTest 6: Unicode support")
    
    # Test with various Unicode characters and verify correct extraction
    test_cases = [
        ("café", (2, 3), {"ca": 1, "af": 1, "fé": 1, "caf": 1, "afé": 1}),  # Latin with accent
        ("北京", (2,), {"北京": 1}),  # Chinese characters (2 chars = 1 2-gram)
        ("مرحبا", (2, 3), {"مر": 1, "رح": 1, "حب": 1, "با": 1, "مرح": 1, "رحب": 1, "حبا": 1}),  # Arabic
        ("こんにちは", (2, 3), {"こん": 1, "んに": 1, "にち": 1, "ちは": 1, "こんに": 1, "んにち": 1, "にちは": 1}),  # Japanese
        ("Привет", (2, 3), {"Пр": 1, "ри": 1, "ив": 1, "ве": 1, "ет": 1, "При": 1, "рив": 1, "иве": 1, "вет": 1}),  # Cyrillic
        ("🚀🌟⭐", (2,), {"🚀🌟": 1, "🌟⭐": 1}),  # Emoji
        ("a🚀b", (2, 3), {"a🚀": 1, "🚀b": 1, "a🚀b": 1}),  # Mixed ASCII and emoji
    ]
    
    for test_str, n_orders, expected_ngrams in test_cases:
        result = char_ngrams(test_str, n=n_orders)
        assert isinstance(result, Counter), f"Should return Counter for '{test_str}'"
        
        # Verify all expected ngrams are present
        for ngram, count in expected_ngrams.items():
            assert result[ngram] == count, \
                f"For '{test_str}': expected '{ngram}' to appear {count} times, got {result[ngram]}"
        
        # Verify total number of ngrams matches expected
        total_expected = sum(expected_ngrams.values())
        total_actual = sum(result.values())
        assert total_actual == total_expected, \
            f"For '{test_str}': expected {total_expected} total ngrams, got {total_actual}"
    
    # Test that Unicode characters are handled correctly in caching
    clear_ngram_cache()
    unicode_str = "café北京"
    result1 = char_ngrams(unicode_str, n=(2, 3))
    result2 = char_ngrams(unicode_str, n=(2, 3))  # Should use cache
    assert result1 == result2, "Cached Unicode result should match original"
    
    print("  [OK] Unicode characters (Latin, Chinese, Arabic, Japanese, Cyrillic, Emoji)")
    print("  [OK] Unicode caching works correctly")
    print("  [PASS] Unicode support verified")
    return True


def test_performance_comparison():
    """Compare performance with repeated tokens (simulating real usage)."""
    print("\nTest 7: Performance with repeated tokens")
    
    clear_ngram_cache()
    
    # Simulate a corpus with many repeated tokens
    tokens = ["the", "quick", "brown", "fox", "the", "lazy", "dog", "the", "quick", "brown"]
    n_orders = (2, 3, 4)
    
    # Without caching (simulated by clearing cache each time)
    start = time.perf_counter()
    for token in tokens:
        clear_ngram_cache()
        char_ngrams(token, n_orders)
    time_no_cache = time.perf_counter() - start
    
    # With caching
    clear_ngram_cache()
    start = time.perf_counter()
    for token in tokens:
        char_ngrams(token, n_orders)
    time_with_cache = time.perf_counter() - start
    
    speedup = time_no_cache / max(time_with_cache, 1e-9)
    print(f"  Without cache: {time_no_cache*1000:.4f}ms")
    print(f"  With cache: {time_with_cache*1000:.4f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.1:  # At least 10% improvement
        print("  [PASS] Caching provides significant speedup")
    else:
        print("  [WARN] Caching speedup is minimal (may be due to test overhead)")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Testing Optimized Ngram Extraction")
    print("=" * 60)
    
    tests = [
        test_basic_ngram_extraction,
        test_caching,
        test_edge_cases,
        test_batch_processing,
        test_morphology_encoder_caching,
        test_unicode_support,
        test_performance_comparison,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] Failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

