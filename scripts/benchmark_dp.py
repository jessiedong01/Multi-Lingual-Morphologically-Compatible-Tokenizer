import math
import sys
import time
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tokenizer_core.utils as core_utils
from tokenizer_core.tokenizer import ScalableTokenizer


def build_demo_tokenizer():
    paragraphs = [
        "Google researchers introduced a scalable tokenizer for multilingual corpora.",
        "Die Universität München erforscht neue Methoden der Sprachverarbeitung.",
        "Türkçe metinler için eklemeli morfoloji çok önemlidir.",
    ]
    langs = ["en", "de", "tr"]
    tok = ScalableTokenizer(
        max_token_len=10,
        min_freq=1,
        top_k_add=2,
        vocab_budget=64,
        device="cpu",
    )
    tok._initialize_stats_and_vocab(paragraphs, langs)
    return tok, paragraphs


def baseline_dp_decode(tokenizer, paragraph_idx: int):
    text = tokenizer._paras[paragraph_idx].text
    info = tokenizer._paras[paragraph_idx]
    T = len(text)
    idx = tokenizer._class2idx
    K = len(tokenizer._classes)

    INF = float("inf")
    dp = torch.full((T + 1, K), INF, dtype=torch.float32, device=tokenizer.device)
    back = [[None] * K for _ in range(T + 1)]
    dp[0, idx["<BOS>"]] = 0.0

    for t in range(1, T + 1):
        max_len = min(tokenizer.max_token_len, t)
        for L in range(1, max_len + 1):
            start = t - L
            if not info.is_legal_span(start, t):
                continue
            tok = text[start:t]
            is_exact_protected = any(start == s and t == e for (s, e) in info.protected)
            if (tok not in tokenizer.tok2id) and (not is_exact_protected):
                is_cjk_override = info.lang == "ja" and core_utils.all_cjk(tok) and len(tok) <= 4
                is_tamil_override = False
                if info.lang == "ta" and core_utils.all_tamil(tok):
                    if len(core_utils.grapheme_clusters(tok)) <= 3:
                        is_tamil_override = True
                if not (is_cjk_override or is_tamil_override):
                    continue
            if is_exact_protected and (tok not in tokenizer.tok2id):
                base = 0.0
            else:
                base = tokenizer._get_token_cost(tok)
            tc = tokenizer._ling.token_class(tok)
            j = idx.get(tc, idx["other"])
            for si in range(K):
                prev_cost = float(dp[start, si].item())
                if not math.isfinite(prev_cost):
                    continue
                add = tokenizer._ling.additive_cost(tok, tokenizer._classes[si], paragraph_idx)
                v = prev_cost + base + add
                if v < float(dp[t, j].item()):
                    dp[t, j] = v
                    back[t][j] = (start, si, L)

    end_cls = int(torch.argmin(dp[T]).item())
    toks = []
    t = T
    c = end_cls
    while t > 0:
        state = back[t][c]
        if state is None:
            break
        start, prev_state, L = state
        toks.append(text[start:t])
        t = start
        c = prev_state
    toks.reverse()
    return toks


def run_benchmark(iterations=20):
    tokenizer, paragraphs = build_demo_tokenizer()
    langs = ["en", "de", "tr"]
    
    # Warm-up
    for pi in range(len(paragraphs)):
        tokenizer._dp_decode(pi)

    new_times = []
    baseline_times = []
    for _ in range(iterations):
        for pi in range(len(paragraphs)):
            t0 = time.perf_counter()
            tokenizer._dp_decode(pi)
            new_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            baseline_dp_decode(tokenizer, pi)
            baseline_times.append(time.perf_counter() - t1)

    avg_new = mean(new_times)
    avg_base = mean(baseline_times)
    improvement = avg_base / avg_new if avg_new > 0 else float("inf")

    print(f"Runs per method: {len(new_times)}")
    print(f"Average new DP time : {avg_new * 1e3:.4f} ms")
    print(f"Average baseline time: {avg_base * 1e3:.4f} ms")
    print(f"Speedup ~ {improvement:.2f}x")
    
    # Show tokenization results
    print("\n" + "=" * 70)
    print("TOKENIZATION RESULTS")
    print("=" * 70)
    for pi, (para, lang) in enumerate(zip(paragraphs, langs)):
        tokens = tokenizer._dp_decode(pi)
        print(f"\n[{lang}] {para}")
        print(f"  -> {len(tokens)} tokens: {tokens}")
    
    # Show vocabulary info
    print("\n" + "=" * 70)
    print("VOCABULARY INFO")
    print("=" * 70)
    print(f"Total vocab size: {len(tokenizer.vocab)}")
    multichar = [t for t in tokenizer.vocab if len(t) > 1]
    print(f"Multi-char tokens ({len(multichar)}): {multichar[:30]}{'...' if len(multichar) > 30 else ''}")


if __name__ == "__main__":
    run_benchmark()

