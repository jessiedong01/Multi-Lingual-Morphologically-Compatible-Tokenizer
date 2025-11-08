#!pip install datasets regex -q

import re
import math
import time
import unicodedata as ud
from collections import Counter, defaultdict
import numpy as np
from datasets import load_dataset
import regex as reg

# --- Knobs (tuned for Colab) ---
PER_LANG      = 200
MAX_ITERS     = 80
TOP_K_ADD     = 20
MIN_FREQ      = 5
MAX_BASE_COST = 18.0

# --- Constants & Regex ---
EPS   = 1e-12
URL_RE   = re.compile(r'(?:https?://|www\.)\S+', re.UNICODE)
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', re.UNICODE)
NUM_RE   = re.compile(r"""(?:[+\-]? (?:(?:\d{1,3}(?:[ ,.\u00A0]\d{3})+|\d+)(?:[.,]\d+)?|\d{4}-\d{2}-\d{2}))""", re.VERBOSE | re.UNICODE)
TRAILING_PUNCT = ".,;:!?)]}›»'\""
REDIRECT_TOKEN_RE = re.compile(r"(?i)(redirect|weiterleitung|yönlendirme|перенаправ|تحويل|リダイレクト)")
WIKI_NOISE_RE = re.compile(r"^(\*{2,}|'{2,}|`{2,}|={2,}|[:;,#–\-•··]+)$")

# --- Helper Functions ---
def _strip_trailing_punct_span(text, s, e):
    while e > s and text[e-1] in TRAILING_PUNCT: e -= 1
    return s, e
def looks_like_redirect(text: str) -> bool: return bool(REDIRECT_TOKEN_RE.search(text))
def _is_mixed_script(tok: str) -> bool:
    S=set()
    for ch in tok:
        if ch.isalpha():
            name = ud.name(ch, "")
            if ("HIRAGANA" in name) or ("KATAKANA" in name) or ("CJK" in name): S.add("Jpn")
            elif "TAMIL" in name: S.add("Tam")
            else: S.add("Latin")
    return len(S) >= 2
def _has_space(s: str) -> bool: return any(ch.isspace() for ch in s)
def is_cjk_char(ch): o = ord(ch); return (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF)
def is_pure_cjk(s): return bool(s) and all(is_cjk_char(c) for c in s)
def is_tamil_char(ch): o = ord(ch); return 0x0B80 <= o <= 0x0BFF
def is_pure_tamil(s): return bool(s) and all(is_tamil_char(c) for c in s)

def grapheme_clusters(s: str): return reg.findall(r'\X', s)
def script_guess(tok: str) -> str:
    for ch in tok:
        if ch.isalpha():
            name = ud.name(ch, "")
            if "HIRAGANA" in name or "KATAKANA" in name or "CJK" in name: return "ja"
            if "TAMIL" in name: return "ta"
            if "TURKISH" in name: return "tr" # Note: unicodedata names are uppercase
            return "en" # Default for Latin scripts
    return "other"

def default_allowed_boundaries(text: str, lang: str = None):
    clusters = grapheme_clusters(text)
    B = np.zeros(len(text)+1, dtype=bool)
    B[0] = True; pos = 0
    for g in clusters:
        pos += len(g)
        B[pos] = True
    return B

def find_protected_spans(text: str):
    spans = []
    for m in URL_RE.finditer(text):
        s, e = _strip_trailing_punct_span(text, m.start(), m.end())
        if e > s: spans.append((s, e))
    for m in EMAIL_RE.finditer(text): spans.append((m.start(), m.end()))
    for m in NUM_RE.finditer(text):   spans.append((m.start(), m.end()))
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]: merged.append([s, e])
        else: merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def span_overlaps_protected(i, j, protected):
    for (s, e) in protected:
        if i < e and j > s and not (i == s and j == e):
            return True
    return False

def ta_grapheme_spans(text: str, max_g=8):
    clusters = grapheme_clusters(text)
    spans, pos, acc = [], [0], 0
    for g in clusters:
        acc += len(g); pos.append(acc)
    for i in range(len(clusters)):
        for L in range(1, max_g + 1):
            if i + L <= len(clusters):
                spans.append((pos[i], pos[i + L]))
    return spans

AFFIXES = {
    "en": {"pre": ["re","un","in","dis"], "suf": ["ing","ed","ly","ness","tion","s"]},
    "de": {"pre": ["ur","ver","be","ent"], "suf": ["ung","en","isch","heit","keit","e"]},
    "ja": {"pre": [], "suf": ["は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで", "ます", "ました", "ません", "でした", "です", "でした", "ない", "た", "て"]},
    "ta": {"pre": [], "suf": ["ன்","க்கு","கள்","த்","டு","ம்","ஆக","உம்","ாமல்","தாக","கின்ற","ப்படும்"]},
    "fr": {"pre": ["re","dé","pré"], "suf": ["ment","tion","sion","able","age","isme","iste"]},
    "tr": {"pre": [], "suf": ["lar", "ler", "de", "da", "den", "dan", "lı", "li", "lık", "lik"]}
}

class LinguisticModels:
    def __init__(self, **kwargs):
        self.lexicon = kwargs.get("lexicon", {})
        self.token_bigram = kwargs.get("token_bigram", {})
        self.paragraph_lang = kwargs.get("paragraph_lang", None)
        self.gamma_boundary = kwargs.get("gamma_boundary", 0.05)
        self.rho_group = kwargs.get("rho_group", 0.08)

    @staticmethod
    def token_class(tok: str) -> str:
        if URL_RE.search(tok):   return "URL"
        if NUM_RE.fullmatch(tok):  return "NUM"
        if tok.istitle():          return "InitCap"
        if tok.isupper():          return "ALLCAPS"
        if tok.islower():          return "lower"
        return "other"

    def _affix_bias(self, token: str, lang: str) -> float:
        suf = AFFIXES.get(lang, {}).get("suf", [])
        pre = AFFIXES.get(lang, {}).get("pre", [])
        b = 0.0
        if any(token.endswith(a) for a in suf): b -= self.rho_group
        if any(token.startswith(a) for a in pre): b -= 0.5 * self.rho_group
        return b

    def additive_cost(self, token: str, prev_class: str, paragraph_idx: int = None) -> float:
        c = 0.0
        if token in self.lexicon: c += -1.0 * self.lexicon.get(token, 0.0)
        tc = self.token_class(token)
        c += self.token_bigram.get((prev_class, tc), 0.0)
        lang = self.paragraph_lang(paragraph_idx) if self.paragraph_lang else None
        if lang: c += self._affix_bias(token, lang)
        if prev_class is not None and prev_class != tc:
            c += self.gamma_boundary
        return c

class ParagraphInfo:
    def __init__(self, text: str, lang: str = None):
        self.text = text
        self.lang = lang or script_guess(text)
        self.boundary_ok = default_allowed_boundaries(text, self.lang)
        self.protected = find_protected_spans(text)

    def is_legal_span(self, i: int, j: int) -> bool:
        if not (0 <= i < j <= len(self.text)): return False
        if not (self.boundary_ok[i] and self.boundary_ok[j]): return False
        if span_overlaps_protected(i, j, self.protected): return False
        return True

class ScalableTokenizer:
    def __init__(self, **kwargs):
        self.max_token_len = kwargs.get('max_token_len', 32)
        self.min_freq = kwargs.get('min_freq', 5)
        self.alpha = kwargs.get('alpha', 1.0) # NLL
        self.beta = kwargs.get('beta', 0.5) # PMI
        self.tau = kwargs.get('tau', 0.01) # Length
        self.top_k_add = kwargs.get('top_k_add', 20)
        self.merge_reward = kwargs.get('merge_reward', 0.05)
        self.short_penalty = kwargs.get('short_penalty', 1.2)
        self.proposer_space_penalty = kwargs.get('proposer_space_penalty', 3.0)

        self.vocab, self.tok2id = [], {}
        self._nll, self._pmi_pen = {}, {}
        self._potential_tokens, self._token_occurrences = set(), defaultdict(list)
        self._cost_cache, self._paras = {}, []
        self._ling = LinguisticModels()
        self._classes = ["<BOS>", "URL", "NUM", "InitCap", "ALLCAPS", "lower", "other"]
        self._class2idx = {c: i for i, c in enumerate(self._classes)}
        self._len_floor = 2

    def set_feature_models(self, **kwargs): self._ling = LinguisticModels(**kwargs)
    def paragraph_lang(self, idx: int): return self._paras[idx].lang if 0 <= idx < len(self._paras) else None

    def _initialize_stats_and_vocab(self, paragraphs_texts, paragraphs_langs):
        print("Step 1: Performing initial corpus analysis...")
        t0 = time.time()
        self._paras = [ParagraphInfo(t, l) for t, l in zip(paragraphs_texts, paragraphs_langs)]

        global_substr_count = Counter()
        for pi, p in enumerate(paragraphs_texts):
            info = self._paras[pi]
            for i in range(len(p)):
                for L in range(1, min(self.max_token_len, len(p) - i) + 1):
                    if info.is_legal_span(i, i + L):
                        global_substr_count[p[i:i+L]] += 1

        cjk_seed, ta_seed = set(), set()
        for pi, p in enumerate(paragraphs_texts):
            if self._paras[pi].lang == 'ja':
                for i in range(len(p)):
                    for L in range(1, 5):
                        if i + L <= len(p) and is_pure_cjk(p[i:i+L]): cjk_seed.add(p[i:i+L])
            if self._paras[pi].lang == 'ta':
                for s, e in ta_grapheme_spans(p): ta_seed.add(p[s:e])

        self._potential_tokens = {
            tok for tok, c in global_substr_count.items()
            if (c >= self.min_freq or tok in cjk_seed or tok in ta_seed) and not WIKI_NOISE_RE.match(tok)
        }

        for pi, p in enumerate(paragraphs_texts):
            info = self._paras[pi]
            for i in range(len(p)):
                for L in range(1, min(self.max_token_len, len(p) - i) + 1):
                    tok = p[i:i+L]
                    if tok in self._potential_tokens and info.is_legal_span(i, i + L):
                        self._token_occurrences[tok].append((pi, i))
        
        corpus_text = "".join(paragraphs_texts)
        char_counts = Counter(corpus_text)
        total_chars = sum(char_counts.values())
        char_prob = {c: count / total_chars for c, count in char_counts.items()}
        
        for tok in self._potential_tokens:
            p_tok = global_substr_count[tok] / total_chars
            p_chars = math.prod(char_prob.get(c, 1/total_chars) for c in tok)
            self._nll[tok] = -math.log(max(p_tok, EPS))
            self._pmi_pen[tok] = -math.log(max(p_tok, EPS) / max(p_chars, EPS))
        
        self.vocab = sorted(list(char_counts.keys()))
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        print(f"Analysis complete in {time.time()-t0:.2f}s. Found {len(self._potential_tokens)} potential tokens.")

    def _base_token_cost(self, token: str) -> float:
        c = (self.alpha * self._nll.get(token, 25.0) +
             self.beta * self._pmi_pen.get(token, 0.0) +
             self.tau * len(token))
        L = len(grapheme_clusters(token))
        if L > 1: c -= self.merge_reward * (L - 1)
        if L <= 1: c += self.short_penalty
        return c

    def _get_token_cost(self, tok): return self._cost_cache.setdefault(tok, self._base_token_cost(tok))
    
    def _proposer_bonus(self, tok: str) -> float:
        delta = 0.0
        if _has_space(tok): delta += self.proposer_space_penalty
        if _is_mixed_script(tok): delta += 0.5
        return delta

    def _dp_costs(self, para_idx):
        text, info = self._paras[para_idx].text, self._paras[para_idx]
        T, K, idx = len(text), len(self._classes), self._class2idx
        dp = np.full((T + 1, K), float('inf'))
        dp[0, idx["<BOS>"]] = 0.0
        
        for t in range(1, T + 1):
            for L in range(1, min(self.max_token_len, t) + 1):
                i = t - L
                if not info.is_legal_span(i, t): continue
                tok = text[i:t]
                
                base_cost = float('inf')
                # This is the full "open arc" logic
                if tok in self.tok2id:
                    base_cost = self._get_token_cost(tok)
                elif info.lang == 'ja' and is_pure_cjk(tok) and len(tok) <= 4:
                    base_cost = self._get_token_cost(tok)
                # Tamil Open Arc Rule
                elif info.lang == 'ta' and is_pure_tamil(tok) and len(grapheme_clusters(tok)) <= 3:
                    base_cost = self._get_token_cost(tok)

                if not np.isfinite(base_cost): continue

                tc = self._ling.token_class(tok)
                j = idx.get(tc, idx["other"])
                for si in range(K):
                    if not np.isfinite(dp[i, si]): continue
                    prev_class = self._classes[si]
                    add_cost = self._ling.additive_cost(tok, prev_class, para_idx)
                    cost = dp[i, si] + base_cost + add_cost
                    if cost < dp[t, j]: dp[t, j] = cost
        return dp

    def _find_best_new_tokens_batch(self, all_dp_costs, top_k):
        candidates = defaultdict(float)
        for tok in self._potential_tokens:
            if tok in self.tok2id or len(tok) < self._len_floor: continue
            
            base_cost = self._get_token_cost(tok)
            if base_cost > MAX_BASE_COST: continue
            
            # Use proposer bonus to quickly estimate token quality
            proposer_cost = base_cost + self._proposer_bonus(tok)
            
            for para_idx, start_pos in self._token_occurrences.get(tok, []):
                end_pos = start_pos + len(tok)
                dp = all_dp_costs[para_idx]
                if end_pos >= dp.shape[0]: continue
                
                baseline_cost = np.min(dp[end_pos, :])
                if not np.isfinite(baseline_cost): continue
                
                best_prev_cost = float('inf')
                for si in range(len(self._classes)):
                    add_cost = self._ling.additive_cost(tok, self._classes[si], para_idx)
                    cost = dp[start_pos, si] + proposer_cost + add_cost # Use proposer cost here
                    if cost < best_prev_cost:
                        best_prev_cost = cost
                
                reduced_cost = best_prev_cost - baseline_cost
                if reduced_cost < 0: candidates[tok] += reduced_cost
                    
        sorted_candidates = sorted(candidates.items(), key=lambda item: item[1])
        return [tok for tok, score in sorted_candidates[:top_k]]

    def train(self, texts, langs, max_iterations):
        self._initialize_stats_and_vocab(texts, langs)
        self._ling.paragraph_lang = self.paragraph_lang
        
        print("\nStep 2: Starting training with full DP pricing...")
        for it in range(1, max_iterations + 1):
            self._cost_cache.clear()
            
            all_dp_costs = [self._dp_costs(i) for i in range(len(texts))]
            new_tokens = self._find_best_new_tokens_batch(all_dp_costs, self.top_k_add)
            
            if not new_tokens:
                print(f"\nConvergence reached at iteration {it}.")
                break
            
            for tok in new_tokens: self.tok2id[tok] = len(self.vocab); self.vocab.append(tok)
            
            preview = ", ".join([f"'{t}'" for t in new_tokens[:4]])
            print(f"Iter {it:02d}: Added {len(new_tokens)} (Vocab: {len(self.vocab)}): {preview}...")
        
        print(f"\nTraining complete. Final vocabulary size: {len(self.vocab)}")

    def tokenize(self, text, lang=None):
        lang = lang or script_guess(text)
        temp_paras = self._paras
        self._paras = [ParagraphInfo(text, lang)]
        
        T, K, idx = len(text), len(self._classes), self._class2idx
        dp = np.full((T + 1, K), float('inf'))
        back = [[None] * K for _ in range(T + 1)]
        dp[0, idx["<BOS>"]] = 0.0
        
        for t in range(1, T + 1):
            for L in range(1, min(self.max_token_len, t) + 1):
                i = t - L
                tok = text[i:t]
                
                base_cost = float('inf')
                if tok in self.tok2id:
                    base_cost = self._get_token_cost(tok)
                elif lang == 'ja' and is_pure_cjk(tok) and len(tok) <= 4:
                    base_cost = self._get_token_cost(tok)
                elif lang == 'ta' and is_pure_tamil(tok) and len(grapheme_clusters(tok)) <= 3:
                    base_cost = self._get_token_cost(tok)

                if not np.isfinite(base_cost): continue

                j = idx.get(self._ling.token_class(tok), idx["other"])
                for si in range(K):
                    if not np.isfinite(dp[i, si]): continue
                    add_cost = self._ling.additive_cost(tok, self._classes[si], 0)
                    cost = dp[i, si] + base_cost + add_cost
                    if cost < dp[t, j]:
                        dp[t, j] = cost; back[t][j] = (i, si)
        
        toks, t, c_idx = [], T, np.argmin(dp[T,:])
        while t > 0:
            if back[t][c_idx] is None: # Fallback for incomplete paths
                t -= 1
                toks.append(text[t:t+1])
                continue
            i, prev_c_idx = back[t][c_idx]
            toks.append(text[i:t])
            t, c_idx = i, prev_c_idx
        
        self._paras = temp_paras
        return list(reversed(toks))

# --- Main Execution ---
def load_wikiann_corpus(codes, per_lang):
    texts, langs = [], []
    print("Loading corpus...")
    for code, name in codes.items():
        try:
            dataset = load_dataset("wikiann", code, split='train', streaming=True).take(per_lang)
            print(f"  -> Loading {name} ({code})")
            for ex in dataset:
                txt = " ".join(ex.get('tokens', []))
                if txt.strip() and not looks_like_redirect(txt):
                    texts.append(txt)
                    langs.append(code)
        except Exception as e:
            print(f"Could not load data for {code}: {e}")
    print("Corpus loaded.\n" + "-"*50)
    return texts, langs

def main():
    # ADDED: Turkish and French language codes
    lang_codes = {
        'en': 'English', 'de': 'German', 'ja': 'Japanese', 'ta': 'Tamil',
        'tr': 'Turkish', 'fr': 'French'
    }
    texts, langs = load_wikiann_corpus(lang_codes, per_lang=PER_LANG)

    tokenizer = ScalableTokenizer(short_penalty=1.2)
    tokenizer.set_feature_models(lexicon={"東京都": 1.5, "Los Angeles": 1.5, "Île-de-France": 1.5})
    tokenizer.train(texts, langs, max_iterations=MAX_ITERS)
    
    print("\n--- Tokenization Examples ---")
    # ADDED: Turkish and French test sentences
    test_sentences = [
        ("東京都の研究チームは評価指標を再定義し、提案した。", "ja"),
        ("இந்த ஆய்வில், குழு அளவுகோல்களை மறுஅர்த்தமளித்தது.", "ta"),
        ("Die Bundesrepublik Deutschland veröffentlichte die Ergebnisse.", "de"),
        ("Afternoon in Los Angeles.", "en"),
        ("Türkiye'deki araştırmacılar yeni yöntemler deniyorlar.", "tr"),
        ("Les chercheurs en Île-de-France ont proposé une nouvelle approche.", "fr"),
    ]
    for sentence, lang in test_sentences:
        tokens = tokenizer.tokenize(sentence, lang=lang)
        print(f"\n[{lang}] {sentence}\n-> {tokens}\n")

if __name__ == "__main__":
    main()
