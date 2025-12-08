import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import torch

from .constants import *
from .utils import *
import tokenizer_core.utils as utils
from .torch_utils import default_device, ensure_tensor, random_choice, randperm
from .uniseg_loader import UniSegLoader

# Global cache for ngram extraction (can be cleared if memory is a concern)
_ngram_cache = {}
_cache_max_size = 100000  # Limit cache size to prevent unbounded growth

def char_ngrams(s, n=(2,3,4)):
    """Extracts character n-grams of specified orders from a string.
    
    Optimized version with caching and efficient string operations.
    Fully Unicode-compatible: handles all Unicode characters including
    emoji, CJK characters, Arabic, etc. String slicing works at the
    code point level, ensuring correct behavior with multi-byte characters.

    Args:
        s (str): The input string (Unicode-compatible).
        n (tuple): A tuple of integers specifying the n-gram orders to extract.

    Returns:
        Counter: A counter mapping each n-gram to its frequency in the string.
        
    Examples:
        >>> char_ngrams("café", n=(2, 3))
        Counter({'ca': 1, 'af': 1, 'fé': 1, 'caf': 1, 'afé': 1})
        >>> char_ngrams("北京", n=(2,))
        Counter({'北京': 1})
    """
    # Create cache key from string and n-gram orders
    cache_key = (s, n) if isinstance(n, tuple) else (s, tuple(n))
    
    # Check cache first
    if cache_key in _ngram_cache:
        return _ngram_cache[cache_key].copy()
    
    # Early exit for empty or very short strings
    s_len = len(s)
    if s_len == 0:
        return Counter()
    
    # Pre-compute valid n-gram orders (filter out orders larger than string)
    valid_orders = [k for k in n if k <= s_len]
    if not valid_orders:
        return Counter()
    
    # Use dict for faster updates, convert to Counter at end
    feats = {}
    
    # Optimized extraction: use dict.get() for faster updates than Counter
    # String slicing in Python is already quite efficient, so we focus on
    # reducing dictionary lookup overhead
    for k in valid_orders:
        max_i = s_len - k + 1
        for i in range(max_i):
            ngram = s[i:i+k]
            feats[ngram] = feats.get(ngram, 0) + 1
    
    # Convert to Counter for consistency with original API
    result = Counter(feats)
    
    # Cache the result (with size limit)
    if len(_ngram_cache) < _cache_max_size:
        _ngram_cache[cache_key] = result.copy()
    elif len(_ngram_cache) >= _cache_max_size:
        # Simple eviction: clear half the cache (could use LRU, but this is simpler)
        keys_to_remove = list(_ngram_cache.keys())[:_cache_max_size // 2]
        for key in keys_to_remove:
            _ngram_cache.pop(key, None)
        _ngram_cache[cache_key] = result.copy()
    
    return result


def char_ngrams_batch(tokens, n=(2,3,4)):
    """Batch version of char_ngrams for processing multiple tokens efficiently.
    
    This is more efficient when processing many tokens at once as it reduces
    function call overhead and can be optimized further.
    
    Args:
        tokens (list): List of token strings.
        n (tuple): A tuple of integers specifying the n-gram orders to extract.
    
    Returns:
        list: List of Counters, one per token.
    """
    return [char_ngrams(tok, n) for tok in tokens]


def clear_ngram_cache():
    """Clears the ngram cache to free memory."""
    global _ngram_cache
    _ngram_cache.clear()

class MorphologyEncoder:
    """Learns vector representations of tokens based on their morphology.

    This class turns tokens into feature vectors based on their character n-grams
    and known prefixes/suffixes. It then uses a matrix factorization technique
    (specifically, eigendecomposition on a Positive Pointwise Mutual Information
    matrix) to learn dense, low-dimensional embeddings for each token.

    These embeddings capture the morphological similarities between words. The encoder
    also computes an average vector, or "prototype," for each language.
    """
    def __init__(self,
                 affixes=AFFIXES,
                 ngram_orders=(2,3,4),
                 k=64,
                 pmi_floor=1e-9,
                 lambda_morph=0.1,
                 gamma=1e-3,
                 refine_lr=0.05,
                 refine_steps=50,
                 embedding_mode="glove",
                 glove_iters=15,
                 glove_lr=0.05,
                 glove_xmax=50.0,
                 glove_alpha=0.75,
                 glove_max_pairs=200000,
                 use_weighted_cross=False,
                 lang_similarity=None,
                 use_semantic_consistency=False,
                 semantic_lr=0.01,
                 semantic_iters=5,
                 use_structure_mapping=False,
                 structure_lr=0.01,
                 structure_iters=5,
                 use_cross_kl=False,
                 kl_weight=0.0,
                 kl_lr=0.01,
                 use_minibatch=True,
                 batch_size_pairs=2048,
                 batch_size_edges=512,
                 batch_size_semantic=512,
                 optimizer="sgd",
                 adagrad_eps=1e-8,
                 adagrad_reset=None,
                 max_tokens=None,
                 device: str | torch.device | None = None,
                 **kwargs):
        """Initializes the MorphologyEncoder.

        Args:
            affixes (dict): A dictionary defining known prefixes and suffixes for
                different languages.
            ngram_orders (tuple): The orders of character n-grams to use as features.
            k (int): The dimensionality of the learned token embeddings.
            pmi_floor (float): A small value to prevent division by zero or log(0)
                during PMI calculation.
            lambda_morph (float): Weight for the localized morphological regularizer.
            gamma (float): Weight decay applied during refinement to keep vectors bounded.
            refine_lr (float): Learning rate for the refinement gradient updates.
            refine_steps (int): Number of refinement iterations after the spectral init.
        """
        self.affixes = affixes
        self.ngram_orders = ngram_orders
        self.k = k
        self.pmi_floor = pmi_floor
        self.lambda_morph = lambda_morph
        self.gamma = gamma
        self.refine_lr = refine_lr
        self.refine_steps = refine_steps
        self.feat2id = {}
        self.token_vec = {}
        self.lang_proto = {}
        self.shared_counts = defaultdict(lambda: defaultdict(int))
        self.embedding_mode = embedding_mode
        self.glove_iters = glove_iters
        self.glove_lr = glove_lr
        self.glove_xmax = glove_xmax
        self.glove_alpha = glove_alpha
        self.glove_max_pairs = glove_max_pairs
        self.use_weighted_cross = use_weighted_cross
        self.lang_similarity = lang_similarity or {}
        self.use_semantic_consistency = use_semantic_consistency
        self.semantic_lr = semantic_lr
        self.semantic_iters = semantic_iters
        self.semantic_proj = {}
        self.use_structure_mapping = use_structure_mapping
        self.structure_lr = structure_lr
        self.structure_iters = structure_iters
        self.structure_maps = {}
        self.use_cross_kl = use_cross_kl
        self.kl_weight = kl_weight
        self.kl_lr = kl_lr
        self.use_minibatch = use_minibatch
        # Pair and edge batch sizes govern stochastic sampling; lower values reduce memory
        # pressure, higher values cut gradient noise. Keep edges <= pairs to avoid oversampling.
        self.batch_size_pairs = batch_size_pairs
        self.batch_size_edges = batch_size_edges
        self.batch_size_semantic = batch_size_semantic
        # DP flags are deprecated: force off to avoid DP paths entirely.
        # DP functionality removed; ignore any DP-related kwargs silently via **kwargs
        # Optimizer can be 'sgd' for constant steps or 'adagrad' to adapt rare pairs.
        # Choose 'sgd' when you want smoother convergence on small corpora; use 'adagrad'
        # for large, diverse data where rare morphs need extra emphasis.
        self.optimizer = optimizer
        self.adagrad_eps = adagrad_eps
        self.adagrad_reset = adagrad_reset
        self._adagrad_W = None
        self._adagrad_C = None
        self._adagrad_steps = 0
        self.max_tokens = max_tokens
        if self.use_minibatch:
            minibatch_cap = max(1, batch_size_pairs)
            if self.max_tokens is None:
                self.max_tokens = minibatch_cap
            else:
                self.max_tokens = min(self.max_tokens, minibatch_cap)
        self.device = torch.device(device) if device else default_device()
        
        # Cache for featurization results (token, lang) -> features
        # This avoids recomputing ngrams for the same token multiple times
        self._featurize_cache = {}
        self._featurize_cache_max_size = 50000

    def set_embedding_mode(self, mode: str):
        """Configures the embedding backend. Currently only 'glove' is supported."""
        mode = mode.lower()
        if mode != "glove":
            raise ValueError("PPMI mode is disabled; only the convex GloVe optimizer is available.")
        self.embedding_mode = mode

    def _affix_feats(self, tok, lang):
        """Extracts prefix and suffix features from a token."""
        pre = self.affixes.get(lang, {}).get("pre", [])
        suf = self.affixes.get(lang, {}).get("suf", [])
        feats = Counter()
        for a in pre:
            if tok.startswith(a): feats[f"^pre:{a}"] += 1
        for a in suf:
            if tok.endswith(a): feats[f"$suf:{a}"] += 1
        return feats

    def _featurize(self, tok, lang):
        """Converts a token into a complete feature set (n-grams + affixes).
        
        Uses caching to avoid recomputing features for the same (token, lang) pair.
        """
        cache_key = (tok, lang)
        
        # Check instance-level cache first
        if cache_key in self._featurize_cache:
            return self._featurize_cache[cache_key].copy()
        
        # Compute features
        f = Counter()
        f.update(char_ngrams(tok, self.ngram_orders))
        f.update(self._affix_feats(tok, lang))
        
        # Cache the result (with size limit)
        if len(self._featurize_cache) < self._featurize_cache_max_size:
            self._featurize_cache[cache_key] = f.copy()
        elif len(self._featurize_cache) >= self._featurize_cache_max_size:
            # Evict half the cache when full
            keys_to_remove = list(self._featurize_cache.keys())[:self._featurize_cache_max_size // 2]
            for key in keys_to_remove:
                self._featurize_cache.pop(key, None)
            self._featurize_cache[cache_key] = f.copy()
        
        return f
    
    def clear_featurize_cache(self):
        """Clears the featurization cache to free memory."""
        self._featurize_cache.clear()

    def _ensure_adagrad_buffers(self, W_shape=None, C_shape=None):
        """Initialises or refreshes AdaGrad accumulators when required."""
        if W_shape is not None and ((self._adagrad_W is None) or (self._adagrad_W.shape != W_shape)):
            self._adagrad_W = torch.zeros(W_shape, dtype=torch.float32, device=self.device)
        if C_shape is not None and ((self._adagrad_C is None) or (self._adagrad_C.shape != C_shape)):
            self._adagrad_C = torch.zeros(C_shape, dtype=torch.float32, device=self.device)

    def _maybe_reset_adagrad(self):
        """Optionally resets AdaGrad accumulators on a fixed schedule."""
        if self.adagrad_reset and self._adagrad_steps >= self.adagrad_reset:
            if self._adagrad_W is not None:
                self._adagrad_W.zero_()
            if self._adagrad_C is not None:
                self._adagrad_C.zero_()
            self._adagrad_steps = 0

    def _apply_optimizer_step(self, matrix, indices, grads, lr, is_token=True):
        """Applies an optimisation step (SGD or AdaGrad) to selected rows.

        SGD simply subtracts lr * grad; AdaGrad divides by the running rms."""
        if not indices:
            return
        opt = (self.optimizer or "sgd").lower()
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        grad_arr = ensure_tensor(grads, dtype=torch.float32, device=self.device)
        if grad_arr.ndim == 1:
            grad_arr = grad_arr.unsqueeze(0)
        if opt == "adagrad":
            if is_token:
                self._adagrad_W[idx] += grad_arr.pow(2)
                denom = torch.sqrt(self._adagrad_W[idx]) + self.adagrad_eps
            else:
                self._adagrad_C[idx] += grad_arr.pow(2)
                denom = torch.sqrt(self._adagrad_C[idx]) + self.adagrad_eps
            matrix[idx] -= lr * (grad_arr / denom)
            self._adagrad_steps += 1
            self._maybe_reset_adagrad()
        elif opt == "sgd":
            matrix[idx] -= lr * grad_arr
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer}'.")

    def _morph_keys(self, tok, lang):
        """Identifies if a token's suffix belongs to a cross-lingual category."""
        keys = set()
        suf_list = self.affixes.get(lang, {}).get("suf", [])
        for a in suf_list:
            if tok.endswith(a):
                for key, mp in CROSS_EQUIV.items():
                    if a in mp.get(lang, set()):
                        keys.add(key)
        return keys

    def _equiv_sets(self, toks, tok_lang):
        """Builds morphological equivalence sets (indices) for localized Laplacians."""
        class_members = defaultdict(list)
        for ti, tok in enumerate(toks):
            lang = tok_lang[tok]
            for key in self._morph_keys(tok, lang):
                class_members[key].append(ti)
        # Only sets with at least two members contribute to a Laplacian term.
        return [
            torch.as_tensor(members, dtype=torch.long, device=self.device)
            for members in class_members.values()
            if len(members) >= 2
        ]

    def _refine_embeddings(self, V, target, equiv_sets, tok_lang_list=None, lang_similarity=None):
        """Refines embeddings with the localized morphological regularizer."""
        if self.refine_steps <= 0:
            return V
        if self.lambda_morph <= 0 and self.gamma <= 0 and target is None:
            return V
        if self.lambda_morph <= 0 or not equiv_sets:
            # No localized Laplacians to apply; only ridge if gamma > 0.
            apply_lap = False
        else:
            apply_lap = True

        for _ in range(self.refine_steps):
            # Only use target-based gradient when a dense target is provided
            if isinstance(target, torch.Tensor):
                grad = 4.0 * ((V @ V.T - target) @ V)
            else:
                grad = torch.zeros_like(V)
            if apply_lap:
                lap_grad = torch.zeros_like(V)
                for idxs in equiv_sets:
                    idx_list = idxs.tolist()
                    if lang_similarity and tok_lang_list is not None:
                        for pos, idx in enumerate(idx_list):
                            lang_i = tok_lang_list[idx]
                            weighted_sum = torch.zeros_like(V[idx])
                            weight_total = 0.0
                            for pos2, idx2 in enumerate(idx_list):
                                if pos2 == pos:
                                    continue
                                lang_j = tok_lang_list[idx2]
                                weight = lang_similarity.get((lang_i, lang_j), lang_similarity.get((lang_j, lang_i), 1.0))
                                weighted_sum += weight * V[idx2]
                                weight_total += weight
                            lap_grad[idx] += weight_total * V[idx] - weighted_sum
                    else:
                        subset = V[idxs]
                        sum_vec = subset.sum(dim=0, keepdim=True)
                        lap_grad[idxs] += len(idx_list) * subset - sum_vec
                grad += 2.0 * self.lambda_morph * lap_grad
            if self.gamma > 0:
                grad += 2.0 * self.gamma * V
            V = V - self.refine_lr * grad
            norms = torch.linalg.norm(V, dim=1, keepdim=True)
            V = V / torch.clamp(norms, min=1e-12)
        return V

    def fit(self, paragraphs, tok_occurrences, paragraph_lang):
        """Trains the encoder on the corpus data to learn token embeddings."""
        # Reset previously stored state.
        self.token_vec.clear()
        self.lang_proto.clear()
        self.shared_counts = defaultdict(lambda: defaultdict(int))

        # --- 1. Determine the primary language for each token ---
        tok_lang = {}
        for tok, occs in tok_occurrences.items():
            langs = Counter(paragraph_lang(pi) or "other" for (pi, _) in occs)
            if langs:
                tok_lang[tok] = langs.most_common(1)[0][0]

        if self.max_tokens and len(tok_lang) > self.max_tokens:
            freq = {tok: len(tok_occurrences.get(tok, [])) for tok in tok_lang.keys()}
            top_tokens = sorted(tok_lang.keys(), key=lambda t: freq.get(t, 0), reverse=True)[: self.max_tokens]
            tok_lang = {tok: tok_lang[tok] for tok in top_tokens}

        # --- 2. Build the feature matrix X (tokens vs. features) ---
        all_feats = set()
        for tok, lang in tok_lang.items():
            all_feats.update(self._featurize(tok, lang).keys())
        self.feat2id = {f: i for i, f in enumerate(sorted(all_feats))}
        F = len(self.feat2id)

        toks = list(tok_lang.keys())
        T = len(toks)
        if T == 0 or F == 0:
            return

        X = torch.zeros((T, F), dtype=torch.float32, device=self.device)
        for ti, tok in enumerate(toks):
            lang = tok_lang[tok]
            for f, v in self._featurize(tok, lang).items():
                X[ti, self.feat2id[f]] = float(v)

        equiv_sets = self._equiv_sets(toks, tok_lang)

        tok_lang_list = [tok_lang[tok] for tok in toks]
        PPMI = self._compute_ppmi(X)
        print(f"[MorphEncoder] embedding_mode='{self.embedding_mode}', tokens={len(toks)}, features={F}")
        if self.embedding_mode != "glove":
            raise RuntimeError("Only the convex GloVe optimizer is supported; set embedding_mode='glove'.")
        V = self._fit_glove_embeddings(X, PPMI, tok_lang_list, equiv_sets)

        if V is None:
            return

        V = V.to(torch.float32)
        norms = torch.linalg.norm(V, dim=1, keepdim=True)
        V = V / torch.clamp(norms, min=1e-9)

        # --- 5. Store the learned token vectors and language prototypes ---
        for ti, tok in enumerate(toks):
            self.token_vec[tok] = V[ti].detach().clone()

        for lang in set(tok_lang.values()):
            idxs = [ti for ti, tok in enumerate(toks) if tok_lang[tok] == lang]
            if idxs:
                idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
                lp = V[idx_tensor].mean(0)
                norm = torch.linalg.norm(lp) + 1e-9
                self.lang_proto[lang] = (lp / norm).detach().clone()

        # --- 6. Pre-calculate counts for the consistency bonus ---
        for tok, lang in tok_lang.items():
            for key in self._morph_keys(tok, lang):
                self.shared_counts[key][lang] += 1
        print("[MorphEncoder] Training complete; stored token vectors:", len(self.token_vec))

    def _compute_ppmi(self, X):
        col_sum = torch.clamp(X.sum(dim=0, keepdim=True), min=self.pmi_floor)
        row_sum = torch.clamp(X.sum(dim=1, keepdim=True), min=self.pmi_floor)
        total = float(X.sum().item()) + 1e-9
        P_xy = X / total
        P_x = row_sum / total
        P_y = col_sum / total
        PMI = torch.log(torch.clamp(P_xy / (P_x @ P_y), min=self.pmi_floor))
        return torch.clamp(PMI, min=0.0)

    def _fit_ppmi_embeddings(self, PPMI, tok_lang_list, equiv_sets):
        """PPMI embeddings via operator-based power iteration (no dense Gram matrix)."""
        eigvals, eigvecs = _eigendecomposition_power_operator(PPMI, self.k, max_iters=100, tol=1e-6, device=self.device)
        if eigvecs is None or eigvals is None or eigvecs.numel() == 0:
            return None
        selected_vals = torch.clamp(eigvals, min=0.0)
        V = eigvecs * torch.sqrt(selected_vals).unsqueeze(0)
        # Refine without explicit Gram target (skip that term), only apply Laplacian/gamma
        V = self._refine_embeddings(V, None, equiv_sets, tok_lang_list, self.lang_similarity if self.use_weighted_cross else None)
        cross_pairs = self._collect_cross_pairs(equiv_sets, tok_lang_list)
        V = self._apply_cross_consistency(V, tok_lang_list, cross_pairs)
        return V

    def _build_morph_graph(self, equiv_sets, n_tokens, tok_lang_list, lang_similarity):
        """Pre-computes degree and neighbor lists for Laplacian gradients."""
        deg = torch.zeros(n_tokens, dtype=torch.float32, device=self.device)
        neighbors = [[] for _ in range(n_tokens)]
        for idxs in equiv_sets:
            idx_list = list(idxs)
            size = len(idx_list)
            if size < 2:
                continue
            for a, i in enumerate(idx_list):
                lang_i = tok_lang_list[i]
                for b, j in enumerate(idx_list):
                    if a == b:
                        continue
                    lang_j = tok_lang_list[j]
                    weight = 1.0
                    if lang_similarity:
                        weight = lang_similarity.get((lang_i, lang_j), lang_similarity.get((lang_j, lang_i), 1.0))
                    deg[i] += weight
                    neighbors[i].append((j, weight))
        return deg, neighbors

    def _collect_morph_edges(self, equiv_sets, tok_lang_list, lang_similarity):
        """Creates an undirected list of Laplacian edges for stochastic sampling."""
        edges = []
        for idxs in equiv_sets:
            idx_list = list(idxs)
            size = len(idx_list)
            if size < 2:
                continue
            for a in range(size):
                i = idx_list[a]
                lang_i = tok_lang_list[i]
                for b in range(a + 1, size):
                    j = idx_list[b]
                    lang_j = tok_lang_list[j]
                    weight = 1.0
                    if lang_similarity:
                        weight = lang_similarity.get((lang_i, lang_j), lang_similarity.get((lang_j, lang_i), 1.0))
                    edges.append((i, j, weight))
        if not edges:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
            )
        src = torch.as_tensor([e[0] for e in edges], dtype=torch.long, device=self.device)
        dst = torch.as_tensor([e[1] for e in edges], dtype=torch.long, device=self.device)
        weights = torch.as_tensor([e[2] for e in edges], dtype=torch.float32, device=self.device)
        return src, dst, weights

    def _collect_cross_pairs(self, equiv_sets, tok_lang_list):
        pairs = []
        seen = set()
        for idxs in equiv_sets:
            indices = list(idxs)
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    i, j = indices[a], indices[b]
                    lang_i = tok_lang_list[i]
                    lang_j = tok_lang_list[j]
                    if lang_i == lang_j:
                        continue
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    weight = 1.0
                    if self.lang_similarity:
                        weight = self.lang_similarity.get((lang_i, lang_j), self.lang_similarity.get((lang_j, lang_i), 1.0))
                    pairs.append((i, j, weight))
        return pairs

    def _apply_cross_consistency(self, V, tok_lang_list, cross_pairs):
        if not cross_pairs:
            return V
        if self.use_semantic_consistency:
            V = self._apply_semantic_consistency(V, tok_lang_list, cross_pairs)
        if self.use_structure_mapping:
            V = self._apply_structure_mapping(V, tok_lang_list, cross_pairs)
        if self.use_cross_kl and self.kl_weight > 0:
            V = self._apply_kl_consistency(V, tok_lang_list, cross_pairs)
        return V

    def _ensure_semantic_proj(self, languages):
        for lang in languages:
            if lang not in self.semantic_proj:
                self.semantic_proj[lang] = torch.eye(self.k, dtype=torch.float32, device=self.device)

    def _apply_semantic_consistency(self, V, tok_lang_list, cross_pairs):
        languages = set(tok_lang_list)
        self._ensure_semantic_proj(languages)
        
        # Always use batched version for performance and consistency
        return self._apply_semantic_consistency_batched(V, tok_lang_list, cross_pairs)
    
    def _apply_semantic_consistency_batched(self, V, tok_lang_list, cross_pairs):
        """Batched version of semantic consistency for better GPU utilization."""
        languages = set(tok_lang_list)
        self._ensure_semantic_proj(languages)
        
        batch_size = getattr(self, 'batch_size_semantic', 512)
        n_pairs = len(cross_pairs)
        
        if n_pairs == 0:
            transformed = torch.zeros_like(V)
            for idx, lang in enumerate(tok_lang_list):
                transformed[idx] = self.semantic_proj[lang] @ V[idx]
            return transformed
        
        for iteration in range(self.semantic_iters):
            # Shuffle for stochastic gradient descent
            shuffled = torch.randperm(n_pairs, device=self.device)
            
            for batch_start in range(0, n_pairs, batch_size):
                batch_end = min(batch_start + batch_size, n_pairs)
                batch_indices = shuffled[batch_start:batch_end]
                
                # Accumulate gradients by language
                lang_grads = {}
                for lang in languages:
                    lang_grads[lang] = torch.zeros(self.k, self.k, dtype=torch.float32, device=self.device)
                
                # Process batch
                for idx in batch_indices:
                    i, j, weight = cross_pairs[int(idx.item())]
                    lang_i = tok_lang_list[i]
                    lang_j = tok_lang_list[j]
                    
                    phi_i = self.semantic_proj[lang_i]
                    phi_j = self.semantic_proj[lang_j]
                    
                    v_i = phi_i @ V[i]
                    v_j = phi_j @ V[j]
                    diff = v_i - v_j
                    
                    grad_i = 2.0 * weight * torch.outer(diff, V[i])
                    grad_j = -2.0 * weight * torch.outer(diff, V[j])
                    
                    lang_grads[lang_i] += grad_i
                    lang_grads[lang_j] += grad_j
                
                # Apply accumulated gradients
                batch_count = batch_end - batch_start
                for lang in languages:
                    if lang_grads[lang].abs().sum() > 0:
                        self.semantic_proj[lang] -= self.semantic_lr * lang_grads[lang] / batch_count
        
        # Transform all embeddings
        transformed = torch.zeros_like(V)
        for idx, lang in enumerate(tok_lang_list):
            transformed[idx] = self.semantic_proj[lang] @ V[idx]
        return transformed

    def _apply_structure_mapping(self, V, tok_lang_list, cross_pairs):
        for _ in range(self.structure_iters):
            for i, j, weight in cross_pairs:
                lang_i = tok_lang_list[i]
                lang_j = tok_lang_list[j]
                M_ij = self.structure_maps.setdefault((lang_i, lang_j), torch.eye(self.k, dtype=torch.float32, device=self.device))
                M_ji = self.structure_maps.setdefault((lang_j, lang_i), torch.eye(self.k, dtype=torch.float32, device=self.device))
                diff_ij = M_ij @ V[i] - V[j]
                grad_M_ij = 2.0 * weight * torch.outer(diff_ij, V[i])
                M_ij -= self.structure_lr * grad_M_ij
                V[i] -= self.structure_lr * weight * (M_ij.T @ diff_ij)
                V[j] += self.structure_lr * weight * diff_ij
                diff_ji = M_ji @ V[j] - V[i]
                grad_M_ji = 2.0 * weight * torch.outer(diff_ji, V[j])
                M_ji -= self.structure_lr * grad_M_ji
        return V

    def _apply_kl_consistency(self, V, tok_lang_list, cross_pairs):
        for i, j, weight in cross_pairs:
            p = self._softmax(V[i])
            q = self._softmax(V[j])
            diff = p - q
            V[i] -= self.kl_lr * self.kl_weight * weight * diff
            V[j] += self.kl_lr * self.kl_weight * weight * diff
        return V

    @staticmethod
    def _softmax(x):
        x = x.to(torch.float32)
        shifted = x - torch.max(x)
        exp_x = torch.exp(shifted)
        return exp_x / torch.clamp(exp_x.sum(), min=1e-9)

    def _fit_glove_embeddings(self, X_counts, PPMI, tok_lang_list, equiv_sets):
        """Solves the convex Laplacian-regularized GloVe objective.
        
        Given fixed subword embeddings S, learns token embeddings U by minimizing:
            (1/2) Σ_{t,f} w_tf (u_t · s_f - X_tf)^2  +  (λ/2) tr(U^T L U)  +  (γ/2) ||U||_F^2
        
        This is strictly convex with a unique solution, computed via conjugate gradient.
        """
        T, F = X_counts.shape
        if T == 0 or F == 0:
            return None

        W = torch.clamp(X_counts.to(torch.float32), min=0.0)  # weights w_tf
        X = PPMI.to(torch.float32)                             # targets X_tf
        k = min(self.k, F)

        # =========================================================
        # Step 1: Compute fixed subword embeddings S ∈ R^{F×k}
        # =========================================================
        # S = top-k eigenvectors of (PPMI^T PPMI), scaled by sqrt(eigenvalues)
        cov = X.T @ X
        cov = (cov + cov.T) * 0.5 + 1e-6 * torch.eye(F, device=self.device)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        topk_idx = torch.argsort(eigvals, descending=True)[:k]
        S = eigvecs[:, topk_idx] * torch.sqrt(torch.clamp(eigvals[topk_idx], min=1e-8))
        
        # Pad to self.k if needed
        if k < self.k:
            S = torch.cat([S, torch.zeros(F, self.k - k, device=self.device)], dim=1)

        # =========================================================
        # Step 2: Build per-token normal equations (batched)
        #   G_t = Σ_f w_tf s_f s_f^T  +  γ I_k
        #   c_t = Σ_f w_tf X_tf s_f
        # =========================================================
        # Batched outer products: G_blocks[t] = S^T diag(W[t]) S + γI
        # Using einsum for clarity: G_blocks = einsum('tf,fj,fk->tjk', W, S, S) + γI
        WS = W.unsqueeze(-1) * S.unsqueeze(0)       # (T, F, k): W_tf * s_f
        G_blocks = torch.einsum('tfi,tfj->tij', WS, S.unsqueeze(0).expand(T, -1, -1))
        G_blocks += (self.gamma if self.gamma > 0 else 1e-6) * torch.eye(self.k, device=self.device)

        # c_vec[t] = Σ_f w_tf X_tf s_f  =  (W * X) @ S
        c_vec = (W * X) @ S  # (T, k)

        # =========================================================
        # Step 3: Solve (G + λL) U = c via conjugate gradient
        # =========================================================
        cross_pairs = self._collect_cross_pairs(equiv_sets, tok_lang_list)
        lambda_reg = float(self.lambda_morph) if self.lambda_morph > 0 else 0.0
        
        # Precompute Laplacian structure for batched matvec
        L_src, L_dst, L_weights = self._build_laplacian_edges(equiv_sets, cross_pairs)

        def apply_operator(U):
            """Computes (G + λL) U using batched ops."""
            # Block-diagonal part: G_blocks @ U (batched matmul)
            out = torch.bmm(G_blocks, U.unsqueeze(-1)).squeeze(-1)
            # Laplacian part
            if lambda_reg > 0 and L_src.numel() > 0:
                out += lambda_reg * self._laplacian_matvec_batched(U, L_src, L_dst, L_weights)
            return out

        # Conjugate gradient
        U = torch.zeros_like(c_vec)
        r = c_vec - apply_operator(U)
        p = r.clone()
        rs_old = torch.sum(r * r)
        tol = 1e-6 * (torch.sum(c_vec * c_vec) + 1e-12)
        
        for _ in range(256):  # max CG iterations
            Ap = apply_operator(p)
            pAp = torch.sum(p * Ap)
            if pAp < 1e-12:
                break
            alpha = rs_old / pAp
            U = U + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)
            if rs_new < tol:
                break
            p = r + (rs_new / (rs_old + 1e-12)) * p
            rs_old = rs_new

        # =========================================================
        # Step 4: Optional cross-lingual consistency refinement
        # =========================================================
        if cross_pairs:
            U = self._apply_cross_consistency(U, tok_lang_list, cross_pairs)

        return U

    def _build_laplacian_edges(self, equiv_sets, cross_pairs):
        """Precomputes edge lists for batched Laplacian matvec."""
        edges_src, edges_dst, edges_w = [], [], []
        
        # Edges from morphological equivalence sets
        for idxs in equiv_sets:
            idx_list = idxs.tolist() if hasattr(idxs, 'tolist') else list(idxs)
            m = len(idx_list)
            if m < 2:
                continue
            for i in idx_list:
                for j in idx_list:
                    if i != j:
                        edges_src.append(i)
                        edges_dst.append(j)
                        edges_w.append(1.0)
        
        # Edges from cross-lingual pairs
        for i, j, w in cross_pairs:
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            edges_w.extend([w, w])
        
        if not edges_src:
            return (torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.float32, device=self.device))
        
        return (torch.tensor(edges_src, dtype=torch.long, device=self.device),
                torch.tensor(edges_dst, dtype=torch.long, device=self.device),
                torch.tensor(edges_w, dtype=torch.float32, device=self.device))

    def _laplacian_matvec_batched(self, U, src, dst, weights):
        """Computes L @ U where L is the graph Laplacian, using scatter ops."""
        T, k = U.shape
        out = torch.zeros_like(U)
        
        if src.numel() == 0:
            return out
        
        # L_ij = -w_ij for i≠j, L_ii = Σ_j w_ij (degree)
        # (L @ U)_i = deg_i * U_i - Σ_j w_ij U_j
        
        # Accumulate weighted neighbor contributions
        neighbor_contrib = weights.unsqueeze(-1) * U[dst]  # (E, k)
        out.index_add_(0, src, neighbor_contrib)           # Σ_j w_ij U_j at each i
        
        # Compute degrees and apply
        deg = torch.zeros(T, device=self.device)
        deg.index_add_(0, src, weights)
        out = deg.unsqueeze(-1) * U - out                  # deg_i U_i - Σ_j w_ij U_j
        
        return out

    def score(self, tok, lang):
        """Calculates the morphological fit of a token for a given language.

        This score is the cosine similarity between the token's vector and the
        language's prototype vector. A higher score means the token is more
        morphologically typical for that language.

        Returns:
            float: A similarity score, typically between -1 and 1.
        """
        v = self.token_vec.get(tok)
        lp = self.lang_proto.get(lang)
        if v is None or lp is None:
            return 0.0
        return float(torch.dot(v, lp).item())

    def consistency_bonus(self, tok, lang):
        """Calculates a bonus for tokens with suffixes that are shared across languages.

        For example, if the plural suffix '-s' in English and '-s' in French are
        both identified, tokens using them get a small bonus, encouraging the
        model to recognize cross-lingual grammatical patterns.

        Returns:
            float: A small non-negative bonus value.
        """
        bonus = 0.0
        for key, mp in CROSS_EQUIV.items():
            if any(tok.endswith(a) for a in mp.get(lang, set())):
                langs_present = sum(1 for _l, c in self.shared_counts[key].items() if c > 0)
                if langs_present >= 2:
                    bonus += 0.05 * (langs_present - 1)
        return bonus

class LinguisticModels:
    """A container for all linguistic feature models and cost calculations.

    This class serves as a central "cost engine" for the main tokenizer. It
    aggregates various linguistic signals—lexical, sequential, morphological,
    and syntactic—into a single, unified cost. This cost is then used by the
    dynamic programming decoder (`_dp_decode` in ScalableTokenizer) to evaluate
    potential token segmentations.

    By incorporating these features, the tokenizer can make more linguistically
    informed decisions beyond purely statistical metrics like frequency or PMI.
    For example, it can be encouraged to:
    - Keep known multi-word expressions like "New York" together.
    - Recognize that a sequence of capitalized words is likely a single name.
    - Favor tokens that are morphologically consistent with the language of
      the surrounding text.
    """
    def __init__(self,
                 lexicon=None, mwe=None, ne_gaz=None,
                 token_bigram=None, lm_token_prob=None,
                 paragraph_lang=None,
                 gamma_boundary=10,
                 mu_morph=0.2,
                 prefix_reward=0.025,
                 suffix_reward=0.01,
                 space_penalty=0.1,
                 email_reward=0.0,
                 url_reward=0.0,
                 hashtag_reward=0.0,
                 morphology_kwargs=None,
                 uniseg_root=None,
                 uniseg_reward=0.1):
        """
        Attributes:
            lexicon (dict): A dictionary mapping known tokens to a score (reward).
            mwe (set): A set of known multi-word expressions.
            ne_gaz (dict): A named entity gazetteer, mapping entity types (e.g., "LOC")
                to sets of known entities.
            token_bigram (dict): A dictionary defining costs for transitions between
                token classes (e.g., from "InitCap" to "InitCap").
            lm_token_prob (callable | dict): An external language model providing token
                probabilities.
            paragraph_lang (callable): A function that returns the language for a given
                paragraph index.
            gamma_boundary (float): A penalty for changing token classes, which
                encourages sequences of similar token types.
            mu_morph (float): The weight applied to the score from the morphology
                encoder.
            prefix_reward (float): A small reward for tokens that contain known prefixes.
            suffix_reward (float): A small reward for tokens that contain known suffixes.
            morph_encoder (MorphologyEncoder | None): An instance of the morphology
                encoder, which is trained to score the morphological "fit" of a
                token for a given language.
            uniseg_root (str | Path | None): Root directory for UniSegments data.
                If provided, enables boundary alignment rewards during DP decoding.
            uniseg_reward (float): Reward (negative cost) for token boundaries that
                align with gold morpheme boundaries from UniSegments.
        """
        
        self.lexicon = lexicon or {}
        self.mwe = mwe or set()
        self.ne_gaz = ne_gaz or {}
        self.token_bigram = token_bigram or {}
        self.lm_token_prob = lm_token_prob
        self.paragraph_lang = paragraph_lang
        self.gamma_boundary = gamma_boundary
        self.mu_morph = mu_morph
        self.prefix_reward = prefix_reward
        self.suffix_reward = suffix_reward
        self.space_penalty = space_penalty
        self.email_reward = email_reward
        self.url_reward = url_reward
        self.hashtag_reward = hashtag_reward
        self.morph_encoder = None
        self.morphology_kwargs = morphology_kwargs or {}
        
        # UniSeg boundary alignment - uses dedicated loader
        self.uniseg_reward = uniseg_reward
        self._uniseg_loader = UniSegLoader(uniseg_root) if uniseg_root else None

    def create_morph_encoder(self):
        return MorphologyEncoder(**self.morphology_kwargs)

    # ---- UniSeg boundary alignment methods ----
    # All methods now delegate to UniSegLoader for JSONL file access
    
    def load_uniseg_for_lang(self, lang: str) -> bool:
        """Load UniSeg morpheme boundaries for a language.
        
        Delegates to UniSegLoader which reads from JSONL files.
        
        Args:
            lang: ISO 639-1 language code (e.g., 'en', 'de')
            
        Returns:
            True if data was loaded successfully, False otherwise.
        """
        if self._uniseg_loader is None:
            return False
        return self._uniseg_loader.load_language(lang)
    
    def get_uniseg_boundaries(self, word: str, lang: str) -> Optional[Set[int]]:
        """Get gold morpheme boundaries for a word from UniSeg JSONL.
        
        Args:
            word: The word to look up (case-insensitive)
            lang: ISO 639-1 language code
            
        Returns:
            Set of character positions where morpheme boundaries occur,
            or None if word is not in UniSeg database.
        """
        if self._uniseg_loader is None:
            return None
        return self._uniseg_loader.get_boundaries(word, lang)
    
    def boundary_alignment_reward(self, word: str, token_start: int, token_end: int, lang: str) -> float:
        """Calculate reward for a token that aligns with morpheme boundaries.
        
        Args:
            word: The full word being tokenized
            token_start: Start position of token within word
            token_end: End position of token within word
            lang: Language code
            
        Returns:
            Reward value (positive = good alignment). Returns reward only if
            BOTH start and end align with valid boundaries (word edges or
            internal morpheme splits).
        """
        word_len = len(word)
        
        # Get internal morpheme boundaries
        gold_boundaries = self.get_uniseg_boundaries(word, lang)
        
        # Valid positions include: 0, word_len, and any internal morpheme boundaries
        valid_positions = {0, word_len}
        if gold_boundaries:
            valid_positions.update(gold_boundaries)
        
        # Reward only if BOTH start and end are valid
        if token_start in valid_positions and token_end in valid_positions:
            return self.uniseg_reward
        
        return 0.0
    
    def precompute_paragraph_boundaries(self, text: str, lang: str) -> Set[int]:
        """Precompute all valid morpheme boundaries for a paragraph.
        
        This extracts words from the paragraph, looks up each word in the UniSeg
        JSONL database, and returns all positions where a valid token boundary could be:
        - Word start positions (beginning of each word)
        - Word end positions (end of each word)  
        - Internal morpheme boundaries (from UniSeg JSONL data)
        
        Args:
            text: The full paragraph text
            lang: Language code
            
        Returns:
            Set of character positions in the paragraph where valid token
            boundaries can occur. A token is morpheme-aligned if BOTH its
            start AND end are in this set.
        """
        if not text or not lang or self._uniseg_loader is None:
            return set()
        
        gold_positions: Set[int] = set()
        
        # Simple word extraction: split on whitespace and punctuation
        import re
        word_pattern = re.compile(r'\b\w+\b', re.UNICODE)
        
        for match in word_pattern.finditer(text):
            word = match.group()
            word_start = match.start()
            word_end = match.end()
            
            # Add word boundaries (start and end of each word)
            gold_positions.add(word_start)
            gold_positions.add(word_end)
            
            # Add internal morpheme boundaries from UniSeg JSONL
            word_boundaries = self._uniseg_loader.get_boundaries(word, lang)
            if word_boundaries:
                for b in word_boundaries:
                    paragraph_pos = word_start + b
                    if word_start < paragraph_pos < word_end:
                        gold_positions.add(paragraph_pos)
        
        return gold_positions

    @staticmethod
    def token_class(tok: str) -> str:
        """Classifies a token into a predefined category (e.g., URL, PUNCT)."""
        if tok in ('.', '!', '?'): return "EOS"
        if tok in (',', ';', ':'): return "PUNCT"
        if URL_RE.search(tok):     return "URL"
        if EMAIL_RE.fullmatch(tok):return "EMAIL"
        if NUM_RE.fullmatch(tok):  return "NUM"
        if EMOJI_RE.search(tok):   return "EMOJI"
        if tok.istitle():          return "InitCap"
        if tok.isupper():          return "ALLCAPS"
        if tok.islower():          return "lower"
        if '-' in tok:             return "hyphen"
        return "other"

    def lm_neglogp(self, token: str) -> float:
        """Calculates the cost contribution from an external language model."""
        if self.lm_token_prob is None: return 0.0
        p = self.lm_token_prob(token) if callable(self.lm_token_prob) else self.lm_token_prob.get(token, None)
        return -math.log(max(p, EPS)) if p and p > 0 else 0.0

    def _affix_bias(self, token: str, lang: str) -> float:
        """Calculates a small reward for tokens matching affixes from UniSeg JSONL.
        
        This uses prefixes/suffixes extracted from the UniSeg JSONL files.
        If UniSeg data isn't available, falls back to the legacy AFFIXES dict.
        """
        # Get affixes from UniSeg loader
        if self._uniseg_loader is not None:
            suffixes = self._uniseg_loader.get_suffixes(lang)
            prefixes = self._uniseg_loader.get_prefixes(lang)
        else:
            suffixes = set()
            prefixes = set()
        
        # Fallback to hardcoded AFFIXES if UniSeg has no data
        if not suffixes and not prefixes:
            suffixes = set(AFFIXES.get(lang, {}).get("suf", []))
            prefixes = set(AFFIXES.get(lang, {}).get("pre", []))
        
        b = 0.0
        token_lower = token.lower()
        
        # Check suffix matches
        if suffixes:
            for suf in suffixes:
                if token_lower.endswith(suf) and len(token_lower) > len(suf):
                    b += self.suffix_reward
                    break
        
        # Check prefix matches
        if prefixes:
            for pre in prefixes:
                if token_lower.startswith(pre) and len(token_lower) > len(pre):
                    b += self.prefix_reward
                    break
        
        return b
    
    def get_uniseg_affixes(self, lang: str) -> Dict[str, Set[str]]:
        """Get the affixes extracted from UniSeg JSONL for a language.
        
        Returns:
            Dict with 'prefixes' and 'suffixes' sets, or empty sets if not loaded.
        """
        if self._uniseg_loader is None:
            return {"prefixes": set(), "suffixes": set()}
        
        return {
            "prefixes": self._uniseg_loader.get_prefixes(lang),
            "suffixes": self._uniseg_loader.get_suffixes(lang),
        }

    def intrinsic_linguistic_cost(self, token: str, lang: str = None) -> float:
        """
        Calculates the portion of the linguistic cost that is context-free.
        """

        c = 0.0
        # 1. Lexicon and Named Entity costs (rewards)
        if token in self.lexicon:
            c += -1.0 * self.lexicon[token]
        if token in self.mwe:     c += -1.0
        for _tag, entity_dict in self.ne_gaz.items():
            if token in entity_dict:
                # Look up the specific reward score for the token
                reward_score = entity_dict[token]
                # Apply it as a negative cost
                c += -1.0 * reward_score
        # 2. Penalize internal spaces
        c += self.space_penalty * token.strip().count(" ")

        # 2b. Structural rewards for e-mail, URLs, hashtags
        if self.email_reward and EMAIL_RE.fullmatch(token):
            c += self.email_reward
        if self.url_reward and URL_RE.search(token):
            c += self.url_reward
        if self.hashtag_reward and token.startswith("#") and len(token) > 1:
            c += self.hashtag_reward

        if lang:
            # 3. Morphology-based costs
            if self.morph_encoder is not None:
                morph_fit = self.morph_encoder.score(token, lang)
                consistency_bonus = self.morph_encoder.consistency_bonus(token, lang)
                #print(f"{token}: {-self.mu_morph*morph_fit:.2f}, {-consistency_bonus:.2f}")
                c += - (self.mu_morph * morph_fit + consistency_bonus)

            # 4. reward for prefix/suffix
            c += -self._affix_bias(token, lang)

            # 5. Script consistency
            token_script = utils.script_guess(token)
            if token_script != "other" and token_script != lang:
                c += 0.05

        # Note: We cannot add token_bigram or prev_class costs here.
        return c
    # Add this new method inside the ScalableTokenizer class

    def _calculate_average_linguistic_cost(self, token: str, occurrences: list) -> float:
        """
        Calculates the average intrinsic linguistic cost for a token over all its
        occurrences in the corpus.

        This is the "Average Score" strategy. It provides a highly accurate,
        context-aware cost for the candidate proposal phase.

        Args:
            token (str): The candidate token string.
            occurrences (list): A list of (paragraph_idx, start_pos) tuples.

        Returns:
            float: The average linguistic cost.
        """
        if not occurrences:
            return 0.0

        total_linguistic_cost = 0.0
        valid_occurrences = 0

        for (pi, _start) in occurrences:
            # For each occurrence, get the language of its paragraph.
            lang = self.paragraph_lang(pi)

            # We can only score it if the language is known.
            if lang:
                # Call the comprehensive cost function from the LinguisticModels class.
                # This function includes morphology, affixes, lexicon checks, etc.
                cost = self.intrinsic_linguistic_cost(token, lang=lang)
                total_linguistic_cost += cost
                valid_occurrences += 1

        # Return the average cost.
        if valid_occurrences == 0:
            return 0.0
        return total_linguistic_cost / valid_occurrences

    def additive_cost(self, token: str, prev_class: str, paragraph_idx: int = None) -> float:
        """Aggregates all linguistic feature costs for a token in context.

        This is the primary method used by the main DP decoder. It calculates a
        single cost value by summing up contributions from all enabled linguistic
        models. A negative cost represents a reward or bonus.

        Args:
            token (str): The current token being evaluated.
            prev_class (str): The class of the preceding token.
            paragraph_idx (int): The index of the paragraph, used to get its language.

        Returns:
            float: The total additive linguistic cost.
        """
        c = 0.0

        # --- Token sequence costs (bigrams) ---
        tc = self.token_class(token)
        #print(token, prev_class, tc, self.token_bigram.get((prev_class, tc), 0.0))
        c += self.token_bigram.get((prev_class, tc), 0.0)

        # --- External Language Model cost ---
        c += self.lm_neglogp(token)

        # --- Morphology-based costs ---
        lang = None
        if paragraph_idx is not None and self.paragraph_lang:
            lang = self.paragraph_lang(paragraph_idx)
        c += self.intrinsic_linguistic_cost(token, lang)

        # --- Boundary and Script costs ---
        # Penalty for switching token types.
        if prev_class is not None and prev_class != tc:
            c += self.gamma_boundary

        return c

    def batch_additive_cost(
        self,
        token: str,
        prev_class_indices,
        class_list,
        paragraph_idx: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Vectorized additive cost for a token given multiple previous classes.

        Args:
            token: Current token string.
            prev_class_indices: Iterable of class indices from the DP lattice.
            class_list: Full list of class labels (index -> name).
            paragraph_idx: Paragraph index for contextual language lookup.
            device: Torch device for the returned tensor.

        Returns:
            torch.Tensor of shape [len(prev_class_indices)] with additive costs.
        """
        indices = list(prev_class_indices)
        if not indices:
            return torch.empty(0, dtype=torch.float32, device=device)

        tc = self.token_class(token)
        prev_classes = [class_list[i] for i in indices]
        lang = None
        if paragraph_idx is not None and self.paragraph_lang:
            lang = self.paragraph_lang(paragraph_idx)

        lm_cost = self.lm_neglogp(token)
        morph_cost = self.intrinsic_linguistic_cost(token, lang)
        base = lm_cost + morph_cost

        bigram = torch.tensor(
            [self.token_bigram.get((prev, tc), 0.0) for prev in prev_classes],
            dtype=torch.float32,
            device=device,
        )
        boundary = torch.tensor(
            [self.gamma_boundary if (prev is not None and prev != tc) else 0.0 for prev in prev_classes],
            dtype=torch.float32,
            device=device,
        )
        combined = bigram + boundary
        if base != 0.0:
            combined = combined + base
        return combined


# ============================================================================
# (DP helpers removed)
# ============================================================================


def _eigendecomposition_power_operator(
    PPMI: torch.Tensor,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-6,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    Top-k eigenpairs of G = PPMI @ PPMI.T via implicit operator y = PPMI @ (PPMI.T @ x),
    avoiding construction of the dense Gram matrix. Uses power iteration with
    Gram-Schmidt orthogonalization across components.
    """
    if device is None:
        device = PPMI.device
    n = PPMI.shape[0]
    if n == 0 or k <= 0:
        return None, None
    eigvecs_list = []
    eigvals_list = []
    for comp in range(min(k, n)):
        v = torch.randn(n, dtype=torch.float32, device=device)
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-9)
        prev = v
        for it in range(max_iters):
            # Apply operator: G v = PPMI @ (PPMI.T @ v)
            t = PPMI.T @ v
            v_new = PPMI @ t
            # Orthogonalize against previously found components
            if eigvecs_list:
                V_prev = torch.stack(eigvecs_list, dim=1)  # [n, comp]
                proj = V_prev @ (V_prev.T @ v_new)
                v_new = v_new - proj
            # Normalize
            v_new = v_new / torch.clamp(torch.linalg.norm(v_new), min=1e-12)
            if torch.norm(v_new - prev) < tol:
                v = v_new
                break
            prev = v_new
            v = v_new
        # Rayleigh quotient for eigenvalue
        t = PPMI.T @ v
        Gv = PPMI @ t
        lambda_val = torch.dot(v, Gv).item()
        eigvals_list.append(lambda_val)
        eigvecs_list.append(v)
    eigvals = torch.tensor(eigvals_list, dtype=torch.float32, device=device)
    eigvecs = torch.stack(eigvecs_list, dim=1) if eigvecs_list else None
    return eigvals, eigvecs


# (DP semantic alignment helper removed)
