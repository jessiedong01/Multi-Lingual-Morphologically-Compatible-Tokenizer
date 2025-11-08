import math
import numpy as np
from collections import Counter, defaultdict
from constants import *
from utils import *
import utils

def char_ngrams(s, n=(2,3,4)):
    """Extracts character n-grams of specified orders from a string.

    Args:
        s (str): The input string.
        n (tuple): A tuple of integers specifying the n-gram orders to extract.

    Returns:
        Counter: A counter mapping each n-gram to its frequency in the string.
    """
    feats = Counter()
    for k in n:
        for i in range(len(s)-k+1):
            feats[s[i:i+k]] += 1
    return feats

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
                 embedding_mode="ppmi",
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
                 use_minibatch=False,
                 batch_size_pairs=2048,
                 batch_size_edges=512,
                 optimizer="sgd",
                 adagrad_eps=1e-8,
                 adagrad_reset=None):
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
        # Optimizer can be 'sgd' for constant steps or 'adagrad' to adapt rare pairs.
        # Choose 'sgd' when you want smoother convergence on small corpora; use 'adagrad'
        # for large, diverse data where rare morphs need extra emphasis.
        self.optimizer = optimizer
        self.adagrad_eps = adagrad_eps
        self.adagrad_reset = adagrad_reset
        self._adagrad_W = None
        self._adagrad_C = None
        self._adagrad_steps = 0

    def set_embedding_mode(self, mode: str):
        """Switches between 'ppmi' and 'glove' embedding training."""
        mode = mode.lower()
        if mode not in {"ppmi", "glove"}:
            raise ValueError(f"Unsupported embedding mode '{mode}'. Use 'ppmi' or 'glove'.")
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
        """Converts a token into a complete feature set (n-grams + affixes)."""
        f = Counter()
        f.update(char_ngrams(tok, self.ngram_orders))
        f.update(self._affix_feats(tok, lang))
        return f

    def _ensure_adagrad_buffers(self, W_shape=None, C_shape=None):
        """Initialises or refreshes AdaGrad accumulators when required."""
        if W_shape is not None and ((self._adagrad_W is None) or (self._adagrad_W.shape != W_shape)):
            self._adagrad_W = np.zeros(W_shape, dtype=np.float32)
        if C_shape is not None and ((self._adagrad_C is None) or (self._adagrad_C.shape != C_shape)):
            self._adagrad_C = np.zeros(C_shape, dtype=np.float32)

    def _maybe_reset_adagrad(self):
        """Optionally resets AdaGrad accumulators on a fixed schedule."""
        if self.adagrad_reset and self._adagrad_steps >= self.adagrad_reset:
            self._adagrad_W.fill(0.0)
            self._adagrad_C.fill(0.0)
            self._adagrad_steps = 0

    def _apply_optimizer_step(self, matrix, indices, grads, lr, is_token=True):
        """Applies an optimisation step (SGD or AdaGrad) to selected rows.

        SGD simply subtracts lr * grad; AdaGrad divides by the running rms."""
        if not indices:
            return
        opt = (self.optimizer or "sgd").lower()
        idx = np.asarray(indices, dtype=np.int32)
        grad_arr = np.asarray(grads, dtype=np.float32)
        if grad_arr.ndim == 1:
            grad_arr = grad_arr[None, :]
        if opt == "adagrad":
            if is_token:
                self._adagrad_W[idx] += grad_arr ** 2
                denom = np.sqrt(self._adagrad_W[idx]) + self.adagrad_eps
            else:
                self._adagrad_C[idx] += grad_arr ** 2
                denom = np.sqrt(self._adagrad_C[idx]) + self.adagrad_eps
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
        return [np.array(members, dtype=np.int32)
                for members in class_members.values()
                if len(members) >= 2]

    def _refine_embeddings(self, V, target, equiv_sets, tok_lang_list=None, lang_similarity=None):
        """Refines embeddings with the localized morphological regularizer."""
        if self.refine_steps <= 0:
            return V
        if self.lambda_morph <= 0 and self.gamma <= 0:
            return V
        if self.lambda_morph <= 0 or not equiv_sets:
            # No localized Laplacians to apply; only ridge if gamma > 0.
            apply_lap = False
        else:
            apply_lap = True

        for _ in range(self.refine_steps):
            grad = 4.0 * ((V @ V.T - target) @ V)
            if apply_lap:
                lap_grad = np.zeros_like(V)
                for idxs in equiv_sets:
                    if lang_similarity and tok_lang_list is not None:
                        for pos, idx in enumerate(idxs):
                            lang_i = tok_lang_list[idx]
                            weighted_sum = 0.0
                            weight_total = 0.0
                            for pos2, idx2 in enumerate(idxs):
                                if pos2 == pos:
                                    continue
                                lang_j = tok_lang_list[idx2]
                                weight = lang_similarity.get((lang_i, lang_j), lang_similarity.get((lang_j, lang_i), 1.0))
                                weighted_sum += weight * V[idx2]
                                weight_total += weight
                            lap_grad[idx] += weight_total * V[idx] - weighted_sum
                    else:
                        subset = V[idxs]
                        sum_vec = subset.sum(axis=0, keepdims=True)
                        lap_grad[idxs] += len(idxs) * subset - sum_vec
                grad += 2.0 * self.lambda_morph * lap_grad
            if self.gamma > 0:
                grad += 2.0 * self.gamma * V
            V = V - self.refine_lr * grad
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            V = V / np.maximum(norms, 1e-12)
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

        # --- 2. Build features and choose dense vs streaming path ---
        all_feats = set()
        for tok, lang in tok_lang.items():
            all_feats.update(self._featurize(tok, lang).keys())
        self.feat2id = {f: i for i, f in enumerate(sorted(all_feats))}
        F = len(self.feat2id)

        toks = list(tok_lang.keys())
        T = len(toks)
        if T == 0 or F == 0:
            return

        equiv_sets = self._equiv_sets(toks, tok_lang)
        tok_lang_list = [tok_lang[tok] for tok in toks]

        # Heuristic: for GloVe/minibatch (or very large problems), avoid dense X/PPMI.
        use_streaming = (self.embedding_mode == "glove") and (self.use_minibatch or (T * F > 5_000_000))

        if use_streaming:
            # Build sparse pairs and marginals without allocating (T x F).
            rows, cols, counts, row_sum, col_sum, total = [], [], [], np.zeros(T, dtype=np.float64), np.zeros(F, dtype=np.float64), 0.0
            for ti, tok in enumerate(toks):
                lang = tok_lang[tok]
                feats = self._featurize(tok, lang)
                if not feats:
                    continue
                for f, v in feats.items():
                    fj = self.feat2id.get(f)
                    if fj is None:
                        continue
                    rows.append(ti)
                    cols.append(fj)
                    counts.append(float(v))
                    row_sum[ti] += float(v)
                    col_sum[fj] += float(v)
                    total += float(v)

            if not rows:
                return

            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            counts = np.asarray(counts, dtype=np.float32)
            row_sum = np.asarray(row_sum, dtype=np.float64)
            col_sum = np.asarray(col_sum, dtype=np.float64)
            total = float(total) if total > 0 else 1.0

            # Compute PPMI targets per pair without dense matrices.
            p_xy = counts / total
            p_x = row_sum[rows] / total
            p_y = col_sum[cols] / total
            with np.errstate(divide='ignore', invalid='ignore'):
                pmi = np.log(np.maximum(p_xy / np.maximum(p_x * p_y, self.pmi_floor), self.pmi_floor))
            targets = np.maximum(pmi, 0.0).astype(np.float32)

            print(f"[MorphEncoder] embedding_mode='glove' (streaming), tokens={T}, features={F}, pairs={rows.size}")
            V = self._fit_glove_embeddings_from_pairs(rows, cols, counts, targets, T, F, tok_lang_list, equiv_sets)
        else:
            # Dense path for small problems or explicit PPMI mode.
            X = np.zeros((T, F), dtype=np.float32)
            for ti, tok in enumerate(toks):
                lang = tok_lang[tok]
                for f, v in self._featurize(tok, lang).items():
                    X[ti, self.feat2id[f]] = v

            PPMI = self._compute_ppmi(X)
            print(f"[MorphEncoder] embedding_mode='{self.embedding_mode}', tokens={len(toks)}, features={F}")
            if self.embedding_mode == "glove":
                V = self._fit_glove_embeddings(X, PPMI, tok_lang_list, equiv_sets)
            else:
                V = self._fit_ppmi_embeddings(PPMI, tok_lang_list, equiv_sets)

        if V is None:
            return

        V = V.astype(np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.maximum(norms, 1e-9)

        # --- 5. Store the learned token vectors and language prototypes ---
        for ti, tok in enumerate(toks):
            self.token_vec[tok] = V[ti]

        for lang in set(tok_lang.values()):
            idxs = [ti for ti, tok in enumerate(toks) if tok_lang[tok] == lang]
            if idxs:
                lp = V[idxs].mean(0)
                self.lang_proto[lang] = lp / (np.linalg.norm(lp) + 1e-9)

        # --- 6. Pre-calculate counts for the consistency bonus ---
        for tok, lang in tok_lang.items():
            for key in self._morph_keys(tok, lang):
                self.shared_counts[key][lang] += 1
        print("[MorphEncoder] Training complete; stored token vectors:", len(self.token_vec))

    def _compute_ppmi(self, X):
        col_sum = np.maximum(X.sum(0, keepdims=True), self.pmi_floor)
        row_sum = np.maximum(X.sum(1, keepdims=True), self.pmi_floor)
        total = float(X.sum()) + 1e-9
        P_xy = X / total
        P_x = row_sum / total
        P_y = col_sum / total
        PMI = np.log(np.maximum(P_xy / (P_x @ P_y), self.pmi_floor))
        PPMI = np.maximum(PMI, 0)
        return PPMI

    def _fit_ppmi_embeddings(self, PPMI, tok_lang_list, equiv_sets):
        """Produces embeddings using the original PPMI + eigendecomposition routine."""
        G = PPMI @ PPMI.T
        G = (G + G.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1][:min(self.k, G.shape[0])]
        if idx.size == 0:
            return None
        V = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))[None, :]
        V = self._refine_embeddings(V, G, equiv_sets, tok_lang_list, self.lang_similarity if self.use_weighted_cross else None)
        cross_pairs = self._collect_cross_pairs(equiv_sets, tok_lang_list)
        V = self._apply_cross_consistency(V, tok_lang_list, cross_pairs)
        return V

    def _build_morph_graph(self, equiv_sets, n_tokens, tok_lang_list, lang_similarity):
        """Pre-computes degree and neighbor lists for Laplacian gradients."""
        deg = np.zeros(n_tokens, dtype=np.float32)
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
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        src = np.array([e[0] for e in edges], dtype=np.int32)
        dst = np.array([e[1] for e in edges], dtype=np.int32)
        weights = np.array([e[2] for e in edges], dtype=np.float32)
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
                self.semantic_proj[lang] = np.eye(self.k, dtype=np.float32)

    def _apply_semantic_consistency(self, V, tok_lang_list, cross_pairs):
        languages = set(tok_lang_list)
        self._ensure_semantic_proj(languages)
        for _ in range(self.semantic_iters):
            for i, j, weight in cross_pairs:
                lang_i = tok_lang_list[i]
                lang_j = tok_lang_list[j]
                phi_i = self.semantic_proj[lang_i]
                phi_j = self.semantic_proj[lang_j]
                v_i = phi_i @ V[i]
                v_j = phi_j @ V[j]
                diff = v_i - v_j
                grad_i = 2.0 * weight * np.outer(diff, V[i])
                grad_j = -2.0 * weight * np.outer(diff, V[j])
                self.semantic_proj[lang_i] -= self.semantic_lr * grad_i
                self.semantic_proj[lang_j] -= self.semantic_lr * grad_j
        transformed = np.zeros_like(V)
        for idx, lang in enumerate(tok_lang_list):
            transformed[idx] = self.semantic_proj[lang] @ V[idx]
        return transformed

    def _apply_structure_mapping(self, V, tok_lang_list, cross_pairs):
        for _ in range(self.structure_iters):
            for i, j, weight in cross_pairs:
                lang_i = tok_lang_list[i]
                lang_j = tok_lang_list[j]
                M_ij = self.structure_maps.setdefault((lang_i, lang_j), np.eye(self.k, dtype=np.float32))
                M_ji = self.structure_maps.setdefault((lang_j, lang_i), np.eye(self.k, dtype=np.float32))
                diff_ij = M_ij @ V[i] - V[j]
                grad_M_ij = 2.0 * weight * np.outer(diff_ij, V[i])
                M_ij -= self.structure_lr * grad_M_ij
                V[i] -= self.structure_lr * weight * (M_ij.T @ diff_ij)
                V[j] += self.structure_lr * weight * diff_ij
                diff_ji = M_ji @ V[j] - V[i]
                grad_M_ji = 2.0 * weight * np.outer(diff_ji, V[j])
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
        x = np.asarray(x, dtype=np.float32)
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.maximum(exp_x.sum(), 1e-9)

    def _fit_glove_embeddings(self, X_counts, PPMI, tok_lang_list, equiv_sets):
        """Learns embeddings with a lightweight (possibly mini-batched) GloVe objective."""
        rows, cols = np.nonzero(PPMI)
        if rows.size == 0:
            return None
        counts = X_counts[rows, cols].astype(np.float32)
        targets = PPMI[rows, cols].astype(np.float32)
        pair_count = len(rows)
        if pair_count > self.glove_max_pairs:
            sample_idx = np.random.choice(pair_count, self.glove_max_pairs, replace=False)
            rows = rows[sample_idx]
            cols = cols[sample_idx]
            counts = counts[sample_idx]
            targets = targets[sample_idx]
            pair_count = len(rows)

        positive_mask = counts > 0
        if not np.all(positive_mask):
            rows = rows[positive_mask]
            cols = cols[positive_mask]
            counts = counts[positive_mask]
            targets = targets[positive_mask]
            pair_count = len(rows)
            if pair_count == 0:
                return None

        xmax = float(self.glove_xmax)
        alpha = float(self.glove_alpha)
        weights = np.ones(pair_count, dtype=np.float32)
        mask = counts < xmax
        weights[mask] = (counts[mask] / xmax) ** alpha

        T, F = X_counts.shape
        W = np.random.normal(scale=0.01, size=(T, self.k)).astype(np.float32)
        C = np.random.normal(scale=0.01, size=(F, self.k)).astype(np.float32)

        lr = self.glove_lr
        opt = (self.optimizer or "sgd").lower()
        if opt == "adagrad":
            self._ensure_adagrad_buffers(W.shape, C.shape)
            self._adagrad_steps = 0
        else:
            self._adagrad_W = None
            self._adagrad_C = None
            self._adagrad_steps = 0

        lang_similarity = self.lang_similarity if self.use_weighted_cross else None

        deg = neighbors = None
        edge_src = edge_dst = edge_weight = None
        total_edges = 0
        if self.lambda_morph > 0 and equiv_sets:
            if self.use_minibatch:
                edge_src, edge_dst, edge_weight = self._collect_morph_edges(equiv_sets, tok_lang_list, lang_similarity)
                total_edges = edge_src.size
            else:
                deg, neighbors = self._build_morph_graph(equiv_sets, T, tok_lang_list, lang_similarity)

        cross_pairs = self._collect_cross_pairs(equiv_sets, tok_lang_list)

        if self.use_minibatch:
            batch_size = max(1, min(self.batch_size_pairs, pair_count))
            edge_batch = max(0, int(self.batch_size_edges))
            print(f"[MorphEncoder] Using mini-batch GloVe: pairs={pair_count}, batch_size={batch_size}, edges={total_edges}")
            for epoch in range(self.glove_iters):
                print(f"[MorphEncoder]  Epoch {epoch + 1}/{self.glove_iters}")
                order = np.random.permutation(pair_count)
                for start in range(0, pair_count, batch_size):
                    batch_idx = order[start:start + batch_size]
                    tokens = rows[batch_idx]
                    features = cols[batch_idx]
                    targets_b = targets[batch_idx]
                    weights_b = weights[batch_idx]

                    W_batch = W[tokens]
                    C_batch = C[features]
                    preds = np.einsum("ik,ik->i", W_batch, C_batch)
                    diffs = preds - targets_b
                    grad_scalar = weights_b * diffs

                    grad_w = defaultdict(lambda: np.zeros(self.k, dtype=np.float32))
                    grad_c = defaultdict(lambda: np.zeros(self.k, dtype=np.float32))
                    for n, (g, i, j) in enumerate(zip(grad_scalar, tokens, features)):
                        if g == 0.0:
                            continue
                        grad_w[i] += g * C_batch[n]
                        grad_c[j] += g * W_batch[n]

                    for i in list(grad_w.keys()):
                        grad_w[i] += 2.0 * self.gamma * W[i]
                    for j in list(grad_c.keys()):
                        grad_c[j] += 2.0 * self.gamma * C[j]

                    if total_edges > 0 and self.lambda_morph > 0:
                        requested_edges = edge_batch or total_edges
                        current_edges = min(requested_edges, total_edges)
                        if current_edges > 0:
                            replace = requested_edges > total_edges
                            chosen = np.random.choice(total_edges, size=current_edges, replace=replace)
                            scale = float(total_edges) / float(current_edges) if current_edges > 0 else 1.0
                            for idx in chosen:
                                i = int(edge_src[idx])
                                j = int(edge_dst[idx])
                                weight = float(edge_weight[idx])
                                diff_vec = W[i] - W[j]
                                grad_vec = 2.0 * self.lambda_morph * scale * weight * diff_vec
                                grad_w[i] += grad_vec
                                grad_w[j] -= grad_vec

                    if grad_w:
                        token_ids = list(grad_w.keys())
                        grad_mat = np.stack([grad_w[i] for i in token_ids], axis=0)
                        self._apply_optimizer_step(W, token_ids, grad_mat, lr, is_token=True)
                    if grad_c:
                        feat_ids = list(grad_c.keys())
                        grad_feat = np.stack([grad_c[j] for j in feat_ids], axis=0)
                        self._apply_optimizer_step(C, feat_ids, grad_feat, lr, is_token=False)
        else:
            print(f"[MorphEncoder] Using full-batch optimizer='{opt}' with {pair_count} pairs")
            for epoch in range(self.glove_iters):
                print(f"[MorphEncoder]  Epoch {epoch + 1}/{self.glove_iters}")
                order = np.random.permutation(pair_count)
                for idx in order:
                    i = rows[idx]
                    j = cols[idx]
                    x = counts[idx]
                    if x <= 0:
                        continue
                    w_i = W[i].copy()
                    c_j = C[j].copy()
                    target = targets[idx]
                    diff = float(np.dot(w_i, c_j) - target)
                    grad = weights[idx] * diff
                    grad_w = grad * c_j + 2.0 * self.gamma * w_i
                    grad_c = grad * w_i + 2.0 * self.gamma * c_j
                    self._apply_optimizer_step(W, [i], grad_w, lr, is_token=True)
                    self._apply_optimizer_step(C, [j], grad_c, lr, is_token=False)

                if deg is not None and self.lambda_morph > 0:
                    for i in range(T):
                        if deg[i] <= 0:
                            continue
                        weighted_sum = np.zeros_like(W[i])
                        for j, weight in neighbors[i]:
                            weighted_sum += weight * W[j]
                        morph_grad = 2.0 * self.lambda_morph * (deg[i] * W[i] - weighted_sum)
                        self._apply_optimizer_step(W, [i], morph_grad, lr, is_token=True)

        W = self._apply_cross_consistency(W, tok_lang_list, cross_pairs)
        return W

    def _fit_glove_embeddings_from_pairs(self, rows, cols, counts, targets, T, F, tok_lang_list, equiv_sets):
        """GloVe-style training using precomputed sparse pairs (streaming path).

        Args:
            rows, cols (np.ndarray[int32]): Indices for token-feature co-occurrences.
            counts (np.ndarray[float32]): Raw co-occurrence counts per (token, feature).
            targets (np.ndarray[float32]): PPMI targets per pair.
            T (int): Number of tokens.
            F (int): Number of features.
            tok_lang_list (list[str]): Language label per token index.
            equiv_sets (list[np.ndarray]): Morphological equivalence sets over token indices.
        """
        pair_count = int(rows.size)
        if pair_count == 0:
            return None

        # Subsample if necessary
        if pair_count > self.glove_max_pairs:
            sample_idx = np.random.choice(pair_count, self.glove_max_pairs, replace=False)
            rows = rows[sample_idx]
            cols = cols[sample_idx]
            counts = counts[sample_idx]
            targets = targets[sample_idx]
            pair_count = int(rows.size)
            if pair_count == 0:
                return None

        # Ensure strictly positive counts
        positive_mask = counts > 0
        if not np.all(positive_mask):
            rows = rows[positive_mask]
            cols = cols[positive_mask]
            counts = counts[positive_mask]
            targets = targets[positive_mask]
            pair_count = int(rows.size)
            if pair_count == 0:
                return None

        xmax = float(self.glove_xmax)
        alpha = float(self.glove_alpha)
        weights = np.ones(pair_count, dtype=np.float32)
        mask = counts < xmax
        weights[mask] = (counts[mask] / xmax) ** alpha

        # Initialise embeddings
        W = np.random.normal(scale=0.01, size=(T, self.k)).astype(np.float32)
        C = np.random.normal(scale=0.01, size=(F, self.k)).astype(np.float32)

        lr = self.glove_lr
        opt = (self.optimizer or "sgd").lower()
        if opt == "adagrad":
            self._ensure_adagrad_buffers(W.shape, C.shape)
            self._adagrad_steps = 0
        else:
            self._adagrad_W = None
            self._adagrad_C = None
            self._adagrad_steps = 0

        lang_similarity = self.lang_similarity if self.use_weighted_cross else None

        # Morphology regulariser setup
        deg = neighbors = None
        edge_src = edge_dst = edge_weight = None
        total_edges = 0
        if self.lambda_morph > 0 and equiv_sets:
            if self.use_minibatch:
                edge_src, edge_dst, edge_weight = self._collect_morph_edges(equiv_sets, tok_lang_list, lang_similarity)
                total_edges = edge_src.size
            else:
                deg, neighbors = self._build_morph_graph(equiv_sets, T, tok_lang_list, lang_similarity)

        cross_pairs = self._collect_cross_pairs(equiv_sets, tok_lang_list)

        if self.use_minibatch:
            batch_size = max(1, min(self.batch_size_pairs, pair_count))
            edge_batch = max(0, int(self.batch_size_edges))
            print(f"[MorphEncoder] Using mini-batch GloVe (stream): pairs={pair_count}, batch_size={batch_size}, edges={total_edges}")
            for epoch in range(self.glove_iters):
                print(f"[MorphEncoder]  Epoch {epoch + 1}/{self.glove_iters}")
                order = np.random.permutation(pair_count)
                for start in range(0, pair_count, batch_size):
                    batch_idx = order[start:start + batch_size]
                    tokens = rows[batch_idx]
                    features = cols[batch_idx]
                    targets_b = targets[batch_idx]
                    weights_b = weights[batch_idx]

                    W_batch = W[tokens]
                    C_batch = C[features]
                    preds = np.einsum("ik,ik->i", W_batch, C_batch)
                    diffs = preds - targets_b
                    grad_scalar = weights_b * diffs

                    from collections import defaultdict as _dd
                    grad_w = _dd(lambda: np.zeros(self.k, dtype=np.float32))
                    grad_c = _dd(lambda: np.zeros(self.k, dtype=np.float32))
                    for n, (g, i, j) in enumerate(zip(grad_scalar, tokens, features)):
                        if g == 0.0:
                            continue
                        grad_w[int(i)] += g * C_batch[n]
                        grad_c[int(j)] += g * W_batch[n]

                    for i in list(grad_w.keys()):
                        grad_w[i] += 2.0 * self.gamma * W[i]
                    for j in list(grad_c.keys()):
                        grad_c[j] += 2.0 * self.gamma * C[j]

                    if total_edges > 0 and self.lambda_morph > 0:
                        requested_edges = edge_batch or total_edges
                        current_edges = min(requested_edges, total_edges)
                        if current_edges > 0:
                            replace = requested_edges > total_edges
                            chosen = np.random.choice(total_edges, size=current_edges, replace=replace)
                            scale = float(total_edges) / float(current_edges) if current_edges > 0 else 1.0
                            for idx in chosen:
                                i = int(edge_src[idx])
                                j = int(edge_dst[idx])
                                weight = float(edge_weight[idx])
                                diff_vec = W[i] - W[j]
                                grad_vec = 2.0 * self.lambda_morph * scale * weight * diff_vec
                                grad_w[i] += grad_vec
                                grad_w[j] -= grad_vec

                    if grad_w:
                        token_ids = list(grad_w.keys())
                        grad_mat = np.stack([grad_w[i] for i in token_ids], axis=0)
                        self._apply_optimizer_step(W, token_ids, grad_mat, lr, is_token=True)
                    if grad_c:
                        feat_ids = list(grad_c.keys())
                        grad_feat = np.stack([grad_c[j] for j in feat_ids], axis=0)
                        self._apply_optimizer_step(C, feat_ids, grad_feat, lr, is_token=False)
        else:
            print(f"[MorphEncoder] Using full-batch optimizer='{opt}' with {pair_count} pairs (stream)")
            for epoch in range(self.glove_iters):
                print(f"[MorphEncoder]  Epoch {epoch + 1}/{self.glove_iters}")
                order = np.random.permutation(pair_count)
                for idx in order:
                    i = int(rows[idx])
                    j = int(cols[idx])
                    x = float(counts[idx])
                    if x <= 0:
                        continue
                    w_i = W[i].copy()
                    c_j = C[j].copy()
                    target = float(targets[idx])
                    diff = float(np.dot(w_i, c_j) - target)
                    grad = weights[idx] * diff
                    grad_w = grad * c_j + 2.0 * self.gamma * w_i
                    grad_c = grad * w_i + 2.0 * self.gamma * c_j
                    self._apply_optimizer_step(W, [i], grad_w, lr, is_token=True)
                    self._apply_optimizer_step(C, [j], grad_c, lr, is_token=False)

                if deg is not None and self.lambda_morph > 0:
                    for i in range(T):
                        if deg[i] <= 0:
                            continue
                        weighted_sum = np.zeros_like(W[i])
                        for j, weight in neighbors[i]:
                            weighted_sum += weight * W[j]
                        morph_grad = 2.0 * self.lambda_morph * (deg[i] * W[i] - weighted_sum)
                        self._apply_optimizer_step(W, [i], morph_grad, lr, is_token=True)

        W = self._apply_cross_consistency(W, tok_lang_list, cross_pairs)
        return W

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
        if v is None or lp is None: return 0.0
        return float(np.dot(v, lp))

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
                 morphology_kwargs=None):
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

    def create_morph_encoder(self):
        return MorphologyEncoder(**self.morphology_kwargs)

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
        """Calculates a small reward (negative cost) for tokens with known affixes."""
        suf = AFFIXES.get(lang, {}).get("suf", [])
        pre = AFFIXES.get(lang, {}).get("pre", [])
        b = 0.0
        if any(token.endswith(a) for a in suf): b+=self.suffix_reward
        if any(token.startswith(a) for a in pre): b+=self.prefix_reward
        return b

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
