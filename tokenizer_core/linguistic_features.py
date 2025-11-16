import math
from collections import Counter, defaultdict

import torch

from .constants import *
from .utils import *
import tokenizer_core.utils as utils
from .torch_utils import default_device, ensure_tensor, random_choice, randperm

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
                 batch_size_semantic=512,
                 use_dp_semantic=False,
                 use_dp_eig=False,
                 use_iterative_eig=False,
                 optimizer="sgd",
                 adagrad_eps=1e-8,
                 adagrad_reset=None,
                 max_tokens=None,
                 device: str | torch.device | None = None):
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
        self.use_dp_semantic = use_dp_semantic
        self.use_dp_eig = use_dp_eig
        self.use_iterative_eig = use_iterative_eig
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
        self.device = torch.device(device) if device else default_device()

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
        if self.embedding_mode == "glove":
            V = self._fit_glove_embeddings(X, PPMI, tok_lang_list, equiv_sets)
        else:
            V = self._fit_ppmi_embeddings(PPMI, tok_lang_list, equiv_sets)

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
        """Produces embeddings using the original PPMI + eigendecomposition routine."""
        G = PPMI @ PPMI.T
        G = (G + G.T) * 0.5
        
        # Use DP method if requested
        use_dp_eig = getattr(self, 'use_dp_eig', False)
        if use_dp_eig:
            eigvals, eigvecs = _eigendecomposition_dp_incremental(G, self.k, device=self.device)
            selected_vals = torch.clamp(eigvals, min=0.0)
            V = eigvecs * torch.sqrt(selected_vals).unsqueeze(0)
        else:
            # Use iterative method for very large matrices if requested
            use_iterative = getattr(self, 'use_iterative_eig', False) and G.shape[0] > 5000
            if use_iterative:
                eigvals, eigvecs = _eigendecomposition_via_power_iteration(
                    G, self.k, max_iters=100, tol=1e-6, device=self.device
                )
                selected_vals = torch.clamp(eigvals, min=0.0)
                V = eigvecs * torch.sqrt(selected_vals).unsqueeze(0)
            else:
                # Standard eigendecomposition (most efficient for moderate sizes)
                eigvals, eigvecs = torch.linalg.eigh(G)
                idx = torch.argsort(eigvals, descending=True)[: min(self.k, G.shape[0])]
                if idx.numel() == 0:
                    return None
                selected_vals = torch.clamp(eigvals[idx], min=0.0)
                V = eigvecs[:, idx] * torch.sqrt(selected_vals).unsqueeze(0)
        
        V = self._refine_embeddings(V, G, equiv_sets, tok_lang_list, self.lang_similarity if self.use_weighted_cross else None)
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
        
        # Use DP version if requested
        use_dp = getattr(self, 'use_dp_semantic', False)
        if use_dp:
            return _semantic_consistency_dp_alignment(
                V, tok_lang_list, cross_pairs,
                self.semantic_proj, self.semantic_lr, self.semantic_iters,
                device=self.device
            )
        
        # Use batched version if batch_size_semantic is set
        if hasattr(self, 'batch_size_semantic') and self.batch_size_semantic > 0:
            return self._apply_semantic_consistency_batched(V, tok_lang_list, cross_pairs)
        
        # Original sequential version (slower but simpler)
        for _ in range(self.semantic_iters):
            for i, j, weight in cross_pairs:
                lang_i = tok_lang_list[i]
                lang_j = tok_lang_list[j]
                phi_i = self.semantic_proj[lang_i]
                phi_j = self.semantic_proj[lang_j]
                v_i = phi_i @ V[i]
                v_j = phi_j @ V[j]
                diff = v_i - v_j
                grad_i = 2.0 * weight * torch.outer(diff, V[i])
                grad_j = -2.0 * weight * torch.outer(diff, V[j])
                self.semantic_proj[lang_i] -= self.semantic_lr * grad_i
                self.semantic_proj[lang_j] -= self.semantic_lr * grad_j
        transformed = torch.zeros_like(V)
        for idx, lang in enumerate(tok_lang_list):
            transformed[idx] = self.semantic_proj[lang] @ V[idx]
        return transformed
    
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
        """Solves the convex Laplacian-regularized GloVe objective described in the new paradigm."""
        T, F = X_counts.shape
        if T == 0 or F == 0:
            return None

        PPMI = PPMI.to(torch.float32)
        counts = torch.clamp(X_counts.to(torch.float32), min=0.0)

        # Compute fixed subword embeddings S \in R^{F x k}
        k = min(self.k, F)
        try:
            cov = PPMI.T @ PPMI
            cov = (cov + cov.T) * 0.5 + 1e-6 * torch.eye(F, device=self.device)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            idx = torch.argsort(eigvals, descending=True)[:k]
            eigvals = torch.clamp(eigvals[idx], min=1e-8)
            eigvecs = eigvecs[:, idx]
            S_core = eigvecs * torch.sqrt(eigvals).unsqueeze(0)
        except RuntimeError:
            S_core = torch.randn(F, k, device=self.device) * 0.01

        if k < self.k:
            pad = torch.zeros(F, self.k - k, dtype=torch.float32, device=self.device)
            S = torch.cat([S_core, pad], dim=1)
        else:
            S = S_core

        # Pre-compute per-token normal equations H_t and rhs g_t
        H_blocks = torch.zeros((T, self.k, self.k), dtype=torch.float32, device=self.device)
        rhs = torch.zeros((T, self.k), dtype=torch.float32, device=self.device)
        identity_k = torch.eye(self.k, dtype=torch.float32, device=self.device)
        for t in range(T):
            w_t = counts[t]
            mask = w_t > 0
            if mask.any():
                S_sel = S[mask]
                w_sel = w_t[mask].unsqueeze(1)
                H_blocks[t] = (S_sel * w_sel).T @ S_sel
                rhs[t] = ((w_t[mask] * PPMI[t, mask]).unsqueeze(0) @ S_sel).squeeze(0)
            if self.gamma > 0:
                H_blocks[t] += self.gamma * identity_k
            else:
                H_blocks[t] += 1e-6 * identity_k  # ensure positive definiteness

        cross_pairs = self._collect_cross_pairs(equiv_sets, tok_lang_list)
        lambda_reg = float(self.lambda_morph) if self.lambda_morph > 0 else 0.0

        def apply_operator(U: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(U)
            for idx in range(T):
                result[idx] = H_blocks[idx] @ U[idx]
            if lambda_reg > 0:
                result += lambda_reg * self._laplacian_matvec(U, equiv_sets, cross_pairs)
            return result

        def conjugate_gradient(b: torch.Tensor, max_iters: int = 512, tol: float = 1e-6) -> torch.Tensor:
            x = torch.zeros_like(b)
            r = b - apply_operator(x)
            p = r.clone()
            rs_old = torch.sum(r * r)
            if rs_old.item() <= tol:
                return x
            b_norm = torch.sqrt(torch.sum(b * b)) + 1e-12
            for _ in range(max_iters):
                Ap = apply_operator(p)
                alpha = rs_old / (torch.sum(p * Ap) + 1e-12)
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = torch.sum(r * r)
                if torch.sqrt(rs_new) <= tol * b_norm:
                    break
                p = r + (rs_new / (rs_old + 1e-12)) * p
                rs_old = rs_new
            return x

        U = conjugate_gradient(rhs)
        U = U.to(torch.float32)

        if cross_pairs:
            U = self._apply_cross_consistency(U, tok_lang_list, cross_pairs)
        return U

    def _laplacian_matvec(self, U: torch.Tensor, equiv_sets, cross_pairs):
        out = torch.zeros_like(U)
        for idxs in equiv_sets:
            if idxs.numel() < 2:
                continue
            subset = U[idxs]
            sum_subset = subset.sum(dim=0, keepdim=True)
            m = subset.shape[0]
            out[idxs] += m * subset - sum_subset
        for i, j, weight in cross_pairs:
            w = float(weight)
            if w == 0.0:
                continue
            diff = U[i] - U[j]
            out[i] += w * diff
            out[j] -= w * diff
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


# ============================================================================
# Helper functions for DP and iterative eigendecomposition
# ============================================================================

def _eigendecomposition_dp_incremental(
    G: torch.Tensor,
    k: int,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition using Dynamic Programming approach.
    
    Frames the problem as: find optimal sequence of rank-1 updates to approximate G.
    This is essentially power iteration with DP-style memoization of intermediate results.
    
    Args:
        G: Symmetric Gram matrix [n, n]
        k: Number of eigenvectors to compute
        device: Torch device
    
    Returns:
        eigvals: Top-k eigenvalues [k]
        eigvecs: Top-k eigenvectors [n, k]
    """
    if device is None:
        device = G.device
    
    n = G.shape[0]
    eigvals = []
    eigvecs = []
    
    # Residual matrix (what we haven't explained yet)
    R = G.clone()
    
    for comp in range(min(k, n)):
        # Current best eigenvector (via power iteration on residual)
        v = torch.randn(n, dtype=torch.float32, device=device)
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-9)
        
        # Power iteration on residual (this is the "DP step")
        prev_error = float('inf')
        for it in range(100):
            v_new = R @ v
            v_new = v_new / torch.clamp(torch.linalg.norm(v_new), min=1e-9)
            
            # Check convergence
            if torch.norm(v_new - v) < 1e-6:
                break
            v = v_new
            
            # Compute current approximation error (DP cost)
            approx = torch.outer(v, v) * (v.T @ R @ v)
            error = torch.norm(R - approx).item()
            if abs(error - prev_error) < 1e-6:
                break
            prev_error = error
        
        # Extract eigenvalue via Rayleigh quotient
        lambda_val = (v.T @ R @ v).item()
        eigvals.append(lambda_val)
        eigvecs.append(v)
        
        # Update residual (DP transition)
        if comp < k - 1:
            R = R - lambda_val * torch.outer(v, v)
    
    eigvals = torch.tensor(eigvals, dtype=torch.float32, device=device)
    eigvecs = torch.stack(eigvecs, dim=1)  # [n, k]
    
    return eigvals, eigvecs


def _eigendecomposition_via_power_iteration(
    G: torch.Tensor,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-6,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate eigendecomposition using power iteration (iterative method).
    
    More memory-efficient than full eigendecomposition for large matrices.
    
    Args:
        G: Symmetric Gram matrix [n, n]
        k: Number of top eigenvectors to compute
        max_iters: Maximum iterations per eigenvector
        tol: Convergence tolerance
        device: Torch device
    
    Returns:
        eigvals: Top-k eigenvalues [k]
        eigvecs: Top-k eigenvectors [n, k]
    """
    if device is None:
        device = G.device
    
    n = G.shape[0]
    eigvals = []
    eigvecs = []
    
    # Deflated matrix (we'll subtract found components)
    G_deflated = G.clone()
    
    for comp in range(min(k, n)):
        # Initialize random vector
        v = torch.randn(n, dtype=torch.float32, device=device)
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-9)
        
        for it in range(max_iters):
            v_old = v.clone()
            v = G_deflated @ v
            v = v / torch.clamp(torch.linalg.norm(v), min=1e-9)
            
            # Check convergence
            if torch.norm(v - v_old) < tol:
                break
        
        # Compute eigenvalue via Rayleigh quotient
        lambda_val = (v.T @ G @ v).item()
        eigvals.append(lambda_val)
        eigvecs.append(v)
        
        # Deflate: remove this component from G
        if comp < k - 1:
            G_deflated = G_deflated - lambda_val * torch.outer(v, v)
    
    eigvals = torch.tensor(eigvals, dtype=torch.float32, device=device)
    eigvecs = torch.stack(eigvecs, dim=1)  # [n, k]
    
    return eigvals, eigvecs


def _semantic_consistency_dp_alignment(
    V: torch.Tensor,
    tok_lang_list: list[str],
    cross_pairs: list[tuple[int, int, float]],
    semantic_proj: dict[str, torch.Tensor],
    semantic_lr: float,
    semantic_iters: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Semantic consistency using Dynamic Programming for optimal alignment.
    
    Frames the problem as: find optimal sequence of projection updates to minimize
    total alignment error. Processes language pairs in optimal order (greedy DP).
    
    Args:
        V: Token embedding matrix [n_tokens, k]
        tok_lang_list: Language for each token
        cross_pairs: List of (i, j, weight) tuples
        semantic_proj: Dict mapping language -> projection matrix [k, k]
        semantic_lr: Learning rate
        semantic_iters: Number of iterations
        device: Torch device
    
    Returns:
        Transformed V matrix
    """
    if device is None:
        device = V.device
    
    languages = set(tok_lang_list)
    k = V.shape[1]
    
    # Initialize projections
    for lang in languages:
        if lang not in semantic_proj:
            semantic_proj[lang] = torch.eye(k, dtype=torch.float32, device=device)
    
    # Group pairs by language pairs for DP processing
    lang_pair_groups = defaultdict(list)
    for i, j, weight in cross_pairs:
        lang_i = tok_lang_list[i]
        lang_j = tok_lang_list[j]
        key = tuple(sorted([lang_i, lang_j]))
        lang_pair_groups[key].append((i, j, weight))
    
    # DP: Process language pairs in optimal order
    for iteration in range(semantic_iters):
        # Sort language pairs by alignment error (DP ordering)
        pair_errors = []
        for (lang_i, lang_j), pairs in lang_pair_groups.items():
            phi_i = semantic_proj[lang_i]
            phi_j = semantic_proj[lang_j]
            
            total_error = 0.0
            for i, j, weight in pairs:
                v_i = phi_i @ V[i]
                v_j = phi_j @ V[j]
                error = weight * torch.norm(v_i - v_j).item() ** 2
                total_error += error
            
            pair_errors.append((total_error, (lang_i, lang_j), pairs))
        
        # Process in order of decreasing error (greedy DP)
        pair_errors.sort(reverse=True)
        
        for total_error, (lang_i, lang_j), pairs in pair_errors:
            phi_i = semantic_proj[lang_i]
            phi_j = semantic_proj[lang_j]
            
            # Accumulate gradients for this language pair
            grad_i = torch.zeros(k, k, dtype=torch.float32, device=device)
            grad_j = torch.zeros(k, k, dtype=torch.float32, device=device)
            
            for i, j, weight in pairs:
                v_i = phi_i @ V[i]
                v_j = phi_j @ V[j]
                diff = v_i - v_j
                
                grad_i += 2.0 * weight * torch.outer(diff, V[i])
                grad_j -= 2.0 * weight * torch.outer(diff, V[j])
            
            # Update (DP transition)
            if len(pairs) > 0:
                semantic_proj[lang_i] -= semantic_lr * grad_i / len(pairs)
                semantic_proj[lang_j] -= semantic_lr * grad_j / len(pairs)
    
    # Transform all embeddings
    transformed = torch.zeros_like(V)
    for idx, lang in enumerate(tok_lang_list):
        transformed[idx] = semantic_proj[lang] @ V[idx]
    
    return transformed
