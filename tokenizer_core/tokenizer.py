import json
import math
import time
import unicodedata as ud
from pathlib import Path
from collections import Counter, defaultdict

import torch

from .constants import *
import tokenizer_core.utils as utils
from .linguistic_features import LinguisticModels, MorphologyEncoder
from .torch_utils import default_device, ensure_tensor

class ScalableTokenizer:
    """Implements a scalable, linguistically-aware, unsupervised tokenizer.

    This tokenizer learns a vocabulary from a raw text corpus using an iterative
    optimization algorithm based on column generation (batch pricing). It frames
    tokenization as a shortest-path problem on a graph of characters, where the
    goal is to find the segmentation with the minimum total cost.

    The process involves three main steps:
    1.  **Corpus Analysis**: It first analyzes the corpus to find all potential
        candidate tokens and calculates their statistical properties (e.g.,
        frequency, PMI-like cohesion scores).
    2.  **Iterative Training**: It iteratively builds a vocabulary by finding and
        adding new tokens that offer the greatest reduction in the overall
        tokenization cost for the corpus.
    3.  **Vocabulary Budgeting**: After training, it can enforce a strict
        vocabulary size by using a Lagrangian relaxation technique (bisection
        search on a penalty term `lambda`) to prune less valuable tokens.

    Attributes:
        vocab (list[str]): The list of tokens in the learned vocabulary.
        tok2id (dict[str, int]): A mapping from token strings to their integer IDs.
    """
    def __init__(self, max_token_len=14, min_freq=5, alpha=1.0, beta=0.5, tau=0.01,
                 top_k_add=8, vocab_budget=None, lambda_lo=0.0, lambda_hi=2.0,
                 merge_reward=0.05, short_penalty=0.1, space_penalty=0.1,
                 device: str | torch.device | None = None):
        """Initializes the ScalableTokenizer with its configuration.

        Args:
            max_token_len (int): The maximum length in characters for any learned token.
            min_freq (int): A candidate token must appear at least this many times
                in the corpus to be considered for the vocabulary.
            alpha (float): A weighting factor for the token's negative log-likelihood cost.
            beta (float): A weighting factor for the token's PMI-based cohesion penalty.
            tau (float): A weighting factor for the token's length penalty.
            top_k_add (int): The number of best new tokens to add to the vocabulary
                in each training iteration.
            vocab_budget (int | None): The target size for the final vocabulary. If None,
                no budget is enforced.
            lambda_lo (float): The lower bound for the Lagrangian multiplier used
                in vocabulary budgeting.
            lambda_hi (float): The upper bound for the Lagrangian multiplier.
            merge_reward (float): Reward for combining consecutive tokens into longer spans.
            short_penalty (float): Penalty discouraging very short tokens.
            space_penalty (float): Penalty applied when tokens introduce internal spaces.
            device (str | torch.device | None): Torch device for tensorized computations.
        """
        # --- Hyperparameters ---
        self.max_token_len = max_token_len
        self.min_freq = min_freq
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.top_k_add = top_k_add
        self.vocab_budget = vocab_budget
        self.lambda_lo = lambda_lo
        self.lambda_hi = lambda_hi
        self.merge_reward = merge_reward
        self.short_penalty = short_penalty
        self.space_penalty = space_penalty
        self.device = torch.device(device) if device else default_device()

        # --- Model State ---
        self.vocab = []
        self.tok2id = {}
        self._nll = {}
        self._pmi_pen = {}
        self._potential_tokens = set()
        self._token_occurrences = defaultdict(list)
        self._cost_cache = {}
        self._lambda_global = 0.0
        self._ling = LinguisticModels(space_penalty=space_penalty)
        self._paras = []

        # Pre-defined classes for tokens to enable class-based features.
        self._classes = ["<BOS>", "MWE","URL","EMAIL","NUM","EMOJI",
                         "InitCap","ALLCAPS","lower","hyphen","PUNCT","EOS","other"]
        self._class2idx = {c:i for i,c in enumerate(self._classes)}

    def set_feature_models(self, **kwargs):
        """Configures the linguistic feature models for the tokenizer.

        This method allows injecting external knowledge (lexicons, named entities, etc.)
        to guide the tokenization process.

        Args:
            **kwargs: Keyword arguments passed directly to the LinguisticModels constructor.
                Examples: `lexicon`, `ne_gaz`, `token_bigram`.
        """
        morph_kwargs = kwargs.pop("morphology_kwargs", None)
        if "space_penalty" not in kwargs:
            kwargs["space_penalty"] = self._ling.space_penalty
        self._ling = LinguisticModels(morphology_kwargs=morph_kwargs, **kwargs)
        self._cost_cache.clear()

    def paragraph_lang(self, idx: int):
        """Retrieves the language code for a paragraph by its index."""
        if 0 <= idx < len(self._paras):
            return self._paras[idx].lang
        return None

    def _initialize_stats_and_vocab(self, paragraphs_texts, paragraphs_langs):
        """Performs initial analysis of the corpus to gather statistics."""
        print("Step 1: Performing initial corpus analysis...")
        t0 = time.time()

        self._paras = [utils.ParagraphInfo(t, l) for t,l in zip(paragraphs_texts, paragraphs_langs)]

        # --- 1. Count characters and substrings ---
        corpus_text = "\n".join(paragraphs_texts)
        char_count = Counter(corpus_text)
        total_chars = sum(char_count.values())
        char_prob = {c: cnt / max(total_chars, 1) for c, cnt in char_count.items()}

        substr_count = Counter()
        for pi, p in enumerate(paragraphs_texts):
            T = len(p); info = self._paras[pi]
            for i in range(T):
                maxL = min(self.max_token_len, T - i)
                for L in range(1, maxL + 1):
                    j = i + L
                    if info.is_legal_span(i, j):
                        tok = p[i:j]
                        substr_count[tok] += 1

        # --- 2. Filter to create a set of "potential tokens" ---
        self._potential_tokens = set()
        # for debugging we keep track of why they failed
        self._filtered_tokens = {
            "low_freq": set(),
            "punct_space_low_freq": set(),
            "redirect": set(),
            "wiki_noise": set(),
            "quote_mixed_script": set(),
            "too_many_spaces": set(),
        }
        for t, c in substr_count.items():
            if c < self.min_freq:
                self._filtered_tokens["low_freq"].add(t)
                continue
            if utils.is_all_punct_or_space(t) and c < (2 * self.min_freq):
                self._filtered_tokens["punct_space_low_freq"].add(t)
                continue
            if REDIRECT_TOKEN_RE.search(t):
                self._filtered_tokens["redirect"].add(t)
                continue
            if WIKI_NOISE_RE.match(t):
                self._filtered_tokens["wiki_noise"].add(t)
                continue
            if QUOTE_RUN_EDGE_RE.search(t) and utils._is_mixed_script(t):
                self._filtered_tokens["quote_mixed_script"].add(t)
                continue
            if utils._too_many_internal_spaces(t) and t not in self._ling.lexicon:
                self._filtered_tokens["too_many_spaces"].add(t)
                continue
            # survived all checks
            self._potential_tokens.add(t)

        lexicon = {
            "Google": 100.0, # should be included
            "Microsoft": 100.0, # not in the corpus
            "’s": 0.5, # not in the corpus
            "'s": 100.0, # should be included
            "tokenizers": 100.0, # should be included
            "representations": 100.0, # too long, not considered
            "Entwicklungen": 100.0, # should be included
            "Weiterleitung": 100.0, # redirect token, removed
            "Universität": 10.0 # should be included
        }
        # --- 3. Index all occurrences of potential tokens for fast access ---
        self._token_occurrences.clear()
        for pi, p in enumerate(paragraphs_texts):
            T = len(p); info = self._paras[pi]
            for i in range(T):
                maxL = min(self.max_token_len, T - i)
                for L in range(1, maxL + 1):
                    j = i + L
                    tok = p[i:j]
                    if tok in self._potential_tokens and info.is_legal_span(i, j):
                        self._token_occurrences[tok].append((pi, i))

        # --- 4. Calculate statistical costs for each potential token ---
        # Length-conditioned probs and PMI-like cohesion
        len_totals = defaultdict(int)
        for tok in self._potential_tokens:
            len_totals[len(tok)] += substr_count[tok]

        alpha_len, alpha_char = 1.0, 1e-3
        for token in self._potential_tokens:
            L = len(token)
            count = substr_count[token]
            denom = len_totals[L] + alpha_len * max(1, len(len_totals))
            p_len = (count + alpha_len) / max(denom, EPS)

            prod_char = 1.0
            for c in token:
                prod_char *= (char_prob.get(c, 0.0) + alpha_char)

            self._nll[token] = -math.log(max(p_len, EPS))
            pmi = math.log(max(p_len / max(prod_char, EPS), EPS))
            self._pmi_pen[token] = -max(min(pmi, 20.0), -20.0)

        # --- 5. Initialize vocabulary with single characters as a fallback ---
        # Seed vocab = all observed single characters (fallback alphabet)
        self.vocab = sorted(set(char_count.keys()))
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self._cost_cache.clear()

        print(f"Analysis complete in {time.time()-t0:.2f}s. "
              f"Found {len(self._potential_tokens)} potential tokens; "
              f"seed vocab chars = {len(self.vocab)}.")

        # --- 6. Train the morphology encoder on the corpus data ---
        # Morphology encoder
        self._ling.morph_encoder = self._ling.create_morph_encoder()
        self._ling.paragraph_lang = self.paragraph_lang
        self._ling.morph_encoder.fit(self._paras, self._token_occurrences, self.paragraph_lang)

    # ------------------------------------------------------------------
    # Serialization / debugging helpers
    # ------------------------------------------------------------------

    def save(self, path, include_morphology=True):
        """Persists the tokenizer state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        morph_encoder = getattr(self._ling, "morph_encoder", None)
        morph_payload = None
        if include_morphology and morph_encoder is not None:
            morph_payload = {
                "embedding_mode": getattr(morph_encoder, "embedding_mode", "ppmi"),
                "token_vec": {tok: vec.tolist() for tok, vec in morph_encoder.token_vec.items()},
                "lang_proto": {lang: vec.tolist() for lang, vec in morph_encoder.lang_proto.items()},
            }
        payload = {
            "config": {
                "max_token_len": self.max_token_len,
                "min_freq": self.min_freq,
                "alpha": self.alpha,
                "beta": self.beta,
                "tau": self.tau,
                "top_k_add": self.top_k_add,
                "vocab_budget": self.vocab_budget,
                "lambda_lo": self.lambda_lo,
                "lambda_hi": self.lambda_hi,
                "merge_reward": self.merge_reward,
                "short_penalty": self.short_penalty,
                "linguistic_kwargs": {
                    "lexicon": self._ling.lexicon,
                    "mwe": list(self._ling.mwe),
                    "ne_gaz": {k: list(v) for k, v in self._ling.ne_gaz.items()},
                    "token_bigram": {f"{k[0]}|||{k[1]}": v for k, v in self._ling.token_bigram.items()},
                    "gamma_boundary": self._ling.gamma_boundary,
                    "mu_morph": self._ling.mu_morph,
                    "prefix_reward": self._ling.prefix_reward,
                    "suffix_reward": self._ling.suffix_reward,
                    "space_penalty": self._ling.space_penalty,
                    "morphology_kwargs": getattr(self._ling, "morphology_kwargs", {}),
                },
            },
            "model_state": {
                "vocab": self.vocab,
                "tok2id": self.tok2id,
                "lambda_global": self._lambda_global,
            },
            "morphology": morph_payload,
        }
        # Explicit UTF-8 avoids Windows cp1252 write failures on multilingual data.
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path):
        """Loads a tokenizer instance from disk."""
        path = Path(path)
        payload = json.loads(path.read_text())
        cfg = payload["config"]
        tok = cls(
            max_token_len=cfg["max_token_len"],
            min_freq=cfg["min_freq"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            tau=cfg["tau"],
            top_k_add=cfg["top_k_add"],
            vocab_budget=cfg["vocab_budget"],
            lambda_lo=cfg["lambda_lo"],
            lambda_hi=cfg["lambda_hi"],
            merge_reward=cfg["merge_reward"],
            short_penalty=cfg["short_penalty"],
            space_penalty=cfg.get("space_penalty", cfg["linguistic_kwargs"].get("space_penalty", 0.1)),
        )
        ling_kwargs = cfg["linguistic_kwargs"]
        token_bigram = {}
        for key, val in ling_kwargs["token_bigram"].items():
            prev, curr = key.split("|||")
            token_bigram[(prev, curr)] = val
        tok.set_feature_models(
            lexicon=ling_kwargs["lexicon"],
            mwe=set(ling_kwargs["mwe"]),
            ne_gaz={k: set(v) for k, v in ling_kwargs["ne_gaz"].items()},
            token_bigram=token_bigram,
            gamma_boundary=ling_kwargs["gamma_boundary"],
            mu_morph=ling_kwargs["mu_morph"],
            prefix_reward=ling_kwargs["prefix_reward"],
            suffix_reward=ling_kwargs["suffix_reward"],
            space_penalty=ling_kwargs["space_penalty"],
            morphology_kwargs=ling_kwargs.get("morphology_kwargs"),
        )
        tok_state = payload["model_state"]
        tok.vocab = tok_state["vocab"]
        tok.tok2id = tok_state["tok2id"]
        tok._lambda_global = tok_state["lambda_global"]

        morph_payload = payload.get("morphology")
        if morph_payload:
            morph = tok._ling.create_morph_encoder()
            morph.embedding_mode = morph_payload.get("embedding_mode", "ppmi")
            morph.token_vec = {
                k: ensure_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in morph_payload.get("token_vec", {}).items()
            }
            morph.lang_proto = {
                k: ensure_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in morph_payload.get("lang_proto", {}).items()
            }
            tok._ling.morph_encoder = morph
        return tok

    def dump_debug_info(self, path, max_tokens=50, include_filtered=True):
        """Writes out useful debugging statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "vocab_size": len(self.vocab),
            "potential_tokens": len(getattr(self, "_potential_tokens", [])),
            "lambda_global": self._lambda_global,
            "top_vocab_samples": self.vocab[:max_tokens],
            "config": {
                "max_token_len": self.max_token_len,
                "min_freq": self.min_freq,
                "merge_reward": self.merge_reward,
                "short_penalty": self.short_penalty,
            },
        }
        if include_filtered and hasattr(self, "_filtered_tokens"):
            info["filtered_tokens"] = {k: len(v) for k, v in self._filtered_tokens.items()}
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2))

    def _is_protected_token(self, tok: str) -> bool:
        """Checks if a token matches protected patterns like URLs or emails."""
        return (URL_RE.search(tok) is not None) or \
               (EMAIL_RE.fullmatch(tok) is not None) or \
               (NUM_RE.fullmatch(tok) is not None)

    def _base_token_cost(self, token: str) -> float:
        """Calculates the fundamental statistical cost of a token."""

        cost = (self.alpha * self._nll.get(token, 0.0) +        # Frequency cost 
                self.beta  * self._pmi_pen.get(token, 0.0) +    # Cohesion cost
                self.tau   * len(token))                        # Length penalty
        # Add these new hyperparameters to your __init__

        num_graphemes = utils.count_graphemes(token)
        if num_graphemes > 1:
            cost -= self.merge_reward * (num_graphemes - 1)
        if num_graphemes <= 1:
            cost += self.short_penalty
        return cost

    def _get_token_cost(self, token: str) -> float:
        """Retrieves the total cost of a token, including the global lambda penalty."""
        key = (token, self._lambda_global)
        if key in self._cost_cache:
            return self._cost_cache[key]

        # Apply the Lagrangian penalty only to multi-character tokens.
        lam = self._lambda_global if len(token) > 1 else 0.0
        c = self._base_token_cost(token) + lam
        if not math.isfinite(c): c = 1e6
        self._cost_cache[key] = c
        return c

    # --- DP decode (with atomic protected spans + post-clean + CJK merge) ---
    def _dp_decode(self, paragraph_idx: int, restrict_vocab=None, decode_only=True):
        """Tokenizes a single paragraph using dynamic programming to find the lowest-cost path.

        This function implements the Viterbi algorithm on a lattice representing
        all possible segmentations of the text.

        Args:
            paragraph_idx (int): The index of the paragraph to decode.
            decode_only (bool): If True, returns only the token list. If False, also
                returns the set of used vocabulary and the DP cost array.

        Returns:
            If `decode_only` is True, a list of strings (the tokens).
            Otherwise, a tuple containing (tokens, used_vocab_set, dp_min_costs).
        """
        text = self._paras[paragraph_idx].text
        info = self._paras[paragraph_idx]
        T = len(text); K = len(self._classes); idx = self._class2idx
        INF = float('inf')

        # dp[t, k] = min cost to segment text[0...t] ending with a token of class k.
        dp = torch.full((T + 1, K), INF, dtype=torch.float32, device=self.device)
        # back[t, k] = backpointer to reconstruct the optimal path.
        back = [[None] * K for _ in range(T + 1)]
        dp[0, idx["<BOS>"]] = 0.0

        allowed_vocab = None if restrict_vocab is None else set(restrict_vocab)

        for t in range(1, T + 1):
            maxL = min(self.max_token_len, t)
            for L in range(1, maxL + 1):
                i = t - L
                if not info.is_legal_span(i, t):
                    continue

                tok = text[i:t]
                is_exact_protected = any(i == s and t == e for (s, e) in info.protected)

                # is_exact_protected = []
                # A token is valid if it's in our vocab or it's an exact protected span.
                # allow exact protected spans even if not in vocab (virtual arc)
                if (tok not in self.tok2id) and (not is_exact_protected):
                    # Check for "open arc" conditions
                    is_cjk_override = (info.lang == 'ja' and utils.all_cjk(tok) and len(tok) <= 4)
                    is_tamil_override = (info.lang == 'ta' and utils.all_tamil(tok) and len(utils.grapheme_clusters(tok)) <= 3)

                    if not (is_cjk_override or is_tamil_override):
                        continue # If not in vocab AND not a special override, then skip.
                if (allowed_vocab is not None) and (not is_exact_protected) and (tok not in allowed_vocab):
                    continue

                # Protected spans not in the vocab are treated as "free" atomic units.
                if is_exact_protected and (tok not in self.tok2id):
                    base = 0.0  # free atomic arc for protected spans
                else:
                    base = self._get_token_cost(tok)

                tc = self._ling.token_class(tok)
                j = idx.get(tc, idx["other"])

                # Find the best previous state to transition from.
                for si in range(K):
                    prev_cost = float(dp[i, si].item())
                    if not math.isfinite(prev_cost): continue

                    # Additive cost includes linguistic features (bigrams, morphology, etc.)
                    add = self._ling.additive_cost(tok, self._classes[si], paragraph_idx)
                    v = prev_cost + base + add

                    # If we found a cheaper path, update the DP table.
                    current = float(dp[t, j].item())
                    if v < current:
                        dp[t, j] = v
                        back[t][j] = (i, si, L)

        end_cls = int(torch.argmin(dp[T]).item())
        dp_min = torch.min(dp, dim=1).values.detach().to("cpu")

        def backtrace():
            """Reconstructs the token sequence from the backpointers."""
            if not torch.isfinite(dp[T, end_cls]).item():
                return list(text), set(text)
            toks = []; t = T; c = end_cls; used = set()
            while t > 0:
                i, si, L = back[t][c]
                if i is None: break
                tok = text[i:t]
                toks.append(tok); used.add(tok)
                t, c = i, si
            toks.reverse()
            # Apply post-processing to clean up the final token sequence.
            toks = utils.clean_junk_runs(toks)
            toks = utils.merge_cjk_runs(toks)
            return toks, used

        if decode_only:
            toks, _ = backtrace()
            return toks

        toks, used = backtrace()
        return toks, used, dp_min


    def _find_best_new_tokens_batch(self, all_dp_min, top_k=8, max_base_cost=50.0):
        """Finds the best candidate tokens to add to the vocabulary.

        This is the "pricing" or "column generation" step. It calculates the
        "reduced cost" for every potential token not yet in the vocabulary. A
        negative reduced cost means adding this token would improve the overall
        segmentation.

        Args:
            all_dp_min (list[torch.Tensor]): Minimum DP costs for each paragraph.
            top_k (int): The number of best tokens to return.
            max_base_cost (float): A threshold to prune highly unlikely candidates early.

        Returns:
            A tuple containing (list of best token strings, list of full candidate info).
        """
        candidates = []

        for tok in self._potential_tokens:
            if tok in self.tok2id: continue
            if utils.is_all_punct_or_space(tok): continue
            if REDIRECT_TOKEN_RE.search(tok): continue
            if WIKI_NOISE_RE.match(tok): continue
            if QUOTE_RUN_EDGE_RE.search(tok) and utils._is_mixed_script(tok): continue
            if utils._too_many_internal_spaces(tok) and tok not in self._ling.lexicon: continue

            # 1. Get the simple, context-free statistical base cost.
            base_cost = self._get_token_cost(tok)
            # 2. Get all occurrences to calculate the average linguistic cost.
            occurrences = self._token_occurrences.get(tok, [])
            if not occurrences:
                continue
            # 3. Call for accurate average linguistic cost.
            avg_ling_cost = self._ling._calculate_average_linguistic_cost(tok, occurrences)
            # 4. The final proposal cost is the sum of the two.
            proposal_cost = base_cost + avg_ling_cost

            if proposal_cost > max_base_cost:
                continue
            rc_sum = 0.0; occs = 0; L = len(tok)
            # Calculate total reduced cost summed over all occurrences of the token.
            for (pi, start) in self._token_occurrences.get(tok, []):
                dpmin = all_dp_min[pi]; end = start + L
                if not self._paras[pi].is_legal_span(start, end): continue
                if end >= len(dpmin) or start >= len(dpmin): continue
                start_cost = float(dpmin[start].item())
                end_cost = float(dpmin[end].item())
                if not (math.isfinite(proposal_cost) and math.isfinite(start_cost) and math.isfinite(end_cost)): continue
                # Reduced Cost = (cost of new path) - (cost of old path)
                # rc = (cost of `tok`) + (cost to reach `start`) - (cost to reach `end`)
                rc = proposal_cost + start_cost - end_cost
                if math.isfinite(rc) and rc < 0:
                    rc_sum += rc; occs += 1
            if occs > 0:
                candidates.append((rc_sum, tok, occs))

        candidates.sort(key=lambda x: x[0])  # most negative first
        # use this here to debug what candidates is being proposed. You can also use pdb (preferable)
        #print(candidates)
        #assert False

        chosen = [tok for _, tok, _ in candidates[:top_k]]
        return chosen, candidates[:top_k]


    def train(self, paragraphs_texts, paragraphs_langs, max_iterations=1000, rc_stop_tol=-1e-6, verbose=True):
        """Trains the tokenizer by iteratively building the vocabulary.

        Args:
            paragraphs_texts (list[str]): The list of raw text paragraphs for training.
            paragraphs_langs (list[str]): The corresponding language codes for each paragraph.
            max_iterations (int): The maximum number of training iterations.
            rc_stop_tol (float): The training stops if the best candidate's reduced
                cost is greater than this tolerance (i.e., close to zero).
            verbose (bool): If True, prints progress updates.
        """
        self._initialize_stats_and_vocab(paragraphs_texts, paragraphs_langs)

        if verbose: print("\nStep 2: Starting training with batch pricing...")
        for it in range(1, max_iterations + 1):
            self._cost_cache.clear()
            # Primal step: Decode the whole corpus with the current vocabulary.
            all_dp_min = []
            for pi in range(len(paragraphs_texts)):
                _, _, dpmin = self._dp_decode(pi, decode_only=False)
                all_dp_min.append(dpmin)

            # Pricing step: Find the best new tokens to add.
            new_tokens, topk_info = self._find_best_new_tokens_batch(all_dp_min, top_k=self.top_k_add)
            if not new_tokens:
                min_rc = topk_info[0][0] if topk_info else 0.0
                if verbose:
                    print(f"\nConvergence: no negative reduced-cost tokens (min summed RC = {min_rc:.6f}).")
                break
            min_rc = topk_info[0][0] if topk_info else 0.0

            # Check for convergence.
            if min_rc >= rc_stop_tol:
                if verbose:
                    print(f"\nConvergence: min summed RC = {min_rc:.6f} >= tol {rc_stop_tol:.1e}.")
                break

            # Add the new tokens to the vocabulary.
            for tok in new_tokens:
                tok = ud.normalize("NFC", tok)
                if tok not in self.tok2id:
                    self.tok2id[tok] = len(self.vocab)
                    self.vocab.append(tok)

            if verbose:
                preview = ", ".join([f"'{t}'" for t in new_tokens[:6]])
                more = "" if len(new_tokens) <= 6 else f" ... (+{len(new_tokens)-6})"
                print(f"Iter {it:02d}: Added {len(new_tokens)} tokens: {preview}{more} (best summed RC={min_rc:.4f})")
        else:
            if verbose: print("\nMax iterations reached.")

        # If a vocab budget is set, enforce it now.
        if self.vocab_budget is not None:
            self._enforce_vocab_budget_bisection(paragraphs_texts, self.vocab_budget, verbose=verbose)
            # Prune exported vocab to active multi-char tokens (plus alphabet)
            self._prune_vocab_to_active(paragraphs_texts)

        if verbose:
            print(f"\nTraining complete. Final vocabulary size: {len(self.vocab)}")


    def _count_types_used(self, paragraphs_texts):
        """Counts the number of unique multi-character tokens used to segment the corpus."""
        used = set()
        for pi in range(len(paragraphs_texts)):
            toks, U, _ = self._dp_decode(pi, decode_only=False)
            used |= {u for u in U if (len(u) > 1) and (not self._is_protected_token(u))}
        return len(used)

    def _collect_active_multichar(self, paragraphs_texts):
        active = set()
        for pi in range(len(paragraphs_texts)):
            _, U, _ = self._dp_decode(pi, decode_only=False)
            for u in U:
                if len(u) > 1 and not self._is_protected_token(u):
                    active.add(ud.normalize("NFC", u))
        return active

    def _prune_vocab_to_active(self, paragraphs_texts):
        """Removes any unused tokens from the final vocabulary."""
        active = self._collect_active_multichar(paragraphs_texts)
        singles = [t for t in self.vocab if len(t) == 1]
        # Final vocab is the union of all single characters and active multi-char tokens.
        new_vocab = sorted(set(singles)) + sorted(active)
        self.vocab = new_vocab
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}

    def _enforce_vocab_budget_bisection(self, paragraphs_texts, K_target, tol=1e-2, max_steps=25, verbose=True):
        """Uses a bisection search to find the lambda that produces the target vocab size."""
        if verbose:
            print(f"\nStep 3: Enforcing vocabulary budget K={K_target} via Lagrangian bisection on lambda.")
        lo, hi = self.lambda_lo, self.lambda_hi

        # First, ensure our search range is wide enough.
        self._lambda_global = hi; self._cost_cache.clear()
        types_hi = self._count_types_used(paragraphs_texts)
        while types_hi > K_target and hi < 1e6:
            hi *= 2.0
            self._lambda_global = hi; self._cost_cache.clear()
            types_hi = self._count_types_used(paragraphs_texts)

        best_lambda = self._lambda_global
        best_gap = float('inf')

        # Perform bisection search.
        for step in range(max_steps):
            mid = 0.5*(lo+hi)
            self._lambda_global = mid
            self._cost_cache.clear()
            types = self._count_types_used(paragraphs_texts)
            gap = abs(types - K_target)
            if gap < best_gap:
                best_gap = gap
                best_lambda = mid
            if verbose:
                print(f"  lambda={mid:.6f} -> used types = {types} (target {K_target})")
            if types > K_target:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break

        # Set the best lambda and finalize.
        self._lambda_global = best_lambda
        self._cost_cache.clear()
        if verbose:
            print(f"Chosen lambda*={best_lambda:.6f} (gap={best_gap}).")


    def tokenize(self, text, lang=None):
        """Tokenizes a new piece of text using the trained vocabulary.

        Args:
            text (str): The input string to tokenize.
            lang (str, optional): The language code of the text. If None, it will be guessed.

        Returns:
            list[str]: A list of token strings.
        """
        for idx, para in enumerate(self._paras):
            if para.text is text or para.text == text:
                return self._dp_decode(idx)
        backup = self._paras
        try:
            self._paras = [utils.ParagraphInfo(text, lang or utils.script_guess(text))]
            # Decode using the main DP algorithm.
            tokens = self._dp_decode(0)
        finally:
            # Restore the original corpus paragraphs state.
            self._paras = backup
        return tokens
