import json
import math
import re
import time
import unicodedata as ud
from pathlib import Path
from collections import Counter, defaultdict

import torch

from .constants import *
import tokenizer_core.utils as utils
from .distributed_utils import (
    DistributedContext,
    all_gather_object,
    broadcast_object,
    merge_sets,
    shard_indices,
)
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
    def __init__(
        self,
        max_token_len=14,
        min_freq=3,
        alpha=1.0,
        beta=0.5,
        tau=0.01,
        top_k_add=8,
        vocab_budget=None,
        lambda_lo=0.0,
        lambda_hi=2.0,
        merge_reward=0.05,
        short_penalty=0.2,
        space_penalty=0.25,
        device: str | torch.device | None = None,
        distributed_context: DistributedContext | None = None,
        uniseg_root: str | Path | None = None,
        uniseg_reward: float = 0.3,
        use_morph_encoder: bool = True,
        seed_uniseg_segments: bool = False,
        force_seed_uniseg_tokens: bool = False,
        pricing_device: str | torch.device | None = None,
    ):
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
            uniseg_root (str | Path | None): Root directory for UniSegments morpheme data.
                If provided, token boundaries that align with gold morpheme boundaries
                receive a cost reduction, encouraging morphologically-aware tokenization.
            uniseg_reward (float): Reward (cost reduction) for boundaries aligning with
                gold morpheme boundaries. Default 0.1. Only used if uniseg_root is set.
            use_morph_encoder (bool): If False, disables the morphological encoder entirely.
                The tokenizer will rely only on statistical costs and UniSeg alignment.
            seed_uniseg_segments (bool): If True, seed the candidate pool with UniSeg
                morpheme segments for any words that actually occur in the corpus.
            force_seed_uniseg_tokens (bool): If True, add all seeded UniSeg morpheme tokens
                into the vocabulary (after the usual reduced-cost selection).
            pricing_device (str | torch.device | None): Device to use for batch-pricing
                vectorized cost calculation (default CPU).
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
        if distributed_context is None:
            resolved = torch.device(device) if device else None
            distributed_context = DistributedContext.standalone(resolved)
        if device:
            distributed_context.device = torch.device(device)
        elif distributed_context.device is None:
            distributed_context.device = default_device()
        self.dist = distributed_context
        self.device = distributed_context.device

        # Surface which device we are using (CPU vs CUDA)
        if self._is_primary_rank():
            device_str = getattr(self.device, "type", str(self.device))
            print(f"[Tokenizer] Using device: {device_str}")

        # --- Model State ---
        self.vocab = []
        self.tok2id = {}
        self._nll = {}
        self._pmi_pen = {}
        self._potential_tokens = set()
        self._token_occurrences = defaultdict(list)
        self._cost_cache = {}
        self._lambda_global = 0.0
        self.use_morph_encoder = use_morph_encoder
        self.seed_uniseg_segments = seed_uniseg_segments
        self.force_seed_uniseg_tokens = force_seed_uniseg_tokens
        self._seeded_uniseg_tokens: set[str] = set()
        if pricing_device is None:
            self.pricing_device = torch.device("cpu")
        else:
            dev = torch.device(pricing_device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                if self._is_primary_rank():
                    print(f"[Tokenizer] Requested CUDA pricing device '{dev}' but CUDA is unavailable. Falling back to CPU.")
                dev = torch.device("cpu")
            self.pricing_device = dev
        self._ling = LinguisticModels(
            space_penalty=space_penalty,
            uniseg_root=uniseg_root,
            uniseg_reward=uniseg_reward,
        )
        self._paras = []
        self._curated_prefixes = self._load_curated_prefixes()
        self._vocab_prefixes: set[str] = set(self._curated_prefixes)

        # Pre-defined classes for tokens to enable class-based features.
        self._classes = ["<BOS>", "MWE","URL","EMAIL","NUM","EMOJI",
                         "InitCap","ALLCAPS","lower","hyphen","PUNCT","EOS","other"]
        self._class2idx = {c:i for i,c in enumerate(self._classes)}

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    def _is_primary_rank(self) -> bool:
        return (not self.dist.is_distributed) or self.dist.is_primary

    def _log(self, message: str) -> None:
        if self._is_primary_rank():
            print(message)

    def _collect_dp_min(self, total_paragraphs: int):
        local_pairs = []
        for pi in shard_indices(total_paragraphs, self.dist):
            _, _, dpmin = self._dp_decode(pi, decode_only=False)
            local_pairs.append((pi, dpmin.detach().to("cpu")))
        gathered = all_gather_object(local_pairs, self.dist)
        merged = []
        for chunk in gathered:
            merged.extend(chunk)
        merged.sort(key=lambda pair: pair[0])
        # Move to pricing device on the primary rank so batch pricing can stay tensorized.
        dp_list = []
        for _, dp in merged:
            if dp.device != self.pricing_device:
                dp = dp.to(self.pricing_device)
            dp_list.append(dp)
        return dp_list

    def _broadcast_candidates(self, new_tokens, topk_info):
        payload = {"new_tokens": new_tokens, "topk_info": topk_info}
        payload = broadcast_object(payload, self.dist)
        return payload["new_tokens"], payload["topk_info"]

    def _sync_morphology_encoder(self):
        morph_encoder = getattr(self._ling, "morph_encoder", None)
        payload = None
        if morph_encoder and self._is_primary_rank():
            payload = {
                "embedding_mode": getattr(morph_encoder, "embedding_mode", "ppmi"),
                "token_vec": {k: v.tolist() for k, v in morph_encoder.token_vec.items()},
                "lang_proto": {k: v.tolist() for k, v in morph_encoder.lang_proto.items()},
            }
        payload = broadcast_object(payload, self.dist)
        if payload:
            morph = self._ling.create_morph_encoder()
            morph.embedding_mode = payload.get("embedding_mode", "ppmi")
            morph.token_vec = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device)
                for k, v in payload.get("token_vec", {}).items()
            }
            morph.lang_proto = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device)
                for k, v in payload.get("lang_proto", {}).items()
            }
            self._ling.morph_encoder = morph

    def _distributed_barrier(self):
        if self.dist.is_distributed:
            self.dist.barrier()

    # ------------------------------------------------------------------
    # Vocabulary prefix helpers for fast lattice pruning
    # ------------------------------------------------------------------

    def _rebuild_vocab_prefixes(self):
        """Recomputes the set of all prefixes in the current vocabulary."""
        prefixes: set[str] = set(self._curated_prefixes)
        for tok in self.vocab:
            for k in range(1, len(tok) + 1):
                prefixes.add(tok[:k])
        self._vocab_prefixes = prefixes

    def _update_vocab_prefixes(self, tokens):
        """Incrementally inserts prefixes for newly added tokens."""
        if not self._vocab_prefixes:
            self._vocab_prefixes = set(self._curated_prefixes)
        for tok in tokens or []:
            for k in range(1, len(tok) + 1):
                self._vocab_prefixes.add(tok[:k])

    def _has_vocab_prefix(self, prefix: str) -> bool:
        """Returns True if any vocab token begins with the given prefix."""
        if not prefix:
            return True
        return prefix in self._vocab_prefixes

    def _load_curated_prefixes(self) -> set[str]:
        """Loads curated prefix fragments from UniSegments/affix data."""
        curated: set[str] = set()
        data_root = Path(__file__).resolve().parents[1] / "data"
        uniseg_prefix_file = data_root / "uniseg_prefixes.txt"

        def _add_partial(prefix: str):
            for k in range(1, len(prefix) + 1):
                curated.add(prefix[:k])

        if uniseg_prefix_file.exists():
            try:
                for line in uniseg_prefix_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    _add_partial(line)
            except OSError:
                # Fall back to affix lists if the curated file cannot be read.
                pass

        if not curated:
            for lang_cfg in AFFIXES.values():
                for prefix in lang_cfg.get("pre", []):
                    if prefix:
                        _add_partial(prefix)

        return curated

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

        # --- 3b. Optionally seed UniSeg morpheme segments into the candidate pool ---
        self._seed_uniseg_segments()

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
        self._rebuild_vocab_prefixes()
        self._cost_cache.clear()

        print(f"Analysis complete in {time.time()-t0:.2f}s. "
              f"Found {len(self._potential_tokens)} potential tokens; "
              f"seed vocab chars = {len(self.vocab)}.")

        # --- 6. Train the morphology encoder on the corpus data (optional) ---
        # Always set paragraph_lang for linguistic cost calculations
        self._ling.paragraph_lang = self.paragraph_lang
        
        if self.use_morph_encoder:
            if self._is_primary_rank() or not self.dist.is_distributed:
                self._ling.morph_encoder = self._ling.create_morph_encoder()
                self._ling.morph_encoder.fit(self._paras, self._token_occurrences, self.paragraph_lang)
            self._sync_morphology_encoder()
        else:
            print("[MorphEncoder] Disabled - using only statistical costs and UniSeg alignment.")

    def _register_seed_token(self, token: str, paragraph_idx: int, start_idx: int):
        """Ensure a token exists in potential set and record its occurrence."""
        if not token:
            return
        if utils.is_all_punct_or_space(token):
            return
        if len(token) > self.max_token_len:
            return
        self._potential_tokens.add(token)
        self._token_occurrences[token].append((paragraph_idx, start_idx))
        self._seeded_uniseg_tokens.add(token)

    def _seed_uniseg_segments(self):
        """Optionally seed UniSeg morpheme segments into candidate pool."""
        if not self.seed_uniseg_segments:
            return
        loader = getattr(self._ling, "_uniseg_loader", None)
        if loader is None:
            return
        word_pattern = re.compile(r"\b\w+\b", re.UNICODE)
        added = 0
        for pi, para in enumerate(self._paras):
            lang = para.lang or self.paragraph_lang(pi) or "en"
            text = para.text
            if not text:
                continue
            for match in word_pattern.finditer(text):
                word = match.group()
                boundaries = loader.get_boundaries(word, lang)
                if not boundaries:
                    continue
                positions = [0] + sorted(boundaries) + [len(word)]
                for i in range(len(positions) - 1):
                    rel_start = positions[i]
                    rel_end = positions[i + 1]
                    abs_start = match.start() + rel_start
                    abs_end = match.start() + rel_end
                    if abs_end - abs_start <= 0 or abs_end - abs_start > self.max_token_len:
                        continue
                    if not para.is_legal_span(abs_start, abs_end):
                        continue
                    token = text[abs_start:abs_end]
                    self._register_seed_token(token, pi, abs_start)
                    added += 1
        if added and self._is_primary_rank():
            print(f"[UniSeg Seed] Added {added} morpheme spans from UniSeg.")

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
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        tok._rebuild_vocab_prefixes()
        tok._lambda_global = tok_state["lambda_global"]

        morph_payload = payload.get("morphology")
        if morph_payload:
            morph = tok._ling.create_morph_encoder()
            morph.embedding_mode = morph_payload.get("embedding_mode", "ppmi")
            morph.token_vec = {
                k: ensure_tensor(v, dtype=torch.float32, device=tok.device)
                for k, v in morph_payload.get("token_vec", {}).items()
            }
            morph.lang_proto = {
                k: ensure_tensor(v, dtype=torch.float32, device=tok.device)
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

    def _prepare_span_metadata(self, text: str, info, class_index: dict[str, int]):
        """Precomputes legal spans ending at each position for DP decoding.
        
        Includes UniSeg boundary alignment rewards: spans whose boundaries
        align with gold morpheme boundaries receive a cost reduction.
        """
        T = len(text)
        spans_by_end = [[] for _ in range(T + 1)]
        protected_lookup = {(s, e) for (s, e) in getattr(info, "protected", [])}
        lang = getattr(info, "lang", None)
        
        # Precompute gold morpheme boundaries for this paragraph (UniSeg alignment)
        gold_boundaries = self._ling.precompute_paragraph_boundaries(text, lang)
        uniseg_reward = self._ling.uniseg_reward if gold_boundaries else 0.0

        for start in range(T):
            max_len = min(self.max_token_len, T - start)
            for L in range(1, max_len + 1):
                end = start + L
                if not info.is_legal_span(start, end):
                    continue

                tok = text[start:end]
                is_exact_protected = (start, end) in protected_lookup
                in_vocab = tok in self.tok2id
                is_cjk_override = (
                    lang == "ja"
                    and utils.all_cjk(tok)
                    and len(tok) <= 4
                )
                is_tamil_override = False
                if lang == "ta" and utils.all_tamil(tok):
                    grapheme_len = len(utils.grapheme_clusters(tok))
                    if grapheme_len <= 3:
                        is_tamil_override = True

                allow_open_arc = is_exact_protected or is_cjk_override or is_tamil_override

                if (not in_vocab) and (not allow_open_arc):
                    # If no vocab token extends this prefix, we can stop expanding.
                    if not self._has_vocab_prefix(tok):
                        break
                    continue

                base_cost = 0.0 if (is_exact_protected and not in_vocab) else self._get_token_cost(tok)
                
                # UniSeg boundary alignment reward: reduce cost if BOTH boundaries align
                # This ensures the token is a complete morpheme segment
                if uniseg_reward > 0:
                    # A token gets reward only if:
                    # - Its start is at a word boundary (start in word_starts) OR a morpheme boundary
                    # - AND its end is at a word boundary OR a morpheme boundary
                    # For simplicity: reward if BOTH start and end are at gold positions
                    # (gold_boundaries includes internal morpheme splits)
                    start_aligned = (start == 0) or (start in gold_boundaries)
                    end_aligned = (end == T) or (end in gold_boundaries)
                    
                    if start_aligned and end_aligned:
                        base_cost -= uniseg_reward  # Reward = negative cost
                
                token_class = self._ling.token_class(tok)
                class_idx = class_index.get(token_class, class_index["other"])

                spans_by_end[end].append(
                    {
                        "start": start,
                        "length": L,
                        "token": tok,
                        "token_class_idx": class_idx,
                        "base_cost": base_cost,
                        "is_exact_protected": is_exact_protected,
                    }
                )

        return spans_by_end

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
        spans_by_end = self._prepare_span_metadata(text, info, idx)
        active_states = [[] for _ in range(T + 1)]
        active_states[0].append(idx["<BOS>"])

        for t in range(1, T + 1):
            spans_here = spans_by_end[t]
            if not spans_here:
                continue
            new_active = set()
            for span in spans_here:
                tok = span["token"]
                if (allowed_vocab is not None) and (not span["is_exact_protected"]) and (tok not in allowed_vocab):
                    continue

                start = span["start"]
                prev_states = active_states[start]
                if not prev_states:
                    continue

                base = span["base_cost"]
                j = span["token_class_idx"]

                prev_idx_tensor = torch.tensor(prev_states, dtype=torch.long, device=self.device)
                prev_costs = dp[start, prev_idx_tensor]
                add_vec = self._ling.batch_additive_cost(
                    tok,
                    prev_states,
                    self._classes,
                    paragraph_idx,
                    device=self.device,
                )
                if add_vec.numel() == 0:
                    continue
                candidate = prev_costs + base + add_vec
                best_val, best_pos = torch.min(candidate, dim=0)
                best_prev_idx = prev_states[int(best_pos.item())]

                current_val = float(dp[t, j].item())
                best_val_float = float(best_val.item())
                if best_val_float < current_val:
                    dp[t, j] = best_val
                    back[t][j] = (start, best_prev_idx, span["length"])
                    new_active.add(j)
            active_states[t] = sorted(new_active)

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
        dp_deltas = self._precompute_dp_span_deltas(all_dp_min)
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
            L = len(tok)
            span_deltas = []
            occ_by_para = defaultdict(list)
            for (pi, start) in self._token_occurrences.get(tok, []):
                end = start + L
                if not self._paras[pi].is_legal_span(start, end):
                    continue
                dpmin = all_dp_min[pi]
                if end >= dpmin.shape[0]:
                    continue
                occ_by_para[pi].append(start)

            for pi, starts in occ_by_para.items():
                delta_cache = dp_deltas[pi].get(L)
                if delta_cache is None:
                    continue
                idx = torch.as_tensor(starts, dtype=torch.long, device=self.pricing_device)
                valid = idx < delta_cache.shape[0]
                if not torch.any(valid):
                    continue
                idx = idx[valid]
                gathered = delta_cache.index_select(0, idx)
                if gathered.numel() == 0:
                    continue
                span_deltas.append(gathered)

            if not span_deltas:
                continue

            deltas_tensor = torch.cat(span_deltas, dim=0)
            if deltas_tensor.numel() == 0:
                continue
            rc = deltas_tensor + proposal_cost
            mask = torch.isfinite(rc) & (rc < 0)
            occs = int(mask.sum().item())
            if occs > 0:
                rc_sum = float(rc[mask].sum().item())
                candidates.append((rc_sum, tok, occs))

        candidates.sort(key=lambda x: x[0])  # most negative first
        # use this here to debug what candidates is being proposed. You can also use pdb (preferable)
        #print(candidates)
        #assert False

        chosen = [tok for _, tok, _ in candidates[:top_k]]
        return chosen, candidates[:top_k]

    def _precompute_dp_span_deltas(self, dp_min_list):
        """Precompute dp[start] - dp[start+L] tensors for every paragraph and length."""
        cache = []
        for idx, dp in enumerate(dp_min_list):
            if dp.device != self.pricing_device:
                dp = dp.to(self.pricing_device)
                dp_min_list[idx] = dp
            dp_cache = {}
            dp_len = dp.shape[0]
            max_len = min(self.max_token_len, max(dp_len - 1, 0))
            for L in range(1, max_len + 1):
                if dp_len <= L:
                    break
                dp_cache[L] = dp[:-L] - dp[L:]
            cache.append(dp_cache)
        return cache


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

        if verbose and self._is_primary_rank():
            print("\nStep 2: Starting training with batch pricing...")
        for it in range(1, max_iterations + 1):
            self._cost_cache.clear()
            all_dp_min = self._collect_dp_min(len(paragraphs_texts))

            if self._is_primary_rank():
                new_tokens, topk_info = self._find_best_new_tokens_batch(all_dp_min, top_k=self.top_k_add)
            else:
                new_tokens, topk_info = [], []
            new_tokens, topk_info = self._broadcast_candidates(new_tokens, topk_info)

            forced_tokens = []
            if self.force_seed_uniseg_tokens:
                if self._is_primary_rank():
                    forced_tokens = [
                        tok for tok in self._seeded_uniseg_tokens
                        if tok not in self.tok2id
                    ]
                forced_tokens = broadcast_object(forced_tokens, self.dist)
            if forced_tokens:
                combined = []
                seen = set()
                for tok in new_tokens + forced_tokens:
                    if tok not in seen:
                        seen.add(tok)
                        combined.append(tok)
                new_tokens = combined

            min_rc = topk_info[0][0] if topk_info else 0.0
            if not new_tokens:
                if verbose and self._is_primary_rank():
                    print(f"\nConvergence: no negative reduced-cost tokens (min summed RC = {min_rc:.6f}).")
                break

            # Check for convergence.
            if min_rc >= rc_stop_tol:
                if verbose and self._is_primary_rank():
                    print(f"\nConvergence: min summed RC = {min_rc:.6f} >= tol {rc_stop_tol:.1e}.")
                break

            # Add the new tokens to the vocabulary.
            for tok in new_tokens:
                tok = ud.normalize("NFC", tok)
                if tok not in self.tok2id:
                    self.tok2id[tok] = len(self.vocab)
                    self.vocab.append(tok)
                    self._update_vocab_prefixes([tok])
                self._seeded_uniseg_tokens.discard(tok)

            if verbose and self._is_primary_rank():
                preview = ", ".join([f"'{t}'" for t in new_tokens[:6]])
                more = "" if len(new_tokens) <= 6 else f" ... (+{len(new_tokens)-6})"
                print(f"Iter {it:02d}: Added {len(new_tokens)} tokens: {preview}{more} (best summed RC={min_rc:.4f})")
            self._distributed_barrier()
        else:
            if verbose and self._is_primary_rank():
                print("\nMax iterations reached.")

        # If a vocab budget is set, enforce it now.
        if self.vocab_budget is not None and len(self.vocab) > self.vocab_budget:
            self._enforce_vocab_budget_bisection(paragraphs_texts, self.vocab_budget, verbose=verbose)
            # Prune exported vocab to active multi-char tokens (plus alphabet)
            self._prune_vocab_to_active(paragraphs_texts)

        if verbose and self._is_primary_rank():
            print(f"\nTraining complete. Final vocabulary size: {len(self.vocab)}")


    def _count_types_used(self, paragraphs_texts):
        """Counts the number of unique multi-character tokens used to segment the corpus."""
        used = set()
        for pi in shard_indices(len(paragraphs_texts), self.dist):
            toks, U, _ = self._dp_decode(pi, decode_only=False)
            used |= {u for u in U if (len(u) > 1) and (not self._is_protected_token(u))}
        used = merge_sets(used, self.dist)
        return len(used)

    def _collect_active_multichar(self, paragraphs_texts):
        active = set()
        for pi in shard_indices(len(paragraphs_texts), self.dist):
            _, U, _ = self._dp_decode(pi, decode_only=False)
            for u in U:
                if len(u) > 1 and not self._is_protected_token(u):
                    active.add(ud.normalize("NFC", u))
        active = merge_sets(active, self.dist)
        return active

    def _prune_vocab_to_active(self, paragraphs_texts):
        """Removes any unused tokens from the final vocabulary."""
        active = self._collect_active_multichar(paragraphs_texts)
        singles = [t for t in self.vocab if len(t) == 1]
        # Final vocab is the union of all single characters and active multi-char tokens.
        new_vocab = sorted(set(singles)) + sorted(active)
        self.vocab = new_vocab
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}
        self._rebuild_vocab_prefixes()

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
