from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from .constants import CROSS_EQUIV

eps = 1e-9


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _collect_class_tokens(encoder, class_key: str) -> Dict[str, List[torch.Tensor]]:
    """Returns vectors for tokens that match a morphological class."""
    suffix_map = CROSS_EQUIV.get(class_key, {})
    lang_tokens: Dict[str, List[torch.Tensor]] = {}
    if not suffix_map:
        return lang_tokens
    for tok, vec in encoder.token_vec.items():
        for lang, suffixes in suffix_map.items():
            if any(tok.endswith(suf) for suf in suffixes):
                lang_tokens.setdefault(lang, []).append(vec)
                break
    return lang_tokens


def _collect_other_class_tokens(
    class_token_cache: Mapping[str, Dict[str, List[torch.Tensor]]],
    exclude_key: str,
    langs_subset: Optional[Iterable[str]],
) -> Dict[str, List[torch.Tensor]]:
    """Collects tokens from all other classes restricted to a language subset."""
    result: Dict[str, List[torch.Tensor]] = {}
    langs = set(langs_subset) if langs_subset is not None else None
    if langs is not None and not langs:
        return result
    for key, lang_tokens in class_token_cache.items():
        if key == exclude_key:
            continue
        for lang, vecs in lang_tokens.items():
            if not vecs:
                continue
            if langs is not None and lang not in langs:
                continue
            result.setdefault(lang, []).extend(vecs)
    return result


def _average_cosine(vecs: Sequence[torch.Tensor]) -> Optional[float]:
    if len(vecs) < 2:
        return None
    mat = torch.stack(vecs, dim=0)
    mat = _normalize(mat)
    sim = mat @ mat.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    if mask.sum() == 0:
        return None
    return float(sim[mask].mean().item())


def _cross_language_cosine(lang_tokens: Mapping[str, Sequence[torch.Tensor]]) -> Tuple[Optional[float], int]:
    langs = [lang for lang, toks in lang_tokens.items() if toks]
    if len(langs) < 2:
        return None, 0
    sims: List[float] = []
    for i, lang_a in enumerate(langs):
        Va = _normalize(torch.stack(lang_tokens[lang_a], dim=0))
        for lang_b in langs[i + 1 :]:
            Vb = _normalize(torch.stack(lang_tokens[lang_b], dim=0))
            pairwise = Va @ Vb.T
            sims.extend(pairwise.flatten().tolist())
    if not sims:
        return None, 0
    avg = sum(sims) / len(sims)
    return float(avg), len(sims)


def _random_pair_similarity(encoder, n_pairs: int = 256) -> float:
    items = list(encoder.token_vec.items())
    if len(items) < 2:
        return 0.0
    sims: List[float] = []
    for _ in range(min(n_pairs, len(items) // 2)):
        (tok_a, vec_a), (tok_b, vec_b) = random.sample(items, 2)
        va = _normalize(vec_a.unsqueeze(0))[0]
        vb = _normalize(vec_b.unsqueeze(0))[0]
        sims.append(float(torch.dot(va, vb).item()))
    return sum(sims) / len(sims) if sims else 0.0


def _length_correlation(encoder) -> float:
    lengths: List[float] = []
    norms: List[float] = []
    for tok, vec in encoder.token_vec.items():
        lengths.append(float(len(tok)))
        norms.append(float(torch.linalg.norm(vec).item()))
    if len(lengths) < 2:
        return 0.0
    L = torch.tensor(lengths, dtype=torch.float32)
    N = torch.tensor(norms, dtype=torch.float32)
    L = L - L.mean()
    N = N - N.mean()
    denom = torch.linalg.norm(L) * torch.linalg.norm(N)
    if denom.item() == 0.0:
        return 0.0
    return float(torch.dot(L, N) / denom)


@dataclass
class ClassEval:
    token_count: int
    languages: int
    avg_similarity: Optional[float]
    gain_vs_random: Optional[float]
    cross_similarity: Optional[float]
    cross_pairs: int
    cross_baseline: Optional[float]
    cross_baseline_pairs: int
    cross_gain_vs_baseline: Optional[float]


@dataclass
class MorphologyEvalResult:
    random_similarity: float
    length_correlation: float
    classes: Dict[str, ClassEval]


def evaluate_morphology_encoder(
    encoder,
    class_keys: Optional[Sequence[str]] = None,
    random_pairs: int = 512,
) -> MorphologyEvalResult:
    """
    Computes intrinsic quality metrics for a trained MorphologyEncoder.

    Args:
        encoder: A fitted MorphologyEncoder instance (must have token_vec populated).
        class_keys: Optional subset of CROSS_EQUIV keys to evaluate.
        random_pairs: Number of random token pairs for the baseline cosine similarity.
    """
    if not getattr(encoder, "token_vec", None):
        raise ValueError("MorphologyEncoder has no learned token vectors; fit() must be called first.")
    keys = list(class_keys or CROSS_EQUIV.keys())
    baseline = _random_pair_similarity(encoder, n_pairs=random_pairs)
    length_corr = _length_correlation(encoder)
    per_class: Dict[str, ClassEval] = {}
    class_token_cache: Dict[str, Dict[str, List[torch.Tensor]]] = {
        key: _collect_class_tokens(encoder, key) for key in keys
    }
    for key in keys:
        lang_tokens = class_token_cache.get(key, {})
        token_count = sum(len(v) for v in lang_tokens.values())
        if token_count < 2:
            continue
        langs_present = sum(1 for v in lang_tokens.values() if v)
        all_vecs: List[torch.Tensor] = [vec for vecs in lang_tokens.values() for vec in vecs]
        avg_sim = _average_cosine(all_vecs)
        cross_sim, cross_pairs = _cross_language_cosine(lang_tokens)
        gain = avg_sim - baseline if avg_sim is not None else None
        baseline_tokens = _collect_other_class_tokens(class_token_cache, key, lang_tokens.keys())
        if len(baseline_tokens) < 2:
            baseline_tokens = _collect_other_class_tokens(class_token_cache, key, None)
        cross_other_sim, cross_other_pairs = _cross_language_cosine(baseline_tokens)
        cross_gain = (
            None
            if cross_sim is None or cross_other_sim is None
            else cross_sim - cross_other_sim
        )
        per_class[key] = ClassEval(
            token_count=token_count,
            languages=langs_present,
            avg_similarity=avg_sim,
            gain_vs_random=gain,
            cross_similarity=cross_sim,
            cross_pairs=cross_pairs,
            cross_baseline=cross_other_sim,
            cross_baseline_pairs=cross_other_pairs,
            cross_gain_vs_baseline=cross_gain,
        )
    return MorphologyEvalResult(
        random_similarity=baseline,
        length_correlation=length_corr,
        classes=per_class,
    )


def summarize_morphology_eval(result: MorphologyEvalResult) -> str:
    """Creates a short human-readable summary of the evaluation."""
    lines = [
        f"Random pair cosine: {result.random_similarity:.4f}",
        f"Length correlation  : {result.length_correlation:.4f}",
        "",
        "Per-class cohesion (avg cosine | delta vs rand | cross-lingual)",
    ]
    if not result.classes:
        lines.append("  (no classes with enough tokens)")
        return "\n".join(lines)
    for key, stats in sorted(result.classes.items()):
        avg = "n/a" if stats.avg_similarity is None else f"{stats.avg_similarity:.4f}"
        gain = "n/a" if stats.gain_vs_random is None else f"{stats.gain_vs_random:+.4f}"
        cross = (
            "n/a"
            if stats.cross_similarity is None
            else f"{stats.cross_similarity:.4f} ({stats.cross_pairs} pairs)"
        )
        cross_base = (
            "n/a"
            if stats.cross_baseline is None
            else f"{stats.cross_baseline:.4f} ({stats.cross_baseline_pairs} pairs)"
        )
        cross_gain = (
            "n/a"
            if stats.cross_gain_vs_baseline is None
            else f"{stats.cross_gain_vs_baseline:+.4f}"
        )
        lines.append(
            f"  {key:8s} tokens={stats.token_count:3d} langs={stats.languages:2d} "
            f"avg={avg} gain={gain} cross={cross} vs_other={cross_base} (delta={cross_gain})"
        )
    return "\n".join(lines)

