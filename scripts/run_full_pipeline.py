import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from tokenizer_core.tokenizer import ScalableTokenizer


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def ensure_langs(langs_path: Path | None, corpus: List[str]) -> List[str]:
    if langs_path is None:
        # Default to language guessed by tokenizer (None) using placeholder
        return [None] * len(corpus)
    langs = read_lines(langs_path)
    if len(langs) != len(corpus):
        raise ValueError("Language file length must match corpus length.")
    return langs


def tokenize_corpus(tokenizer: ScalableTokenizer, docs: List[str], langs: List[str]) -> List[List[str]]:
    tokenized = []
    for doc, lang in zip(docs, langs):
        tokens = tokenizer.tokenize(doc, lang=lang)
        tokenized.append(tokens)
    return tokenized


def compute_compactness(tokenized: List[List[str]], docs: List[str]) -> dict:
    total_tokens = sum(len(toks) for toks in tokenized)
    total_chars = sum(len(doc) for doc in docs)
    total_words = sum(len(doc.split()) for doc in docs)
    tpc = total_tokens / max(total_chars, 1)
    tpw = total_tokens / max(total_words, 1)
    return {"tpc": tpc, "tpw": tpw}


def compute_distribution_metrics(tokenized: List[List[str]], rare_threshold: int = 100) -> dict:
    counts = Counter(token for doc in tokenized for token in doc)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())
    sorted_counts = sorted(counts.values())
    V = len(sorted_counts)
    gini = 1 - (2 / max(V, 1)) * sum(((V + 1 - i) / max(V, 1)) * (cnt / total) for i, cnt in enumerate(sorted_counts, start=1))
    rare_frac = sum(1 for c in counts.values() if c <= rare_threshold) / max(V, 1)
    return {"entropy": entropy, "gini": gini, "rare_type_fraction": rare_frac}


def load_morph_dataset(path: Path, limit: int | None = None) -> List[Tuple[str, List[int]]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        word = obj.get("word", "")
        if not word:
            continue
        bounds = obj.get("boundaries", [])
        entries.append((word, [int(b) for b in bounds]))
        if limit and len(entries) >= limit:
            break
    return entries


def get_token_spans(tokens: List[str]) -> List[Tuple[int, int]]:
    spans = []
    cursor = 0
    for tok in tokens:
        length = len(tok)
        spans.append((cursor, cursor + length))
        cursor += length
    return spans


def calc_overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> bool:
    return max(span_a[0], span_b[0]) < min(span_a[1], span_b[1])


def compute_morph_metrics(tokenizer: ScalableTokenizer, morph_file: Path, morph_lang: str, limit: int | None = None) -> dict:
    dataset = load_morph_dataset(morph_file, limit)
    if not dataset:
        return {}
    total_spm = 0.0
    words_seen = 0
    total_prec_num = 0
    total_prec_den = 0
    total_rec_num = 0
    total_rec_den = 0

    for word, boundaries in dataset:
        tokens = tokenizer.tokenize(word, lang=morph_lang)
        if not tokens:
            continue
        word_len = len(word)
        morpheme_positions = [0] + sorted(boundaries) + [word_len]
        morpheme_spans = [(morpheme_positions[i], morpheme_positions[i + 1]) for i in range(len(morpheme_positions) - 1)]
        token_spans = get_token_spans(tokens)
        # SPM calculation
        per_word_spm = 0.0
        for span in morpheme_spans:
            overlaps = sum(1 for token_span in token_spans if calc_overlap(token_span, span))
            per_word_spm += overlaps
        per_word_spm /= max(len(morpheme_spans), 1)
        total_spm += per_word_spm
        words_seen += 1

        morpheme_boundaries = set(boundaries)
        token_boundaries = set(span[1] for span in token_spans[:-1])  # exclude end-of-word boundary
        tp = len(token_boundaries & morpheme_boundaries)
        total_prec_num += tp
        total_prec_den += max(len(token_boundaries), 1)
        total_rec_num += tp
        total_rec_den += max(len(morpheme_boundaries), 1)

    if words_seen == 0:
        return {}
    precision = total_prec_num / max(total_prec_den, 1)
    recall = total_rec_num / max(total_rec_den, 1)
    f1 = 2 * precision * recall / max((precision + recall), 1e-8)
    spm = total_spm / words_seen
    return {"spm": spm, "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        logits = self.linear(out)
        return logits


def build_windows(
    token_ids: List[int],
    block_size: int,
    max_windows: int,
    stride: int | None = None,
) -> List[List[int]]:
    """Builds possibly-overlapping training windows."""
    if block_size < 1 or len(token_ids) <= block_size:
        return []
    step = stride or max(1, block_size // 2)
    limit = len(token_ids) - (block_size + 1)
    if limit < 0:
        return []
    windows: List[List[int]] = []
    for start in range(0, limit + 1, step):
        end = start + block_size + 1
        window = token_ids[start:end]
        if len(window) == block_size + 1:
            windows.append(window)
        if len(windows) >= max_windows:
            break
    return windows


def _ensure_min_length(seq: List[int], min_len: int) -> List[int]:
    if len(seq) >= min_len or not seq:
        return seq
    out = list(seq)
    while len(out) < min_len:
        take = min(len(seq), min_len - len(out))
        out.extend(seq[:take])
    return out


def _split_train_val_ids(
    token_ids: List[int],
    block_size: int,
    val_ratio: float,
) -> Tuple[List[int], List[int]]:
    min_len = block_size + 1
    total = len(token_ids)
    if total < min_len * 2:
        # Extremely small corpora: fall back to using the same material for train/val.
        repeated = _ensure_min_length(token_ids, min_len)
        return repeated, repeated
    split_idx = int(total * (1 - val_ratio))
    split_idx = max(min_len, min(split_idx, total - min_len))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    return _ensure_min_length(train_ids, min_len), _ensure_min_length(val_ids, min_len)


def train_lm(
    tokenized_docs: List[List[str]],
    tok2id: dict,
    args,
    doc_langs: List[str] | None = None,
) -> dict:
    flat_tokens = [tok for doc in tokenized_docs for tok in doc]
    token_ids = [tok2id[tok] for tok in flat_tokens if tok in tok2id]
    total_tokens = len(token_ids)
    if total_tokens < 4:
        return {"lm_nll": float("nan"), "lm_perplexity": float("nan")}
    effective_block = min(args.lm_block_size, max(4, total_tokens - 2))
    train_ids, val_ids = _split_train_val_ids(token_ids, effective_block, args.lm_val_ratio)
    train_windows = build_windows(train_ids, effective_block, args.lm_max_windows)
    val_windows = build_windows(val_ids, effective_block, max(args.lm_max_windows // 5, 1))
    stride_candidates = [max(1, effective_block // 2), 1]
    while (not train_windows or not val_windows) and effective_block > 4:
        effective_block = max(4, effective_block // 2)
        train_ids, val_ids = _split_train_val_ids(token_ids, effective_block, args.lm_val_ratio)
        train_windows = build_windows(train_ids, effective_block, args.lm_max_windows)
        val_windows = build_windows(val_ids, effective_block, max(args.lm_max_windows // 5, 1))
    if not train_windows or not val_windows:
        for stride in stride_candidates:
            if not train_windows:
                train_windows = build_windows(train_ids, effective_block, args.lm_max_windows, stride=stride)
            if not val_windows:
                val_windows = build_windows(val_ids, effective_block, max(args.lm_max_windows // 5, 1), stride=stride)
            if train_windows and val_windows:
                break
    if not train_windows or not val_windows:
        return {"lm_nll": float("nan"), "lm_perplexity": float("nan")}

    device = torch.device(args.lm_device)
    model = TinyLM(len(tok2id), args.lm_embed_dim, args.lm_hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lm_lr)
    criterion = nn.CrossEntropyLoss()

    rng = random.Random(getattr(args, "lm_seed", 13))

    def iter_batches(windows: List[List[int]], shuffle: bool) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not windows:
            return []
        order = list(range(len(windows)))
        if shuffle:
            rng.shuffle(order)
        batches = []
        for i in range(0, len(order), args.lm_batch_size):
            batch_idx = order[i : i + args.lm_batch_size]
            batch = [windows[idx] for idx in batch_idx]
            xs = torch.tensor([w[:-1] for w in batch], dtype=torch.long, device=device)
            ys = torch.tensor([w[1:] for w in batch], dtype=torch.long, device=device)
            batches.append((xs, ys))
        return batches

    def evaluate_windows(windows: List[List[int]]) -> dict | None:
        if not windows:
            return None
        eval_batches = iter_batches(windows, shuffle=False)
        if not eval_batches:
            return None
        prev_mode = model.training
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for xs, ys in eval_batches:
                logits = model(xs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), ys.reshape(-1))
                total_loss += loss.item() * xs.numel()
                total_tokens += xs.numel()
        if prev_mode:
            model.train()
        if total_tokens == 0:
            return None
        nll = total_loss / total_tokens
        ppl = math.exp(nll)
        return {"lm_nll": nll, "lm_perplexity": ppl}

    patience = max(0, getattr(args, "lm_patience", 0))
    wait = 0
    best_loss = float("inf")
    best_state = None

    model.train()
    for _ in range(args.lm_epochs):
        for xs, ys in iter_batches(train_windows, shuffle=True):
            optimizer.zero_grad()
            logits = model(xs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), ys.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.lm_clip)
            optimizer.step()
        val_snapshot = evaluate_windows(val_windows)
        if val_snapshot is None:
            continue
        current_loss = val_snapshot["lm_nll"]
        if current_loss + 1e-6 < best_loss:
            best_loss = current_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience and wait >= patience:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    global_metrics = evaluate_windows(val_windows)
    if global_metrics is None:
        return {"lm_nll": float("nan"), "lm_perplexity": float("nan")}
    global_metrics["lm_block_size_used"] = effective_block

    per_lang_metrics = {}
    if doc_langs:
        lang_to_ids: dict[str | None, List[int]] = {}
        for doc_tokens, lang in zip(tokenized_docs, doc_langs):
            ids = [tok2id[tok] for tok in doc_tokens if tok in tok2id]
            if ids:
                lang_to_ids.setdefault(lang, []).extend(ids)
        for lang, ids in lang_to_ids.items():
            if len(ids) <= effective_block + 1:
                continue
            split_idx = int(len(ids) * (1 - args.lm_val_ratio))
            split_idx = max(effective_block + 1, min(split_idx, len(ids) - (effective_block + 1)))
            lang_val_ids = ids[split_idx:]
            lang_val_ids = _ensure_min_length(lang_val_ids, effective_block + 1)
            lang_windows = build_windows(
                lang_val_ids,
                effective_block,
                max(args.lm_max_windows // 5, 1),
            )
            metrics = evaluate_windows(lang_windows)
            if metrics:
                per_lang_metrics[lang or "unknown"] = metrics

    if per_lang_metrics:
        global_metrics["lm_per_lang"] = per_lang_metrics
    return global_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Full tokenizer training + evaluation pipeline.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--langs", type=Path)
    parser.add_argument("--eval_corpus", type=Path, help="Optional separate corpus for evaluation metrics.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--max_token_len", type=int, default=12)
    parser.add_argument("--min_freq", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--top_k_add", type=int, default=32)
    parser.add_argument("--vocab_budget", type=int, default=4000)
    parser.add_argument("--merge_reward", type=float, default=0.05)
    parser.add_argument("--short_penalty", type=float, default=0.2)
    parser.add_argument("--space_penalty", type=float, default=0.25)
    parser.add_argument("--uniseg_root", type=Path, default=Path("data/uniseg_word_segments"))
    parser.add_argument("--uniseg_reward", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pricing_device", default="cpu")
    parser.add_argument("--seed_uniseg_segments", action="store_true")
    parser.add_argument("--force_seed_uniseg_tokens", action="store_true")
    parser.add_argument("--morph_file", type=Path, default=Path("data/uniseg_word_segments/eng/MorphoLex.jsonl"))
    parser.add_argument("--morph_lang", default="en")
    parser.add_argument("--morph_limit", type=int, default=1000)
    parser.add_argument("--lm_device", default="cuda")
    parser.add_argument("--lm_epochs", type=int, default=8)
    parser.add_argument("--lm_block_size", type=int, default=128)
    parser.add_argument("--lm_batch_size", type=int, default=32)
    parser.add_argument("--lm_embed_dim", type=int, default=128)
    parser.add_argument("--lm_hidden_dim", type=int, default=256)
    parser.add_argument("--lm_lr", type=float, default=1e-3)
    parser.add_argument("--lm_clip", type=float, default=1.0)
    parser.add_argument("--lm_max_windows", type=int, default=20000)
    parser.add_argument("--lm_val_ratio", type=float, default=0.15)
    parser.add_argument("--lm_patience", type=int, default=3, help="Early-stop GRU training after N non-improving epochs (0 disables).")
    parser.add_argument("--lm_seed", type=int, default=13)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    corpus = read_lines(args.corpus)
    langs = ensure_langs(args.langs, corpus)
    eval_corpus = read_lines(args.eval_corpus) if args.eval_corpus else corpus
    eval_langs = ensure_langs(args.langs, corpus) if args.eval_corpus else langs

    tokenizer = ScalableTokenizer(
        max_token_len=args.max_token_len,
        min_freq=args.min_freq,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        top_k_add=args.top_k_add,
        vocab_budget=args.vocab_budget,
        merge_reward=args.merge_reward,
        short_penalty=args.short_penalty,
        space_penalty=args.space_penalty,
        device=args.device,
        uniseg_root=args.uniseg_root,
        uniseg_reward=args.uniseg_reward,
        use_morph_encoder=False,
        seed_uniseg_segments=args.seed_uniseg_segments,
        force_seed_uniseg_tokens=args.force_seed_uniseg_tokens,
        pricing_device=args.pricing_device,
    )
    tokenizer.train(corpus, langs, max_iterations=args.max_iterations, verbose=not args.quiet)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tokenizer_checkpoint.json"
    tokenizer.save(checkpoint_path, include_morphology=True)

    tokenized_eval = tokenize_corpus(tokenizer, eval_corpus, eval_langs)
    metrics = {}
    metrics.update(compute_compactness(tokenized_eval, eval_corpus))
    metrics.update(compute_distribution_metrics(tokenized_eval))
    metrics.update(compute_morph_metrics(tokenizer, args.morph_file, args.morph_lang, args.morph_limit))
    metrics.update(train_lm(tokenized_eval, tokenizer.tok2id, args, doc_langs=eval_langs))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

