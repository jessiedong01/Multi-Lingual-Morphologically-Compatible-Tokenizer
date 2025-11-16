"""
Quick manual smoke-test for the morphology encoder's mini-batch + optimiser paths.

Run this script whenever you touch `linguistic_features.py` to confirm that both
SGD and AdaGrad modes finish without numerical explosions. The corpus is tiny on
purpose so it runs in under a second.
"""

from __future__ import annotations

import torch

from linguistic_features import MorphologyEncoder


def tiny_corpus():
    paragraphs = [
        "The cats are napping on the mat.",
        "Les chats dorment sur le tapis.",
        "Koty śpią na dywanie.",
    ]
    tok_occurrences = {
        "cats": [(0, (4, 8))],
        "cat": [(0, (4, 7))],
        "chats": [(1, (4, 9))],
        "koty": [(2, (0, 4))],
        "kot": [(2, (0, 3))],
        "nap": [(0, (13, 16))],
        "dorment": [(1, (10, 17))],
    }
    lang_map = {0: "en", 1: "fr", 2: "pl"}
    paragraph_lang = lambda idx: lang_map.get(idx, "en")
    return paragraphs, tok_occurrences, paragraph_lang


def run_variant(name: str, **kwargs):
    print(f"== Running {name} ==")
    paragraphs, tok_occurrences, paragraph_lang = tiny_corpus()
    glove_iters = kwargs.pop("glove_iters", 3)
    encoder = MorphologyEncoder(
        embedding_mode="glove",
        glove_iters=glove_iters,
        glove_lr=kwargs.pop("glove_lr", 0.02),
        use_minibatch=kwargs.pop("use_minibatch", True),
        batch_size_pairs=kwargs.pop("batch_size_pairs", 16),
        batch_size_edges=kwargs.pop("batch_size_edges", 8),
        optimizer=kwargs.pop("optimizer", "adagrad"),
        lambda_morph=0.05,
        gamma=1e-3,
        **kwargs,
    )
    encoder.fit(paragraphs, tok_occurrences, paragraph_lang)
    norms = [float(torch.linalg.norm(v).item()) for v in encoder.token_vec.values()]
    print(f"tokens learned: {len(encoder.token_vec)} | norm range: ({min(norms):.3f}, {max(norms):.3f})")
    print(f"languages covered: {sorted(encoder.lang_proto)}")
    print()


if __name__ == "__main__":
    run_variant("AdaGrad minibatch")
    run_variant("SGD minibatch", optimizer="sgd", glove_lr=0.05)
    run_variant("AdaGrad full batch", use_minibatch=False, glove_lr=0.01, glove_iters=2)
