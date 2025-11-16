from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenizer_core.main_experiments import maybe_run_segmentation_eval
from tokenizer_core.segmentation_eval import evaluate_sentences_with_uniseg


class _WhitespaceTokenizer:
    """Minimal tokenizer that splits on whitespace for testing."""

    def tokenize(self, text: str, lang: str | None = None):
        return text.split()


def _write_uniseg_fixture(root: Path, lang: str, dataset: str, words: list[str]) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_path = dataset_dir / f"UniSegments-1.0-{dataset}.useg"
    with file_path.open("w", encoding="utf-8") as handle:
        for word in words:
            meta = {"segmentation": [{"morpheme": word, "type": "stem"}]}
            fields = [word, "_", "_", "_", json.dumps(meta, ensure_ascii=False)]
            handle.write("\t".join(fields) + "\n")


class SentenceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.lang = "xx"
        self.dataset = "xx-mock"
        _write_uniseg_fixture(self.root, self.lang, self.dataset, ["alpha", "beta", "gamma"])

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_sentence_level_metrics(self) -> None:
        tokenizer = _WhitespaceTokenizer()
        sentences = ["alpha beta", "gamma alpha"]
        languages = [self.lang, self.lang]

        result = evaluate_sentences_with_uniseg(
            tokenizer,
            sentences,
            languages,
            uniseg_root=self.root,
            lang_map={self.lang: [self.dataset]},
        )

        self.assertGreater(result["sentences_evaluated"], 0)
        expected_similarity = 0.5 * result["boundary_f1"] + 0.5 * result["morphological_score"]
        self.assertAlmostEqual(result["sentence_similarity"], expected_similarity, places=6)
        self.assertGreaterEqual(result["morphological_score"], 0.0)
        self.assertIn("morphology", result)

    def test_maybe_run_segmentation_eval_aggregates(self) -> None:
        tokenizer = _WhitespaceTokenizer()
        lang_codes = {self.lang: "Mock"}
        references = {"space": lambda text, lang=None: text.split()}
        eval_samples = [
            {"text": "alpha beta", "language": self.lang},
            {"text": "gamma alpha", "language": self.lang},
        ]
        external_cfg = {
            "languages": [self.lang],
            "uniseg_root": str(self.root),
            "lang_map": {self.lang: [self.dataset]},
            "evaluate_sentences": True,
            "compare_references": True,
        }

        results = maybe_run_segmentation_eval(
            tokenizer,
            lang_codes,
            external_cfg,
            references,
            eval_samples=eval_samples,
        )

        self.assertIn("word_level", results)
        self.assertIn("sentence_level", results)

        word_level = results["word_level"]
        self.assertIn("per_language", word_level)
        self.assertIn(self.lang, word_level["per_language"])
        self.assertIn("trained", word_level["per_language"][self.lang])
        self.assertEqual(word_level["per_language"][self.lang]["trained"].get("mode"), "uniseg")

        sentence_level = results["sentence_level"]
        self.assertIn("aggregate", sentence_level)
        aggregate = sentence_level["aggregate"].get("trained")
        self.assertIsNotNone(aggregate)
        agg_similarity = aggregate["sentence_similarity"]
        expected_similarity = 0.5 * aggregate.get("boundary_f1", 0.0) + 0.5 * aggregate.get("morphological_score", 0.0)
        self.assertAlmostEqual(agg_similarity, expected_similarity, places=6)
        self.assertGreaterEqual(aggregate.get("words_evaluated", 0.0), 0.0)
        self.assertIn("meta", sentence_level)
        self.assertIn(self.lang, sentence_level["meta"].get("evaluated_languages", []))


if __name__ == "__main__":
    unittest.main()
