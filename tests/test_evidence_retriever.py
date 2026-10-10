"""
Tests for evidence_retriever.py / dataset_evidence.py   (run: python -m unittest -v)

ALL data below is SYNTHETIC. The files are written in the documented
TruthfulQA / HaluEval layouts so the loaders can be tested, but none of these
rows are real dataset records. The embedder is a simple word-count TEST DOUBLE
(no model download), so these tests check the logic, not the quality of MiniLM.
"""
import csv
import json
import os
import re
import tempfile
import unittest

import numpy as np

from dataset_evidence import Passage, load_halueval, load_truthfulqa
from evidence_retriever import (EvidenceIndex, build_output, cosine_similarity,
                                is_checkable_claim, retrieve_for_claims)


class FakeEmbedder:
    """TEST DOUBLE: hashes words into a 64-dim count vector."""
    model_name = "fake-bow-test-double"

    def encode(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, sum(map(ord, w)) % 64] += 1
        return out


def P(pid, text, role="reference_correct_answer"):
    return Passage(pid, text, "dataset_reference_answer", "synthetic", pid, role)


PASSAGES = [
    P("p1", "Paris is the capital city of France"),
    P("p2", "Water boils at one hundred degrees Celsius at sea level"),
    P("p3", "Paris is not the capital of France"),            # synthetic contradiction
]


class TestSimilarity(unittest.TestCase):
    def test_cosine_values(self):
        s = cosine_similarity([1, 0], [[1, 0], [0, 1], [-1, 0]])
        np.testing.assert_allclose(s, [1, 0, -1], atol=1e-6)

    def test_zero_vector_is_safe(self):
        s = cosine_similarity([0, 0], [[1, 0]])
        self.assertTrue(np.isfinite(s).all())

    def test_empty_evidence(self):
        self.assertEqual(len(cosine_similarity([1, 0], np.zeros((0, 2)))), 0)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.index = EvidenceIndex(PASSAGES, FakeEmbedder())

    def run_claims(self, texts, **kw):
        claims = [{"claim_id": f"C{i+1}", "claim": t} for i, t in enumerate(texts)]
        return retrieve_for_claims(claims, self.index, min_similarity=0.5, **kw)

    def test_supporting_evidence_ranked_first(self):
        r = self.run_claims(["Paris is the capital city of France"])[0]
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["evidence"][0]["passage_id"], "p1")
        self.assertEqual(r["evidence"][0]["rank"], 1)
        sims = [e["similarity"] for e in r["evidence"]]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_contradicting_passage_is_retrieved_not_judged(self):
        # Retrieval only returns relevant text; deciding "contradiction" is the NLI module's job.
        r = self.run_claims(["Paris is the capital city of France"], top_k=3)[0]
        self.assertIn("p3", [e["passage_id"] for e in r["evidence"]])

    def test_no_relevant_evidence(self):
        r = self.run_claims(["Quantum entanglement violates Bell inequalities"])[0]
        self.assertEqual(r["status"], "no_relevant_evidence")
        self.assertEqual(r["evidence"], [])
        self.assertIsNotNone(r["max_similarity"])

    def test_multiple_claims_keep_order_and_ids(self):
        rs = self.run_claims(["Paris is the capital city of France",
                              "Water boils at one hundred degrees Celsius"])
        self.assertEqual([x["claim_id"] for x in rs], ["C1", "C2"])
        self.assertEqual(rs[1]["evidence"][0]["passage_id"], "p2")

    def test_invalid_inputs(self):
        rs = retrieve_for_claims(
            [{"claim_id": "C1", "claim": ""}, {"claim_id": "C2", "claim": None},
             {"claim_id": "C3", "claim": "These tasks include things like:"}, "oops", {}],
            self.index)
        self.assertTrue(all(r["status"] == "invalid_claim" for r in rs))

    def test_empty_claim_list_and_empty_index(self):
        self.assertEqual(retrieve_for_claims([], self.index), [])
        empty = EvidenceIndex([], FakeEmbedder())
        r = retrieve_for_claims([{"claim_id": "C1", "claim": "Paris is the capital of France"}], empty)[0]
        self.assertEqual(r["status"], "no_relevant_evidence")

    def test_duplicates_removed(self):
        idx = EvidenceIndex([P("a", "Same text here"), P("b", "same  TEXT here"), P("c", "")],
                            FakeEmbedder())
        self.assertEqual(len(idx.passages), 1)

    def test_checkable_claim(self):
        self.assertTrue(is_checkable_claim("The sky is blue today"))
        self.assertFalse(is_checkable_claim("Status:"))
        self.assertFalse(is_checkable_claim("   "))


class TestOutputContract(unittest.TestCase):
    """Checks the output shape proposed for the NLI module (the real NLI code was not available)."""
    def test_output_is_json_and_has_required_fields(self):
        index = EvidenceIndex(PASSAGES, FakeEmbedder())
        out = build_output({"query": "q", "claims": [
            {"claim_id": "C1", "claim": "Paris is the capital city of France"}]}, index, 3, 0.5)
        out = json.loads(json.dumps(out))                    # must be serialisable
        item = out["results"][0]
        for key in ("claim_id", "claim", "status", "evidence", "max_similarity"):
            self.assertIn(key, item)
        for key in ("rank", "passage_id", "text", "similarity", "source_type", "record_id"):
            self.assertIn(key, item["evidence"][0])


class TestLoaders(unittest.TestCase):
    def test_truthfulqa_csv_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tqa.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Type", "Category", "Question", "Best Answer", "Best Incorrect Answer",
                            "Correct Answers", "Incorrect Answers", "Source"])
                w.writerow(["Adversarial", "Test", "SYNTH question?", "Right one",
                            "WRONG ONE", "Right one; Another right", "WRONG ONE; Wrong two", "http://x"])
            ps = load_truthfulqa(path)
            texts = " | ".join(p.text for p in ps)
            self.assertEqual(len(ps), 2)                      # duplicates merged
            self.assertNotIn("WRONG", texts)                  # incorrect answers never loaded
            self.assertEqual(ps[0].source_type, "dataset_reference_answer")

    def test_truthfulqa_underscore_columns_and_list_text(self):
        # SYNTHETIC: HF-style lower-case/underscore headers, answers saved as list text
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "generation_validation.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["type", "category", "question", "best_answer",
                            "correct_answers", "incorrect_answers", "source"])
                w.writerow(["Adversarial", "Test", "SYNTH q?", "Right one",
                            "['Right one', 'Other right; with semicolon']", "['WRONG']", "http://x"])
            ps = load_truthfulqa(path)
            self.assertEqual(len(ps), 2)
            self.assertNotIn("WRONG", " ".join(p.text for p in ps))

    def test_halueval_jsonl_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "qa.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"knowledge": "Synthetic knowledge text.", "question": "SYNTH q?",
                                    "right_answer": "good", "hallucinated_answer": "BAD"}) + "\n")
            ps = load_halueval(path)
            self.assertEqual({p.source_type for p in ps},
                             {"dataset_knowledge_context", "dataset_reference_answer"})
            self.assertNotIn("BAD", " ".join(p.text for p in ps))
            only_ctx = load_halueval(path, use_reference_answers=False)
            self.assertEqual(len(only_ctx), 1)

    def test_missing_column_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                f.write(json.dumps({"foo": 1}) + "\n")
            with self.assertRaises(ValueError) as cm:
                load_halueval(path)
            self.assertIn("foo", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
