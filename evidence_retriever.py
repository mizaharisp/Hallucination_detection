"""
evidence_retriever.py  -  Module: Evidence Retrieval and Similarity Analysis

Input : claims_output.json   (written by claim_processor.py)
          {"query": "...", "claims": [{"claim_id": "C1", "claim": "..."}]}
Output: evidence_output.json (proposed format, see build_output())

Steps:  claim -> embedding -> cosine similarity vs. every evidence passage
        -> rank -> keep top-k above a threshold -> hand to the NLI module.

This module does NOT decide true/false. Low similarity only means
"no relevant evidence found"; the NLI module judges entail/contradict/neutral.
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np

from dataset_evidence import Passage, load_halueval, load_truthfulqa

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
# Starting value only. It is NOT calibrated on your data - tune it (see README steps).
DEFAULT_MIN_SIMILARITY = 0.30


# ---------------------------------------------------------------- embeddings
class SentenceEmbedder:
    """Turns text into vectors (lists of numbers) with a sentence-embedding model."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def encode(self, texts):
        if self._model is None:                         # load lazily (first use)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(list(texts), convert_to_numpy=True),
                          dtype=np.float32)


# ---------------------------------------------------------------- similarity
def cosine_similarity(claim_vec, evidence_matrix):
    """
    cosine(a, b) = (a . b) / (|a| * |b|)
    1 = same direction (very similar meaning), 0 = unrelated.
    claim_vec: (d,)   evidence_matrix: (n, d)   ->   (n,) scores
    """
    claim_vec = np.asarray(claim_vec, dtype=np.float32)
    evidence_matrix = np.asarray(evidence_matrix, dtype=np.float32)
    if evidence_matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    denom = np.linalg.norm(claim_vec) * np.linalg.norm(evidence_matrix, axis=1)
    denom = np.where(denom == 0, 1e-12, denom)          # avoid divide-by-zero
    return (evidence_matrix @ claim_vec) / denom


# ---------------------------------------------------------------- claim checks
def is_checkable_claim(claim):
    """
    Cheap safety check on text coming from the claim extractor.
    Rejects empty text, non-strings, headings ending in ':' and fragments with
    fewer than 3 alphabetic words (e.g. '└── Machine Learning (ML)' still passes;
    that is fine - we only block clear non-claims, we do not judge truth here).
    """
    if not isinstance(claim, str) or not claim.strip():
        return False
    text = claim.strip()
    if text.endswith(":"):
        return False
    return len(re.findall(r"[A-Za-z]{2,}", text)) >= 3


# ---------------------------------------------------------------- index
class EvidenceIndex:
    def __init__(self, passages, embedder=None, cache_path=None):
        self.embedder = embedder or SentenceEmbedder()
        self.passages = self._dedupe(passages)
        self.vectors = self._build_vectors(cache_path)

    @staticmethod
    def _dedupe(passages):
        seen, unique = set(), []
        for p in passages:
            key = re.sub(r"\s+", " ", p.text.strip().lower())
            if key and key not in seen:                 # skip empty + duplicate text
                seen.add(key)
                unique.append(p)
        return unique

    def _build_vectors(self, cache_path):
        if not self.passages:
            return np.zeros((0, 1), dtype=np.float32)
        fingerprint = hashlib.sha256(
            (getattr(self.embedder, "model_name", "custom") + "".join(
                p.passage_id + p.text for p in self.passages)).encode("utf-8")
        ).hexdigest()
        if cache_path and os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["fingerprint"]) == fingerprint:
                return cached["vectors"]
        vectors = self.embedder.encode([p.text for p in self.passages])
        if cache_path:
            np.savez(cache_path, vectors=vectors, fingerprint=fingerprint)
        return vectors

    def search(self, claim_vec, top_k=DEFAULT_TOP_K, min_similarity=DEFAULT_MIN_SIMILARITY):
        """Return (kept_passages, best_similarity_seen)."""
        if len(self.passages) == 0:
            return [], None
        scores = cosine_similarity(claim_vec, self.vectors)
        order = np.argsort(-scores)                     # highest score first
        best = float(scores[order[0]])
        kept = []
        for idx in order[:max(top_k, 0)]:
            if scores[idx] < min_similarity:
                break
            p = self.passages[idx]
            kept.append({
                "rank": len(kept) + 1,
                "passage_id": p.passage_id,
                "text": p.text,
                "similarity": round(float(scores[idx]), 4),
                "source_type": p.source_type,
                "dataset": p.dataset,
                "record_id": p.record_id,
                "evidence_role": p.evidence_role,
                "metadata": p.metadata,
            })
        return kept, best


# ---------------------------------------------------------------- main logic
def retrieve_for_claims(claims, index, top_k=DEFAULT_TOP_K, min_similarity=DEFAULT_MIN_SIMILARITY):
    """
    claims: list of {"claim_id": ..., "claim": ...}  (format from claim_processor.py)
    returns one result dict per input claim, same order.
    """
    results = [None] * len(claims)
    to_embed = []                                       # (position, text) of valid claims
    for pos, item in enumerate(claims):
        item = item if isinstance(item, dict) else {}
        claim_id, text = item.get("claim_id", f"C{pos + 1}"), item.get("claim")
        base = {"claim_id": claim_id, "claim": text, "evidence": [], "max_similarity": None}
        if is_checkable_claim(text):
            results[pos] = base
            to_embed.append((pos, text.strip()))
        else:
            results[pos] = {**base, "status": "invalid_claim"}

    if to_embed:
        vectors = index.embedder.encode([t for _, t in to_embed])   # one batch call
        for (pos, _), vec in zip(to_embed, vectors):
            kept, best = index.search(vec, top_k, min_similarity)
            results[pos]["evidence"] = kept
            results[pos]["max_similarity"] = None if best is None else round(best, 4)
            results[pos]["status"] = "ok" if kept else "no_relevant_evidence"
    return results


def build_output(claim_data, index, top_k, min_similarity):
    return {
        "query": claim_data.get("query"),
        "retrieval_config": {
            "embedding_model": getattr(index.embedder, "model_name", "custom"),
            "similarity": "cosine", "top_k": top_k, "min_similarity": min_similarity,
            "num_indexed_passages": len(index.passages),
        },
        "note": ("Evidence comes from evaluation-dataset fields, not an independent "
                 "knowledge base. Similarity is relevance, not truth."),
        "results": retrieve_for_claims(claim_data.get("claims", []), index, top_k, min_similarity),
    }


def load_claim_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError(f"{path}: expected {{'query': ..., 'claims': [...]}}")
    return data


def main():
    ap = argparse.ArgumentParser(description="Evidence retrieval + similarity for extracted claims")
    ap.add_argument("--claims", default="claims_output.json")
    ap.add_argument("--truthfulqa", help="path to TruthfulQA .csv/.json/.jsonl")
    ap.add_argument("--halueval", help="path to HaluEval QA .json/.jsonl")
    ap.add_argument("--out", default="evidence_output.json")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    ap.add_argument("--limit", type=int, help="only use first N dataset records (quick tests)")
    ap.add_argument("--cache", default="evidence_embeddings.npz")
    args = ap.parse_args()

    if not (args.truthfulqa or args.halueval):
        raise SystemExit("Give at least one evidence file: --truthfulqa and/or --halueval")

    passages = []
    if args.truthfulqa:
        passages += load_truthfulqa(args.truthfulqa, limit=args.limit)
    if args.halueval:
        passages += load_halueval(args.halueval, limit=args.limit)

    claim_data = load_claim_file(args.claims)
    index = EvidenceIndex(passages, cache_path=args.cache)
    output = build_output(claim_data, index, args.top_k, args.min_similarity)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    counts = {}
    for r in output["results"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Indexed {len(index.passages)} passages. Claim statuses: {counts}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
