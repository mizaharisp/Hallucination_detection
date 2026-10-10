"""
dataset_evidence.py
Turns TruthfulQA / HaluEval files into a flat list of evidence passages.

IMPORTANT: these datasets are EVALUATION datasets, not encyclopedias.
Every passage keeps a `source_type` so later modules (and you, in the viva)
can tell the difference between:
  - "dataset_knowledge_context" : a supporting passage shipped with a record
                                  (HaluEval 'knowledge' field)
  - "dataset_reference_answer"  : the dataset's own correct answer (the answer key)
Incorrect / hallucinated answers are NEVER loaded as evidence.

The column names below come from the public dataset documentation. They have NOT
been checked against your actual files (the ZIP did not contain any dataset).
If a column is missing, the loader stops and prints the columns it did find.
"""
import csv
import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class Passage:
    passage_id: str
    text: str                      # text given to the embedder and to NLI
    source_type: str               # see module docstring
    dataset: str                   # "truthfulqa" or "halueval"
    record_id: str                 # row number in the source file
    evidence_role: str             # same idea as source_type, kept for clarity
    metadata: dict = field(default_factory=dict)


def _norm(name):
    return re.sub(r"[\s_]+", " ", str(name).strip().lower())


def _read_records(path):
    """Read .csv, .json (list) or .jsonl into a list of dicts."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:                                # JSON Lines: one record per line
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(data, dict):
        for value in data.values():                             # e.g. {"validation": [...]}
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [data]                                           # a single record
    return data


def _find(record, candidates):
    """Return the value of the first matching column name (case/underscore-insensitive)."""
    lookup = {_norm(k): k for k in record}
    for c in candidates:
        if c in lookup:
            return record[lookup[c]]
    return None


def _require(records, candidates, label, path):
    if records and _find(records[0], candidates) is None:
        raise ValueError(
            f"{path}: could not find a '{label}' column (looked for {candidates}). "
            f"Columns actually present: {list(records[0].keys())}"
        )


def _as_list(value):
    """TruthfulQA CSV stores answers as 'a; b; c'; HF JSON stores real lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):             # list saved as text in a CSV
        import ast
        try:
            return _as_list(list(ast.literal_eval(text)))
        except (ValueError, SyntaxError):
            pass
    return [p.strip() for p in text.split(";") if p.strip()]


def load_truthfulqa(path, use_reference_answers=True, limit=None):
    records = _read_records(path)[:limit]
    q_cols = ["question"]
    best_cols = ["best answer"]
    correct_cols = ["correct answers"]
    src_cols = ["source"]
    _require(records, q_cols, "Question", path)
    if not use_reference_answers:
        return []                      # TruthfulQA has no other trusted evidence field
    passages = []
    for i, rec in enumerate(records):
        question = str(_find(rec, q_cols) or "").strip()
        answers = _as_list(_find(rec, best_cols)) + _as_list(_find(rec, correct_cols))
        source = _find(rec, src_cols)
        for j, ans in enumerate(dict.fromkeys(answers)):        # drop duplicates, keep order
            passages.append(Passage(
                passage_id=f"tqa-{i}-{j}",
                text=f"{question} {ans}".strip(),
                source_type="dataset_reference_answer",
                dataset="truthfulqa",
                record_id=str(i),
                evidence_role="reference_correct_answer",
                metadata={"question": question, "answer": ans, "source_url": source},
            ))
    return passages


def load_halueval(path, use_reference_answers=True, limit=None):
    """Supports the HaluEval QA format (knowledge / question / right_answer)."""
    records = _read_records(path)[:limit]
    _require(records, ["question"], "question", path)
    _require(records, ["knowledge"], "knowledge", path)
    passages = []
    for i, rec in enumerate(records):
        question = str(_find(rec, ["question"]) or "").strip()
        knowledge = str(_find(rec, ["knowledge"]) or "").strip()
        right = str(_find(rec, ["right answer", "right_answer"]) or "").strip()
        if knowledge:
            passages.append(Passage(
                passage_id=f"halu-{i}-k", text=knowledge,
                source_type="dataset_knowledge_context", dataset="halueval",
                record_id=str(i), evidence_role="knowledge_context",
                metadata={"question": question},
            ))
        if use_reference_answers and right:
            passages.append(Passage(
                passage_id=f"halu-{i}-a", text=f"{question} {right}".strip(),
                source_type="dataset_reference_answer", dataset="halueval",
                record_id=str(i), evidence_role="reference_correct_answer",
                metadata={"question": question, "answer": right},
            ))
    return passages
