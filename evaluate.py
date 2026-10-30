"""Evaluate the pipeline on a question set and compute a hallucination score.

Usage:  python evaluate.py data/document.pdf eval_set.json
"""

import json
import sys
import time

from llm import is_abstention
from pipeline import answer, normalize
from rag import load_chunks

PAUSE_SECONDS = 2.5  # stay under the Groq free-tier rate limit


def evaluate_item(item, chunks):
    r = answer(item["question"], chunks)
    abstained = is_abstention(r.get("answer"))

    if item["answerable"]:
        keywords_ok = all(normalize(k) in normalize(r.get("answer")) for k in item["expected_keywords"])
        correct = not abstained and keywords_ok
        # Answered, but the quote does not exist in the document -> invented source
        hallucinated = not abstained and r.get("grounded") is False
    else:
        correct = abstained
        # Should have said "not found" but answered anyway -> invented answer
        hallucinated = not abstained

    return {
        "question": item["question"],
        "answerable": item["answerable"],
        "model_answer": r.get("answer"),
        "quote": r.get("quote"),
        "page": r.get("page"),
        "abstained": abstained,
        "correct": correct,
        "hallucinated": hallucinated,
    }


def rate(part, total):
    return round(100 * part / total, 1) if total else None


def main(pdf_path, eval_path):
    chunks = load_chunks(pdf_path)
    with open(eval_path, encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for i, item in enumerate(items, start=1):
        row = evaluate_item(item, chunks)
        rows.append(row)
        status = "OK " if row["correct"] else "ERR"
        flag = "  <- HALLUCINATION" if row["hallucinated"] else ""
        print(f"[{i}/{len(items)}] {status} {item['question']}{flag}")
        time.sleep(PAUSE_SECONDS)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    metrics = {
        "n_questions": len(rows),
        "accuracy_%": rate(sum(r["correct"] for r in rows), len(rows)),
        "hallucination_rate_%": rate(sum(r["hallucinated"] for r in rows), len(rows)),
        "answerable_accuracy_%": rate(sum(r["correct"] for r in answerable), len(answerable)),
        "false_abstention_%": rate(sum(r["abstained"] for r in answerable), len(answerable)),
        "correct_abstention_%": rate(sum(r["correct"] for r in unanswerable), len(unanswerable)),
    }

    print("\n=== By question type ===")
    for qtype in ("direct", "paraphrase", "trap"):
        group = [r for r, item in zip(rows, items) if item.get("type") == qtype]
        if group:
            print(f"{qtype:12} {sum(r['correct'] for r in group)} / {len(group)} correct")
            
    print("\n=== Results ===")
    for name, value in metrics.items():
        print(f"{name:24} {value}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": rows}, f, ensure_ascii=False, indent=2)
    print("\nDetails saved to results.json")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py data/document.pdf eval_set.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])