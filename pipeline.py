"""Full pipeline: PDF -> retrieval -> LLM -> source citation check.

Usage:  python pipeline.py document.pdf "your question"
"""

import re
import sys

from llm import NOT_FOUND, ask_llm, is_abstention
from rag import load_chunks, retrieve


def normalize(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def answer(question, chunks, k=3):
    context = retrieve(question, chunks, k)
    if not context:
        # Nothing relevant retrieved: abstain without calling the LLM
        return {"answer": NOT_FOUND, "quote": "", "page": None, "grounded": None}

    result = ask_llm(question, context)
    if is_abstention(result.get("answer")):
        result["grounded"] = None  # abstention: nothing to verify
    else:
        # Hallucination check #1: does the quote really exist in the retrieved text?
        quote = normalize(result.get("quote"))
        result["grounded"] = bool(quote) and any(quote in normalize(c["text"]) for c in context)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python pipeline.py document.pdf "your question"')
        sys.exit(1)

    chunks = load_chunks(sys.argv[1])
    if not chunks:
        print("No text extracted (scanned PDF?).")
        sys.exit(1)

    r = answer(sys.argv[2], chunks)
    print("Answer:", r["answer"])
    if r.get("quote"):
        print(f'Source (page {r.get("page")}): "{r["quote"]}"')
    print("Citation verified:", r["grounded"])