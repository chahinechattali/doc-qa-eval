"""Interactive demo: ask several questions about a PDF without retyping commands.

Usage:  python demo.py data/sample.pdf
"""

import sys

from pipeline import answer
from rag import load_chunks

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
chunks = load_chunks(pdf_path)
print(f"Loaded {pdf_path} ({len(chunks)} chunks). Type 'exit' to quit.\n")

while True:
    question = input("Question: ").strip()
    if question.lower() in ("exit", "quit", ""):
        break
    r = answer(question, chunks)
    print("\nAnswer:", r["answer"])
    if r.get("quote"):
        print(f'Source (page {r.get("page")}): "{r["quote"]}"')
    print("Citation verified:", r["grounded"])
    print("-" * 60 + "\n")