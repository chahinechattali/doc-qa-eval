"""Mini-RAG: split a PDF into chunks (keeping page numbers) and retrieve
the most relevant ones for a question with BM25 (no extra dependency)."""

import math
import re
from collections import Counter

from pypdf import PdfReader

STOPWORDS = {
    # English
    "the", "and", "for", "are", "was", "what", "which", "who", "how", "does", "with", "this", "that",
    # French
    "les", "des", "une", "est", "que", "qui", "quoi", "dans", "pour", "par", "sur", "avec", "quel", "quelle",
    # German
    "der", "die", "das", "und", "ist", "ein", "eine", "wer", "was", "wie", "mit", "von", "für", "den",
}


def load_chunks(pdf_path, chunk_size=800, overlap=150):
    """Return a list of {"page": int, "text": str}."""
    reader = PdfReader(pdf_path)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        start = 0
        while start < len(text):
            chunks.append({"page": page_num, "text": text[start:start + chunk_size]})
            start += chunk_size - overlap
    return chunks


def tokenize(text):
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def retrieve(question, chunks, k=3, k1=1.5, b=0.75):
    """Return the k chunks with the best BM25 score (only those with score > 0)."""
    if not chunks:
        return []
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) or 1
    df = Counter(w for d in docs for w in set(d))

    scored = []
    for chunk, doc in zip(chunks, docs):
        tf = Counter(doc)
        score = 0.0
        for w in tokenize(question):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(doc) / avgdl))
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:k] if score > 0]
