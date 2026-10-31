"""Unit tests for chunking and BM25 retrieval (no API call)."""

from rag import load_chunks, retrieve, tokenize

CHUNKS = [
    {"page": 1, "text": "BERT wurde 2018 von Google veröffentlicht."},
    {"page": 2, "text": "GPT-2 hat 1,5 Milliarden Parameter."},
    {"page": 3, "text": "Rechenzentren benötigen sehr viel Strom."},
]


def test_tokenize_lowercases_and_removes_stopwords():
    assert tokenize("Die Plattform und der BERT") == ["plattform", "bert"]


def test_tokenize_ignores_short_words():
    assert tokenize("KI ist gut") == ["gut"]


def test_retrieve_ranks_relevant_chunk_first():
    results = retrieve("Wann wurde BERT veröffentlicht?", CHUNKS, k=2)
    assert results[0]["page"] == 1


def test_retrieve_returns_nothing_without_common_words():
    assert retrieve("Wie heißt die Hauptstadt von Frankreich?", CHUNKS) == []


def test_retrieve_respects_k():
    assert len(retrieve("BERT GPT Strom", CHUNKS, k=2)) == 2


def test_retrieve_handles_empty_document():
    assert retrieve("BERT", []) == []


def test_load_chunks_keeps_page_numbers_and_size():
    chunks = load_chunks("data/sample.pdf", chunk_size=800)
    assert len(chunks) > 0
    assert all(c["page"] >= 1 for c in chunks)
    assert all(len(c["text"]) <= 800 for c in chunks)