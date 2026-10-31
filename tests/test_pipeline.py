"""Tests for the citation check: the LLM is replaced by a fake (mock), so no API call."""

import pipeline
from llm import NOT_FOUND, is_abstention

CHUNKS = [{"page": 4, "text": "BERT wurde 2018 von Google veröffentlicht."}]


def fake_llm(response):
    """Return a fake ask_llm function that always gives the same response."""
    return lambda question, chunks: dict(response)


def test_is_abstention():
    assert is_abstention(NOT_FOUND)
    assert is_abstention("not found in the document")
    assert not is_abstention("2018")


def test_real_quote_is_verified(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", fake_llm(
        {"answer": "2018", "quote": "BERT wurde 2018 von Google veröffentlicht.", "page": 4}))
    assert pipeline.answer("Wann wurde BERT veröffentlicht?", CHUNKS)["grounded"] is True


def test_invented_quote_is_detected(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", fake_llm(
        {"answer": "2019", "quote": "BERT wurde 2019 von Microsoft entwickelt.", "page": 4}))
    assert pipeline.answer("Wann wurde BERT veröffentlicht?", CHUNKS)["grounded"] is False


def test_answer_without_quote_is_not_verified(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", fake_llm({"answer": "2018", "quote": "", "page": 4}))
    assert pipeline.answer("Wann wurde BERT veröffentlicht?", CHUNKS)["grounded"] is False


def test_quote_check_ignores_case_and_whitespace(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", fake_llm(
        {"answer": "2018", "quote": "bert  wurde 2018\nvon GOOGLE veröffentlicht.", "page": 4}))
    assert pipeline.answer("Wann wurde BERT veröffentlicht?", CHUNKS)["grounded"] is True


def test_abstention_has_nothing_to_verify(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", fake_llm({"answer": NOT_FOUND, "quote": "", "page": None}))
    assert pipeline.answer("Wann wurde BERT verkauft?", CHUNKS)["grounded"] is None


def test_llm_not_called_when_nothing_is_retrieved(monkeypatch):
    def must_not_be_called(question, chunks):
        raise AssertionError("The LLM should not be called")

    monkeypatch.setattr(pipeline, "ask_llm", must_not_be_called)
    result = pipeline.answer("Hauptstadt von Frankreich?", CHUNKS)
    assert is_abstention(result["answer"])
    assert result["grounded"] is None