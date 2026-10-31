"""Tests for the scoring logic of the evaluation (correct / hallucinated)."""

import evaluate
from llm import NOT_FOUND


def fake_answer(result):
    return lambda question, chunks: dict(result)


def test_answered_trap_counts_as_hallucination(monkeypatch):
    monkeypatch.setattr(evaluate, "answer", fake_answer({"answer": "500 GWh", "quote": "", "grounded": False}))
    row = evaluate.evaluate_item({"question": "?", "answerable": False}, chunks=[])
    assert row["hallucinated"] is True
    assert row["correct"] is False


def test_abstention_on_trap_is_correct(monkeypatch):
    monkeypatch.setattr(evaluate, "answer", fake_answer({"answer": NOT_FOUND, "grounded": None}))
    row = evaluate.evaluate_item({"question": "?", "answerable": False}, chunks=[])
    assert row["correct"] is True
    assert row["hallucinated"] is False


def test_correct_answer_with_expected_keyword(monkeypatch):
    monkeypatch.setattr(evaluate, "answer", fake_answer({"answer": "Im Jahr 2018", "grounded": True}))
    item = {"question": "?", "answerable": True, "expected_keywords": ["2018"]}
    row = evaluate.evaluate_item(item, chunks=[])
    assert row["correct"] is True
    assert row["hallucinated"] is False


def test_invented_quote_on_answerable_question_is_hallucination(monkeypatch):
    monkeypatch.setattr(evaluate, "answer", fake_answer({"answer": "2018", "grounded": False}))
    item = {"question": "?", "answerable": True, "expected_keywords": ["2018"]}
    assert evaluate.evaluate_item(item, chunks=[])["hallucinated"] is True


def test_false_abstention_on_answerable_question(monkeypatch):
    monkeypatch.setattr(evaluate, "answer", fake_answer({"answer": NOT_FOUND, "grounded": None}))
    item = {"question": "?", "answerable": True, "expected_keywords": ["2018"]}
    row = evaluate.evaluate_item(item, chunks=[])
    assert row["abstained"] is True
    assert row["correct"] is False
    assert row["hallucinated"] is False