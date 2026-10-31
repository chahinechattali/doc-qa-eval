# doc-qa-eval

![tests](https://github.com/chahinechattali/doc-qa-eval/actions/workflows/tests.yml/badge.svg)

**Grounded question answering over PDF documents, with automatic hallucination evaluation.**

Ask a question about a PDF. The tool retrieves the relevant passages, asks an LLM to answer **only** from those passages, returns the exact source quote with its page number, and checks that this quote really exists in the document. An evaluation script measures accuracy, abstentions and hallucinations on a labelled question set.

## How it works

```
PDF ──► chunking (with page numbers) ──► BM25 retrieval (top 3 chunks)
                                               │
                                               ▼
                         LLM with strict prompt + JSON output
                         {"answer", "quote", "page"}
                                               │
                                               ▼
                 citation check: does the quote exist verbatim in the text?
```

- **Retrieval:** BM25, implemented in pure Python (`rag.py`).
- **Generation:** `openai/gpt-oss-120b` via the Groq API, `temperature=0`, JSON mode (`llm.py`).
- **Strict prompt:** no outside knowledge; if the answer is not in the excerpts, the model must reply exactly `Not found in the document.`
- **Grounding check:** the returned quote is searched verbatim in the retrieved text (`pipeline.py`).
- **If nothing relevant is retrieved,** the pipeline abstains without calling the LLM.

## Results (v1 — BM25 retrieval)

**Document:** German Wikipedia article *Large Language Model* ([fixed version](https://de.wikipedia.org/w/index.php?title=Large_Language_Model&oldid=271533446)), 15 pages, 142 chunks.
**Question set:** 20 hand-written questions in German (`eval_set.json`).

| Question type | Description | Correct |
|---|---|---|
| Direct | Uses the same words as the document | **8 / 8** |
| Paraphrased | Same information, different wording | **0 / 5** |
| Trap | On-topic, but the answer is **not** in the document | **7 / 7** |

| Metric | Value |
|---|---|
| Accuracy | 75.0 % |
| **Hallucination rate** | **0.0 %** |
| Correct abstention (traps) | 100.0 % |
| False abstention (answerable questions) | 38.5 % |

### Findings

- **No hallucinations:** the model refused all 7 trap questions (e.g. the training energy of *GPT-5*, when the document only gives it for *GPT-4*), and never returned a quote that does not exist in the text.
- **All 5 errors are false abstentions on paraphrased questions, and they come from retrieval, not from the LLM.** BM25 matches exact words: when the question uses synonyms (*"Firma steckt hinter"*, *"Anlernen"*), the relevant passage is not among the retrieved chunks, so the model correctly answers "not found".
- **Noise:** pages 9–15 of the PDF are the bibliography, and their chunks sometimes outrank the relevant text.
- **Conclusion:** the system is cautious but incomplete. The bottleneck is lexical retrieval, which is the target of the next version (semantic search with embeddings, then re-running the same evaluation).

### Limitations

- 20 questions is a small sample: "0 % hallucinations" means *none on this test set*, not *never*.
- The citation check verifies that the quote **exists** in the text, not that it **supports** the answer.
- Answers are scored by expected keywords, which can miss correct answers phrased differently.

## Quick start

Requirements: Python 3.10+ and a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/<your-username>/doc-qa-eval.git
cd doc-qa-eval
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# then put your key in .env
```

Ask a question:

```bash
python pipeline.py data/sample.pdf "In welchem Jahr wurde BERT von Google veröffentlicht?"
```

Run the evaluation (results are written to `results.json`):

```bash
python evaluate.py data/sample.pdf eval_set.json
```

The model can be changed in `.env` (`LLM_MODEL=...`) without touching the code.

## Project structure

| File | Role |
|---|---|
| `rag.py` | PDF loading, chunking with page numbers, BM25 retrieval |
| `llm.py` | Groq client, strict system prompt, JSON parsing |
| `pipeline.py` | Retrieval + LLM + citation check, command-line entry point |
| `evaluate.py` | Runs the question set and computes the metrics |
| `eval_set.json` | 20 labelled questions (direct / paraphrased / trap) |
| `results.json` | Detailed results of the last evaluation run |

## Roadmap

- [x] v1: retrieval, grounded answers, citation check, hallucination evaluation
- [ ] v2: REST API with FastAPI (`/ask`, `/upload`, `/evaluate`) + pytest tests
- [ ] v3: web frontend with React and TypeScript
- [ ] v4: semantic retrieval with embeddings, before/after comparison on the same question set
- [ ] Docker and GitHub Actions

## Data and license

`data/sample.pdf` is an export of the German Wikipedia article [*Large Language Model*](https://de.wikipedia.org/w/index.php?title=Large_Language_Model&oldid=271533446) (revision 271533446), written by Wikipedia contributors and licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Author

Chahine Chattali, B.Eng. student in Technische Informatik at Berliner Hochschule für Technik (BHT).
