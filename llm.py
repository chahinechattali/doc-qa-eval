"""LLM call (Groq) with a strict prompt: answer only from the provided excerpts."""

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

NOT_FOUND = "Not found in the document."

SYSTEM_PROMPT = f"""You answer questions using ONLY the document excerpts provided.
Rules:
1. Never use outside knowledge.
2. If the answer is not in the excerpts, the answer must be exactly: "{NOT_FOUND}"
3. Copy word for word the sentence from the excerpts that supports your answer.
4. Answer in the language of the question, EXCEPT the not-found message, which must stay exactly as written in rule 2.
Reply with valid JSON only, no text around it:
{{"answer": "...", "quote": "exact sentence from the excerpts, or empty string", "page": page number or null}}"""

_client = None


def is_abstention(answer):
    """True if the model said the answer is not in the document (robust to small variations)."""
    return "not found" in (answer or "").lower()


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing: create a .env file (see .env.example).")
        _client = Groq(api_key=key)
    return _client


def ask_llm(question, chunks):
    context = "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in chunks)
    response = get_client().chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"},
        ],
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "quote": "", "page": None, "parse_error": True}