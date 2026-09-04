"""
Generation: takes a question + retrieved chunks, and asks an LLM to answer
USING ONLY those chunks - with citations, and an explicit refusal if the
retrieved text doesn't actually contain the answer.

This version uses Groq's free API (openai-compatible) instead of a paid
provider, running open models like Llama. Groq's free tier needs no credit
card, which makes it the easiest zero-cost option for testing and for a
Streamlit Cloud deployment later, since the model is called from the cloud
(unlike a fully local option such as Ollama, which can't run on Streamlit's
servers).

--- Design decisions, explained ---

1. GROUNDING VIA PROMPT INSTRUCTIONS.
   The system prompt explicitly tells the model to only use the provided
   context and to say so if the answer isn't in it. This doesn't
   perfectly eliminate hallucination (nothing does), but it's the standard
   first line of defense, and it's what the eval harness in eval/run_eval.py
   measures - "faithfulness" scoring checks whether the model actually
   respected this instruction.

2. CITATIONS ARE STRUCTURAL, NOT JUST TEXTUAL.
   Rather than trusting the model to remember which source said what, we
   pass each chunk into the prompt already labeled with its source file and
   page number, and ask the model to cite using those exact labels. This
   makes citations checkable - a human (or eval script) can verify a citation
   is real by checking whether that chunk was actually retrieved.

3. THIS IS DELIBERATELY NOT A DIAGNOSTIC TOOL.
   The system prompt includes a standing instruction to never give
   personalized medical advice, and to frame all answers as "what the
   guideline document says" rather than "what you should do." This matters
   both ethically and as an interview talking point: knowing how to put
   guardrails on a GenAI system in a sensitive domain is exactly the kind
   of judgment companies want to see.
"""

import os
import time
from groq import Groq
from groq import RateLimitError
from dotenv import load_dotenv

from retrieve import retrieve

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

# KNOWN ISSUE, DIAGNOSED VIA THE EVAL HARNESS:
# gpt-oss-20b is a reasoning model - it spends part of its token budget on
# internal "thinking" before writing the visible answer. With a low
# max_tokens value, that reasoning step can consume most of the budget and
# cut the actual answer off mid-sentence (this is exactly what the eval
# harness caught on the "malaria_3" question - the visible answer was
# truncated to "The WHO guideline documents"). Fix: raise max_tokens and
# explicitly set reasoning_effort="low" so more of the budget goes to the
# answer itself.


def _call_with_retry(create_fn, max_retries: int = 5):
    """
    Calls create_fn() and automatically retries on Groq's free-tier rate
    limit (429) errors, waiting the amount of time Groq tells us to wait
    (plus a small buffer). This matters for eval.py, which fires many
    requests in a row and reliably hits the per-minute token cap.
    """
    for attempt in range(max_retries):
        try:
            return create_fn()
        except RateLimitError as e:
            wait_seconds = 3 * (attempt + 1)  # simple linear backoff
            print(f"  Rate limited, waiting {wait_seconds}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
    # Final attempt - let the exception propagate if it still fails.
    return create_fn()

SYSTEM_PROMPT = """You are an assistant that answers questions using ONLY the \
excerpts from WHO guideline documents provided below. You are a software \
demo, not a medical professional, and your answers describe what the \
guideline documents say - you are not giving personalized medical advice.

Formatting rules - follow these exactly:
- Write in plain prose. Do NOT use markdown formatting: no asterisks for \
bold/italic, no headers, no bullet-point markdown. Use plain paragraphs and \
plain numbered lists like "1. ..." if a list is needed.
- Every factual claim must be followed by a citation using EXACTLY this \
format, with plain ASCII square brackets (not any other bracket style): \
[source: filename, page X]. Only cite sources that were actually provided \
to you below. Do not use any other bracket characters for citations.

Content rules you must follow:
1. Answer using ONLY information present in the provided excerpts. Do not \
use outside knowledge, even if you know the answer.
2. If the excerpts do not contain enough information to answer the question, \
say so plainly instead of guessing: "The retrieved guideline excerpts don't \
contain enough information to answer this."
3. Frame answers as describing the guideline's recommendations, e.g. "The \
guideline recommends..." rather than "You should...".
4. Never provide dosing, treatment, or diagnostic advice for an individual's \
specific situation - only describe what the general guideline document \
states.
"""


def format_context(chunks: list[dict]) -> str:
    """Turns retrieved chunks into a labeled block the model can cite from."""
    parts = []
    for c in chunks:
        parts.append(f"[source: {c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, top_k: int = 8) -> dict:
    """
    Runs the full RAG pipeline: retrieve chunks, ask Claude to answer
    grounded in them. Returns the answer text plus the raw chunks used,
    so the UI can show "sources" alongside the answer.
    """
    chunks = retrieve(question, top_k=top_k)
    context = format_context(chunks)

    user_message = f"""Context excerpts from WHO guideline documents:

{context}

---

Question: {question}"""

    response = _call_with_retry(lambda: client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=2000,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    ))

    answer_text = response.choices[0].message.content

    return {
        "answer": answer_text,
        "chunks_used": chunks,
    }


if __name__ == "__main__":
    # Quick manual test - run: python src/generate.py
    test_question = "What is the recommended treatment duration for MDR-TB?"
    result = answer_question(test_question)
    print(f"Question: {test_question}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"(Used {len(result['chunks_used'])} retrieved chunks)")
