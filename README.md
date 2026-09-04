---
title: WHO Guideline Assistant
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# WHO Guideline Assistant (RAG)

A retrieval-augmented question-answering assistant over public WHO health guidelines
(malaria, drug-resistant tuberculosis, maternal & newborn care, immunization,
mental health, diabetes, hepatitis C, child nutrition, chronic low back pain).
Built as a learning/
portfolio project — **not a diagnostic or medical advice tool.**

Live demo: _add your deployed link here once Phase 7 is done_

## Why this project

Most RAG demos wire up an API and call it done. This project is built to demonstrate
understanding of the whole pipeline, not just "it returns an answer":
- A deliberate chunking strategy, with documented tradeoffs
- An embedding model choice made for cost/speed, verified against a real eval set
- Retrieval quality measured with real metrics (Hit@k, MRR), not eyeballed
- Groundedness and refusal behavior when the corpus doesn't contain the answer
- A documented debugging trail: real failures found via the eval harness, diagnosed,
  and fixed — see "What I found while building this" below

## Architecture

```
WHO guideline PDFs (malaria, TB, maternal health)
      │
      ▼
[src/ingest.py]     → extract text → chunk (1000 chars, 200 overlap) → embed
                       (sentence-transformers, all-MiniLM-L6-v2) → store in ChromaDB
      │
      ▼
[src/retrieve.py]   → embed the question, fetch top-k relevant chunks (k=8)
      │
      ▼
[src/generate.py]   → question + retrieved chunks → Groq (openai/gpt-oss-20b)
                       → grounded answer with [source: file, page] citations
      │
      ▼
[backend/main.py]   → FastAPI server exposing POST /api/ask, serves the frontend
      │
      ▼
[frontend/]         → custom HTML/CSS/JS chat UI, citations rendered as numbered
                       footnotes (not chat-bubble decoration) to reinforce that
                       every claim is traceable to a real source document

[src/eval/]          → ground-truth Q&A set (17 questions) + script measuring
                        retrieval quality, answer faithfulness/relevance
                        (LLM-as-judge), and refusal correctness
```

## Project status

- [x] Phase 1: Project scaffold
- [x] Phase 2: Corpus acquisition
- [x] Phase 3: Ingestion pipeline
- [x] Phase 4: RAG core (retrieval + generation)
- [x] Phase 5: Eval harness
- [x] Phase 6: Custom UI (FastAPI + vanilla HTML/CSS/JS)
- [ ] Phase 7: Deployment

## Results

Latest eval run, 37 questions (33 answerable across all 9 documents, 4 deliberately
out-of-corpus to test refusal), `top_k=8`:

| Metric | Score |
|---|---|
| Retrieval Hit@8 | 100% |
| Mean Reciprocal Rank | 1.00 |
| Faithfulness (LLM-as-judge, 1-5) | 4.88 |
| Relevance (LLM-as-judge, 1-5) | 4.97 |
| Keyword coverage | 68% |
| Refusal correctness (out-of-corpus) | 100% |

Full per-question results: `src/eval/results.csv`. Aggregate summary regenerated
each run: `src/eval/results_summary.md`.

**Honest caveats:**
- Retrieval Hit@k is now a genuinely meaningful metric with 9 documents to choose
  between (versus only 3 previously), and it still held at 100% - a stronger signal
  than the earlier 3-document result.
- The 68% keyword coverage (down from 73% with the smaller corpus) partly reflects
  harder, more specific expected keywords (exact numbers like "115", technical
  abbreviations like "MUAC") rather than a real drop in answer quality - the
  faithfulness/relevance scores stayed high across the expansion.

## What I found while building this

The eval harness surfaced three real, distinct issues - not hypothetical ones:

1. **Reasoning-model token truncation.** The generation model (`gpt-oss-20b`) spends
   part of its token budget on internal reasoning before writing the visible answer.
   With a low `max_tokens`, that reasoning silently ate the whole budget and cut a
   real answer off mid-sentence (`"The WHO guideline documents"` — nothing more).
   Fixed by raising `max_tokens` and setting `reasoning_effort="low"`.

2. **Retrieval ranking, not retrieval failure.** One question ("first-line malaria
   treatment") scored badly even though the correct passage existed verbatim in the
   corpus. Manual inspection (`src/debug_page.py`, `src/debug_retrieval.py`) showed
   the passage WAS being retrieved — ranked 6th, just outside the `top_k=5` cutoff
   used at the time. Raising `top_k` to 8 fixed it, without needing to change
   chunking or embeddings at all.

3. **Brittle refusal detection, twice.** The eval's "did the model correctly refuse"
   check used exact string matching, which broke on (a) a curly vs. straight
   apostrophe difference between two otherwise-identical refusals, and (b) natural
   paraphrasing of the refusal wording across runs. Fixed by matching against a list
   of characteristic phrase fragments instead of one exact string, and normalizing
   apostrophe characters before comparing.

4. **Inequality-sign inversion in numeric thresholds.** After expanding the corpus
   to 9 documents, the eval flagged a low faithfulness score on a question about
   severe wasting thresholds. The source text reads: *"WHZ or WLZ greater than 3 SD
   below the median (WHZ or WLZ < -3 SD)"*. The model's answer correctly paraphrased
   the prose ("greater than 3 SD below") but then generated its OWN symbolic
   shorthand and flipped the sign: `(WHZ or WLZ > -3 SD)` - self-contradicting its
   own sentence. Retrieval was correct, source text was correct, the paraphrase was
   correct; the model introduced an error only when converting prose into symbolic
   notation it wasn't explicitly asked to produce. This is a known LLM weak spot
   (verbal-to-symbolic translation of inequalities) worth being aware of for any
   RAG system answering questions with numeric thresholds - a stricter system
   prompt (e.g. "quote numeric thresholds verbatim, do not add your own notation")
   is the logical next fix, tracked as future work rather than applied blindly
   without re-testing.

## Setup

```bash
python -m venv venv
# Use Python 3.11 or 3.12 - newer versions may lack prebuilt wheels for some
# dependencies (this bit me during setup - see git history / dev notes).
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then add a free Groq API key: console.groq.com
```

Build the corpus and index (one-time, or whenever PDFs change):
```bash
python src/download_guidelines.py
python src/ingest.py
```

Run the eval harness:
```bash
python src/eval/run_eval.py
```

Run the app locally:
```bash
uvicorn backend.main:app --reload
```
Then open http://127.0.0.1:8000

## Disclaimer

This tool answers questions using only the text of public WHO guideline documents.
It is a software engineering demo, not a medical device, and must not be used for
real clinical or personal health decisions.
