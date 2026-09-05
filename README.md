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
Built as a learning/portfolio project — **not a diagnostic or medical advice tool.**

**Live demo: https://who-guideline-assistant.onrender.com**
(Free-tier hosting — the app sleeps after inactivity, so the first request after
a while may take 30-60 seconds to wake up. Subsequent requests are fast.)

## Why this project

Most RAG demos wire up an API and call it done. This project is built to demonstrate
understanding of the whole pipeline, not just "it returns an answer":
- A deliberate chunking strategy, with documented tradeoffs
- An embedding model choice made deliberately for memory/cost, and changed once
  during the project when the first choice didn't survive deployment (see below)
- Retrieval quality measured with real metrics (Hit@k, MRR), not eyeballed
- Groundedness and refusal behavior when the corpus doesn't contain the answer
- A documented debugging trail: real failures found via the eval harness and via
  deployment, diagnosed, and fixed — see "What I found while building this" below

## Architecture

```
WHO guideline PDFs (9 documents, 9 health topics)
      │
      ▼
[src/ingest.py]     → extract text → chunk (1000 chars, 200 overlap) → embed
                       (ChromaDB DefaultEmbeddingFunction, onnxruntime-based
                       MiniLM) → store in ChromaDB
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

[src/eval/]          → ground-truth Q&A set (37 questions across all 9 documents)
                        + script measuring retrieval quality, answer
                        faithfulness/relevance (LLM-as-judge), and refusal
                        correctness

[Dockerfile]         → containerized for identical behavior locally and in
                        production; deployed on Render (free tier) directly
                        from GitHub, auto-redeploys on every push
```

## Project status

- [x] Phase 1: Project scaffold
- [x] Phase 2: Corpus acquisition (9 documents)
- [x] Phase 3: Ingestion pipeline
- [x] Phase 4: RAG core (retrieval + generation)
- [x] Phase 5: Eval harness (37 questions)
- [x] Phase 6: Custom UI (FastAPI + vanilla HTML/CSS/JS)
- [x] Phase 7: Deployment (Docker + Render, live)

**Planned next (not yet built — see "Future work" below):** CI pipeline running
the eval harness on every push, API rate limiting, response streaming, hybrid
(dense + keyword) retrieval, automated tests.

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
- Retrieval Hit@k is a genuinely meaningful metric with 9 documents to choose
  between, and it held at 100%.
- The 68% keyword coverage partly reflects harder, more specific expected
  keywords (exact numbers, technical abbreviations) rather than a real quality
  problem — faithfulness/relevance scores stayed high across the full corpus.
- The pre-built ChromaDB index is committed directly to this repo so deployment
  doesn't depend on re-downloading PDFs or re-embedding at build time. This is a
  reasonable shortcut for a portfolio project's scale, not something I'd do for a
  large production corpus (a managed vector DB would be the real answer there).

## What I found while building this

Real, distinct issues surfaced by the eval harness and by deployment — not
hypothetical ones:

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
   shorthand and flipped the sign: `(WHZ or WLZ > -3 SD)` — self-contradicting its
   own sentence. Retrieval was correct, source text was correct, the paraphrase was
   correct; the model introduced an error only when converting prose into symbolic
   notation it wasn't explicitly asked to produce. A stricter system prompt (e.g.
   "quote numeric thresholds verbatim") is the logical next fix, tracked as future
   work rather than applied blindly without re-testing.

5. **Free-tier memory limit hit on deployment.** The app worked locally, but the
   first deployed version crashed with an out-of-memory error on Render's free
   512MB tier as soon as a real question triggered embedding + retrieval. Root
   cause: `sentence-transformers` pulls in the full PyTorch stack, which alone can
   exceed 512MB at runtime. Fixed by switching to ChromaDB's built-in
   `DefaultEmbeddingFunction` (onnxruntime-based, no PyTorch dependency) — same
   underlying MiniLM model, dramatically smaller memory footprint. Required a full
   re-embed of the corpus, since embeddings from different embedding functions
   aren't compatible with each other.

6. **Dependency version mismatch after a clean reinstall.** After removing
   `sentence-transformers`/PyTorch, a fresh `pip install` resolved `groq` to an
   older pinned version (`0.11.0`) that predated support for the `reasoning_effort`
   parameter used in fix #1 — a working feature broke silently because of an
   unrelated dependency change. Fixed by pinning `groq==0.30.0` explicitly, and
   removing a now-unnecessary `httpx` version pin that was only needed for a
   different SDK no longer used in the project.

## Future work

Concrete next improvements identified but not yet built:
- **CI pipeline**: run `run_eval.py` automatically on every push (GitHub Actions),
  fail the build if faithfulness/retrieval scores regress below a threshold —
  turns the eval harness into real regression testing for an AI system.
- **API rate limiting**: the deployed endpoint currently has no abuse protection
  against a shared API key.
- **Response streaming**: token-by-token output instead of waiting for the full
  answer, for a faster-feeling UI.
- **Hybrid retrieval**: add BM25/keyword search alongside vector search and
  measure the combined result against the existing eval set.
- **Confidence-based refusal**: refuse automatically when top retrieval distance
  exceeds a threshold, instead of relying entirely on the model's own judgment.
- **Automated tests**: unit tests for chunking, citation parsing, and refusal
  detection logic.

## Setup

```bash
python -m venv venv
# Use Python 3.11 or 3.12 - newer versions may lack prebuilt wheels for some
# dependencies.
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

## Deployment

Containerized with the included `Dockerfile` and deployed on
[Render](https://render.com) (free tier), connected directly to this GitHub repo.
Every push to `main` triggers an automatic rebuild and redeploy. To deploy your
own copy: fork this repo, create a Render Web Service pointed at it with
environment "Docker," and add your own `GROQ_API_KEY` as an environment variable
in Render's dashboard.

## Disclaimer

This tool answers questions using only the text of public WHO guideline documents.
It is a software engineering demo, not a medical device, and must not be used for
real clinical or personal health decisions.
