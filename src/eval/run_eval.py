"""
Eval harness: runs every question in eval_set.json through the RAG pipeline
and measures two separate things, on purpose kept separate:

1. RETRIEVAL QUALITY - did the search step find the right document at all?
   Measured independently of generation, because if retrieval fails, no
   amount of good prompting can fix the answer - you'd be optimizing the
   wrong stage. Metrics:
   - Hit@k: was the expected source document present anywhere in the top-k
     retrieved chunks?
   - MRR (Mean Reciprocal Rank): rewards the expected source appearing
     EARLIER in the ranked results, not just present somewhere in top-k.

2. ANSWER QUALITY - given what was retrieved, was the generated answer good?
   Measured two ways:
   - Keyword check: crude but fast and free - does the answer contain the
     expected key terms? Good for catching obvious failures.
   - LLM-as-judge: a second model call scores the answer 1-5 on faithfulness
     (is it grounded in the retrieved text, no hallucination) and relevance
     (does it actually answer the question). This is what a real eval
     pipeline uses when keyword matching isn't nuanced enough - e.g. an
     answer can contain all the right keywords while still being poorly
     reasoned, or vice versa.
   - Refusal correctness: for questions marked "answerable: false", checks
     whether the model correctly declined instead of hallucinating an answer.
     This matters as much as answering correctly does.

Run with:
    python src/eval/run_eval.py

Outputs:
    src/eval/results.csv       - one row per question, all metrics
    src/eval/results_summary.md - aggregate scores, readable in a README
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))
from retrieve import retrieve  # noqa: E402
from generate import answer_question, MODEL_NAME  # noqa: E402

load_dotenv()

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_CSV_PATH = Path(__file__).parent / "results.csv"
RESULTS_MD_PATH = Path(__file__).parent / "results_summary.md"

TOP_K = 8
# Match on a phrase fragment with no apostrophe in it, since models
# inconsistently use straight (') vs curly/smart (') apostrophes - matching
# on "don't contain enough information" would silently miss the curly-quote
# variant, causing a false "incorrect refusal" even when the model refused
# correctly. This was caught by manually inspecting a "failed" eval row.
REFUSAL_PHRASES = [
    "don't contain enough information",
    "do not contain enough information",
    "doesn't contain enough information",
    "does not contain enough information",
    "don't contain information",
    "do not contain information",
    "doesn't contain information",
    "does not contain information",
    "isn't enough information",
    "is not enough information",
    "cannot answer this",
    "can't answer this",
    "unable to answer",
]
judge_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring an AI assistant's answer \
to a question, based on a set of source excerpts it was given. Score strictly.

Return ONLY a JSON object, no other text, in exactly this format:
{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence>"}

faithfulness: 5 = every claim in the answer is directly supported by the \
excerpts, with no invented facts. 1 = the answer contains claims not \
supported by the excerpts (hallucination).

relevance: 5 = the answer directly and completely addresses the question. \
1 = the answer is off-topic or fails to address the question.
"""


def judge_answer(question: str, answer: str, retrieved_chunks: list[dict]) -> dict:
    """Uses a second LLM call to score faithfulness and relevance 1-5."""
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    user_message = f"""Source excerpts the assistant was given:
{context}

Question: {question}

Assistant's answer: {answer}

Score this answer."""

    response = None
    for attempt in range(5):
        try:
            response = judge_client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=500,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            break
        except RateLimitError:
            wait_seconds = 3 * (attempt + 1)
            print(f"    (judge call rate limited, waiting {wait_seconds}s...)")
            time.sleep(wait_seconds)
    if response is None:
        return {"faithfulness": None, "relevance": None, "judge_reasoning": "RATE_LIMITED_GAVE_UP"}

    raw = response.choices[0].message.content.strip()
    # Models sometimes wrap JSON in ```json fences despite instructions - strip them.
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return {
            "faithfulness": parsed.get("faithfulness"),
            "relevance": parsed.get("relevance"),
            "judge_reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "judge_reasoning": f"PARSE_FAILED: {raw[:100]}"}


def evaluate_retrieval(expected_source: str, retrieved_chunks: list[dict]) -> dict:
    """Hit@k and reciprocal rank for a single question."""
    if expected_source is None:
        return {"retrieval_hit": None, "reciprocal_rank": None}

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["source"] == expected_source:
            return {"retrieval_hit": True, "reciprocal_rank": 1 / rank}
    return {"retrieval_hit": False, "reciprocal_rank": 0.0}


def evaluate_keywords(expected_keywords: list[str], answer: str) -> dict:
    """What fraction of expected keywords appear (case-insensitive) in the answer."""
    if not expected_keywords:
        return {"keyword_match_rate": None}
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return {"keyword_match_rate": found / len(expected_keywords)}


def evaluate_refusal(answerable: bool, answer: str) -> dict:
    """For unanswerable questions, did the model correctly decline?"""
    if answerable:
        return {"correct_refusal": None}
    normalized = answer.lower().replace("\u2019", "'")
    refused = any(phrase in normalized for phrase in REFUSAL_PHRASES)
    return {"correct_refusal": refused}

def run_eval():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    questions = eval_data["questions"]
    print(f"Running eval on {len(questions)} questions...\n")

    rows = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['id']}: {item['question']}")

        retrieved_chunks = retrieve(item["question"], top_k=TOP_K)
        result = answer_question(item["question"], top_k=TOP_K)
        answer = result["answer"]

        row = {"id": item["id"], "question": item["question"], "answer": answer}
        row.update(evaluate_retrieval(item["expected_source_doc"], retrieved_chunks))
        row.update(evaluate_keywords(item["expected_keywords"], answer))
        row.update(evaluate_refusal(item["answerable"], answer))

        if item["answerable"]:
            row.update(judge_answer(item["question"], answer, retrieved_chunks))
        else:
            row.update({"faithfulness": None, "relevance": None, "judge_reasoning": ""})

        rows.append(row)

        # Small pause between questions - each question makes 2 API calls
        # (answer + judge), and Groq's free tier caps tokens/minute, so
        # this keeps us comfortably under the limit instead of relying
        # purely on retry-after-failure.
        time.sleep(2)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nFull results saved to {RESULTS_CSV_PATH}")

    write_summary(df)


def write_summary(df: pd.DataFrame):
    answerable_df = df[df["correct_refusal"].isna()]
    unanswerable_df = df[df["correct_refusal"].notna()]

    lines = ["# Eval Results Summary\n"]

    lines.append("## Retrieval quality\n")
    hit_rate = answerable_df["retrieval_hit"].mean()
    mrr = answerable_df["reciprocal_rank"].mean()
    lines.append(f"- Hit@{TOP_K} (expected source found in top-{TOP_K} results): **{hit_rate:.0%}**")
    lines.append(f"- Mean Reciprocal Rank: **{mrr:.2f}**\n")

    lines.append("## Answer quality (LLM-as-judge, 1-5 scale)\n")
    avg_faithfulness = answerable_df["faithfulness"].mean()
    avg_relevance = answerable_df["relevance"].mean()
    lines.append(f"- Average faithfulness: **{avg_faithfulness:.2f}/5**")
    lines.append(f"- Average relevance: **{avg_relevance:.2f}/5**\n")

    avg_keyword_match = answerable_df["keyword_match_rate"].mean()
    lines.append(f"## Keyword coverage\n")
    lines.append(f"- Average expected-keyword match rate: **{avg_keyword_match:.0%}**\n")

    if len(unanswerable_df) > 0:
        refusal_rate = unanswerable_df["correct_refusal"].mean()
        lines.append("## Refusal correctness (out-of-corpus questions)\n")
        lines.append(f"- Correctly declined to answer: **{refusal_rate:.0%}** "
                      f"({int(unanswerable_df['correct_refusal'].sum())}/{len(unanswerable_df)})\n")

    lines.append("## Worst-performing questions (lowest faithfulness score)\n")
    worst = answerable_df.dropna(subset=["faithfulness"]).sort_values("faithfulness").head(3)
    for _, r in worst.iterrows():
        lines.append(f"- **{r['id']}** (faithfulness {r['faithfulness']}/5): {r['question']}")
        lines.append(f"  - Judge notes: {r['judge_reasoning']}")

    summary_text = "\n".join(lines)
    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nSummary saved to {RESULTS_MD_PATH}\n")
    print(summary_text)


if __name__ == "__main__":
    run_eval()
