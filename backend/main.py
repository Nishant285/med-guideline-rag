"""
FastAPI backend for the WHO Guideline Assistant.

Serves two things:
1. POST /api/ask - runs the RAG pipeline (retrieve.py + generate.py) and
   returns a JSON answer with sources.
2. The static frontend (frontend/) at the root path, so the whole app is
   one deployable service with one URL - no separate frontend hosting needed.

Run locally with:
    uvicorn backend.main:app --reload

(Run this from the project root, not from inside backend/, so the src/
imports below resolve correctly.)
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from generate import answer_question  # noqa: E402

app = FastAPI(title="WHO Guideline Assistant API")

# Permissive CORS is fine here since this is a public-data read-only demo
# with no user accounts or sensitive actions - not something to copy as-is
# for an app with real user data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    return AskResponse(
        answer=result["answer"],
        sources=[
            SourceChunk(
                text=c["text"],
                source=c["source"],
                page=c["page"],
                distance=c["distance"],
            )
            for c in result["chunks_used"]
        ],
    )


# Mount the frontend LAST, so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")
