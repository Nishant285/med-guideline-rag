"""
Retrieval: given a question, embed it and fetch the most relevant chunks
from the ChromaDB index we built in ingest.py.

This file has no main() to run directly - it's imported by generate.py
and eval/run_eval.py. Keeping retrieval separate from generation means we
can test/measure retrieval quality on its own (see eval/run_eval.py),
which is the whole point of having an eval harness instead of just eyeballing
chat answers.
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "who_guidelines"

_client = None
_collection = None


def _get_collection():
    """Lazy-load the Chroma collection so importing this module is cheap."""
    global _client, _collection
    if _collection is None:
        # DefaultEmbeddingFunction (onnxruntime-based) instead of
        # SentenceTransformerEmbeddingFunction (PyTorch-based) - see
        # ingest.py for why. Must match whatever embedding function built
        # the index, or similarity scores become meaningless.
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k most relevant chunks for a question.

    Each result is a dict: {"text": ..., "source": ..., "page": ..., "distance": ...}
    'distance' is how far the chunk's embedding is from the question's embedding -
    LOWER means more similar/relevant. We keep it in the output so the eval
    harness and UI can show retrieval confidence.
    """
    collection = _get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    chunks = []
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, distance in zip(docs, metadatas, distances):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "distance": distance,
        })
    return chunks


if __name__ == "__main__":
    # Quick manual test - run: python src/retrieve.py
    test_question = "What is the recommended treatment duration for MDR-TB?"
    print(f"Question: {test_question}\n")
    results = retrieve(test_question, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (source: {r['source']}, page: {r['page']}, distance: {r['distance']:.3f}) ---")
        print(r["text"][:300] + "...\n")
