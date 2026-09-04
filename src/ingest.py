"""
Ingestion pipeline.

Takes every PDF in data/raw_pdfs/, extracts text, splits it into overlapping
chunks, embeds each chunk, and stores everything in a local ChromaDB.

Run this once (and again any time you add/change PDFs):
    python src/ingest.py

--- Design decisions, explained (know these for interviews) ---

1. CHUNKING STRATEGY: fixed-size character chunks with overlap.
   We split each page's text into ~1000-character chunks with a 200-character
   overlap between consecutive chunks. Why:
     - Too large a chunk -> retrieval pulls back irrelevant surrounding text,
       diluting relevance and wasting context window.
     - Too small a chunk -> you lose surrounding context needed to understand
       a sentence (e.g. "the recommended dose is 10mg" with no drug name
       nearby because the drug name was in the previous sentence).
     - Overlap prevents a fact from being split exactly at a chunk boundary
       and becoming unretrievable from either half.
   This is a *baseline* strategy, not the best possible one. A stronger
   version (worth doing later, and worth mentioning in interviews as "next
   step") is chunking by semantic unit - e.g. by section/heading, so a chunk
   never crosses a topic boundary.

2. EMBEDDING MODEL: sentence-transformers 'all-MiniLM-L6-v2'.
   Small (80MB), fast, runs on CPU, free, no API key. Good enough to prove
   the RAG pipeline works. In eval.py we'll measure retrieval quality
   directly, so if this model underperforms, the eval numbers will show it -
   and swapping to a stronger model (e.g. 'BAAI/bge-base-en-v1.5') is a
   one-line change. That's the whole point of having an eval harness.

3. METADATA: every chunk stores its source filename and page number.
   This is what lets the assistant cite "malaria_treatment_guidelines.pdf,
   page 14" instead of giving an ungrounded answer - critical for a medical
   context where you must be able to verify claims against the source.
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

RAW_PDF_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "who_guidelines"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # collapse whitespace/newlines
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping fixed-size chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_PDF_DIR}. Run download_guidelines.py first.")
        return

    print("Loading embedding model (first run downloads ~80MB, then it's cached)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Reset the collection each run so re-ingesting doesn't duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    all_ids, all_docs, all_metadatas = [], [], []

    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path.name} ...")
        pages = extract_pages(pdf_path)
        print(f"  Extracted {len(pages)} pages with text.")

        chunk_count = 0
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                chunk_id = f"{pdf_path.stem}_p{page_num}_c{chunk_count}"
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_metadatas.append({"source": pdf_path.name, "page": page_num})
                chunk_count += 1

        print(f"  Created {chunk_count} chunks.")

    print(f"\nEmbedding and storing {len(all_docs)} chunks total (this may take a minute)...")
    # Chroma can hit issues with very large single batches - add in slices of 500.
    batch_size = 500
    for i in range(0, len(all_docs), batch_size):
        collection.add(
            ids=all_ids[i : i + batch_size],
            documents=all_docs[i : i + batch_size],
            metadatas=all_metadatas[i : i + batch_size],
        )

    print(f"\nDone. Index stored at {CHROMA_DIR}")
    print(f"Total chunks indexed: {collection.count()}")


if __name__ == "__main__":
    build_index()
