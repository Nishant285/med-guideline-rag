"""
Dumps the FULL raw extracted text of one page from a PDF, with no chunking
applied - useful for telling apart two different failure modes:
1. Chunking cut a good sentence in half (text itself reads fine, just split
   at the wrong point)
2. Text EXTRACTION scrambled the page (e.g. a two-column PDF layout got
   read left-to-right straight across both columns, jumbling the words) -
   this is a worse problem than chunking, since no amount of re-chunking
   fixes already-garbled text.

Edit PDF_NAME and PAGE_NUM below and run:
    python src\\debug_page.py
"""

from pathlib import Path
from pypdf import PdfReader

PDF_NAME = "child_wasting_nutrition_guideline.pdf"
PAGE_NUM = 30  # 1-indexed, matches what ingest.py stores as metadata

pdf_path = Path(__file__).parent.parent / "data" / "raw_pdfs" / PDF_NAME
reader = PdfReader(str(pdf_path))

page = reader.pages[PAGE_NUM - 1]
text = page.extract_text() or ""

print(f"=== Raw extracted text, {PDF_NAME}, page {PAGE_NUM} ===\n")
print(text)
