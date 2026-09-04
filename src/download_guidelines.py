"""
Downloads the WHO guideline PDFs that make up our corpus: 9 documents spanning
malaria, tuberculosis, maternal/newborn care, immunization, mental health,
diabetes, hepatitis C, child nutrition, and chronic low back pain.

Run this once, locally, after setting up your virtual environment:
    python src/download_guidelines.py

Deliberately spans multiple WHO teams, years, and document structures - good
for testing whether chunking/retrieval generalizes instead of overfitting to
one document's layout, and a harder, more realistic retrieval test than a
narrow 3-document corpus (more topics to confuse a weak retriever with).
"""

import requests
from pathlib import Path

GUIDELINES = {
    "malaria_treatment_guidelines.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/NBK588130/pdf/Bookshelf_NBK588130.pdf"
    ),
    "tuberculosis_drug_resistant_treatment.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who311389/pdf/"
    ),
    "maternal_newborn_care_guidelines.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who352658/pdf/"
    ),
    "immunization_refugees_migrants.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who9789240051843/pdf/"
    ),
    "mental_health_at_work.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who363177/pdf/"
    ),
    "diabetes_second_third_line_medicines.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who272433/pdf/"
    ),
    "hepatitis_c_treatment_guidelines.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who273174/pdf/"
    ),
    "child_wasting_nutrition_guideline.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who376075/pdf/"
    ),
    "chronic_low_back_pain_guideline.pdf": (
        "https://www.ncbi.nlm.nih.gov/books/n/who374726/pdf/"
    ),
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"


def download_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (educational RAG project)"}

    for filename, url in GUIDELINES.items():
        dest = OUTPUT_DIR / filename
        if dest.exists():
            print(f"✓ Already have {filename}, skipping.")
            continue

        print(f"Downloading {filename} ...")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            # Sanity check: a real guideline PDF is at least a few hundred KB.
            # A tiny response usually means we got an HTML error/redirect page
            # instead of the actual file, so we catch that here instead of
            # silently saving garbage.
            content = resp.content
            looks_like_pdf = content[:4] == b"%PDF"
            if not looks_like_pdf or len(content) < 50_000:
                print(f"  WARNING: response doesn't look like a real PDF "
                      f"({len(content) / 1024:.1f} KB, starts with {content[:20]!r}).")
                print(f"  Skipping save. Try downloading manually from: {url}")
                continue

            dest.write_bytes(content)
            print(f"  Saved to {dest} ({len(content) / 1024:.0f} KB)")
        except requests.RequestException as e:
            print(f"  FAILED to download {filename}: {e}")
            print(f"  You can manually download it from: {url}")


if __name__ == "__main__":
    download_all()
