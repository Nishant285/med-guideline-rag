"""
Quick debugging tool: shows exactly what chunks get retrieved for a given
question, so you can eyeball whether the RIGHT passage was actually found -
not just the right document. Useful any time the eval harness or the app
gives a surprising answer and you want to know if it's a retrieval problem
or a generation problem.

Edit QUESTION below and run:
    python src\\debug_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import retrieve

QUESTION = "What is the first-line treatment for uncomplicated P. falciparum malaria?"

results = retrieve(QUESTION, top_k=10)

print(f"Question: {QUESTION}\n")
for i, r in enumerate(results, 1):
    print(f"--- Result {i} | source: {r['source']} | page: {r['page']} | distance: {r['distance']:.3f} ---")
    print(r["text"])
    print()
