# Eval Results Summary

## Retrieval quality

- Hit@8 (expected source found in top-8 results): **100%**
- Mean Reciprocal Rank: **1.00**

## Answer quality (LLM-as-judge, 1-5 scale)

- Average faithfulness: **4.88/5**
- Average relevance: **4.97/5**

## Keyword coverage

- Average expected-keyword match rate: **68%**

## Refusal correctness (out-of-corpus questions)

- Correctly declined to answer: **100%** (4/4)

## Worst-performing questions (lowest faithfulness score)

- **nutrition_1** (faithfulness 1.0/5): How is severe wasting defined in children under 5, according to WHO?
  - Judge notes: The answer misstates the direction of the threshold for severe wasting, claiming a z‑score greater than -3 SD rather than less than -3 SD, which is not supported by the WHO guideline excerpts.
- **malaria_1** (faithfulness 5.0/5): What is the first-line treatment for uncomplicated P. falciparum malaria?
  - Judge notes: The answer correctly states that artemisinin-based combination therapy (ACT) is the first‑line treatment, which is directly supported by the excerpts.
- **immunization_3** (faithfulness 5.0/5): What barriers do refugees and migrants face in accessing immunization services?
  - Judge notes: The answer accurately reflects the barriers detailed in the excerpts, covering all categories and examples without adding unsupported claims.