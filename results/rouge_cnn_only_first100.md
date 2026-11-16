# ROUGE on CNN/DailyMail test (first 100 articles)

Extractive methods select 3 sentences. ROUGE values are F1 x 100; ROUGE-L is the summary-level variant (rougeLsum).

| Method | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg words | Novel bigrams % | Sec/doc |
|---|---|---:|---:|---:|---:|---:|---:|
| lead | baseline | 30.59 | 12.68 | 27.29 | 66 | 0.0 | 0.00 |
| frequency | extractive | 27.55 | 10.43 | 24.54 | 88 | 0.0 | 0.00 |
| textrank | extractive | 29.18 | 10.72 | 26.00 | 74 | 0.0 | 0.01 |
| lsa | extractive | 26.67 | 8.45 | 23.46 | 64 | 0.0 | 0.01 |
| abstractive (distilbart) | abstractive | 35.48 | 15.71 | 32.46 | 41 | 12.2 | 1.42 |
| hybrid (distilbart) | hybrid | 35.80 | 15.88 | 32.58 | 42 | 12.7 | 0.99 |
