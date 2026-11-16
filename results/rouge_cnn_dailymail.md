# ROUGE on CNN/DailyMail test (100 articles, evenly spread over the test split)

Extractive methods select 3 sentences. ROUGE values are F1 x 100; ROUGE-L is the summary-level variant (rougeLsum).

| Method | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg words | Novel bigrams % | Sec/doc |
|---|---|---:|---:|---:|---:|---:|---:|
| lead | baseline | 41.44 | 20.13 | 37.66 | 76 | 0.0 | 0.00 |
| frequency | extractive | 34.13 | 13.89 | 30.75 | 98 | 0.0 | 0.00 |
| textrank | extractive | 36.11 | 15.27 | 32.59 | 78 | 0.0 | 0.01 |
| lsa | extractive | 32.41 | 12.49 | 29.35 | 69 | 0.0 | 0.01 |
| abstractive (distilbart) | abstractive | 43.32 | 21.45 | 40.41 | 57 | 12.0 | 2.21 |
| hybrid (distilbart) | hybrid | 44.25 | 22.10 | 41.31 | 62 | 11.4 | 1.57 |
