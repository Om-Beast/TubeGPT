# RAG evaluation plan

Do not report an accuracy or latency number until the measurement has actually been run and reviewed.

## Included retrieval smoke test

The bundled file data/evaluation/retrieval_gold.json contains a small hand-labelled corpus of 12 short passages and 10 questions around retrieval engineering concepts. It allows repeatable comparison of dense-only FAISS ranking and hybrid dense+BM25 ranking.

Run from the project root after installing requirements and setting GOOGLE_API_KEY:

    python scripts/evaluate_retrieval.py --top-k 4 --candidate-k 12 --output evaluation-report.json

The script builds a FAISS index for the bundled corpus, retrieves dense candidates per query, calculates BM25 ranking, fuses ranks with Reciprocal Rank Fusion, and reports:

- Hit@k: 1 if any gold evidence chunk appears in the top k, else 0.
- Gold evidence Recall@k: fraction of the manually labelled gold chunk set found in the top k.
- MRR: reciprocal rank of the first gold chunk, averaged across questions.

It reports dense and hybrid results side by side. The smoke dataset is tiny, hand-authored and focused on retrieval engineering; it is not representative of arbitrary YouTube videos, does not measure generated-answer quality, and must not be used alone to claim production accuracy.

## Build a meaningful dataset for a resume claim

1. Select 8–10 videos with diverse topics, languages and transcript lengths that you are permitted to analyse.
2. Write factual and paraphrased questions manually, and record one or more supporting chunk IDs/timestamp ranges.
3. Include multi-passage questions, unanswerable questions, exact-term questions, Hindi/multilingual questions, and adversarial transcript instructions.
4. Freeze the dataset before tuning settings. Do not use model-generated answers as the only ground truth.
5. Compare dense-only and hybrid retrieval using the same questions and chunk IDs.
6. Human-review whether each factual answer claim is supported by its cited passage. Count valid label IDs separately from semantic citation support.
7. Record model IDs, embedding model, chunk size, overlap, top-k, date, data size, median/p95 latency and provider usage/cost where available.
8. Re-run the same data after a code change and report failures as well as successes.

## Metrics

- Retrieval: Hit@k, gold evidence Recall@k, MRR.
- Citation support: fraction of answer claims with source passages that actually support them, human-rated with a defined rubric.
- Abstention: correct refusal on unanswerable questions and unnecessary refusal rate on answerable questions.
- Latency/cost: separately measure indexing, retrieval and generation.
- Robustness: prompt-injection attempts, malformed source labels, no-caption cases, large transcripts, and rate-limit behavior.

No representative end-to-end answer-quality benchmark is currently published. The API smoke tests use mocks and do not contact Google models.
