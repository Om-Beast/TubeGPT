# TubeGPT interview guide

Use this as a learning guide, not a script to memorize. Be able to open the implementation and explain each stage in your own words.

## 30-second explanation

TubeGPT is a YouTube transcript Q&A application built with Flask and a retrieval-augmented generation pipeline. It chunks captions with timestamps, embeds them in FAISS, combines vector retrieval with BM25 keyword ranking using Reciprocal Rank Fusion, and asks Gemini to answer from retrieved passages. The answer's source labels are programmatically checked, and users can open the transcript evidence at the corresponding video time.

## Two-minute explanation

A full transcript is usually too large and unfocused to include in every prompt. I split captions into bounded overlapping chunks and keep each chunk's time range. I embed the chunks and index them with FAISS. For each question, I retrieve dense candidates, rank the transcript corpus with BM25 so exact terms can surface, and combine both rankings using RRF. The selected passages are assigned source labels and sent to Gemini with instructions to answer only from those passages. After generation, I validate that each cited source label exists in the retrieved set; if labels are missing or invalid, the model gets one repair attempt. The UI renders the answer and clickable source cards so users can verify the transcript. The system remains a single-process MVP with an in-memory index cache, and retrieval/answer quality must be measured rather than assumed.

## Important implementation modules

- rag_core.py: URL validation, timestamp parsing, chunk construction.
- hybrid_retrieval.py: Unicode tokenization, BM25 and Reciprocal Rank Fusion.
- citation_validation.py: source-label integrity check, not semantic entailment.
- retrieval_metrics.py: Hit@k, gold evidence recall@k and reciprocal rank.
- app.py: Flask API, index creation, source retrieval, Gemini call and bounded citation repair.
- scripts/evaluate_retrieval.py: compare dense-only and hybrid retrieval on a manually labelled corpus.

## Interview questions

### Why use RAG instead of passing the full transcript?

RAG selects a limited number of question-relevant passages. This can reduce prompt size and gives the answer a concrete evidence set. But a retrieval miss can still cause an incomplete answer; that is why the retriever needs evaluation.

### Why combine dense search and BM25?

Dense embeddings can match semantically similar paraphrases. BM25 is good at surfacing exact words such as error codes, product names, acronyms and identifiers. Their raw scores have different scales, so I combine rank positions with Reciprocal Rank Fusion instead of adding the raw scores.

### How does Reciprocal Rank Fusion work?

Each document receives a contribution based on its rank in each list, commonly 1 divided by a rank constant plus its rank. Contributions from lists are added and the highest fused scores are returned. This is easy to reason about and avoids calibrating two different score distributions. The right retrieval strategy still has to be verified against a labelled dataset.

### Does citation validation prove the answer is true?

No. It only confirms that labels like S1 and S2 exist among the retrieved sources. A valid label can still point to irrelevant evidence. Semantic citation support needs manual evaluation or a separately validated entailment method.

### Why repair the answer only once?

A bounded retry can correct missing or unknown source labels without creating an unbounded loop that increases latency and model cost. If the second response still has invalid labels, the UI warns the user instead of claiming success.

### How should the app handle transcript prompt injection?

Transcript content is untrusted input, so the prompt separates it from instructions and tells the model to ignore any commands inside it. That mitigates some risks but cannot guarantee immunity. Adversarial tests and output validation are still needed.

### Why preserve timestamps?

The user can verify the returned passage in context. The timestamp is the chunk's approximate caption start time, not word-level alignment.

### How do you evaluate retrieval?

Use a manually labelled question set with the chunk IDs expected to contain supporting evidence. Compare dense-only versus hybrid rankings using Hit@k, gold evidence Recall@k and MRR. Measure answer factuality, citation support, abstention quality, latency and cost separately. The included dataset is a small smoke test; it is not enough for general performance claims.

### What are the scaling limits?

Indexes and rate-limit state are process-local. A restart clears them and multiple workers do not share the state. Indexing is synchronous. A production design could persist embeddings, move indexing to a job queue, use a shared cache, add authentication, quotas and telemetry.

## Honest limitations

- Automatic caption fetching may be blocked by YouTube; the UI supports user-pasted captions as a fallback.
- The app uses external model calls and does not persist chat history.
- Hybrid retrieval can underperform dense-only retrieval on some content; benchmark it.
- The citation audit checks source-label integrity, not semantic proof.
- The app is an educational MVP, not a production multi-tenant service.
