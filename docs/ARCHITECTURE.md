# TubeGPT architecture

## Request path

Browser UI → Flask route → transcript/index cache → FAISS dense retrieval + BM25 lexical ranking → Reciprocal Rank Fusion → Gemini answer model → citation-label audit → source cards.

## Indexing path

1. Validate a YouTube URL or 11-character video ID.
2. Fetch accessible captions. If the hosting network is blocked, a user can submit transcript text from YouTube's Show transcript panel.
3. For pasted text, parse inline or standalone timestamp lines when present. Plain text without times remains un-timestamped.
4. Normalize captions and build bounded overlapping chunks while preserving approximate caption time ranges.
5. Convert chunks to LangChain Documents containing chunk_index, start_seconds, end_seconds, and timestamp_available metadata.
6. Embed documents and build an in-memory FAISS index.
7. Store both the vector index and source documents in the process-local LRU cache, evicting old video indexes when the configured limit is exceeded.

## Question path

1. Validate the video ID and question length.
2. Retrieve up to three times TOP_K dense candidates from FAISS.
3. Calculate corpus-wide BM25 rankings using normalized Unicode word tokens.
4. Fuse dense and lexical rankings using Reciprocal Rank Fusion (RRF). RRF uses rank positions instead of mixing raw similarity distances and lexical scores.
5. Deduplicate chunks and return the top four excerpts with source IDs and timestamps when available.
6. Ask Gemini to answer from only those excerpts, cite source IDs, ignore instructions embedded in transcript text, and abstain when evidence is insufficient.
7. Check that cited labels exist in the retrieved source list. If labels are absent or invalid, make one bounded repair attempt with the allowed source IDs.
8. Return the answer, source metadata, and citation-label audit status. The frontend renders model text with textContent rather than inserting it as HTML.

## Key design decisions

### Why hybrid retrieval?

Dense retrieval helps with paraphrases and conceptual similarity. BM25 helps with exact names, acronyms, error codes, and technical identifiers. RRF fuses ranked positions without assuming that a dense distance and a BM25 score share the same scale. Hybrid retrieval can still underperform dense-only search on some datasets, so the evaluation script compares both paths instead of assuming the change helps.

### What does citation validation guarantee?

It verifies syntax-level references: a label used in the answer must exist in the current retrieved source set. One repair attempt is made if labels are absent or invalid. It does not prove that a cited passage semantically supports the claim. The UI explains this distinction, and the evaluation plan calls for human-rated citation support.

### Why timestamp-aware chunks?

A retrieved passage is more useful when a person can verify it. Metadata connects the result to a caption interval and allows a clickable video link. The range follows caption boundaries, not word-level alignment.

### Why FAISS?

FAISS gives the project a local vector similarity search without a managed vector database. The index is easy to run for a single-process demo but is not durable and is not shared between multiple workers.

### Why an LRU cache?

Repeated questions about an indexed video should not require rebuilding embeddings each time. A bounded LRU cache limits memory growth, but a restart or eviction means the user must re-index the video.

### Why no confidence percentage?

Similarity distance is not calibrated probability. Showing a confidence score without validation would be misleading. The UI exposes retrieved excerpts so users can judge the evidence.

## Tests and evaluation

- Dependency-free tests cover URL validation, timestamp parsing, chunking, BM25 ranking, RRF, citation-label validation, retrieval metrics, and rate limiting.
- Mocked Flask API tests exercise route validation, security headers, manual transcript ingestion, the citation-repair retry, and rate-limit responses.
- The retrieval evaluation script compares dense-only ranking against hybrid ranking on a manually labelled dataset. The bundled smoke dataset is small and must not be presented as a representative production benchmark.
- Model-backed evaluation and citation-claim support still require configured credentials and a larger manually validated set.

## Operational limitations

- Indexes and rate limiter state are process-local and lost on restart.
- Concurrent first-time requests for one video may duplicate indexing work.
- There is no account system, durable chat history, shared vector database, background indexing queue, or calibrated quality metric.
- The API applies best-effort in-process rate limits; provider quotas and platform-level abuse controls are still required for public traffic.
- The app does not download audio/video or bypass caption access restrictions.
