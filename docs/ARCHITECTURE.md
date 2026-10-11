# TubeGPT architecture

## Request path

Browser UI → Flask route → transcript/index cache → FAISS retrieval → Gemini answer model → answer + source cards.

## Indexing path

1. Validate the user's input and extract an 11-character YouTube video ID from an allow-listed host/path.
2. Request captions with youtube-transcript-api. Prefer English where available and fall back to an available transcript.
3. Normalize whitespace and build caption groups up to a character budget. Retain whole-caption overlap and the caption time range.
4. Convert each chunk to a LangChain Document containing chunk_index, start_seconds, and end_seconds metadata.
5. Embed the documents and build an in-memory FAISS index.
6. Insert the index into an OrderedDict LRU cache. Evict the oldest index when the configured cache limit is exceeded.

## Question path

1. Validate the video ID and question length.
2. Retrieve the top four chunks from the selected video's FAISS index.
3. Deduplicate returned chunks by chunk index.
4. Assign response-local source labels such as S1 and S2. These labels and time ranges are included in the context passed to the model.
5. Instruct the model to answer only from those excerpts, cite the labels, ignore instructions embedded in transcript text, and abstain when the evidence is insufficient.
6. Return the text answer and the evidence excerpts as structured JSON. The client creates DOM nodes with textContent instead of inserting model output as HTML.

## Key design decisions

### Why timestamp-aware chunks?
A plain text splitter loses the connection between a retrieved passage and the moment it was spoken. Each chunk tracks the first caption's start and last caption's end, so a source card can link directly to the relevant video moment. The time range follows caption boundaries; it is not word-level timing.

### Why FAISS?
FAISS gives the project a real local vector similarity-search step without a managed vector database. The index is simple to run for a single demo process. It is not durable and it is not shared between multiple workers.

### Why an LRU cache?
Repeated questions on an already indexed video should not require fetching captions and embedding every chunk again. The cache is bounded to avoid unbounded memory growth, but a restart or eviction means the user must re-index the video.

### Why not display a retrieval confidence percentage?
Similarity distances depend on the embedding model and index configuration. A distance is not the same as a probability that the answer is correct. This UI surfaces the retrieved excerpts so users can judge evidence instead of trusting an invented confidence score.

## Operational limitations

- Indexes are process-local and lost on restart.
- Concurrent first-time requests for the same video may duplicate some indexing work.
- The demo does not include account management, durable chat history, a distributed cache, a durable indexing queue, or calibrated retrieval evaluation.
- The application does not download audio/video or bypass caption access restrictions.
- The API applies a best-effort in-process rolling-window limit of 4 video-index requests/minute and 20 chat requests/minute per client. Client IP extraction trusts the configured reverse-proxy hop count; use TRUST_PROXY_HOPS=0 when running directly and only trust known proxies. This limiter is process-local and is not a substitute for provider quotas or platform-level controls.
