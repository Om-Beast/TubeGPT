# TubeGPT: interview guide

Use this as a learning guide, not as a script to memorize. Be able to open the code and explain each stage in your own words.

## 30-second explanation

TubeGPT is a YouTube transcript question-answering app built with Flask and a retrieval-augmented generation pipeline. It fetches accessible captions, groups them into overlapping chunks while preserving timestamps, embeds the chunks into FAISS, and retrieves the top passages for each question. Gemini generates an answer constrained to the retrieved context, and the UI shows the source excerpts with links to the corresponding video times.

## Two-minute explanation

A video transcript can be too long to pass into every question prompt, so I first transform it into chunks and embed them. When the user asks a question, I embed the question in the same vector space and use FAISS similarity search to retrieve relevant chunks. I pass only those excerpts to the answer model rather than asking the model to answer from the whole internet. The retrieval results include each chunk's start and end time, so the frontend can show clickable evidence. The system prompt asks the model to cite source labels and admit when the transcript does not contain the answer. The cache keeps a small number of video indexes in process memory to avoid repeating index work during a session.

## Important functions

- extract_video_id: validates supported YouTube URL patterns and raw IDs.
- build_timestamped_chunks: groups captions by character budget and retains a meaningful time interval.
- _get_or_create_index: fetches captions, creates LangChain Documents, generates embeddings, builds FAISS, and applies an LRU cache limit.
- _retrieve_sources: runs semantic similarity retrieval, removes duplicate chunk IDs, and formats stable source labels and timestamp links.
- chat: validates the question, retrieves context, calls Gemini, and returns answer text plus structured sources.

## Interview questions to practise

### Why use RAG instead of putting the whole transcript into the model prompt?
RAG retrieves a small set of passages related to each question. This can lower prompt size and make it easier to associate claims with the transcript evidence. Retrieval quality is a dependency: if the right passage is not retrieved, the answer may be incomplete.

### Why overlap chunks?
A key sentence can cross a chunk boundary. Overlap gives the retrieval step a chance to see that sentence with nearby context. Too much overlap creates redundant vectors and may fill top-k results with near-duplicate passages, so chunk size, overlap and top-k should be evaluated together.

### Why preserve timestamps in metadata?
A retrieved passage is more useful when a person can verify it. Metadata connects the result to a start/end time and lets the interface link back to the original video. The time range follows the caption boundary; it is not word-level alignment.

### What is stored in FAISS?
FAISS stores vector representations and supports nearest-neighbour similarity search. In this demo, LangChain documents also carry the passage text and timestamp metadata. The index exists in process memory and is not durable.

### What happens when the transcript does not answer the question?
The prompt explicitly instructs the model to say that the evidence is insufficient instead of using outside knowledge. This is a guardrail, not a mathematical guarantee. Evaluation with known answerable and unanswerable questions is necessary to measure how often the model follows it.

### Is prompt injection solved?
No. Transcript text is untrusted content and the prompt tells the model not to follow instructions embedded in it. This reduces risk but cannot prove immunity. A stronger production design would add adversarial tests, constrained output validation, separate trusted instructions from retrieved data, and monitor failures.

### Why not show a confidence score?
FAISS distance is not calibrated probability. A number labelled confidence would be misleading unless validated against a suitable dataset. This UI surfaces the retrieved excerpts so the person can inspect them.

### What are the current scalability limits?
Indexes are in process memory and a restart clears them; multiple workers do not share the cache. Large indexing work can be slow and costly. A production architecture could persist vectors, enqueue indexing work, use a shared cache, add authentication/quotas, and instrument latency and failure rates. Those are future improvements, not current features.

## Honest limitations to say out loud

- Transcript availability depends on YouTube and the video's caption settings.
- RAG can retrieve irrelevant chunks or miss evidence; the project needs a repeatable evaluation set before any accuracy claim.
- The cache is ephemeral and process-local.
- The model provider may change model availability, latency and pricing.
- The app is an educational MVP, not a production multi-tenant platform.


### Why provide a pasted-transcript fallback?
The unofficial transcript endpoint can be blocked or rate-limited from some cloud IP addresses. A fallback lets users supply transcript content they can already access from their browser, then reuses the same chunking, embedding, retrieval and generation pipeline. It does not bypass caption access controls. When time codes are absent, the app must disclose that rather than invent timestamp citations.
