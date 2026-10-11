# TubeGPT — Evidence-Grounded YouTube RAG

**Ask a question about a video. Get an answer you can verify.**

TubeGPT turns accessible YouTube captions into a searchable, timestamp-aware knowledge source. It retrieves transcript passages for each question, asks Gemini to answer from those passages, and returns source cards that jump back to the relevant moment in the video.

This is an educational full-stack RAG project, not an official YouTube product. The application does not download video or audio; it uses captions exposed by the YouTube Transcript API.

## Why this project is more than a prompt wrapper

- **Timestamp-aware chunking:** overlapping caption chunks retain source time ranges.
- **Hybrid retrieval:** FAISS dense candidates are combined with BM25 keyword ranking using Reciprocal Rank Fusion (RRF); raw dense and lexical scores are not compared directly.
- **Evidence-linked answers:** retrieved passages receive source IDs and each card links back to the relevant video moment when timestamps are available.
- **Citation-label audit:** generated labels are checked against the IDs actually supplied to the model. The model receives one bounded repair attempt if labels are missing or unknown. This checks label integrity, not semantic entailment.
- **Grounding and abstention:** the prompt asks the model to answer from retrieved excerpts and say when evidence is insufficient.
- **Prompt-injection awareness:** transcript text is treated as untrusted quoted content, not as instructions. Prompt wording is a guardrail, not a proof of immunity.
- **Bounded index cache:** a configurable process-local LRU cache evicts older indexes.
- **Defensive API behavior:** URL and request limits, best-effort per-client rate limits, safe client-facing errors, and security headers.
- **Transcript fallback:** users can paste transcript text from YouTube's Show transcript panel if cloud caption fetching is blocked. Parseable timestamps are preserved; missing timestamps are not fabricated.
- **Testable engineering:** dependency-free unit tests cover chunking, timestamps, rate limiting, citation-label validation, hybrid ranking and retrieval metrics. Mocked API smoke tests exercise Flask routes and citation-repair behavior without calling real model APIs.

## Product flow

1. Validate a supported YouTube URL or 11-character video ID.
2. Fetch accessible captions or accept pasted transcript text.
3. Build bounded overlapping chunks and retain available start/end times.
4. Embed chunks with Google Gemini Embedding 2 and store them in an in-memory FAISS index.
5. Retrieve dense candidates; rank the transcript corpus lexically with BM25; combine ranked lists using Reciprocal Rank Fusion.
6. Generate an answer from selected excerpts with Gemini Flash.
7. Validate source-label references; retry once with correction instructions when labels are missing or unknown.
8. Show the answer, citation-label audit status, and source passages for user verification.

## Stack

- Backend: Python, Flask
- Retrieval: LangChain, FAISS, custom BM25 and Reciprocal Rank Fusion helpers
- Embeddings: Google Gemini Embedding 2
- Answer model: Gemini Flash (configurable through environment variables)
- Transcript source: youtube-transcript-api or text supplied by the user
- Frontend: semantic HTML, CSS, browser JavaScript
- Testing: Python unittest, mocked Flask API tests, GitHub Actions CI

Check the current [Gemini model documentation](https://ai.google.dev/gemini-api/docs/models) before changing model identifiers; provider availability can change.

## Run locally

Requirements: Python 3.11+ and a Google AI API key for model-backed features.

1. Create and activate a virtual environment.

   Windows PowerShell:

       py -m venv .venv
       .venv\Scripts\Activate.ps1

   macOS / Linux:

       python3 -m venv .venv
       source .venv/bin/activate

2. Install dependencies.

       python -m pip install --upgrade pip
       pip install -r requirements.txt

3. Copy .env.example to .env, set GOOGLE_API_KEY, and never commit the secret.
4. Run dependency-free unit tests:

       python -m unittest discover -s tests -p "test_[!a]*.py" -v

5. Run mocked API smoke tests (project dependencies required, no paid API call is made):

       python -m unittest discover -s tests -p "test_api_smoke.py" -v

6. Start the application:

       python app.py

7. Open http://localhost:5000.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| GOOGLE_API_KEY | Required | Server-side Google AI API key |
| GEMINI_CHAT_MODEL | gemini-3.8-flash | Answer model |
| GEMINI_EMBEDDING_MODEL | models/gemini-embedding-2 | Embedding model |
| PORT | 5000 | Server port |
| FLASK_DEBUG | 0 | Local debugging only |
| TRUST_PROXY_HOPS | 0 locally; 1 on Render | Number of trusted proxy hops for client IP extraction |
| LOG_LEVEL | INFO | Server log level |
| MAX_TRANSCRIPT_CHARS | 120000 | Transcript character cap |
| MAX_CHUNKS | 180 | Maximum indexed chunks per video |
| MAX_CACHED_VIDEOS | 5 | In-memory LRU index count |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | Liveness check |
| POST | /api/load-video | Fetch and index accessible captions |
| POST | /api/load-transcript | Parse and index user-pasted transcript text |
| POST | /api/chat | Hybrid retrieval and grounded answer generation |

A chat response includes answer, sources, video_id, retrieved_count, and citation_audit. Each source has a source ID, chunk ID, excerpt, optional timestamps, and YouTube URL. citation_audit.valid means the labels used exist in the retrieved source list; it does **not** prove that each factual claim is entailed by its cited passage.

## Reproducible retrieval evaluation

A small hand-labelled smoke dataset is bundled at data/evaluation/retrieval_gold.json. It is a demonstration corpus, not a representative real-world accuracy benchmark.

With a configured API key and the project dependencies installed, run:

    python scripts/evaluate_retrieval.py --top-k 4 --candidate-k 12 --output evaluation-report.json

The report compares dense-only FAISS rankings with hybrid dense+BM25 rankings using Hit@k, gold evidence recall@k, and Mean Reciprocal Rank (MRR). The script intentionally does not claim that retrieval metrics measure answer factuality. For resume claims, build and manually label a larger set from representative videos, rerun the comparison, and report the dataset size and test conditions.

## Architecture and limitations

This educational MVP is a modular monolith. FAISS indexes and rate-limit state are process-local and cleared on restart; multiple workers do not share either. Indexing is synchronous, chat history is not persisted, and no user authentication or durable vector store is implemented. Broad public usage needs platform-level abuse controls, provider quotas, monitoring, durable/shared index storage, and an operational privacy policy.

Prompt-injection instructions and citation repair reduce particular failure modes but are not guarantees. A valid citation label can still point to irrelevant evidence. Always verify important claims in source cards.

## Security and privacy

- Keep the Google API key only in server-side environment settings.
- No account or server-side chat-history persistence is present.
- Transcript excerpts are sent to the answer model. Do not use the demo with confidential content.
- Only process caption text you are permitted to use. The app does not bypass YouTube caption restrictions.
- The in-memory index stores transcript text until eviction or process restart.
- Do not describe this MVP as production-grade or multi-tenant.

## Project docs

- Architecture notes: docs/ARCHITECTURE.md
- Interview guide: docs/INTERVIEW_GUIDE.md
- Evaluation plan: docs/EVALUATION.md

## License

MIT. See LICENSE.
