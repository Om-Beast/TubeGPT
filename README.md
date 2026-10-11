# TubeGPT — Evidence-Grounded YouTube RAG

**Ask a question about a video. Get an answer you can verify.**

TubeGPT turns accessible YouTube captions into a searchable, timestamp-aware knowledge source. It retrieves transcript passages for each question, asks Gemini to answer from those passages, and returns source cards that jump back to the relevant moment in the video.

This is an educational full-stack RAG project, not an official YouTube product. The application does not download video or audio; it uses captions exposed by the YouTube Transcript API.

## Why this project is more than a prompt wrapper

- **Timestamp-aware chunking:** transcript captions are grouped into overlapping chunks while preserving the start and end time of each group.
- **Evidence-linked answers:** retrieved chunks receive source IDs (such as [S1]); each answer includes passages and links to the corresponding video timestamps.
- **Grounding and abstention:** the generation prompt asks the model to cite retrieved sources and say when the transcript does not provide enough evidence.
- **Prompt-injection awareness:** transcript text is treated as untrusted quoted data, not as instructions to the assistant.
- **Real retrieval pipeline:** Google embeddings + FAISS similarity search + Gemini answer generation through LangChain.
- **Bounded index cache:** the demo keeps a configurable number of video indexes in process memory and evicts the least recently used index.
- **Defensive API behavior:** validates supported YouTube URL forms, rejects oversized requests/transcripts/questions, avoids returning raw exception details to clients, and uses secure response headers.
- **Premium responsive UI:** loading states, question starters, conversation history for the current page, evidence cards, and mobile layouts with no frontend framework or external JavaScript dependencies.
- **Offline unit tests:** video URL validation, timestamp formatting, and transcript chunking can be tested without Google credentials or network access.

## Product flow

1. **Validate:** accept a raw 11-character video ID or a supported YouTube watch, Shorts, live, embed, or youtu.be URL.
2. **Fetch captions:** prefer an English transcript when available; otherwise use an available transcript.
3. **Chunk with time metadata:** group caption snippets into bounded, overlapping chunks and preserve their time ranges.
4. **Embed:** create a vector for each chunk with Gemini Embedding 2.
5. **Retrieve:** perform FAISS similarity search for the user's question and select the top matching chunks.
6. **Generate:** pass only the retrieved excerpts and question to Gemini Flash, with instructions to cite source labels and ignore instructions embedded in transcript text.
7. **Verify:** render every retrieved source as a clickable card that opens the original video at the start timestamp.

## Stack

- Backend: Python, Flask
- RAG orchestration: LangChain
- Embeddings: Google Gemini Embedding 2
- Answer model: Gemini 3.8 Flash by default
- Vector index: FAISS
- Transcript source: youtube-transcript-api
- Frontend: semantic HTML, CSS, browser JavaScript
- Tests: Python standard-library unittest
- Deployment entry point: Gunicorn / Procfile

Model defaults are configurable through environment variables. Check the [official Gemini model list](https://ai.google.dev/gemini-api/docs/models) and [embedding documentation](https://ai.google.dev/gemini-api/docs/embeddings) before changing model IDs, because provider availability and API behavior can change.

## Run locally

Requirements: Python 3.11 or newer and a Google AI API key.

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

3. Copy .env.example to .env, then set GOOGLE_API_KEY to your own key. Do not commit .env.

4. Run the dependency-free unit tests.

       python -m unittest discover -s tests -v

5. Start the app.

       python app.py

6. Open http://localhost:5000.

A Google API key is required for video indexing and answer generation. The UI and offline unit tests can be inspected without making model calls.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| GOOGLE_API_KEY | Required | Server-side Google AI API key |
| GEMINI_CHAT_MODEL | gemini-3.8-flash | Model used to generate the grounded answer |
| GEMINI_EMBEDDING_MODEL | models/gemini-embedding-2 | Embedding model used to index and query chunks |
| PORT | 5000 | Local / platform port |
| FLASK_DEBUG | 0 | Enable Flask debug mode only for local development |
| TRUST_PROXY_HOPS | 0 locally; 1 on Render | Number of trusted reverse-proxy hops; only enable when the proxy is trusted |
| LOG_LEVEL | INFO | Server logging level |
| MAX_TRANSCRIPT_CHARS | 120000 | Reject transcripts beyond this character limit |
| MAX_CHUNKS | 180 | Cap the number of chunks embedded for one video |
| MAX_CACHED_VIDEOS | 5 | Maximum number of in-memory video indexes |

## API

All endpoints are same-origin JSON endpoints.

| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | Liveness check |
| POST | /api/load-video | Fetch captions, chunk them, embed, and index the video |
| POST | /api/chat | Retrieve relevant chunks and return an answer plus source metadata |

Example load request:

    curl -X POST http://localhost:5000/api/load-video \
      -H "Content-Type: application/json" \
      -d '{"video_url":"https://www.youtube.com/watch?v=Gfr50f6ZBvo"}'

Example question request:

    curl -X POST http://localhost:5000/api/chat \
      -H "Content-Type: application/json" \
      -d '{"video_id":"Gfr50f6ZBvo","question":"What are the main takeaways?"}'

The chat response contains answer, sources, video_id, and retrieved_count. Each source includes a label, transcript excerpt, start/end seconds, formatted timestamps, and a YouTube timestamp link.

## Architecture and trade-offs

The current app is intentionally a small modular monolith. FAISS indexes are process-local and live in memory; a restart clears them, and each worker maintains its own cache. This keeps the demo easy to run but is not a multi-worker, durable, or horizontally scaled architecture. For a production service, move indexes to durable object/vector storage, use a shared cache or job queue for indexing, and add per-user authentication, provider quotas, request rate limiting, metrics, and persistent evaluation data.

The retrieval score is not shown as a confidence percentage. FAISS distances are index/model-specific and must not be presented as calibrated answer confidence without evaluation.

## Security and privacy boundaries

- The Google API key stays on the server; do not expose it with a client-side environment prefix.
- No user account is required and this version does not persist user chat history.
- Retrieved transcript excerpts are included in the model prompt; only use content you are permitted to process.
- Prompt instructions reduce the chance of following malicious transcript instructions, but prompt injection is not fully solved by wording alone.
- The process-local cache can contain indexed transcript text through FAISS documents until eviction or restart.
- Do not use the demo for confidential videos or claim production-grade privacy, security, or uptime.
- Only accessible captions are supported. If captions are disabled, blocked, or unavailable, the app reports that limitation.

## Project docs

- [Architecture notes](docs/ARCHITECTURE.md)
- [RAG and interview guide](docs/INTERVIEW_GUIDE.md)
- [Evaluation plan](docs/EVALUATION.md)

## License

MIT. See LICENSE.
