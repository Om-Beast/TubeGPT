# YouTube RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) web app that lets you ask questions about any YouTube video using its transcript.

## How it works

1. **Load** — Paste a YouTube URL; the app fetches the video transcript.
2. **Index** — The transcript is split into chunks, embedded with OpenAI, and stored in a FAISS vector database.
3. **Chat** — Ask questions; the app retrieves relevant chunks and generates answers with GPT-4o-mini.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Copy `.env.example` to `.env` and add your key:

```
GOOGLE_API_KEY=sk-your-key-here
```

### 3. Run the app

```bash
python app.py
```

## Usage

1. Paste a YouTube URL (e.g. `https://www.youtube.com/watch?v=Gfr50f6ZBvo`) or video ID.
2. Click **Load Video** and wait for indexing to finish.
3. Ask questions like:
   - "Can you summarize the video?"
   - "What topics are discussed?"
   - "Is nuclear fusion mentioned?"

## Project structure

```
Youtube Chatbot/
├── app.py              # Flask backend + RAG pipeline
├── templates/
│   └── index.html      # Main page
├── static/
│   ├── css/style.css
│   └── js/app.js
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- The video must have English captions enabled.
- Vector stores are kept in memory; restarting the server clears loaded videos.
