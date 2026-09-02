import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript, IpBlocked, RequestBlocked

load_dotenv()

app = Flask(__name__)

# In-memory cache: video_id -> FAISS vector store
_vector_stores: dict[str, FAISS] = {}

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant for a YouTube video Q&A chatbot.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"],
)


def extract_video_id(url_or_id: str) -> str | None:
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([\w-]{11})",
        r"youtube\.com/shorts/([\w-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> str:
    # 1. DOCUMENT LOADER: Fetching the transcript from YouTube (WWW -> Document)
    transcript_list = YouTubeTranscriptApi().list(video_id)
    try:
        # Try to get the English transcript first
        transcript = transcript_list.find_transcript(["en"])
    except Exception:
        # If English isn't available, just grab the first available transcript (e.g. Hindi, Spanish, etc.)
        transcript = next(iter(transcript_list))
    
    return " ".join(chunk.text for chunk in transcript.fetch())


def build_vector_store(transcript: str) -> FAISS:
    # 2. TEXT SPLITTER: Breaking the large document into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    # 3. EMBEDDING MODEL: Converting text chunks into numerical vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # 4. VECTOR STORE: Storing the embedded chunks in a FAISS database
    return FAISS.from_documents(chunks, embeddings)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_or_create_store(video_id: str) -> FAISS:
    if video_id not in _vector_stores:
        transcript = fetch_transcript(video_id)
        _vector_stores[video_id] = build_vector_store(transcript)
    return _vector_stores[video_id]


def build_chain(vector_store: FAISS):
    # 5. RETRIEVER (Semantic Search): Setting up the retriever to find relevant context
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # 6. LLM: Initializing the Language Model that will generate the final answer
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

    # 7. COMBINING CONTEXT & QUERY: Getting "Most Relevant Chunks (Context)" + "Query"
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    
    # 8. THE FULL PIPELINE: Passing Context+Query -> Prompt -> LLM -> Response
    return parallel_chain | PROMPT | llm | StrOutputParser()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-video", methods=["POST"])
def load_video():
    if not os.getenv("GOOGLE_API_KEY"):
        return jsonify({"error": "GOOGLE_API_KEY is not set. Add it to your .env file."}), 500

    data = request.get_json(silent=True) or {}
    video_input = data.get("video_url", "").strip()
    video_id = extract_video_id(video_input)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL or video ID."}), 400

    try:
        transcript = fetch_transcript(video_id)
        _vector_stores[video_id] = build_vector_store(transcript)
        word_count = len(transcript.split())
        return jsonify(
            {
                "video_id": video_id,
                "message": f"Video loaded successfully ({word_count:,} words indexed).",
            }
        )
    except TranscriptsDisabled:
        return jsonify({"error": "No captions available for this video."}), 400
    except (IpBlocked, RequestBlocked):
        return jsonify({"error": "YouTube is currently rate-limiting or blocking requests from your IP address. Please wait a few minutes or try another video."}), 429
    except CouldNotRetrieveTranscript:
        return jsonify({"error": "Could not retrieve transcript. YouTube may be rate-limiting requests or captions are unavailable for this video."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    if not os.getenv("GOOGLE_API_KEY"):
        return jsonify({"error": "GOOGLE_API_KEY is not set. Add it to your .env file."}), 500

    data = request.get_json(silent=True) or {}
    video_id = data.get("video_id", "").strip()
    question = data.get("question", "").strip()

    if not video_id or not question:
        return jsonify({"error": "Video ID and question are required."}), 400

    if video_id not in _vector_stores:
        return jsonify({"error": "Video not loaded. Please load a video first."}), 400

    try:
        chain = build_chain(_vector_stores[video_id])
        answer = chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
