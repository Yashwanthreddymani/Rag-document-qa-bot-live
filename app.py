"""
RAG Document Q&A Bot — Web version
Upload documents, then ask questions about them. Answers are grounded in the
uploaded content, with source citations.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> https://share.streamlit.io (Streamlit Community Cloud)
"""

import os
import sys
from datetime import datetime

# Streamlit Cloud's Linux environment ships an SQLite version older than
# ChromaDB requires. Swap in the pysqlite3-binary package before chromadb
# is imported anywhere, or ChromaDB fails with a "tenant" connection error.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # not needed locally on Windows/Mac, only on some Linux hosts

import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==================== CONFIGURATION ====================
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"   # Groq-hosted model, fast + free tier
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
SUPPORTED_TYPES = ["pdf", "docx", "txt"]
# ======================================================

st.set_page_config(page_title="RAG Document Q&A Bot", page_icon="📚", layout="wide")


# ---------- Cached resources ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )


def get_llm(api_key: str):
    return ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=api_key)


PROMPT = ChatPromptTemplate.from_template(
    """You are a warm, natural, conversational assistant embedded in a document Q&A app.

Follow these rules based on what the user says:

1. GREETINGS & SMALL TALK (e.g. "hi", "good morning", "how are you", "thanks", "bye"):
   Respond naturally and briefly, like a friendly person would. Do NOT mention documents,
   context, or sources for these. Just be warm and human. Match their energy — a quick
   "hi" gets a quick "hi" back, not a paragraph.

2. GENERAL KNOWLEDGE QUESTIONS not related to the uploaded documents (e.g. "what's the
   capital of France", "explain photosynthesis"): answer normally and helpfully from your
   own knowledge. You don't need the document context for these.

3. QUESTIONS ABOUT THE UPLOADED DOCUMENTS: answer using ONLY the provided context below.
   If the answer isn't in the context, say so honestly rather than guessing. Mention which
   source(s) you used.

Use your judgment on which category the question falls into — most everyday conversation
is category 1 or 2, and document-specific questions are category 3.

Context from uploaded documents (only relevant for category 3 questions):
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    formatted = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "")
        page_info = f" (page {int(page) + 1})" if page != "" else ""
        formatted.append(f"Source: {source}{page_info}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_upload_history():
    """Read back every unique uploaded file and its stats from the vector store's metadata."""
    try:
        vectorstore = get_vectorstore()
        raw = vectorstore._collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", [])
    except Exception:
        return []

    files = {}
    for m in metadatas:
        if not m:
            continue
        name = m.get("source", "unknown")
        if name not in files:
            files[name] = {
                "name": name,
                "uploaded_at": m.get("uploaded_at", "unknown"),
                "size_kb": m.get("file_size_kb", "?"),
                "chunks": 0,
            }
        files[name]["chunks"] += 1

    return sorted(files.values(), key=lambda f: f["uploaded_at"], reverse=True)


def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Send recorded audio to Groq's Whisper model and return the transcribed text."""
    client = Groq(api_key=api_key)
    transcript = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model="whisper-large-v3-turbo",
    )
    return transcript.text.strip()


def speak_text(text: str):
    """Inject a tiny JS snippet that uses the browser's built-in speech synthesis
    to read the given text aloud. Runs client-side, no extra API calls or cost."""
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    components.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


def load_file(file_path: str, suffix: str):
    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()
    elif suffix in (".docx", ".doc"):
        return Docx2txtLoader(file_path).load()
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    return []


def index_uploaded_files(uploaded_files):
    """Chunk + embed uploaded files and add them to the persistent vector store."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix.lower()
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                docs = load_file(tmp_path, suffix)
                upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for d in docs:
                    d.metadata["source"] = uploaded_file.name
                    d.metadata["uploaded_at"] = upload_time
                    d.metadata["file_size_kb"] = round(len(uploaded_file.getbuffer()) / 1024, 1)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                st.error(f"❌ Could not process {uploaded_file.name}: {e}")

        if all_chunks:
            vectorstore = get_vectorstore()
            vectorstore.add_documents(all_chunks)

    return len(all_chunks)


# ---------- Sidebar: setup + upload ----------
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com")
    else:
        st.success("API key loaded from environment")

    st.divider()
    st.header("📂 Upload documents")
    uploaded_files = st.file_uploader(
        "Add PDF, DOCX, or TXT files to the knowledge base",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index documents", type="primary"):
        with st.spinner("Reading, chunking, and embedding your documents..."):
            n = index_uploaded_files(uploaded_files)
        st.success(f"✅ Added {n} chunks from {len(uploaded_files)} file(s) to the knowledge base.")

    st.divider()
    st.header("🕒 Upload History")
    history = get_upload_history()
    if not history:
        st.caption("No documents uploaded yet.")
    else:
        for f in history:
            with st.expander(f"📄 {f['name']}"):
                st.write(f"**Uploaded:** {f['uploaded_at']}")
                st.write(f"**Size:** {f['size_kb']} KB")
                st.write(f"**Chunks indexed:** {f['chunks']}")

    st.divider()
    st.header("🔊 Voice")
    speak_answers = st.checkbox("Read answers aloud", value=False)

    st.divider()
    if st.button("🗑️ Clear entire knowledge base"):
        import shutil
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        st.cache_resource.clear()
        st.success("Knowledge base cleared.")

# ---------- Main: chat ----------
st.title("📚 RAG Document Q&A Bot")
st.caption("Upload documents in the sidebar, then ask questions about them below.")

if not api_key:
    st.info("👈 Enter a Groq API key in the sidebar to get started. It's free at console.groq.com")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

col1, col2 = st.columns([6, 1])
with col1:
    typed_question = st.chat_input("Ask a question about your documents...")
with col2:
    audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", just_once=True, key="mic")

question = typed_question

if audio and audio.get("bytes"):
    with st.spinner("Transcribing your question..."):
        try:
            question = transcribe_audio(audio["bytes"], api_key)
            st.caption(f"🎤 Heard: \"{question}\"")
        except Exception as e:
            st.error(f"❌ Could not transcribe audio: {e}")
            question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vectorstore = get_vectorstore()
                retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
                llm = get_llm(api_key)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | PROMPT
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke(question)
            except Exception as e:
                answer = f"❌ Error: {e}\n\nMake sure you've uploaded and indexed at least one document, and that your API key is valid."

        st.markdown(answer)
        if speak_answers:
            speak_text(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
