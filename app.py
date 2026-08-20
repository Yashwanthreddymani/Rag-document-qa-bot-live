"""
RAG Document Q&A Bot — Web version
Upload documents, then ask questions about them. Answers are grounded in the
uploaded content, with source citations.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> https://share.streamlit.io (Streamlit Community Cloud)
"""

import os
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

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
    """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I don't have enough information in the documents to answer this."
Always mention which source(s) you used.

Context (with sources):
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
                for d in docs:
                    d.metadata["source"] = uploaded_file.name  # keep clean filename, not temp path
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

question = st.chat_input("Ask a question about your documents...")

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
    st.session_state.messages.append({"role": "assistant", "content": answer})
