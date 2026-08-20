# RAG Document Q&A Bot — Web Version

A Retrieval-Augmented Generation (RAG) web app: upload documents, ask questions,
get grounded answers with source citations. Anyone with the link can use it.

## What changed from the CLI version

| | Before | Now |
|---|---|---|
| Interface | Terminal only | Web app (Streamlit) — shareable link |
| Uploading docs | Manually copy files into `/data`, rerun `index.py` | Upload button in the browser, indexes instantly |
| LLM | Ollama (local machine only, not public) | Groq API (hosted, free tier, works from anywhere) |
| Embeddings | Local HuggingFace model | Same — still free, still local (runs on the server) |

## Run it locally first

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a **free** Groq API key at https://console.groq.com (takes ~1 minute, no credit card).

```bash
cp .env.example .env
# edit .env and paste your key: GROQ_API_KEY=gsk_...
```

Run it:
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Upload a document in the sidebar, click "Index documents", then ask a question.

## Make it public (free) — Streamlit Community Cloud

1. Push this project to a **public GitHub repo** (don't commit your `.env` — it's already git-ignored).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo and branch, set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. You'll get a public URL like `https://your-app-name.streamlit.app` that anyone can open — no installation needed on their end.

That's it — the app is now "live." Anyone who opens the link can upload documents and ask questions.

## Notes on the shared knowledge base

Everyone who uses your public link shares the **same** document collection (same as
the original design — one `data/` folder, one vector store). Anyone can add documents,
and anyone can query all previously uploaded documents. If you want each visitor to
have their own private/isolated set of documents instead, that's a bigger change
(per-session storage) — let me know if you'd like that instead.

## Cost

Groq's free tier is generous (rate-limited but no cost) and is normally enough for demos
and small internships/portfolio projects. If you outgrow it, Groq, OpenAI, and Anthropic
all offer pay-as-you-go pricing in the cents-per-1000-tokens range.

## Known limitations

- Shared knowledge base (see above) — no per-user privacy.
- Free-tier Groq API has rate limits; heavy concurrent traffic may need a paid tier.
- Very large PDFs may take a few seconds to embed on first upload.
- Video files are not currently processed into the knowledge base (only PDF, DOCX, TXT).
