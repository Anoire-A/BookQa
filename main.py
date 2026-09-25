import os
import time
import uuid

import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in your .env file")

app = FastAPI(title="BookQA")
client = genai.Client(api_key=API_KEY)

BOOKS = {}  # book_id -> {"chunks", "emb", "bm25"}  (in-memory for the MVP)

# --- chunking ---
CHUNK_WORDS = 300
OVERLAP = 50
MAX_CHUNKS = 900  # keeps a full upload within the free-tier daily quota

# --- retrieval ---
TOP_K = 8

# --- embedding throttling (free tier: 100 RPM / 30k TPM / 1k RPD) ---
EMBED_BATCH = 10        # chunks per API call -> 1 request, not 10
TPM_SAFE_DELAY = 8.5    # seconds between batches

SUPPORTED_EXT = {".pdf": "pdf", ".epub": "epub"}

SYSTEM = """You answer questions about a document using ONLY the excerpts provided.
Rules:
- If the excerpts do not clearly contain the answer, reply exactly: "I couldn't find this in the document."
- Never guess and never use outside knowledge.
- Cite page numbers like (p. 12) after each claim.
- For quotes, give the exact wording from the excerpt, and say who speaks it only if the excerpt states it.
- Be concise."""


def tokenize(text: str):
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def extract_chunks(file_bytes: bytes, filetype: str):
    """filetype: 'pdf' or 'epub'. For epub, 'page' means internal chapter-fragment
    index, not a printed page number."""
    doc = fitz.open(stream=file_bytes, filetype=filetype)
    chunks = []
    step = CHUNK_WORDS - OVERLAP
    for page_num, page in enumerate(doc, start=1):
        words = page.get_text().split()
        for i in range(0, max(len(words), 1), step):
            piece = " ".join(words[i : i + CHUNK_WORDS])
            if len(piece) > 30:
                chunks.append({"page": page_num, "text": piece})
    return chunks


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    """Embed strings via the Gemini API in batches, paced to stay under the
    free tier's tokens-per-minute limit."""
    all_vecs = []
    total_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        batch_num = i // EMBED_BATCH + 1
        print(f"Embedding batch {batch_num}/{total_batches}...")
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_vecs.extend(e.values for e in result.embeddings)
        if batch_num < total_batches:
            time.sleep(TPM_SAFE_DELAY)
    vecs = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vecs / norms  # normalized so dot product = cosine similarity


def retrieve(book, question: str, k: int = TOP_K):
    q_emb = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    sem_rank = np.argsort(-(book["emb"] @ q_emb))[:30]

    kw_scores = book["bm25"].get_scores(tokenize(question))
    kw_rank = np.argsort(-kw_scores)[:30]

    score = {}
    for ranking in (sem_rank, kw_rank):
        for rank, idx in enumerate(ranking):
            score[int(idx)] = score.get(int(idx), 0) + 1 / (60 + rank)
    top = sorted(score, key=score.get, reverse=True)[:k]
    return [book["chunks"][i] for i in top]


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename.lower()
    ext = next((e for e in SUPPORTED_EXT if name.endswith(e)), None)
    if not ext:
        raise HTTPException(400, "Please upload a PDF or EPUB file.")
    filetype = SUPPORTED_EXT[ext]

    chunks = extract_chunks(await file.read(), filetype)
    if not chunks:
        raise HTTPException(
            400, "No text found. This file may be scanned/image-based and would need OCR."
        )
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(
            400,
            f"This file has {len(chunks)} chunks, over the {MAX_CHUNKS} limit for the "
            "free tier. Try a shorter document, or split this one into two parts "
            "(e.g. first half / second half) and upload each separately.",
        )

    texts = [c["text"] for c in chunks]
    try:
        embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Free-tier quota hit while indexing. Wait a bit and retry.")
        raise HTTPException(500, f"Embedding error: {msg[:300]}")

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    book_id = str(uuid.uuid4())
    BOOKS[book_id] = {"chunks": chunks, "emb": embeddings, "bm25": bm25}
    return {
        "book_id": book_id,
        "chunks": len(chunks),
        "filename": file.filename,
        "filetype": filetype,
    }


class Ask(BaseModel):
    book_id: str
    question: str


@app.post("/ask")
async def ask(req: Ask):
    book = BOOKS.get(req.book_id)
    if not book:
        raise HTTPException(404, "Unknown book. Please upload it again.")
    hits = retrieve(book, req.question)
    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
    prompt = f"Excerpts:\n{context}\n\nQuestion: {req.question}"
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0,
                max_output_tokens=1000,
            ),
        )
        answer = resp.text or "I couldn't find this in the document."
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(429, "Free-tier rate limit hit. Wait a minute and retry.")
        raise HTTPException(500, f"AI error: {msg[:300]}")
    sources = [{"page": h["page"], "text": h["text"]} for h in hits]
    return {"answer": answer, "sources": sources}


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")