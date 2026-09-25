# BookQA

Upload a PDF or EPUB and ask questions about it — character names, quotes, relationships, minor details — and get answers grounded strictly in the document itself, with page citations. No relying on the model's memory or the open web, so the answers can't be mixed up with other books, fan wikis, or inaccurate summaries.

## Why

Reading assignments and research often come with follow-up questions that are annoying to track down by hand: "who is this minor character," "who said this line," "how are these two related." Re-reading or skimming a whole book for one detail is slow. BookQA reads the file once and lets you ask.

## How it works

1. **Extract** — the uploaded PDF/EPUB is split into overlapping ~300-word chunks, each tagged with its page number.
2. **Index** — every chunk is embedded (semantic search) and also indexed with BM25 (keyword search), so both "meaning" matches and exact names/quotes are found.
3. **Retrieve** — for each question, both search methods run and their results are merged (reciprocal rank fusion).
4. **Answer** — the top matching excerpts are sent to Gemini with a strict system prompt: answer only from the excerpts, cite the page, or say it isn't in the document.

## Tech stack

- **Backend:** Python, FastAPI
- **PDF/EPUB parsing:** PyMuPDF (`fitz`)
- **Keyword search:** `rank_bm25`
- **Embeddings + answers:** Google Gemini API (`gemini-embedding-001` + `gemini-2.5-flash`), free tier
- **Frontend:** vanilla HTML/CSS/JS (no framework)

## Setup

**Requirements:** Python 3.11+, a free [Gemini API key](https://aistudio.google.com/apikey)

```bash
git clone https://github.com/YOUR_USERNAME/bookqa.git
cd bookqa
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001


Run it:
```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`.

## Usage

1. Upload a PDF or EPUB (drag-and-drop or click to browse).
2. Wait for indexing to finish — the status message turns green when ready.
3. Ask a question in plain language. Expand "Show source excerpts" under any answer to verify it against the actual text.

## Limitations

- **Free-tier quota:** Gemini's free tier caps embedding requests (100/min, 1,000/day), so very long books (600+ pages) may exceed the per-upload chunk limit or take several minutes to index. Splitting a long book into two files is a workaround.
- **Scanned PDFs:** files without a real text layer (image-only scans) aren't supported yet — would need OCR.
- **EPUB page numbers:** for EPUBs, the "page" shown is an internal chapter-fragment index, not a printed page number.
- **In-memory storage:** uploaded books are lost on server restart; nothing is saved to disk yet.

## Roadmap

- [ ] Persistent storage (SQLite/ChromaDB) so books survive restarts
- [ ] OCR support for scanned documents
- [ ] `.txt` / `.docx` support
- [ ] Multiple books per session, with a switcher
- [ ] Deploy (Render/Railway)

## License

MIT