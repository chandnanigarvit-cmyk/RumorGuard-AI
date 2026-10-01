# RumorGuard AI – Backend

FastAPI backend that takes a rumor (text, screenshot or voice note), extracts the claim, retrieves **official documents** with RAG, verifies the claim, detects **claim drift**, and returns an evidence-backed correction.

```
Input (text | image | audio)
  → type detection → OCR (Tesseract) / Speech-to-text (Whisper)
  → claim extraction (LLM)
  → RAG retrieval (ChromaDB + sentence-transformers) from knowledge_base/
  → verification + claim drift (LLM, evidence-only)
  → correction generation (LLM)
  → structured JSON
```

## Folder structure
```
backend/
├── main.py              FastAPI app, routes, CORS, input-type detection
├── config.py            env-based settings
├── models.py            Pydantic models
├── rag.py               ingestion (TXT/PDF), chunking, embeddings, ChromaDB retrieval
├── claim_extractor.py   claim → {claim, location, entity, event, scope, duration, date_time}
├── verifier.py          verification, drift, correction, full pipeline + guardrails
├── prompts.py           all LLM prompts
├── llm.py               async LLM wrapper (only file to edit to change provider)
├── ocr.py               image → text
├── speech.py            audio → text (Whisper)
├── requirements.txt
├── .env.example
├── knowledge_base/official_notice_001.txt   ← demo official notice
└── data/                ChromaDB persistence (auto-created)
```

## Setup (exact steps)

**1. System dependency – Tesseract OCR**
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, then set `TESSERACT_CMD` in `.env`

(Whisper needs no ffmpeg: `faster-whisper` bundles its audio decoder.)

**2. Python 3.11+ environment**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure**
```bash
cp .env.example .env               # Windows: copy .env.example .env
# edit .env → set ANTHROPIC_API_KEY
```

**4. Run**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
On startup every file in `knowledge_base/` is indexed. First run downloads the embedding model (~90 MB); the Whisper model (~150 MB for `base`) downloads on the first audio request.

Swagger UI: http://localhost:8000/docs

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/verify` | multipart: `text` and/or `file` (image or audio; `image`/`audio` also accepted). Input type auto-detected |
| POST | `/verify/text` | JSON `{"text": "..."}` shortcut |
| GET | `/health` | status, LLM configured?, indexed chunk count |
| GET | `/documents` | list knowledge-base documents + chunk counts |
| POST | `/documents` | upload a `.txt`/`.pdf` official document (saved + indexed immediately) |
| POST | `/documents/reindex` | rescan `knowledge_base/` |

Tip: separate distinct notices inside one TXT file with a line containing only `---` to force separate chunks.

## Example requests
```bash
# text
curl -X POST http://localhost:8000/verify \
  -F "text=All service centres in the city are closed indefinitely."

# JSON
curl -X POST http://localhost:8000/verify/text -H "Content-Type: application/json" \
  -d '{"text": "All service centres in the city are closed indefinitely."}'

# screenshot
curl -X POST http://localhost:8000/verify -F "file=@rumor_screenshot.png"

# voice note
curl -X POST http://localhost:8000/verify -F "file=@voice_note.m4a"

# add an official document
curl -X POST http://localhost:8000/documents -F "file=@official_notice_002.pdf"
# (if ADMIN_TOKEN is set:  -H "X-Admin-Token: <token>")

curl http://localhost:8000/documents
curl http://localhost:8000/health
```

## Expected response for the demo rumor
Illustrative – exact wording varies by LLM run.
```json
{
  "verdict": "MISLEADING",
  "confidence": 94,
  "claim": "All service centres in the city are closed indefinitely",
  "input_type": "text",
  "source_text": "All service centres in the city are closed indefinitely.",
  "extracted_claim": {
    "claim": "All service centres in the city are closed indefinitely",
    "location": "city", "entity": "service centres", "event": "closure",
    "scope": "all branches", "duration": "indefinite", "date_time": null
  },
  "evidence": [
    {
      "source": "official_notice_001.txt",
      "chunk_id": "official_notice_001.txt::0",
      "text": "This closure applies only to the Central Service Branch.",
      "similarity": 0.61
    }
  ],
  "contradictions": ["The notice limits the closure to the Central Service Branch, not all centres."],
  "missing_context": ["Reopening time (12:00 PM)"],
  "claim_drift": {
    "drift_detected": true,
    "scope_change": "Central Branch → All Branches",
    "location_change": null,
    "duration_change": "Temporary (8:00 AM–12:00 PM) → Indefinite",
    "date_time_change": null,
    "context_removed": ["Temporary closure", "Reopening time"]
  },
  "explanation": "The notice announces a temporary maintenance closure of one branch only. The rumor widens it to all centres and removes the time limit.",
  "correction": "The official notice states that only the Central Service Branch is temporarily closed from 8:00 AM to 12:00 PM and will reopen at 12:00 PM. Other branches remain operational.",
  "alternatives": ["Other service branches remain operational."],
  "sources": ["official_notice_001.txt"],
  "notes": "'evidence' and 'sources' are retrieved from official documents. All other analysis text is AI-generated from that evidence."
}
```
If nothing relevant is retrieved the response is `"verdict": "INSUFFICIENT_EVIDENCE"` with `confidence: 0`, empty `evidence`/`sources`, and the LLM is never asked to judge.

## Anti-hallucination guardrails
- The LLM only sees retrieved chunks, each with an id, and is told to use nothing else.
- `evidence` in the response is rebuilt server-side from real chunks: quotes not found in the chunk are replaced by the real chunk text, unknown chunk ids are dropped.
- A verdict with zero verifiable evidence is automatically downgraded to `INSUFFICIENT_EVIDENCE`.
- Below `MIN_SIMILARITY` (default 0.30) chunks are not treated as evidence. Tune this in `.env` if you see false "insufficient" or false matches.
- User/rumor text is passed as delimited, untrusted data (basic prompt-injection resistance).

## Connecting your Lovable frontend
Lovable runs in the cloud, so it can't reach `localhost` reliably. Expose your backend with a tunnel:
```bash
ngrok http 8000                         # or: cloudflared tunnel --url http://localhost:8000
```
Copy the https URL (e.g. `https://abc123.ngrok-free.app`) and use it as the API base URL. CORS is open (`*`) by default; for deployment set `CORS_ORIGINS=https://your-app.lovable.app`.

**Prompt you can paste into Lovable**
> Add a "Verify rumor" page. Base URL comes from `VITE_API_URL`. Provide a textarea, an image upload and an audio upload/record button. On submit, POST `FormData` (field `text` for text, field `file` for image/audio) to `${VITE_API_URL}/verify`. Render the JSON: a verdict badge (SUPPORTED green, MISLEADING amber, CONTRADICTED red, INSUFFICIENT_EVIDENCE grey) with confidence %, the claim, a "Claim drift" card (scope_change, duration_change, location_change, date_time_change, context_removed), the "Official evidence" list (text + source), the explanation, and a copyable correction box with alternatives and sources. Show `detail` from error responses.

**Fetch code**
```ts
const API = import.meta.env.VITE_API_URL;

export async function verify({ text, file }: { text?: string; file?: File | Blob }) {
  const fd = new FormData();
  if (text) fd.append("text", text);
  if (file) fd.append("file", file, (file as File).name ?? "recording.webm");
  const res = await fetch(`${API}/verify`, {
    method: "POST",
    body: fd,
    headers: { "ngrok-skip-browser-warning": "1" }, // only needed for ngrok free URLs
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed");
  return res.json();
}
```
Do not set `Content-Type` manually for FormData. Browser-recorded audio (`audio/webm`) is accepted.

## Troubleshooting
- `503 ANTHROPIC_API_KEY is not set` → fill `.env`, restart.
- `503 Tesseract is not installed` → install it / set `TESSERACT_CMD`.
- Everything returns INSUFFICIENT_EVIDENCE → check `GET /documents` shows `indexed: true`; lower `MIN_SIMILARITY`.
- Slow first request → models are loading/downloading; subsequent calls are fast.
- Change `LLM_MODEL` in `.env` to use another model; to use a different provider, rewrite `call_llm_json` in `llm.py`.

## Known MVP limits
No auth beyond the optional `ADMIN_TOKEN`, no rate limiting, English-first OCR (`OCR_LANG` configurable), scanned (image-only) PDFs aren't OCR'd, and `/verify` makes 3 sequential LLM calls (≈ 3–8 s).
