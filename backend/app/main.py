"""HireLens FastAPI service.

Endpoints
  GET  /api/health          status + which LLM provider is active
  GET  /api/metrics         model evaluation metrics
  POST /api/match           {resume, jd} -> fit score, matched/missing skills
  POST /api/extract-text    upload a PDF/TXT resume -> plain text
  POST /api/coach           {resume, jd, question} -> RAG answer + sources
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .features import FEATURE_NAMES, build_features, skill_breakdown
from .rag import coach, get_llm
from .skills import category_of

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fit_model.joblib"

app = FastAPI(title="HireLens API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bundle: dict | None = None


def bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            from .train import main as train_main
            train_main()
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class MatchIn(BaseModel):
    resume: str = Field(min_length=20)
    jd: str = Field(min_length=20)


class CoachIn(MatchIn):
    question: str = Field(min_length=3)


def _group(skills: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in sorted(skills):
        out.setdefault(category_of(s), []).append(s)
    return out


@app.get("/api/health")
def health():
    _, provider = get_llm()
    return {"status": "ok", "llm_provider": provider}


@app.get("/api/metrics")
def metrics():
    return bundle()["metrics"]


@app.post("/api/match")
def match(body: MatchIn):
    b = bundle()
    x = build_features(body.resume, body.jd, b["vectorizer"])
    score = float(b["model"].predict_proba(x.reshape(1, -1))[0, 1])
    sk = skill_breakdown(body.resume, body.jd)
    verdict = "Strong fit" if score >= 0.7 else "Possible fit" if score >= 0.4 else "Weak fit"
    return {
        "fit_score": round(score * 100, 1),
        "verdict": verdict,
        "matched": _group(sk["matched"]),
        "missing": _group(sk["missing"]),
        "extra": _group(sk["extra"]),
        "features": {n: round(float(v), 3) for n, v in zip(FEATURE_NAMES, x)},
    }


@app.post("/api/extract-text")
async def extract_text(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Upload a .pdf or .txt file")
    return {"text": text.strip()}


@app.post("/api/coach")
def coach_endpoint(body: CoachIn):
    try:
        return coach(body.question, body.resume, body.jd)
    except Exception as e:  # LLM/network errors shouldn't crash the UI
        raise HTTPException(502, f"Coach failed: {e}") from e


# In production the built React app is served from the same origin.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
