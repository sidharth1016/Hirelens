# HireLens: AI Resume ↔ Job Matcher with RAG Interview Coach

Paste (or upload) a resume and a job description. HireLens:

1. **Extracts skills** from both with an NLP skill taxonomy (154 skills, aliases such as `sklearn → Scikit-learn`, `ReactJS → React`).
2. **Scores the fit** (0–100) with a trained scikit-learn classifier, then shows matched, missing and extra skills grouped by category.
3. **Coaches you** with a RAG pipeline (LangChain): it chunks and indexes the resume + JD, retrieves the relevant chunks, and asks an LLM (OpenAI / Gemini / Claude) to explain skill gaps, generate interview questions, or help you pitch yourself. With no API key it runs in offline rule-based mode.

## Architecture

```
React (Vite) ──/api──► FastAPI
                         ├─ skills.py    regex NLP skill extraction (154-skill taxonomy)
                         ├─ features.py  8 features: TF-IDF cosine, skill/core coverage, Jaccard, ...
                         ├─ train.py     dataset → 3 models compared by 5-fold CV → best saved
                         └─ rag.py       LangChain splitter → TF-IDF retriever → prompt → LLM
```

## Model results

| Model | 5-fold CV ROC-AUC |
|---|---|
| **Logistic Regression (selected)** | **0.915** |
| Gradient Boosting | 0.914 |
| Random Forest | 0.913 |

Held-out test set: **accuracy 0.907 · F1 0.874 · ROC-AUC 0.911** (3,000 pairs).

> **Data note:** public labelled resume↔job pairs are rare, so `train.py` synthesises 3,000 pairs from 8 role skill-pools and labels them with a recruiter-style screening rule plus 8% label noise. Swapping in a real labelled dataset only means replacing `make_dataset()`.

## Run locally

```bash
# backend
cd backend
python -m venv .venv && .venv\Scripts\activate      # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m app.train                                  # trains + writes models/metrics.json
uvicorn app.main:app --port 8000

# frontend (new terminal)
cd frontend
npm install
npm run dev                                          # http://localhost:5173
```

Optional: copy `.env.example` to `.env` and set one of `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` for full LLM answers.

### Docker

```bash
docker compose up --build      # web on :5173, api on :8000
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | status, active LLM provider |
| GET | `/api/metrics` | | model evaluation metrics |
| POST | `/api/match` | `{resume, jd}` | fit score, verdict, matched/missing/extra skills, features |
| POST | `/api/extract-text` | file (.pdf/.txt) | extracted text |
| POST | `/api/coach` | `{resume, jd, question}` | answer, provider, retrieved sources |

## Tests

```bash
cd backend && python -m pytest -q tests
```

## Ideas to extend

- Replace the TF-IDF retriever with sentence-transformer embeddings + FAISS/Chroma.
- Train on a real labelled dataset; add SHAP explanations for each fit score.
- Rank many resumes against one JD (recruiter mode).
