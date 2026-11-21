"""RAG interview coach.

1. Split the resume and the JD into chunks (LangChain text splitter).
2. Index chunks with TF-IDF (local, no embedding API needed) behind a LangChain
   BaseRetriever, so it can be swapped for FAISS/Chroma + embeddings later.
3. Retrieve the chunks most relevant to the user's question.
4. Send question + retrieved context to an LLM (OpenAI / Gemini / Claude) via
   LangChain. With no API key set, fall back to an offline rule-based coach so
   the demo always works.
"""
from __future__ import annotations

import os

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .features import skill_breakdown
from .skills import category_of


class TfidfRetriever(BaseRetriever):
    """Minimal in-memory retriever over a list of Documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: list[Document]
    k: int = 4
    vectorizer: TfidfVectorizer | None = None
    matrix: object | None = None

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 4) -> "TfidfRetriever":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform([d.page_content for d in docs])
        return cls(docs=docs, k=k, vectorizer=vec, matrix=mat)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][: self.k]
        return [self.docs[i] for i in top if scores[i] > 0] or self.docs[: self.k]


def build_retriever(resume: str, jd: str, k: int = 4) -> TfidfRetriever:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    docs = [Document(page_content=c, metadata={"source": "resume"}) for c in splitter.split_text(resume)]
    docs += [Document(page_content=c, metadata={"source": "job_description"}) for c in splitter.split_text(jd)]
    return TfidfRetriever.from_documents(docs, k=k)


def get_llm():
    """Pick an LLM from whichever API key is set. Returns (llm, provider_name) or (None, 'offline')."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3), "openai"
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), temperature=0.3), "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), temperature=0.3), "claude"
    return None, "offline"


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are HireLens, a concise and honest interview coach. Use ONLY the context below "
     "(chunks of the candidate's resume and the job description) plus the skill analysis. "
     "Never invent experience the candidate does not have; if something is missing, say so and "
     "suggest how to close the gap.\n\n"
     "Skill analysis:\nMatched: {matched}\nMissing: {missing}\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


def _offline_answer(question: str, b: dict, docs: list[Document]) -> str:
    q = question.lower()
    matched, missing = sorted(b["matched"]), sorted(b["missing"])
    if "question" in q or "interview" in q or "ask" in q:
        lines = ["Likely interview questions for this role:"]
        tech = [s for s in missing + matched if category_of(s) != "Soft Skills"]
        for s in tech[:5]:
            verb = "would use" if s in b["missing"] else "have used"
            lines.append(f"- Explain how you {verb} {s} in a real project. What trade-offs did you face?")
        if "Startup Experience" in b["jd_skills"]:
            lines.append("- Why do you want to work at an early-stage startup, and how do you handle shifting priorities?")
        lines.append("- Walk me through a project end to end: problem, data, model, evaluation, deployment.")
        lines.append("- Tell me about a time you explained something technical to a non-technical client.")
        return "\n".join(lines)
    if "gap" in q or "missing" in q or "improve" in q or "weak" in q:
        if not missing:
            return "No skills from the JD are missing from your resume. Focus on adding measurable results."
        return ("Skills the JD mentions that your resume does not: " + ", ".join(missing) +
                ".\nAdd a project or bullet that shows each one, or prepare to discuss how you'd learn it quickly.")
    ctx = "\n".join(f"- ({d.metadata['source']}) {d.page_content[:200]}" for d in docs[:3])
    return (f"Matched skills: {', '.join(matched) or 'none'}. Missing: {', '.join(missing) or 'none'}.\n"
            f"Most relevant parts for your question:\n{ctx}\n\n"
            "(Offline mode: set OPENAI_API_KEY, GOOGLE_API_KEY or ANTHROPIC_API_KEY for full LLM answers.)")


def coach(question: str, resume: str, jd: str) -> dict:
    b = skill_breakdown(resume, jd)
    retriever = build_retriever(resume, jd)
    docs = retriever.invoke(question)
    llm, provider = get_llm()
    if llm is None:
        answer = _offline_answer(question, b, docs)
    else:
        chain = PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": _format_docs(docs),
            "matched": ", ".join(sorted(b["matched"])) or "none",
            "missing": ", ".join(sorted(b["missing"])) or "none",
        })
    return {
        "answer": answer,
        "provider": provider,
        "sources": [{"source": d.metadata["source"], "text": d.page_content} for d in docs],
    }
