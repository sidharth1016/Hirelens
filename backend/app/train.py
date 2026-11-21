"""Generate a labelled resume/JD dataset and train the fit classifier.

Run:  python -m app.train
Outputs: models/fit_model.joblib  and  models/metrics.json

Data note: real labelled resume<->job pairs are rarely public, so this script
synthesises pairs from role skill pools and labels them with a recruiter-style
screening rule plus 8% label noise (to mimic inconsistent human screening).
Swap `make_dataset()` for a real dataset (e.g. a Kaggle resume/JD set) to retrain.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, build_features, skill_breakdown, CORE_CATEGORIES
from .skills import category_of

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

ROLES: dict[str, list[str]] = {
    "AI/ML Intern": ["Python", "Machine Learning", "NumPy", "Pandas", "Scikit-learn", "Deep Learning",
                     "TensorFlow", "PyTorch", "NLP", "Statistics", "SQL", "Git", "Feature Engineering"],
    "GenAI Engineer": ["Python", "Generative AI", "LLMs", "LangChain", "RAG", "Prompt Engineering",
                       "OpenAI API", "Vector Databases", "Embeddings", "FastAPI", "Docker", "Hugging Face"],
    "Computer Vision Engineer": ["Python", "Computer Vision", "OpenCV", "CNN", "PyTorch", "Object Detection",
                                 "Image Processing", "Deep Learning", "NumPy", "C++"],
    "Data Analyst": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Pandas", "Statistics",
                     "Data Analysis", "Matplotlib", "A/B Testing"],
    "Data Engineer": ["Python", "SQL", "Apache Spark", "Airflow", "ETL", "Kafka", "AWS", "Snowflake",
                      "PostgreSQL", "Docker"],
    "Frontend Developer": ["JavaScript", "TypeScript", "React", "Next.js", "HTML", "CSS", "Tailwind CSS",
                           "Git", "REST APIs", "Figma"],
    "Backend Developer": ["Python", "FastAPI", "Django", "Node.js", "PostgreSQL", "MongoDB", "Redis",
                          "REST APIs", "Docker", "Microservices", "Unit Testing"],
    "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Bash", "Git", "Nginx"],
}
SOFT = ["Communication", "Teamwork", "Problem Solving", "Startup Experience", "Leadership"]

JD_TEMPLATES = [
    "We are hiring a {role}. Candidates should be proficient in {skills}. {soft} is a must.",
    "Role: {role}. Required skills: {skills}. Good {soft} expected. Startup environment.",
    "Join our team as a {role}! You will work with {skills}. We value {soft}.",
]
RESUME_TEMPLATES = [
    "Student and aspiring {role}. Skills: {skills}. Built projects using {proj}. Strong {soft}.",
    "{role} with internship experience. Technical skills - {skills}. Projects: {proj} based application.",
    "Summary: passionate about {proj}. Tools I use daily: {skills}. Known for {soft}.",
]


def _fmt(skills: list[str]) -> str:
    return ", ".join(skills)


def _label(resume: str, jd: str, rng: random.Random) -> int:
    """Recruiter-style screening rule, then 8% label noise."""
    b = skill_breakdown(resume, jd)
    j, matched = b["jd_skills"], b["matched"]
    core = {s for s in j if category_of(s) in CORE_CATEGORIES}
    cov = len(matched) / max(len(j), 1)
    core_cov = len(core & b["resume_skills"]) / len(core) if core else cov
    y = int(core_cov >= 0.55 and cov >= 0.45)
    return 1 - y if rng.random() < 0.08 else y


def make_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    roles = list(ROLES)
    rows = []
    for _ in range(n):
        jd_role = rng.choice(roles)
        jd_sk = rng.sample(ROLES[jd_role], k=rng.randint(4, min(8, len(ROLES[jd_role]))))
        jd = rng.choice(JD_TEMPLATES).format(role=jd_role, skills=_fmt(jd_sk), soft=rng.choice(SOFT))

        # half the candidates come from the same role family, half from others
        cand_role = jd_role if rng.random() < 0.5 else rng.choice(roles)
        pool = ROLES[cand_role]
        k = rng.randint(3, len(pool))
        cand = rng.sample(pool, k=k)
        # sometimes add a few random skills from any role (noise / breadth)
        for _ in range(rng.randint(0, 3)):
            cand.append(rng.choice(ROLES[rng.choice(roles)]))
        cand = list(dict.fromkeys(cand))
        resume = rng.choice(RESUME_TEMPLATES).format(
            role=cand_role, skills=_fmt(cand), proj=rng.choice(cand), soft=rng.choice(SOFT))
        rows.append({"resume": resume, "jd": jd, "label": _label(resume, jd, rng)})
    return pd.DataFrame(rows)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = make_dataset()
    print(f"dataset: {len(df)} pairs, positive rate {df.label.mean():.2f}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    vectorizer.fit(pd.concat([df.resume, df.jd]))
    X = np.vstack([build_features(r, j, vectorizer) for r, j in zip(df.resume, df.jd)])
    y = df.label.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }
    cv = {name: float(cross_val_score(m, X_tr, y_tr, cv=5, scoring="roc_auc").mean())
          for name, m in candidates.items()}
    for name, s in cv.items():
        print(f"  5-fold CV ROC-AUC  {name:<20} {s:.3f}")
    best_name = max(cv, key=cv.get)
    best = candidates[best_name].fit(X_tr, y_tr)

    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": best_name,
        "n_pairs": int(len(df)),
        "test_accuracy": round(float(accuracy_score(y_te, pred)), 3),
        "test_f1": round(float(f1_score(y_te, pred)), 3),
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 3),
        "cv_roc_auc": {k: round(v, 3) for k, v in cv.items()},
        "features": FEATURE_NAMES,
        "label_noise": 0.08,
    }
    if hasattr(best, "feature_importances_"):
        metrics["feature_importance"] = {
            n: round(float(v), 3) for n, v in zip(FEATURE_NAMES, best.feature_importances_)}

    joblib.dump({"model": best, "vectorizer": vectorizer, "metrics": metrics}, MODEL_DIR / "fit_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best: {best_name}  test acc {metrics['test_accuracy']}  f1 {metrics['test_f1']}  "
          f"roc-auc {metrics['test_roc_auc']}")


if __name__ == "__main__":
    main()
