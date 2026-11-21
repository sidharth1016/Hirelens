"""Turn a (resume, job description) pair into a numeric feature vector."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .skills import category_of, extract_skills

CORE_CATEGORIES = {"AI/ML", "ML Libraries", "GenAI", "Languages"}

FEATURE_NAMES = [
    "tfidf_cosine",      # overall text similarity
    "skill_coverage",    # share of JD skills found in resume
    "core_coverage",     # share of JD *technical core* skills found
    "jaccard",           # overlap / union of skill sets
    "category_overlap",  # share of JD skill categories the resume touches
    "n_missing",         # how many JD skills are missing
    "n_jd_skills",       # how demanding the JD is
    "extra_related",     # resume skills in JD's categories that the JD didn't list
]


def skill_breakdown(resume: str, jd: str) -> dict:
    r, j = extract_skills(resume), extract_skills(jd)
    return {
        "resume_skills": r,
        "jd_skills": j,
        "matched": r & j,
        "missing": j - r,
        "extra": r - j,
    }


def build_features(resume: str, jd: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    b = skill_breakdown(resume, jd)
    r, j, matched, missing = b["resume_skills"], b["jd_skills"], b["matched"], b["missing"]

    vecs = vectorizer.transform([resume, jd])
    cos = float(cosine_similarity(vecs[0], vecs[1])[0, 0])

    n_j = max(len(j), 1)
    core_j = {s for s in j if category_of(s) in CORE_CATEGORIES}
    core_cov = len(core_j & r) / len(core_j) if core_j else len(matched) / n_j
    jac = len(matched) / max(len(r | j), 1)
    cats_j = {category_of(s) for s in j}
    cats_r = {category_of(s) for s in r}
    cat_ov = len(cats_j & cats_r) / max(len(cats_j), 1)
    extra_related = len({s for s in (r - j) if category_of(s) in cats_j}) / n_j

    return np.array(
        [cos, len(matched) / n_j, core_cov, jac, cat_ov, len(missing), len(j), extra_related],
        dtype=float,
    )
