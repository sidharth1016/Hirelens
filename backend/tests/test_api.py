from fastapi.testclient import TestClient

from app.main import app
from app.skills import extract_skills

client = TestClient(app)

JD = ("AI Intern. Candidates should be proficient in AI/ML. Knowledge of GenAI is an advantage. "
      "Good communication skills. Startup environment.")
GOOD = ("Python, Machine Learning, Scikit-learn, NumPy, Pandas, LangChain, RAG, OpenAI API, FastAPI. "
        "Co-founder of an AI startup; strong communication with clients.")
BAD = "Frontend developer. HTML, CSS, Figma, Tailwind CSS. Built landing pages."


def test_extract_aliases():
    s = extract_skills("ReactJS, sklearn, Node.js, C++ and GenAI")
    assert {"React", "Scikit-learn", "Node.js", "C++", "Generative AI"} <= s


def test_match_ranks_good_above_bad():
    good = client.post("/api/match", json={"resume": GOOD, "jd": JD}).json()
    bad = client.post("/api/match", json={"resume": BAD, "jd": JD}).json()
    assert good["fit_score"] > bad["fit_score"]
    assert "Machine Learning" in sum(good["matched"].values(), [])


def test_coach_offline_returns_sources():
    out = client.post("/api/coach", json={"resume": GOOD, "jd": JD,
                                          "question": "What are my skill gaps?"}).json()
    assert out["answer"] and out["sources"]
