import { useEffect, useState } from "react";

const SAMPLE_JD = `Exora - AI Intern (Bengaluru). Candidates should be proficient in AI/ML.
Skills required: AI/ML. Add-on: knowledge of GenAI will be an advantage.
Candidates should have good communication skills and be willing to start their career in a startup environment.`;

async function api(path, body) {
  const res = await fetch(`/api${path}`, body instanceof FormData
    ? { method: "POST", body }
    : body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {});
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

function SkillGroups({ groups, kind }) {
  const entries = Object.entries(groups || {});
  if (!entries.length) return <p className="muted">None</p>;
  return entries.map(([cat, skills]) => (
    <div key={cat} className="group">
      <span className="cat">{cat}</span>
      {skills.map((s) => <span key={s} className={`chip ${kind}`}>{s}</span>)}
    </div>
  ));
}

function ScoreRing({ score }) {
  const r = 52, c = 2 * Math.PI * r, off = c * (1 - score / 100);
  const color = score >= 70 ? "var(--good)" : score >= 40 ? "var(--warn)" : "var(--bad)";
  return (
    <svg width="130" height="130" viewBox="0 0 130 130" role="img" aria-label={`Fit score ${score}`}>
      <circle cx="65" cy="65" r={r} fill="none" stroke="var(--line)" strokeWidth="12" />
      <circle cx="65" cy="65" r={r} fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
        strokeDasharray={c} strokeDashoffset={off} transform="rotate(-90 65 65)" />
      <text x="65" y="72" textAnchor="middle" fontSize="26" fontWeight="700" fill="var(--ink)">{Math.round(score)}</text>
    </svg>
  );
}

export default function App() {
  const [resume, setResume] = useState("");
  const [jd, setJd] = useState(SAMPLE_JD);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [chat, setChat] = useState([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    api("/health").then(setHealth).catch(() => setErr("Backend not reachable on :8000"));
    api("/metrics").then(setMetrics).catch(() => {});
  }, []);

  const run = async (label, fn) => {
    setBusy(label); setErr("");
    try { await fn(); } catch (e) { setErr(e.message); } finally { setBusy(""); }
  };

  const onUpload = (e) => {
    const f = e.target.files?.[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    run("upload", async () => setResume((await api("/extract-text", fd)).text));
  };

  const analyze = () => run("match", async () => setResult(await api("/match", { resume, jd })));

  const ask = (question) => run("coach", async () => {
    setChat((c) => [...c, { role: "you", text: question }]); setQ("");
    const out = await api("/coach", { resume, jd, question });
    setChat((c) => [...c, { role: "coach", text: out.answer, sources: out.sources, provider: out.provider }]);
  });

  const ready = resume.length >= 20 && jd.length >= 20;

  return (
    <div className="wrap">
      <header>
        <h1>Hire<span>Lens</span></h1>
        <p>ML fit scoring + RAG interview coaching for any resume and job description.</p>
        {health && <span className="pill">LLM: {health.llm_provider}</span>}
      </header>

      {err && <div className="error">{err}</div>}

      <section className="inputs">
        <div className="card">
          <div className="row"><h2>Resume</h2>
            <label className="btn ghost">Upload PDF/TXT<input type="file" accept=".pdf,.txt" hidden onChange={onUpload} /></label>
          </div>
          <textarea value={resume} onChange={(e) => setResume(e.target.value)} placeholder="Paste your resume text or upload a file…" />
        </div>
        <div className="card">
          <h2>Job description</h2>
          <textarea value={jd} onChange={(e) => setJd(e.target.value)} />
        </div>
      </section>

      <button className="btn primary big" disabled={!ready || !!busy} onClick={analyze}>
        {busy === "match" ? "Analyzing…" : "Analyze fit"}
      </button>

      {result && (
        <section className="results">
          <div className="card score">
            <ScoreRing score={result.fit_score} />
            <div><h2>{result.verdict}</h2><p className="muted">Model probability that this resume passes screening.</p></div>
          </div>
          <div className="card"><h3>Matched skills</h3><SkillGroups groups={result.matched} kind="ok" /></div>
          <div className="card"><h3>Missing skills</h3><SkillGroups groups={result.missing} kind="miss" /></div>
          <div className="card"><h3>Extra skills you bring</h3><SkillGroups groups={result.extra} kind="extra" /></div>
        </section>
      )}

      {ready && (
        <section className="card coach">
          <h2>Interview coach (RAG)</h2>
          <div className="quick">
            {["What are my skill gaps for this role?", "Generate 5 interview questions for me", "How should I pitch my experience?"]
              .map((s) => <button key={s} className="btn ghost" disabled={!!busy} onClick={() => ask(s)}>{s}</button>)}
          </div>
          <div className="chat">
            {chat.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <pre>{m.text}</pre>
                {m.sources && <details><summary>{m.sources.length} retrieved chunks · {m.provider}</summary>
                  {m.sources.map((s, j) => <p key={j} className="src"><b>{s.source}</b>: {s.text}</p>)}</details>}
              </div>
            ))}
            {busy === "coach" && <div className="msg coach"><pre>Thinking…</pre></div>}
          </div>
          <form className="ask" onSubmit={(e) => { e.preventDefault(); q.trim() && ask(q.trim()); }}>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask anything about this resume vs. this job…" />
            <button className="btn primary" disabled={!!busy || !q.trim()}>Ask</button>
          </form>
        </section>
      )}

      {metrics && (
        <footer className="muted">
          Model: {metrics.model.replace("_", " ")} · test accuracy {metrics.test_accuracy} · F1 {metrics.test_f1} · ROC-AUC {metrics.test_roc_auc} · {metrics.n_pairs} labelled pairs
        </footer>
      )}
    </div>
  );
}
