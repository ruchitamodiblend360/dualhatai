import { useState, useRef } from "react";

/* ─── AI prompt ─────────────────────────────────────────────────────────── */
const SYSTEM_PROMPT = `You are an expert agile coach and product manager specializing in user story quality assessment. Your job is to review user stories or epics before sprint planning and return a structured readiness report.

Evaluate the story across these 5 dimensions, each scored 0–20 (total: 0–100):

1. COMPLETENESS (0–20): Is the "As a [persona]… I want… so that…" format present? Are acceptance criteria defined? Are edge cases covered?
2. CLARITY (0–20): Is language unambiguous? Are vague terms like "fast", "easy", "user-friendly", "simple", "good" used without definition?
3. TESTABILITY (0–20): Can acceptance criteria be verified? Are there measurable outcomes?
4. SIZE (0–20): Is the story appropriately sized for a sprint (not an epic in disguise)? Does it have a clear, singular goal?
5. DEPENDENCY RISK (0–20): Are there implied dependencies on other systems, teams, or stories?

Return ONLY valid JSON, no markdown, no explanation outside the JSON. Format exactly as:
{
  "scores": { "completeness": <0-20>, "clarity": <0-20>, "testability": <0-20>, "size": <0-20>, "dependency_risk": <0-20> },
  "total": <0-100>,
  "readiness_level": "<Not Ready | Needs Work | Almost Ready | Sprint Ready>",
  "summary": "<2-3 sentence overall assessment>",
  "gaps": [{ "severity": "<critical|warning|info>", "area": "<area name>", "issue": "<specific issue>", "fix": "<actionable fix>" }],
  "ambiguities": [{ "phrase": "<quoted vague phrase>", "question": "<specific clarifying question>" }],
  "dependencies": [{ "type": "<team|api|story|system>", "description": "<what the dependency is>", "confidence": "<high|medium|low>" }],
  "improved_story": "<rewritten story with clearer language and stronger ACs>",
  "suggested_acs": ["<AC 1 in Given/When/Then format>", "<AC 2>", "<AC 3>"]
}`;

/* ─── Constants ─────────────────────────────────────────────────────────── */
const EXAMPLE_STORY = `As a registered user, I want to reset my password so that I can regain access to my account if I forget it.

Acceptance Criteria:
- User can click "Forgot password" on login page
- System sends an email
- User can set a new password`;

const SEVERITY_CFG = {
  critical: { color: "#A32D2D", bg: "#FCEBEB", border: "#F09595", label: "Critical", icon: "ti-alert-circle" },
  warning:  { color: "#854F0B", bg: "#FAEEDA", border: "#EF9F27", label: "Warning",  icon: "ti-alert-triangle" },
  info:     { color: "#185FA5", bg: "#E6F1FB", border: "#85B7EB", label: "Info",     icon: "ti-info-circle" },
};

const CONFIDENCE_CFG = {
  high:   { color: "#A32D2D", bg: "#FCEBEB" },
  medium: { color: "#854F0B", bg: "#FAEEDA" },
  low:    { color: "#185FA5", bg: "#E6F1FB" },
};

const DEP_ICONS = { team: "ti-users", api: "ti-api", story: "ti-git-branch", system: "ti-server" };

const SCORE_LABELS = {
  completeness: "Completeness", clarity: "Clarity",
  testability: "Testability", size: "Story size", dependency_risk: "Dependency risk",
};

const READINESS_CFG = {
  "Not Ready":    { color: "#A32D2D", bg: "#FCEBEB" },
  "Needs Work":   { color: "#854F0B", bg: "#FAEEDA" },
  "Almost Ready": { color: "#185FA5", bg: "#E6F1FB" },
  "Sprint Ready": { color: "#0F6E56", bg: "#E1F5EE" },
};

const RECENT_MOCK = [
  { id: "AUTH-14", title: "Password reset via email", score: 38, status: "Not Ready" },
  { id: "NOTIF-03", title: "Email notify on task assign", score: 91, status: "Sprint Ready" },
  { id: "DASH-07", title: "PDF export dashboard", score: 61, status: "Needs Work" },
];

/* ─── Helpers ────────────────────────────────────────────────────────────── */
const scoreColor = v => v >= 16 ? "#1D9E75" : v >= 11 ? "#378ADD" : v >= 6 ? "#EF9F27" : "#E24B4A";

const pill = (status) => {
  const c = READINESS_CFG[status] || READINESS_CFG["Needs Work"];
  return (
    <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 99,
      background: c.bg, color: c.color, whiteSpace: "nowrap" }}>
      {status}
    </span>
  );
};

/* ─── Sub-components ─────────────────────────────────────────────────────── */
function ScoreRing({ score, size = 30 }) {
  const c = scoreColor(score);
  const cfg = READINESS_CFG[
    score >= 80 ? "Sprint Ready" : score >= 60 ? "Almost Ready" : score >= 40 ? "Needs Work" : "Not Ready"
  ];
  return (
    <div style={{ width: size, height: size, borderRadius: "50%", border: `2px solid ${c}`,
      background: cfg.bg, display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: size * 0.33, fontWeight: 500, color: c, flexShrink: 0 }}>
      {score}
    </div>
  );
}

function SidePanel() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* What we check */}
      <div style={cardStyle}>
        <div style={sideTitleStyle}>
          <i className="ti ti-info-circle" style={{ fontSize: 14, color: "#185FA5" }} aria-hidden="true" />
          What we check
        </div>
        {["Completeness of the story format", "Clarity & vague language",
          "Testability of acceptance criteria", "Story size & sprint fit",
          "Dependency signals"].map((t, i) => (
          <div key={i} style={{ display: "flex", gap: 7, alignItems: "flex-start", padding: "5px 0",
            borderBottom: i < 4 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
            <i className="ti ti-check" style={{ fontSize: 13, color: "#1D9E75", flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
            <span style={{ fontSize: 12, color: "var(--color-text-primary)", lineHeight: 1.5 }}>{t}</span>
          </div>
        ))}
      </div>

      {/* Tips */}
      <div style={cardStyle}>
        <div style={sideTitleStyle}>
          <i className="ti ti-bulb" style={{ fontSize: 14, color: "#854F0B" }} aria-hidden="true" />
          Tips for better results
        </div>
        {["Include all existing ACs, even rough ones",
          "Mention the epic for dependency detection",
          "Add your DoR to score against your own bar"].map((t, i) => (
          <div key={i} style={{ display: "flex", gap: 7, alignItems: "flex-start", padding: "5px 0",
            borderBottom: i < 2 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
            <i className="ti ti-arrow-right" style={{ fontSize: 12, color: "var(--color-text-tertiary)", flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>{t}</span>
          </div>
        ))}
      </div>

      {/* Recent checks */}
      <div style={cardStyle}>
        <div style={sideTitleStyle}>
          <i className="ti ti-clock" style={{ fontSize: 14, color: "var(--color-text-secondary)" }} aria-hidden="true" />
          Recent checks
        </div>
        {RECENT_MOCK.map((r, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 9, padding: "8px 0",
            borderBottom: i < RECENT_MOCK.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
            <ScoreRing score={r.score} size={28} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)",
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.title}</div>
              <div style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>{r.id}</div>
            </div>
            {pill(r.status)}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Styles ─────────────────────────────────────────────────────────────── */
const cardStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "14px 16px",
};

const sideTitleStyle = {
  fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)",
  display: "flex", alignItems: "center", gap: 6, marginBottom: 10,
};

/* ─── Main App ───────────────────────────────────────────────────────────── */
export default function App() {
  const [activeNav, setActiveNav]     = useState("checker");
  const [story, setStory]             = useState("");
  const [epic, setEpic]               = useState("");
  const [dor, setDor]                 = useState("");
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [resultTab, setResultTab]     = useState("gaps");
  const [copied, setCopied]           = useState(false);
  const resultRef = useRef(null);

  async function analyzeStory() {
    if (!story.trim()) return;
    setLoading(true); setResult(null); setError(null);
    const context = [
      epic.trim() && `Parent epic: ${epic}`,
      dor.trim()  && `Team Definition of Ready: ${dor}`,
    ].filter(Boolean).join("\n");
    const userContent = context
      ? `User story:\n${story}\n\n${context}`
      : `User story:\n${story}`;
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6", max_tokens: 1000,
          system: SYSTEM_PROMPT,
          messages: [{ role: "user", content: userContent }],
        }),
      });
      const data = await res.json();
      const text = data.content?.map(b => b.text || "").join("") || "";
      const parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
      setResult(parsed); setResultTab("gaps");
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function copyImproved() {
    if (!result?.improved_story) return;
    navigator.clipboard.writeText(result.improved_story);
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  }

  /* ── NAV TABS ── */
  const navTabs = [
    { id: "checker",  label: "Readiness Checker",         icon: "ti-sparkles" },
    { id: "dashboard", label: "Dashboard",                icon: "ti-layout-dashboard" },
    { id: "panel",    label: "Single story analysis panel", icon: "ti-file-analytics" },
  ];

  /* ── RESULT TABS ── */
  const resultTabs = [
    { id: "gaps",          label: "Gaps",           icon: "ti-alert-circle",  count: result?.gaps?.length },
    { id: "ambiguities",   label: "Ambiguities",    icon: "ti-question-mark", count: result?.ambiguities?.length },
    { id: "dependencies",  label: "Dependencies",   icon: "ti-git-branch",    count: result?.dependencies?.length },
    { id: "improved",      label: "Improved story", icon: "ti-sparkles",      count: null },
  ];

  return (
    <div style={{ fontFamily: "var(--font-sans)", background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)",
      overflow: "hidden" }}>
      <h2 className="sr-only">User story readiness checker</h2>

      {/* ── TOP NAV ── */}
      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)",
        background: "var(--color-background-primary)" }}>
        {navTabs.map(t => (
          <button key={t.id} onClick={() => setActiveNav(t.id)}
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "11px 18px",
              fontSize: 13, fontWeight: activeNav === t.id ? 500 : 400,
              color: activeNav === t.id ? "#185FA5" : "var(--color-text-secondary)",
              background: activeNav === t.id ? "#E6F1FB" : "none",
              border: "none", borderRight: "0.5px solid var(--color-border-tertiary)",
              cursor: "pointer" }}>
            <i className={`ti ${t.icon}`} style={{ fontSize: 14 }} aria-hidden="true" />
            {t.label}
          </button>
        ))}
      </div>

      {/* ── CHECKER VIEW ── */}
      {activeNav === "checker" && (
        <div style={{ padding: "24px 20px", background: "var(--color-background-secondary)" }}>
          <div style={{ marginBottom: 20 }}>
            <h1 style={{ fontSize: 20, fontWeight: 500, color: "#185FA5", margin: "0 0 4px" }}>Check a user story</h1>
            <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>
              Paste your story below and get a readiness score, flagged gaps, and an improved version in seconds.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.55fr) minmax(0,0.45fr)", gap: 16, alignItems: "start" }}>

            {/* ── LEFT: input card ── */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={cardStyle}>
                {/* Story textarea */}
                <div style={{ marginBottom: 14 }}>
                  <label style={{ display: "block", fontSize: 12, fontWeight: 500,
                    color: "var(--color-text-secondary)", marginBottom: 7 }}>
                    Paste User Story / Epic here below:
                  </label>
                  <textarea
                    value={story}
                    onChange={e => setStory(e.target.value)}
                    rows={10}
                    placeholder={`As a [persona], I want [goal] so that [reason].\n\nAcceptance Criteria:\n- …`}
                    style={{ width: "100%", resize: "vertical", fontSize: 13,
                      fontFamily: "var(--font-mono)", lineHeight: 1.7,
                      background: "var(--color-background-secondary)",
                      border: "0.5px solid var(--color-border-secondary)",
                      borderRadius: "var(--border-radius-md)", padding: "10px 12px",
                      color: "var(--color-text-primary)", boxSizing: "border-box" }}
                  />
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", textAlign: "right", marginTop: 4 }}>
                    {story.length} chars
                  </div>
                </div>

                <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 14, marginBottom: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 10 }}>
                    Context <span style={{ fontWeight: 400, color: "var(--color-text-tertiary)" }}>(optional — improves accuracy)</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
                    <div>
                      <label style={{ display: "block", fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 5 }}>Parent epic</label>
                      <input value={epic} onChange={e => setEpic(e.target.value)}
                        placeholder="e.g. Auth & security epic"
                        style={{ width: "100%", fontSize: 12, padding: "7px 10px", boxSizing: "border-box",
                          background: "var(--color-background-secondary)",
                          border: "0.5px solid var(--color-border-secondary)",
                          borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }} />
                    </div>
                    <div>
                      <label style={{ display: "block", fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 5 }}>Team Definition of Ready</label>
                      <input value={dor} onChange={e => setDor(e.target.value)}
                        placeholder="e.g. GWT ACs, max 8 pts…"
                        style={{ width: "100%", fontSize: 12, padding: "7px 10px", boxSizing: "border-box",
                          background: "var(--color-background-secondary)",
                          border: "0.5px solid var(--color-border-secondary)",
                          borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }} />
                    </div>
                  </div>
                </div>

                {/* Action row */}
                <div style={{ display: "flex", gap: 10, alignItems: "center", justifyContent: "space-between" }}>
                  <button onClick={() => { setStory(EXAMPLE_STORY); setEpic(""); setDor(""); setResult(null); }}
                    style={{ fontSize: 12, padding: "8px 14px", background: "none",
                      border: "0.5px solid var(--color-border-secondary)",
                      borderRadius: "var(--border-radius-md)", color: "var(--color-text-secondary)", cursor: "pointer" }}>
                    Load example
                  </button>
                  <button onClick={analyzeStory} disabled={loading || !story.trim()}
                    style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 500,
                      padding: "9px 20px", background: loading || !story.trim() ? "#85B7EB" : "#185FA5",
                      color: "#fff", border: "none", borderRadius: "var(--border-radius-md)",
                      cursor: loading || !story.trim() ? "not-allowed" : "pointer" }}>
                    {loading
                      ? <><i className="ti ti-loader-2" style={{ fontSize: 15, animation: "spin 1s linear infinite" }} aria-hidden="true" /> Analysing…</>
                      : <><i className="ti ti-sparkles" style={{ fontSize: 15 }} aria-hidden="true" /> Check Readiness</>}
                  </button>
                </div>
              </div>

              {/* Error */}
              {error && (
                <div style={{ background: "#FCEBEB", border: "0.5px solid #F09595",
                  borderRadius: "var(--border-radius-md)", padding: "10px 14px",
                  color: "#A32D2D", fontSize: 13 }}>
                  <i className="ti ti-alert-circle" aria-hidden="true" /> {error}
                </div>
              )}

              {/* ── RESULTS ── */}
              {result && (
                <div ref={resultRef}>
                  {/* Score header */}
                  <div style={{ ...cardStyle, marginBottom: 0 }}>
                    <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
                      <div>
                        <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 6 }}>Readiness score</div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                          <span style={{ fontSize: 40, fontWeight: 500, color: scoreColor(result.total / 5) }}>{result.total}</span>
                          <span style={{ fontSize: 16, color: "var(--color-text-tertiary)" }}>/100</span>
                        </div>
                        <span style={{ display: "inline-block", marginTop: 6, fontSize: 11, fontWeight: 500,
                          padding: "3px 10px", borderRadius: 99,
                          background: READINESS_CFG[result.readiness_level]?.bg,
                          color: READINESS_CFG[result.readiness_level]?.color }}>
                          {result.readiness_level}
                        </span>
                      </div>
                      <div style={{ flex: 1, minWidth: 180 }}>
                        {Object.entries(result.scores).map(([k, v]) => (
                          <div key={k} style={{ marginBottom: 7 }}>
                            <div style={{ display: "flex", justifyContent: "space-between",
                              fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 3 }}>
                              <span>{SCORE_LABELS[k]}</span>
                              <span style={{ color: scoreColor(v), fontWeight: 500 }}>{v}/20</span>
                            </div>
                            <div style={{ height: 4, background: "var(--color-background-secondary)", borderRadius: 99 }}>
                              <div style={{ height: "100%", width: `${(v/20)*100}%`,
                                background: scoreColor(v), borderRadius: 99 }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    {result.summary && (
                      <p style={{ margin: "12px 0 0", fontSize: 13, color: "var(--color-text-secondary)",
                        lineHeight: 1.7, borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 12 }}>
                        {result.summary}
                      </p>
                    )}
                  </div>

                  {/* Result tabs */}
                  <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)",
                    marginTop: 12 }}>
                    {resultTabs.map(t => (
                      <button key={t.id} onClick={() => setResultTab(t.id)}
                        style={{ display: "flex", alignItems: "center", gap: 5, padding: "9px 14px",
                          fontSize: 12, fontWeight: resultTab === t.id ? 500 : 400,
                          color: resultTab === t.id ? "#185FA5" : "var(--color-text-secondary)",
                          background: "none", border: "none",
                          borderBottom: resultTab === t.id ? "2px solid #185FA5" : "2px solid transparent",
                          cursor: "pointer", marginBottom: -1 }}>
                        <i className={`ti ${t.icon}`} style={{ fontSize: 13 }} aria-hidden="true" />
                        {t.label}
                        {t.count > 0 && (
                          <span style={{ fontSize: 10, fontWeight: 500, padding: "1px 6px", borderRadius: 99,
                            background: resultTab === t.id ? "#E6F1FB" : "var(--color-background-secondary)",
                            color: resultTab === t.id ? "#185FA5" : "var(--color-text-secondary)" }}>
                            {t.count}
                          </span>
                        )}
                      </button>
                    ))}
                  </div>

                  <div style={{ ...cardStyle, borderTopLeftRadius: 0, borderTopRightRadius: 0,
                    borderTop: "none", marginBottom: 8 }}>

                    {/* Gaps */}
                    {resultTab === "gaps" && (
                      <div>
                        {!result.gaps?.length && <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>No significant gaps detected.</p>}
                        {result.gaps?.map((g, i) => {
                          const c = SEVERITY_CFG[g.severity] || SEVERITY_CFG.info;
                          return (
                            <div key={i} style={{ border: `0.5px solid ${c.border}`, borderRadius: "var(--border-radius-md)",
                              padding: "11px 13px", marginBottom: 9, background: c.bg }}>
                              <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
                                <i className={`ti ${c.icon}`} style={{ fontSize: 15, color: c.color }} aria-hidden="true" />
                                <span style={{ fontSize: 11, fontWeight: 500, color: c.color }}>{c.label}</span>
                                <span style={{ fontSize: 11, color: c.color, opacity: 0.7 }}>· {g.area}</span>
                              </div>
                              <p style={{ margin: "0 0 5px", fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.5 }}>{g.issue}</p>
                              <p style={{ margin: 0, fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
                                <i className="ti ti-arrow-right" style={{ fontSize: 12 }} aria-hidden="true" /> {g.fix}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Ambiguities */}
                    {resultTab === "ambiguities" && (
                      <div>
                        {!result.ambiguities?.length && <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>No ambiguous language detected.</p>}
                        {result.ambiguities?.map((a, i) => (
                          <div key={i} style={{ display: "flex", gap: 12, padding: "11px 0",
                            borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                            <div style={{ width: 30, height: 30, borderRadius: "50%", background: "#FAEEDA",
                              display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                              <i className="ti ti-question-mark" style={{ fontSize: 15, color: "#854F0B" }} aria-hidden="true" />
                            </div>
                            <div>
                              <div style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "#854F0B",
                                background: "#FAEEDA", padding: "2px 7px", borderRadius: 4,
                                display: "inline-block", marginBottom: 5 }}>"{a.phrase}"</div>
                              <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.6 }}>{a.question}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Dependencies */}
                    {resultTab === "dependencies" && (
                      <div>
                        {!result.dependencies?.length && <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>No dependency signals detected.</p>}
                        {result.dependencies?.map((d, i) => {
                          const c = CONFIDENCE_CFG[d.confidence] || CONFIDENCE_CFG.low;
                          return (
                            <div key={i} style={{ display: "flex", gap: 11, padding: "11px 0",
                              borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                              <div style={{ width: 30, height: 30, borderRadius: "var(--border-radius-md)",
                                background: "var(--color-background-secondary)", display: "flex",
                                alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                <i className={`ti ${DEP_ICONS[d.type] || "ti-link"}`}
                                  style={{ fontSize: 15, color: "var(--color-text-secondary)" }} aria-hidden="true" />
                              </div>
                              <div style={{ flex: 1 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}>
                                  <span style={{ fontSize: 12, fontWeight: 500, textTransform: "capitalize",
                                    color: "var(--color-text-primary)" }}>{d.type}</span>
                                  <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 7px", borderRadius: 99,
                                    background: c.bg, color: c.color }}>{d.confidence} confidence</span>
                                </div>
                                <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>{d.description}</p>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Improved story */}
                    {resultTab === "improved" && (
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 11 }}>
                          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>AI-suggested rewrite — review before using</span>
                          <button onClick={copyImproved}
                            style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11,
                              color: "var(--color-text-secondary)", background: "none",
                              border: "0.5px solid var(--color-border-secondary)",
                              borderRadius: "var(--border-radius-md)", padding: "4px 9px", cursor: "pointer" }}>
                            <i className={`ti ${copied ? "ti-check" : "ti-copy"}`} style={{ fontSize: 13 }} aria-hidden="true" />
                            {copied ? "Copied" : "Copy"}
                          </button>
                        </div>
                        <pre style={{ margin: "0 0 14px", fontFamily: "var(--font-mono)", fontSize: 12,
                          lineHeight: 1.7, whiteSpace: "pre-wrap", wordBreak: "break-word",
                          background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)",
                          padding: "12px 14px", color: "var(--color-text-primary)" }}>
                          {result.improved_story}
                        </pre>
                        {result.suggested_acs?.length > 0 && (
                          <>
                            <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 9 }}>Suggested acceptance criteria</div>
                            {result.suggested_acs.map((ac, i) => (
                              <div key={i} style={{ display: "flex", gap: 9, padding: "7px 0",
                                borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                                <i className="ti ti-circle-check" style={{ fontSize: 15, color: "#1D9E75",
                                  marginTop: 2, flexShrink: 0 }} aria-hidden="true" />
                                <span style={{ fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.6 }}>{ac}</span>
                              </div>
                            ))}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* ── RIGHT: side panel ── */}
            <SidePanel />
          </div>
        </div>
      )}

      {/* ── DASHBOARD placeholder ── */}
      {activeNav === "dashboard" && (
        <div style={{ padding: "40px 24px", textAlign: "center", color: "var(--color-text-secondary)" }}>
          <i className="ti ti-layout-dashboard" style={{ fontSize: 32, color: "var(--color-text-tertiary)", display: "block", marginBottom: 12 }} aria-hidden="true" />
          <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 6 }}>Sprint dashboard</div>
          <div style={{ fontSize: 13 }}>View all stories for the current sprint and their readiness scores.</div>
        </div>
      )}

      {/* ── SINGLE STORY PANEL placeholder ── */}
      {activeNav === "panel" && (
        <div style={{ padding: "40px 24px", textAlign: "center", color: "var(--color-text-secondary)" }}>
          <i className="ti ti-file-analytics" style={{ fontSize: 32, color: "var(--color-text-tertiary)", display: "block", marginBottom: 12 }} aria-hidden="true" />
          <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 6 }}>Single story analysis panel</div>
          <div style={{ fontSize: 13 }}>Deep-dive view — annotated story text alongside score breakdown and suggested ACs.</div>
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        textarea:focus, input:focus { outline: none; border-color: #378ADD !important; box-shadow: 0 0 0 3px rgba(55,138,221,0.15); }
        button:hover { opacity: 0.88; }
      `}</style>
    </div>
  );
}
