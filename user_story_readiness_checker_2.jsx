import { useState, useRef, useEffect } from "react";

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

const EXAMPLE_STORY = `As a registered user, I want to reset my password so that I can regain access to my account if I forget it.

Acceptance Criteria:
- User can click "Forgot password" on login page
- System sends an email
- User can set a new password`;

const LOADING_STEPS = [
  { label: "Parsing story format & persona" },
  { label: "Detecting acceptance criteria" },
  { label: "Scoring completeness, clarity, testability…" },
  { label: "Detecting dependency signals" },
  { label: "Generating improved story & ACs" },
];

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
  { id: "AUTH-14",  title: "Password reset via email",   score: 38, status: "Not Ready"    },
  { id: "NOTIF-03", title: "Email notify on task assign", score: 91, status: "Sprint Ready" },
  { id: "DASH-07",  title: "PDF export dashboard",        score: 61, status: "Needs Work"   },
];

const scoreColor = v => v >= 16 ? "#1D9E75" : v >= 11 ? "#378ADD" : v >= 6 ? "#EF9F27" : "#E24B4A";

const cardStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  overflow: "hidden",
};

const sideTitleStyle = {
  fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)",
  display: "flex", alignItems: "center", gap: 6, marginBottom: 10,
};

function StatusPill({ status }) {
  const c = READINESS_CFG[status] || READINESS_CFG["Needs Work"];
  return (
    <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 99,
      background: c.bg, color: c.color, whiteSpace: "nowrap" }}>{status}</span>
  );
}

function ScoreRing({ score, size = 28 }) {
  const c = scoreColor(score);
  const bg = READINESS_CFG[score >= 80 ? "Sprint Ready" : score >= 60 ? "Almost Ready" : score >= 40 ? "Needs Work" : "Not Ready"].bg;
  return (
    <div style={{ width: size, height: size, borderRadius: "50%", border: `2px solid ${c}`,
      background: bg, display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: Math.round(size * 0.33), fontWeight: 500, color: c, flexShrink: 0 }}>
      {score}
    </div>
  );
}

/* ── Loading state ── */
function LoadingState({ step }) {
  return (
    <div style={{ ...cardStyle, marginBottom: 0 }}>
      <div style={{ padding: "16px 18px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 36, height: 36, borderRadius: "50%",
              border: "3px solid #E6F1FB", borderTopColor: "#185FA5",
              animation: "spin 0.9s linear infinite", flexShrink: 0 }} aria-hidden="true" />
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>Analysing your story…</div>
              <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 2 }}>This takes about 5–10 seconds</div>
            </div>
          </div>
          <span style={{ fontSize: 11, fontWeight: 500, background: "#E6F1FB", color: "#185FA5",
            padding: "4px 10px", borderRadius: 99 }}>In progress</span>
        </div>

        {LOADING_STEPS.map((s, i) => {
          const done = i < step;
          const active = i === step;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 0",
              borderBottom: i < LOADING_STEPS.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
              <div style={{ width: 22, height: 22, borderRadius: "50%", flexShrink: 0, display: "flex",
                alignItems: "center", justifyContent: "center",
                background: done ? "#E1F5EE" : active ? "#E6F1FB" : "var(--color-background-secondary)",
                border: active ? "1.5px solid #85B7EB" : "none" }}>
                {done
                  ? <i className="ti ti-check" style={{ fontSize: 12, color: "#1D9E75" }} aria-hidden="true" />
                  : active
                    ? <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#185FA5",
                        animation: "pulse 1s ease-in-out infinite" }} />
                    : null}
              </div>
              <span style={{ fontSize: 12, color: done ? "#0F6E56" : active ? "#185FA5" : "var(--color-text-tertiary)" }}>
                {s.label}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "12px 18px", display: "flex", gap: 10 }}>
        <div style={{ height: 12, borderRadius: 4, background: "var(--color-background-secondary)",
          width: "55%", animation: "pulse 1.4s ease-in-out infinite" }} />
        <div style={{ height: 12, borderRadius: 4, background: "var(--color-background-secondary)",
          width: "25%", animation: "pulse 1.4s ease-in-out infinite" }} />
      </div>
    </div>
  );
}

/* ── Error state ── */
function ErrorState({ onRetry, onEdit }) {
  return (
    <div style={{ ...cardStyle }}>
      <div style={{ padding: "36px 24px", display: "flex", flexDirection: "column",
        alignItems: "center", textAlign: "center" }}>
        <div style={{ width: 48, height: 48, borderRadius: "50%", background: "#FCEBEB",
          display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 14 }}>
          <i className="ti ti-alert-circle" style={{ fontSize: 24, color: "#A32D2D" }} aria-hidden="true" />
        </div>
        <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 6 }}>
          Analysis failed
        </div>
        <div style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.7,
          marginBottom: 22, maxWidth: 320 }}>
          Something went wrong while processing your story. Your text is still here — try again or edit the story.
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button onClick={onEdit}
            style={{ fontSize: 13, padding: "8px 16px", background: "none", cursor: "pointer",
              border: "0.5px solid var(--color-border-secondary)",
              borderRadius: "var(--border-radius-md)", color: "var(--color-text-secondary)" }}>
            Edit story
          </button>
          <button onClick={onRetry}
            style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, padding: "8px 16px",
              background: "#185FA5", color: "#fff", border: "none", cursor: "pointer",
              borderRadius: "var(--border-radius-md)" }}>
            <i className="ti ti-refresh" style={{ fontSize: 14 }} aria-hidden="true" />
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Result state ── */
function ResultState({ result, onRecheck }) {
  const [tab, setTab] = useState("gaps");
  const [copied, setCopied] = useState(false);

  function copyImproved() {
    navigator.clipboard.writeText(result.improved_story || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const resultTabs = [
    { id: "gaps",         label: "Gaps",           icon: "ti-alert-circle",  count: result.gaps?.length },
    { id: "ambiguities",  label: "Ambiguities",     icon: "ti-question-mark", count: result.ambiguities?.length },
    { id: "dependencies", label: "Dependencies",    icon: "ti-git-branch",    count: result.dependencies?.length },
    { id: "improved",     label: "Improved story",  icon: "ti-sparkles",      count: null },
  ];

  const criticalCount = result.gaps?.filter(g => g.severity === "critical").length || 0;
  const rc = READINESS_CFG[result.readiness_level] || READINESS_CFG["Needs Work"];

  return (
    <div style={cardStyle}>
      {/* Score header */}
      <div style={{ padding: "16px 18px" }}>
        <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ minWidth: 130 }}>
            <div style={{ fontSize: 11, fontWeight: 500, textTransform: "uppercase",
              letterSpacing: "0.05em", color: "var(--color-text-secondary)", marginBottom: 6 }}>
              Readiness score
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 8 }}>
              <span style={{ fontSize: 44, fontWeight: 500, color: scoreColor(result.total / 5), lineHeight: 1 }}>
                {result.total}
              </span>
              <span style={{ fontSize: 16, color: "var(--color-text-tertiary)" }}>/100</span>
            </div>
            <span style={{ display: "inline-block", fontSize: 11, fontWeight: 500,
              padding: "4px 12px", borderRadius: 99, background: rc.bg, color: rc.color }}>
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
                  <div style={{ height: "100%", width: `${Math.round((v / 20) * 100)}%`,
                    background: scoreColor(v), borderRadius: 99,
                    transition: "width 0.7s ease" }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {result.summary && (
          <p style={{ margin: "14px 0 0", fontSize: 13, color: "var(--color-text-secondary)",
            lineHeight: 1.7, borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 12 }}>
            {result.summary}
          </p>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)",
        borderTop: "0.5px solid var(--color-border-tertiary)", padding: "0 6px",
        background: "var(--color-background-secondary)" }}>
        {resultTabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{ display: "flex", alignItems: "center", gap: 5, padding: "9px 12px",
              fontSize: 12, fontWeight: tab === t.id ? 500 : 400,
              color: tab === t.id ? "#185FA5" : "var(--color-text-secondary)",
              background: "none", border: "none",
              borderBottom: tab === t.id ? "2px solid #185FA5" : "2px solid transparent",
              cursor: "pointer", marginBottom: -1 }}>
            <i className={`ti ${t.icon}`} style={{ fontSize: 13 }} aria-hidden="true" />
            {t.label}
            {t.count > 0 && (
              <span style={{ fontSize: 10, fontWeight: 500, padding: "1px 6px", borderRadius: 99,
                background: tab === t.id ? "#E6F1FB" : "var(--color-background-primary)",
                color: tab === t.id ? "#185FA5" : "var(--color-text-secondary)" }}>
                {t.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ padding: "16px 18px" }}>

        {tab === "gaps" && (
          <div>
            {!result.gaps?.length
              ? <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>No significant gaps detected.</p>
              : result.gaps.map((g, i) => {
                  const c = SEVERITY_CFG[g.severity] || SEVERITY_CFG.info;
                  return (
                    <div key={i} style={{ border: `0.5px solid ${c.border}`, borderRadius: "var(--border-radius-md)",
                      padding: "11px 13px", marginBottom: 9, background: c.bg }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
                        <i className={`ti ${c.icon}`} style={{ fontSize: 14, color: c.color }} aria-hidden="true" />
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

        {tab === "ambiguities" && (
          <div>
            {!result.ambiguities?.length
              ? <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>No ambiguous language detected.</p>
              : result.ambiguities.map((a, i) => (
                  <div key={i} style={{ display: "flex", gap: 12, padding: "11px 0",
                    borderBottom: i < result.ambiguities.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
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

        {tab === "dependencies" && (
          <div>
            {!result.dependencies?.length
              ? <p style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>No dependency signals detected.</p>
              : result.dependencies.map((d, i) => {
                  const c = CONFIDENCE_CFG[d.confidence] || CONFIDENCE_CFG.low;
                  return (
                    <div key={i} style={{ display: "flex", gap: 11, padding: "11px 0",
                      borderBottom: i < result.dependencies.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
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
                          <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 7px",
                            borderRadius: 99, background: c.bg, color: c.color }}>{d.confidence} confidence</span>
                        </div>
                        <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>{d.description}</p>
                      </div>
                    </div>
                  );
                })}
          </div>
        )}

        {tab === "improved" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 11 }}>
              <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>AI-suggested rewrite — review before using</span>
              <button onClick={copyImproved}
                style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11,
                  color: "var(--color-text-secondary)", background: "none", cursor: "pointer",
                  border: "0.5px solid var(--color-border-secondary)",
                  borderRadius: "var(--border-radius-md)", padding: "4px 9px" }}>
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
                <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 9 }}>
                  Suggested acceptance criteria
                </div>
                {result.suggested_acs.map((ac, i) => (
                  <div key={i} style={{ display: "flex", gap: 9, padding: "7px 0",
                    borderBottom: i < result.suggested_acs.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
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

      {/* Footer */}
      <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "11px 18px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "var(--color-background-secondary)" }}>
        <span style={{ fontSize: 12, color: criticalCount > 0 ? "#A32D2D" : "var(--color-text-secondary)" }}>
          {criticalCount > 0
            ? <><i className="ti ti-alert-circle" style={{ fontSize: 13, marginRight: 5 }} aria-hidden="true" />{criticalCount} blocker{criticalCount > 1 ? "s" : ""} must be resolved before sprint planning</>
            : <><i className="ti ti-circle-check" style={{ fontSize: 13, marginRight: 5, color: "#1D9E75" }} aria-hidden="true" />No blockers — story is good to go</>}
        </span>
        <button onClick={onRecheck}
          style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 500,
            padding: "6px 14px", background: "#185FA5", color: "#fff", border: "none",
            borderRadius: "var(--border-radius-md)", cursor: "pointer" }}>
          <i className="ti ti-refresh" style={{ fontSize: 13 }} aria-hidden="true" />
          Re-check
        </button>
      </div>
    </div>
  );
}

/* ── Side panel ── */
function SidePanel() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ ...cardStyle, padding: "14px 16px" }}>
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

      <div style={{ ...cardStyle, padding: "14px 16px" }}>
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

      {/* Recent checks — disabled */}
    </div>
  );
}

/* ── Main App ── */
export default function App() {
  const [activeNav, setActiveNav] = useState("checker");
  const [story, setStory]         = useState("");
  const [epic, setEpic]           = useState("");
  const [dor, setDor]             = useState("");

  // "idle" | "loading" | "done" | "error"
  const [uiState, setUiState]     = useState("idle");
  const [loadStep, setLoadStep]   = useState(0);
  const [result, setResult]       = useState(null);
  const [history, setHistory]     = useState(() => {
    try { return JSON.parse(localStorage.getItem("story_history") || "[]"); } catch { return []; }
  });

  const resultRef = useRef(null);
  const timerRef  = useRef(null);

  // Advance loading steps every ~1.8s
  useEffect(() => {
    if (uiState === "loading") {
      setLoadStep(0);
      timerRef.current = setInterval(() => {
        setLoadStep(s => Math.min(s + 1, LOADING_STEPS.length - 1));
      }, 1800);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [uiState]);

  async function analyzeStory() {
    if (!story.trim()) return;
    setUiState("loading");
    setResult(null);

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
          model: "claude-sonnet-4-6",
          max_tokens: 1000,
          system: SYSTEM_PROMPT,
          messages: [{ role: "user", content: userContent }],
        }),
      });
      const data = await res.json();
      const text = data.content?.map(b => b.text || "").join("") || "";
      const parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
      setResult(parsed);
      setUiState("done");
      const entry = {
        id: Date.now(),
        title: story.trim().split("\n")[0].slice(0, 80),
        score: parsed.total,
        readiness_level: parsed.readiness_level,
        checkedAt: new Date().toLocaleString(),
      };
      setHistory(prev => {
        const updated = [entry, ...prev];
        localStorage.setItem("story_history", JSON.stringify(updated));
        return updated;
      });
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
    } catch {
      setUiState("error");
    }
  }

  const navTabs = [
    { id: "checker",   label: "Readiness Checker",           icon: "ti-sparkles" },
    { id: "dashboard", label: "Dashboard",                   icon: "ti-layout-dashboard" },
    { id: "panel",     label: "Single story analysis panel", icon: "ti-file-analytics" },
  ];

  return (
    <div style={{ fontFamily: "var(--font-sans)", background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)",
      overflow: "hidden" }}>
      <h2 className="sr-only">User story readiness checker</h2>

      {/* Top nav */}
      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        {navTabs.map(t => (
          <button key={t.id} onClick={() => setActiveNav(t.id)}
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "11px 18px",
              fontSize: 13, fontWeight: activeNav === t.id ? 500 : 400,
              color: activeNav === t.id ? "#185FA5" : "var(--color-text-secondary)",
              background: activeNav === t.id ? "#E6F1FB" : "none",
              border: "none", borderRight: "0.5px solid var(--color-border-tertiary)", cursor: "pointer" }}>
            <i className={`ti ${t.icon}`} style={{ fontSize: 14 }} aria-hidden="true" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Checker view */}
      {activeNav === "checker" && (
        <div style={{ padding: "24px 20px", background: "var(--color-background-secondary)" }}>
          <div style={{ marginBottom: 20 }}>
            <h1 style={{ fontSize: 20, fontWeight: 500, color: "#185FA5", margin: "0 0 4px" }}>Check a user story</h1>
            <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>
              Paste your story below and get a readiness score, flagged gaps, and an improved version in seconds.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.55fr) minmax(0,0.45fr)", gap: 16, alignItems: "start" }}>

            {/* Left column */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

              {/* Input card — always visible */}
              <div style={cardStyle}>
                <div style={{ padding: "16px 18px 0" }}>
                  <label style={{ display: "block", fontSize: 12, fontWeight: 500,
                    color: "var(--color-text-secondary)", marginBottom: 7 }}>
                    Paste User Story / Epic here below:
                  </label>
                  <textarea
                    value={story}
                    onChange={e => setStory(e.target.value)}
                    rows={9}
                    placeholder={`As a [persona], I want [goal] so that [reason].\n\nAcceptance Criteria:\n- …`}
                    style={{ width: "100%", resize: "vertical", fontSize: 13,
                      fontFamily: "var(--font-mono)", lineHeight: 1.7,
                      background: "var(--color-background-secondary)",
                      border: "0.5px solid var(--color-border-secondary)",
                      borderRadius: "var(--border-radius-md)", padding: "10px 12px",
                      color: "var(--color-text-primary)", boxSizing: "border-box" }}
                  />
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", textAlign: "right", marginTop: 3, marginBottom: 12 }}>
                    {story.length} chars
                  </div>

                  <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 13, marginBottom: 13 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 9 }}>
                      Context <span style={{ fontWeight: 400, color: "var(--color-text-tertiary)" }}>(optional)</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
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
                        <label style={{ display: "block", fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 5 }}>Definition of Ready</label>
                        <input value={dor} onChange={e => setDor(e.target.value)}
                          placeholder="e.g. GWT ACs, max 8 pts…"
                          style={{ width: "100%", fontSize: 12, padding: "7px 10px", boxSizing: "border-box",
                            background: "var(--color-background-secondary)",
                            border: "0.5px solid var(--color-border-secondary)",
                            borderRadius: "var(--border-radius-md)", color: "var(--color-text-primary)" }} />
                      </div>
                    </div>
                  </div>
                </div>

                <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "12px 18px",
                  display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <button onClick={() => { setStory(EXAMPLE_STORY); setEpic(""); setDor(""); setUiState("idle"); setResult(null); }}
                    style={{ fontSize: 12, padding: "7px 13px", background: "none", cursor: "pointer",
                      border: "0.5px solid var(--color-border-secondary)",
                      borderRadius: "var(--border-radius-md)", color: "var(--color-text-secondary)" }}>
                    Load example
                  </button>
                  <button onClick={analyzeStory}
                    disabled={uiState === "loading" || !story.trim()}
                    style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, fontWeight: 500,
                      padding: "9px 20px", cursor: uiState === "loading" || !story.trim() ? "not-allowed" : "pointer",
                      background: uiState === "loading" || !story.trim() ? "#85B7EB" : "#185FA5",
                      color: "#fff", border: "none", borderRadius: "var(--border-radius-md)" }}>
                    {uiState === "loading"
                      ? <><i className="ti ti-loader-2" style={{ fontSize: 15, animation: "spin 0.9s linear infinite" }} aria-hidden="true" /> Analysing…</>
                      : <><i className="ti ti-sparkles" style={{ fontSize: 15 }} aria-hidden="true" /> Check Readiness</>}
                  </button>
                </div>
              </div>

              {/* Dynamic result area */}
              <div ref={resultRef}>
                {uiState === "loading" && <LoadingState step={loadStep} />}
                {uiState === "error"   && <ErrorState onRetry={analyzeStory} onEdit={() => setUiState("idle")} />}
                {uiState === "done" && result && (
                  <ResultState result={result} onRecheck={() => { setUiState("idle"); setResult(null); }} />
                )}
              </div>
            </div>

            {/* Right side panel */}
            <SidePanel />
          </div>
        </div>
      )}

      {activeNav === "dashboard" && (
        <div style={{ padding: "24px 20px", background: "var(--color-background-secondary)", minHeight: 300 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 500, color: "#185FA5", margin: "0 0 4px" }}>Sprint Dashboard</h2>
              <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>All user stories checked through the tool.</p>
            </div>
            {history.length > 0 && (
              <button onClick={() => { setHistory([]); localStorage.removeItem("story_history"); }}
                style={{ fontSize: 12, padding: "6px 12px", background: "none", cursor: "pointer",
                  border: "0.5px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-md)",
                  color: "var(--color-text-secondary)" }}>
                <i className="ti ti-trash" style={{ fontSize: 13, marginRight: 5 }} aria-hidden="true" />
                Clear all
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div style={{ textAlign: "center", padding: "48px 24px" }}>
              <i className="ti ti-clipboard-list" style={{ fontSize: 32, color: "var(--color-text-tertiary)", display: "block", marginBottom: 12 }} aria-hidden="true" />
              <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 6 }}>No stories checked yet</div>
              <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Go to the Readiness Checker tab and analyse a story — it will appear here.</div>
            </div>
          ) : (
            <div style={{ ...cardStyle, overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ background: "var(--color-background-secondary)" }}>
                    {["#", "Story Title", "Readiness Score", "Status", "Checked At"].map((h, i) => (
                      <th key={i} style={{ textAlign: "left", padding: "10px 14px", fontSize: 11,
                        fontWeight: 500, color: "var(--color-text-secondary)",
                        borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {history.map((row, i) => {
                    const rc = READINESS_CFG[row.readiness_level] || READINESS_CFG["Needs Work"];
                    return (
                      <tr key={row.id} style={{ background: i % 2 === 0 ? "var(--color-background-primary)" : "var(--color-background-secondary)" }}>
                        <td style={{ padding: "11px 14px", color: "var(--color-text-tertiary)", width: 36 }}>{i + 1}</td>
                        <td style={{ padding: "11px 14px", color: "var(--color-text-primary)", fontWeight: 400, maxWidth: 340 }}>
                          <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row.title}</div>
                        </td>
                        <td style={{ padding: "11px 14px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ height: 6, width: 80, background: "var(--color-background-secondary)", borderRadius: 99 }}>
                              <div style={{ height: "100%", width: `${row.score}%`, borderRadius: 99,
                                background: scoreColor(row.score / 5) }} />
                            </div>
                            <span style={{ fontWeight: 500, color: scoreColor(row.score / 5) }}>{row.score}/100</span>
                          </div>
                        </td>
                        <td style={{ padding: "11px 14px" }}>
                          <span style={{ fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 99,
                            background: rc.bg, color: rc.color }}>{row.readiness_level}</span>
                        </td>
                        <td style={{ padding: "11px 14px", color: "var(--color-text-tertiary)", fontSize: 12 }}>{row.checkedAt}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeNav === "panel" && (
        <div style={{ padding: "48px 24px", textAlign: "center" }}>
          <i className="ti ti-file-analytics" style={{ fontSize: 32, color: "var(--color-text-tertiary)", display: "block", marginBottom: 12 }} aria-hidden="true" />
          <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 6 }}>Single story analysis panel</div>
          <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Deep-dive view — annotated story text alongside score breakdown and suggested ACs.</div>
        </div>
      )}

      <style>{`
        @keyframes spin  { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
        textarea:focus, input:focus { outline: none; border-color: #378ADD !important; box-shadow: 0 0 0 3px rgba(55,138,221,0.15); }
        button:not(:disabled):hover { opacity: 0.88; }
      `}</style>
    </div>
  );
}
