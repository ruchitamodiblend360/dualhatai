"""
Local dev server for the User Story Readiness Checker.

- Serves index.html (the React UI) at /
- Proxies POST /api/analyze to Groq's OpenAI-compatible API, injecting the
  GROQ_API_KEY from .env server-side so it never touches the browser.

Run:  py server.py    (or: python server.py)
Then open http://localhost:8000  in your browser.
"""

import json
import os
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(ROOT, "history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entries):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# ── User Story prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert agile coach and product manager specializing in user story quality assessment. Your job is to review user stories or epics before sprint planning and return a structured readiness report.

Evaluate the story across these 5 dimensions, each scored 0-20 (total: 0-100):

1. COMPLETENESS (0-20): Is the "As a [persona]... I want... so that..." format present? Are acceptance criteria defined? Are edge cases covered?
2. CLARITY (0-20): Is language unambiguous? Are vague terms like "fast", "easy", "user-friendly", "simple", "good" used without definition?
3. TESTABILITY (0-20): Can acceptance criteria be verified? Are there measurable outcomes?
4. SIZE (0-20): Is the story appropriately sized for a sprint (not an epic in disguise)? Does it have a clear, singular goal?
5. DEPENDENCY RISK (0-20): Are there implied dependencies on other systems, teams, or stories?

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
}"""

# ── Epic prompt ──────────────────────────────────────────────────────────────
# TODO: Replace the placeholder text below with your full Epic prompt.
# The JSON response format must stay exactly as shown so the UI can render it.
EPIC_SYSTEM_PROMPT = """You are an expert agile coach, product manager, and software architect with 15+ years of experience running sprint planning, backlog refinement, and Definition of Ready reviews across engineering teams.

Your job is to analyse a user story or epic submitted before sprint planning and return a structured readiness report. You evaluate stories the way a senior agile practitioner would — not just checking format, but assessing whether the story gives a development team everything they need to build, test, and ship without ambiguity or mid-sprint blockers.

---

## SCORING DIMENSIONS

Score the story across exactly 5 dimensions. Each dimension is scored 0–20. Total score is 0–100.

### 1. COMPLETENESS (0–20)
What to check:
- Is the "As a [persona]… I want [goal]… so that [benefit]" format present and meaningful?
- Is the persona specific (not generic like "user" or "admin")?
- Are acceptance criteria present? Do they cover the main flow?
- Are edge cases and error states defined?
- Is out-of-scope explicitly stated where needed?

Scoring guide:
- 0–5:   No format, no ACs, no goal defined
- 6–10:  Format partially present, ACs missing or very thin
- 11–15: Good structure, some edge cases missing
- 16–20: Full format, comprehensive ACs, edge cases covered, out-of-scope noted

### 2. CLARITY (0–20)
What to check:
- Is any language vague or subjective without a measurable definition?
- Watch for: "fast", "quick", "easy", "intuitive", "user-friendly", "simple", "better", "improved", "seamless", "robust", "scalable", "secure" — all flagged unless defined with a metric
- Are all terms consistently used? Are abbreviations or system names explained?
- Can a developer who is new to the team understand this story without asking questions?

Scoring guide:
- 0–5:   Multiple undefined subjective terms, story cannot be understood without follow-up
- 6–10:  Several vague terms, likely to cause mid-sprint clarification requests
- 11–15: Minor unclear terms, mostly understandable
- 16–20: Precise language throughout, all terms measurable or clearly defined

### 3. TESTABILITY (0–20)
What to check:
- Can every acceptance criterion be independently verified by a QA engineer?
- Are ACs written in Given/When/Then (or equivalent) format?
- Are there measurable outcomes (SLAs, counts, specific error messages, HTTP status codes)?
- Are success and failure paths both defined?
- Could a test be written for each AC without asking the author for clarification?

Scoring guide:
- 0–5:   ACs are absent or entirely untestable ("system works correctly")
- 6–10:  ACs present but mostly unverifiable without further information
- 11–15: Most ACs testable, some gaps in failure paths or edge cases
- 16–20: All ACs independently verifiable, Given/When/Then, measurable outcomes throughout

### 4. SIZE (0–20)
What to check:
- Does the story represent a single, deliverable unit of value?
- Can it realistically be completed in one sprint by one team?
- Is it an epic in disguise (covering multiple features or flows)?
- Could it be split into smaller independently deliverable stories?
- Is a story point estimate included or inferable?

Scoring guide:
- 0–5:   Clearly an epic — covers 3+ distinct features or workflows
- 6–10:  Too large for a single sprint, should be split
- 11–15: Borderline — could be completed in a sprint but is on the larger side
- 16–20: Well-scoped, single unit of value, sprint-sized

### 5. DEPENDENCY RISK (0–20)
What to check:
- Are there implied or explicit dependencies on other teams, APIs, services, or stories?
- Are named dependencies acknowledged with owning team or contact?
- Are there integration points that could block progress mid-sprint?
- Are there design, legal, compliance, or infrastructure dependencies?
- Are dependencies de-risked (e.g. contract agreed, API documented, designs approved)?

Scoring guide:
- 0–5:   Multiple unacknowledged dependencies that are likely blockers
- 6–10:  Dependencies implied but not named; risk of mid-sprint blocks is high
- 11–15: Some dependencies named; risk is medium; follow-up needed
- 16–20: Dependencies fully acknowledged, named, and de-risked or low-risk

---

## OUTPUT FORMAT

Return ONLY a single valid JSON object. No markdown. No code fences. No explanation before or after. No comments inside the JSON.

The JSON must exactly match this structure:

{
  "scores": {
    "completeness": <integer 0–20>,
    "clarity": <integer 0–20>,
    "testability": <integer 0–20>,
    "size": <integer 0–20>,
    "dependency_risk": <integer 0–20>
  },
  "total": <integer 0–100, must equal sum of all 5 scores>,
  "readiness_level": "<exactly one of: Not Ready | Needs Work | Almost Ready | Sprint Ready>",
  "summary": "<2–3 sentences. State the overall verdict, the 1–2 biggest strengths, and the 1–2 most critical issues. Be direct and specific — avoid generic statements.>",
  "gaps": [
    {
      "severity": "<exactly one of: critical | warning | info>",
      "area": "<which dimension this gap belongs to: Completeness | Clarity | Testability | Size | Dependencies>",
      "issue": "<specific description of the gap — quote the problematic phrase or missing element>",
      "fix": "<concrete, actionable fix — tell the team exactly what to add, change, or define>"
    }
  ],
  "ambiguities": [
    {
      "phrase": "<exact phrase from the story that is ambiguous>",
      "question": "<the specific question the team must answer before this story is sprint-ready>"
    }
  ],
  "dependencies": [
    {
      "type": "<exactly one of: team | api | story | system | design | compliance>",
      "description": "<what the dependency is and why it could block progress>",
      "confidence": "<exactly one of: high | medium | low>",
      "status": "<exactly one of: acknowledged | implied | unresolved>"
    }
  ],
  "improved_story": "<Full rewrite of the story in correct format. Preserve the original intent. Fix vague language with measurable alternatives. Do not add acceptance criteria here — those go in suggested_acs.>",
  "suggested_acs": [
    "<AC in Given [context] / When [action] / Then [measurable outcome] format>",
    "<include at least 3, up to 7 ACs covering happy path, error states, and edge cases>"
  ],
  "split_suggestions": [
    "<If the story is too large (size score ≤ 10), suggest 2–4 smaller stories the epic could be split into. Each suggestion should be a one-sentence story title. Leave this array empty [] if the story is appropriately sized.>"
  ]
}

---

## READINESS LEVEL THRESHOLDS

Map total score to readiness_level as follows:
- 0–39:   "Not Ready"
- 40–59:  "Needs Work"
- 60–79:  "Almost Ready"
- 80–100: "Sprint Ready"

---

## SEVERITY DEFINITIONS

Use these definitions consistently across all gap entries:

- critical: A blocker. The story cannot be built or tested without resolving this. Examples: missing ACs, undefined token expiry, no error states, epic-sized scope.
- warning:  Likely to cause a mid-sprint clarification request or rework. Examples: vague language with no metric, missing edge case, dependency named but not de-risked.
- info:     Nice to have. Will improve quality but won't block delivery. Examples: out-of-scope not stated, story point estimate missing, minor inconsistency in terminology.

---

## TEAM CONTEXT (if provided)

If the user provides team context (Definition of Ready, parent epic, story point scale, team conventions), incorporate it into your scoring:
- Score the story against the team's stated DoR, not a generic one
- Flag any DoR criteria the story fails to meet as critical gaps
- Reference the parent epic when assessing dependency risk and scope

---

## BEHAVIOUR RULES

1. Never invent information. If something is not in the story, flag it as missing — do not assume it exists.
2. Be specific. Quote exact phrases when flagging ambiguities or gaps. "User can log in quickly" is a quote; "vague language present" is not useful.
3. Be proportionate. A story with 1 minor vague word should not score the same as a story with no ACs at all.
4. Gaps array: include all issues found, ordered by severity (critical first, then warning, then info). There is no maximum — include every genuine issue found.
5. Ambiguities array: only include phrases that are genuinely ambiguous. Do not manufacture ambiguity in an otherwise clear story.
6. Dependencies array: include both explicit (named in the story) and strongly implied dependencies. Set confidence to "low" for implied ones.
7. Improved story: rewrite the story narrative only. Do not insert ACs into the improved story field — they belong in suggested_acs.
8. Split suggestions: only populate if size score is ≤ 10. Otherwise return an empty array.
9. Total score must arithmetically equal the sum of the five dimension scores.
10. Return valid JSON only. Any deviation breaks the application."""


# ── Status Deck prompt ──────────────────────────────────────────────────────
STATUS_DECK_SYSTEM_PROMPT = """You are an executive project status generator. Given Jira sprint issue data, synthesize a concise, stakeholder-ready weekly status deck.

Output rules:

1. executiveSummary: 2-3 sentences. State overall project health, the primary focus/win this week, and any critical concern or blocker.

2. accomplishments: Extract from Done issues. For each item the "sentence" field MUST be a past-tense sentence YOU COMPOSE (8-15 words) starting with a strong action verb — do NOT copy the Jira summary verbatim. Use the "Description:" text to build a specific, meaningful sentence about what was actually delivered. If no description, craft the best sentence from the summary.
   Strong verbs: Implemented, Deployed, Resolved, Delivered, Launched, Fixed, Migrated, Integrated, Optimized, Shipped, Automated, Configured, Built, Enabled, Completed, Released, Refactored.
   Examples: summary 'User auth API' + desc 'JWT auth with refresh tokens' -> sentence 'Implemented JWT-based user authentication with automatic token refresh'. Summary 'Fix login bug' + desc 'Fixed redirect loop' -> sentence 'Resolved OAuth callback redirect loop causing login failures'. 4-8 items max.

3. nextSteps: Extract from In Progress + high-priority To Do. Action-oriented, starting with a verb. Include owner and due date where derivable. 4-8 items max.

4. blockers: Items labeled blocked or risk, unassigned critical items, issues with no progress. 5 max.

5. healthStatus: On Track if >60% complete and no critical blockers. At Risk if 40-60% or has blockers. Off Track if <40% or multiple critical blockers.

6. milestones: Extract sprint goals or milestone-labeled issues. Return [] if none.

Return ONLY valid JSON, no markdown:
{
  "projectName": "string",
  "sprintName": "string",
  "weekOf": "string",
  "healthStatus": "On Track | At Risk | Off Track",
  "executiveSummary": "string",
  "accomplishments": [{"sentence": "string (past-tense, AI-composed, NOT the raw Jira summary)"}],
  "nextSteps": [{"action": "string", "owner": "string", "dueDate": "string", "priority": "high|medium|low"}],
  "blockers": [{"title": "string", "impact": "High|Medium|Low", "type": "blocker|risk|dependency", "mitigation": "string", "owner": "string"}],
  "milestones": [{"name": "string", "status": "complete|in_progress|upcoming", "date": "string"}]
}"""


def load_env():
    path = os.path.join(ROOT, ".env")
    cfg = {}
    if not os.path.exists(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


_ENV = load_env()
GROQ_API_KEY   = _ENV.get("GROQ_API_KEY")
JIRA_BASE_URL  = _ENV.get("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL     = _ENV.get("JIRA_EMAIL", "")
JIRA_API_TOKEN = _ENV.get("JIRA_API_TOKEN", "")

import base64 as _b64
JIRA_AUTH = _b64.b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode() if JIRA_EMAIL and JIRA_API_TOKEN else None


def jira_get(path):
    """Make an authenticated GET to the Jira REST API and return parsed JSON."""
    url = f"{JIRA_BASE_URL}/rest/api/3/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Basic {JIRA_AUTH}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def jira_issue_to_text(issue):
    """Convert a Jira issue to plain text suitable for the readiness checker."""
    fields = issue.get("fields", {})
    key    = issue.get("key", "")
    summary = fields.get("summary", "")
    desc_raw = fields.get("description") or {}

    def extract_text(node):
        if not node:
            return ""
        if isinstance(node, str):
            return node
        t = node.get("type", "")
        text = node.get("text", "")
        children = node.get("content", [])
        parts = [extract_text(c) for c in children]
        joined = "".join(parts) if parts else text
        if t in ("paragraph", "heading"):
            return joined + "\n"
        if t == "listItem":
            return "- " + joined
        if t in ("bulletList", "orderedList"):
            return joined + "\n"
        return joined

    description = extract_text(desc_raw).strip() if isinstance(desc_raw, dict) else str(desc_raw or "").strip()
    issue_type = (fields.get("issuetype") or {}).get("name", "")
    status     = (fields.get("status") or {}).get("name", "")
    priority   = (fields.get("priority") or {}).get("name", "")
    assignee   = ((fields.get("assignee") or {}).get("displayName") or "Unassigned")
    story_points = fields.get("story_points") or fields.get("customfield_10016") or ""

    lines = [f"[{key}] {summary}"]
    if issue_type: lines.append(f"Type: {issue_type}")
    if status:     lines.append(f"Status: {status}")
    if priority:   lines.append(f"Priority: {priority}")
    if assignee:   lines.append(f"Assignee: {assignee}")
    if story_points: lines.append(f"Story Points: {story_points}")
    if description:
        lines.append("")
        lines.append(description)
    return "\n".join(lines)


def jira_agile_get(path):
    """Authenticated GET to the Jira Agile (Software) REST API."""
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Basic {JIRA_AUTH}",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_issue_comments(key, max_comments=3, max_chars=300):
    """Fetch the most recent comments for a Jira issue. Returns a list of plain-text strings."""
    try:
        data = jira_get(f"issue/{key}/comment?maxResults=50&orderBy=-created")
        comments = data.get("comments", [])
        recent = comments[-max_comments:] if len(comments) > max_comments else comments

        def _txt(node):
            if not node: return ""
            if isinstance(node, str): return node
            parts = [_txt(c) for c in node.get("content", [])]
            return (" ".join(p for p in parts if p.strip()) or node.get("text", ""))

        result = []
        for c in recent:
            author = (c.get("author") or {}).get("displayName", "Unknown")
            body_raw = c.get("body") or {}
            text = (_txt(body_raw) if isinstance(body_raw, dict) else str(body_raw)).strip()
            if text:
                result.append(f"{author}: {text[:max_chars]}")
        return result
    except Exception:
        return []


def fetch_board_detail(board):
    """Fetch sprints + backlog count for one board. Called in parallel."""
    bid = board["id"]
    project_key = (board.get("location") or {}).get("projectKey", "")
    project_name = (board.get("location") or {}).get("projectName", board.get("name", ""))
    avatar_url = (board.get("location") or {}).get("avatarURI", "")

    # Sprints (active + future)
    try:
        sd = jira_agile_get(f"board/{bid}/sprint?state=active,future&maxResults=10")
        raw_sprints = sd.get("values", [])
    except Exception:
        raw_sprints = []

    # Backlog count
    try:
        bl = jira_agile_get(f"board/{bid}/backlog?maxResults=1&fields=summary")
        backlog_count = bl.get("total", 0)
    except Exception:
        backlog_count = 0

    # Per-sprint issue counts — run in parallel
    def sprint_count(sprint_id):
        try:
            d = jira_agile_get(f"sprint/{sprint_id}/issue?maxResults=1&fields=summary")
            return d.get("total", 0)
        except Exception:
            return 0

    sprints = []
    if raw_sprints:
        with ThreadPoolExecutor(max_workers=min(len(raw_sprints), 5)) as ex:
            futures = {ex.submit(sprint_count, s["id"]): s for s in raw_sprints}
            for fut in as_completed(futures):
                s = futures[fut]
                sprints.append({
                    "id": s["id"],
                    "name": s["name"],
                    "state": s.get("state", "future"),
                    "startDate": s.get("startDate", ""),
                    "endDate": s.get("endDate", ""),
                    "goal": s.get("goal", ""),
                    "count": fut.result(),
                })
        sprints.sort(key=lambda x: (0 if x["state"] == "active" else 1, x["name"]))

    board_url = f"{JIRA_BASE_URL}/jira/software/projects/{project_key}/boards/{bid}"
    return {
        "id": bid,
        "name": board.get("name", ""),
        "type": board.get("type", "scrum"),
        "projectKey": project_key,
        "projectName": project_name,
        "avatarUrl": avatar_url,
        "sprints": sprints,
        "backlogCount": backlog_count,
        "url": board_url,
    }


def build_pptx(deck):
    """Build a Blend-branded PPTX from a status deck dict. Returns bytes."""
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
    import io

    def rgb(hex6):
        h = hex6.lstrip("#")
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    C = dict(
        wb="#053057", nt="#00EDED", lt="#A2F3F3",
        cg="#314550", gr="#1A1A1A", dg="#0B0D0E",
        wh="#FFFFFF", ow="#F4F3F0",
    )

    W, H = Inches(13.33), Inches(7.5)   # 16:9 widescreen

    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    blank_layout = prs.slide_layouts[6]  # completely blank

    def add_rect(slide, x, y, w, h, fill_hex, alpha=None):
        shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill_hex)
        shape.line.fill.background()
        return shape

    def add_text(slide, text, x, y, w, h, size=14, color="#FFFFFF", bold=False,
                 align="left", valign="top", wrap=True):
        txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        txb.word_wrap = wrap
        tf = txb.text_frame
        tf.word_wrap = wrap
        p = tf.paragraphs[0]
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                       "right": PP_ALIGN.RIGHT}.get(align, PP_ALIGN.LEFT)
        run = p.add_run()
        run.text = str(text)
        run.font.size = Pt(size)
        run.font.color.rgb = rgb(color)
        run.font.bold = bold
        return txb

    def bottom_bar(slide):
        add_rect(slide, 0, 7.25, 13.33, 0.25, C["nt"])

    def side_strip(slide, color=None):
        add_rect(slide, 0, 0, 0.22, 7.5, color or C["cg"])

    def section_label(slide, text, y=0.3):
        add_text(slide, text, 0.5, y, 12.5, 0.3, size=8, color=C["nt"], bold=True)
        add_rect(slide, 0.5, y + 0.33, 12.5, 0.02, C["nt"])

    m  = deck.get("metrics") or {}
    pn = deck.get("projectName") or "Project"
    sn = deck.get("sprintName") or ""
    wo = deck.get("weekOf") or ""
    hs = deck.get("healthStatus") or "At Risk"

    # ── Slide 1: Cover ──────────────────────────────────────────────────────
    sl = prs.slides.add_slide(blank_layout)
    add_rect(sl, 0, 0, 13.33, 7.5, C["wb"])           # Washed Blue background (dark mode)
    add_rect(sl, 0, 0, 0.06, 7.5, C["nt"])            # Thin neon-turquoise left accent (spark, not flood)
    bottom_bar(sl)
    add_text(sl, "WEEKLY STATUS UPDATE", 0.5, 0.6, 12.5, 0.35, size=9, color=C["nt"], bold=True)
    add_text(sl, pn, 0.5, 1.1, 12.5, 2.0, size=52, color=C["wh"], bold=True)
    add_text(sl, sn, 0.5, 3.2, 12.5, 0.55, size=24, color=C["lt"])
    add_text(sl, "Week of " + wo, 0.5, 3.9, 9, 0.45, size=16, color=C["lt"])
    h_clr = {"On Track": "#065F46", "At Risk": "#B45309", "Off Track": "#7F1D1D"}.get(hs, "#B45309")
    add_rect(sl, 0.5, 4.55, 2.4, 0.5, h_clr)
    add_text(sl, hs, 0.5, 4.55, 2.4, 0.5, size=14, color=C["wh"], bold=True, align="center")

    # ── Slide 2: Combined — Exec Summary + Metrics + Accomplishments + Next Steps + Blockers ──
    sl = prs.slides.add_slide(blank_layout)
    add_rect(sl, 0, 0, 13.33, 7.5, C["dg"])          # Dark Gray background (dark mode)
    add_rect(sl, 0, 0, 0.06, 7.5, C["nt"])            # Thin neon-turquoise accent strip (spark, not flood)

    blockers = deck.get("blockers") or []
    has_blockers = len(blockers) > 0

    blocker_strip_h = 1.05 if has_blockers else 0.0
    col_bottom      = 7.5 - 0.25 - blocker_strip_h

    # Executive Summary — label in turquoise (small accent), body in Off White
    add_text(sl, "EXECUTIVE SUMMARY", 0.4, 0.15, 12.5, 0.25, size=7, color=C["nt"], bold=True)
    add_rect(sl, 0.4, 0.4, 12.5, 0.018, C["nt"])
    add_text(sl, deck.get("executiveSummary") or "", 0.4, 0.45, 12.5, 0.72, size=12, color=C["ow"])

    # Metrics row — Cool Gray tiles, Neon Turquoise values, Light Turquoise labels
    stats = [
        ("Total",       str(m.get("totalIssues",      0))),
        ("Completed",   str(m.get("doneIssues",       0))),
        ("In Progress", str(m.get("inProgressIssues", 0))),
        ("Completion",  str(m.get("completionPct",    0)) + "%"),
    ]
    tw, sx, sy_m = 2.95, 0.4, 1.22
    for i, (lbl, val) in enumerate(stats):
        x = sx + i * (tw + 0.18)
        add_rect(sl, x, sy_m, tw, 0.72, C["cg"])                                         # Cool Gray surface
        add_text(sl, val, x, sy_m + 0.04, tw, 0.42, size=22, color=C["nt"], bold=True, align="center")  # Neon Turquoise value
        add_text(sl, lbl, x, sy_m + 0.50, tw, 0.20, size=8,  color=C["lt"], align="center")             # Light Turquoise label

    # Two-column section
    col_top = 2.05
    col_h   = col_bottom - col_top
    col_w   = 6.1
    left_x  = 0.4
    right_x = 6.83

    add_text(sl, "KEY ACCOMPLISHMENTS", left_x,  col_top, col_w, 0.22, size=7, color=C["nt"], bold=True)
    add_rect(sl, left_x,  col_top + 0.22, col_w, 0.015, C["nt"])
    add_text(sl, "NEXT STEPS",          right_x, col_top, col_w, 0.22, size=7, color=C["nt"], bold=True)
    add_rect(sl, right_x, col_top + 0.22, col_w, 0.015, C["nt"])

    row_h    = 0.44
    max_rows = max(1, int((col_h - 0.3) / row_h))

    accomplishments = (deck.get("accomplishments") or [])[:max_rows]
    steps           = (deck.get("nextSteps")       or [])[:max_rows]

    for i, item in enumerate(accomplishments):
        ry = col_top + 0.28 + i * row_h
        add_rect(sl, left_x, ry + 0.06, 0.2, 0.2, C["nt"])                               # Turquoise dot accent
        add_text(sl, "✓", left_x, ry + 0.05, 0.22, 0.22, size=8, color=C["wb"], bold=True, align="center")  # Washed Blue on turquoise
        sentence = item.get("sentence") or item.get("title") or ""
        add_text(sl, sentence, left_x + 0.28, ry, col_w - 0.3, 0.4, size=11, color=C["wh"])

    for i, s in enumerate(steps):
        ry = col_top + 0.28 + i * row_h
        if i % 2 == 0:
            add_rect(sl, right_x, ry, col_w, row_h, C["cg"])                             # Cool Gray alternating row (on-palette)
        add_text(sl, "→", right_x + 0.05, ry + 0.06, 0.24, 0.28, size=10, color=C["nt"], bold=True)  # Neon Turquoise arrow
        add_text(sl, s.get("action") or "", right_x + 0.3, ry + 0.04, col_w - 0.35, 0.38, size=11, color=C["wh"])

    # Bottom bar — Neon Turquoise accent strip
    add_rect(sl, 0, 7.25, 13.33, 0.25, C["nt"])

    # Blockers strip — Gray surface, Cool Gray cards, semantic impact colors
    if has_blockers:
        bsy = 7.25 - blocker_strip_h
        add_rect(sl, 0, bsy, 13.33, blocker_strip_h, C["gr"])                            # Gray (#1A1A1A) surface
        add_rect(sl, 0.06, bsy, 0.06, blocker_strip_h, C["cg"])                          # Cool Gray divider
        add_text(sl, "ISSUES / BLOCKERS", 0.2, bsy + 0.08, 2.3, 0.22, size=7, color=C["lt"], bold=True)  # Light Turquoise label
        i_clr = {"High": "#F87171", "Medium": "#FCD34D", "Low": "#6EE7B7"}
        bw = 10.6 / max(len(blockers[:4]), 1)
        for i, b in enumerate(blockers[:4]):
            bx = 2.6 + i * (bw + 0.1)
            ic = i_clr.get(b.get("impact") or "", "#FCD34D")
            add_rect(sl, bx, bsy + 0.08, bw, 0.85, C["cg"])                             # Cool Gray card
            add_rect(sl, bx, bsy + 0.08, 0.05, 0.85, ic)                                # Semantic impact colour bar
            add_text(sl, (b.get("impact") or "").upper(), bx + 0.10, bsy + 0.10, 0.8, 0.2, size=6, color=ic, bold=True)
            add_text(sl, b.get("title") or "", bx + 0.10, bsy + 0.30, bw - 0.15, 0.54, size=9, color=C["ow"])

    # ── Milestones (conditional) ──────────────────────────────────────────────
    milestones = deck.get("milestones") or []
    if milestones:
        sl = prs.slides.add_slide(blank_layout)
        add_rect(sl, 0, 0, 13.33, 7.5, C["dg"])
        add_rect(sl, 0, 0, 0.06, 7.5, C["nt"])                                          # Thin turquoise strip
        section_label(sl, "MILESTONES")
        s_icon = {"complete": "✓", "in_progress": "◎", "upcoming": "○"}
        s_clr  = {"complete": C["nt"], "in_progress": "#FCD34D", "upcoming": C["ow"]}
        for i, ms in enumerate(milestones):
            y = 1.0 + i * 0.95
            st = ms.get("status") or "upcoming"
            add_rect(sl, 0.4, y + 0.05, 12.5, 0.72, C["cg"])                           # Cool Gray row surface
            add_text(sl, s_icon.get(st, "○"), 0.55, y + 0.08, 0.5, 0.55, size=20, bold=True, color=s_clr.get(st, C["ow"]))
            add_text(sl, ms.get("name") or "", 1.15, y + 0.18, 9.5, 0.4, size=15, color=C["wh"])
            if ms.get("date"):
                add_text(sl, ms["date"], 11.3, y + 0.18, 1.5, 0.4, size=13, color=C["lt"], align="right")
        bottom_bar(sl)

    # ── Closing slide ─────────────────────────────────────────────────────────
    sl = prs.slides.add_slide(blank_layout)
    add_rect(sl, 0, 0, 13.33, 7.5, C["wb"])                                             # Washed Blue background
    add_rect(sl, 0, 0, 0.06, 7.5, C["nt"])                                              # Thin turquoise strip
    bottom_bar(sl)
    add_text(sl, "Thank you", 0.5, 1.8, 12.33, 1.5, size=66, bold=True, color=C["wh"], align="center")
    add_text(sl, pn, 0.5, 3.5, 12.33, 0.65, size=26, color=C["lt"], align="center")
    add_text(sl, f"Generated by Blend PM Tools  ·  {wo}",
             0.5, 4.3, 12.33, 0.45, size=13, color=C["lt"], align="center")

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("  " + (fmt % args))

    def _send(self, code, body, content_type="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # ── Jira endpoints ────────────────────────────────────────────────────
        if self.path == "/board.html":
            try:
                with open(os.path.join(ROOT, "board.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "board.html not found", "text/plain")
            return

        if self.path == "/api/jira/board-overview":
            if not JIRA_AUTH:
                self._send(503, json.dumps({"error": "Jira not configured"}))
                return
            try:
                raw_boards = []
                start = 0
                while True:
                    page = jira_agile_get(f"board?maxResults=50&startAt={start}")
                    values = page.get("values", [])
                    raw_boards.extend(values)
                    if page.get("isLast", True) or len(values) < 50:
                        break
                    start += 50
                results = []
                with ThreadPoolExecutor(max_workers=min(len(raw_boards), 8)) as ex:
                    futures = {ex.submit(fetch_board_detail, b): b for b in raw_boards}
                    for fut in as_completed(futures):
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            print(f"  Board detail error: {e}")
                results.sort(key=lambda x: x["projectName"].lower())
                self._send(200, json.dumps(results))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return

        if self.path == "/api/jira/status":
            connected = bool(JIRA_AUTH and JIRA_BASE_URL)
            self._send(200, json.dumps({
                "connected": connected,
                "base_url": JIRA_BASE_URL,
            }))
            return

        if self.path.startswith("/api/jira/issues"):
            if not JIRA_AUTH:
                self._send(503, json.dumps({"error": "Jira not configured"}))
                return
            from urllib.parse import urlparse, parse_qs, urlencode
            qs = parse_qs(urlparse(self.path).query)
            search = qs.get("q", [""])[0].strip()
            max_results = int(qs.get("max", ["25"])[0])
            if search:
                jql = f'(summary ~ "{search}" OR text ~ "{search}") ORDER BY updated DESC'
            else:
                jql = "ORDER BY updated DESC"
            try:
                params = urlencode({"jql": jql, "maxResults": max_results,
                                    "fields": "summary,issuetype,status,priority,assignee,customfield_10016,description"})
                data = jira_get(f"search?{params}")
                issues = []
                for iss in data.get("issues", []):
                    f = iss.get("fields", {})
                    issues.append({
                        "key":   iss["key"],
                        "summary": f.get("summary", ""),
                        "type":  (f.get("issuetype") or {}).get("name", ""),
                        "status": (f.get("status") or {}).get("name", ""),
                        "priority": (f.get("priority") or {}).get("name", ""),
                        "points": f.get("customfield_10016"),
                        "assignee": ((f.get("assignee") or {}).get("displayName") or ""),
                    })
                self._send(200, json.dumps({"issues": issues, "total": data.get("total", 0)}))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return

        if self.path.startswith("/api/jira/issue/"):
            if not JIRA_AUTH:
                self._send(503, json.dumps({"error": "Jira not configured"}))
                return
            key = self.path.split("/api/jira/issue/")[-1].split("?")[0].strip()
            try:
                issue = jira_get(f"issue/{key}?fields=summary,issuetype,status,priority,assignee,customfield_10016,description")
                self._send(200, json.dumps({
                    "key": issue["key"],
                    "text": jira_issue_to_text(issue),
                    "type": (issue["fields"].get("issuetype") or {}).get("name", ""),
                }))
            except urllib.error.HTTPError as e:
                self._send(e.code, json.dumps({"error": f"Jira {e.code}: {e.read().decode('utf-8','replace')}"}))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return
        if self.path.startswith("/api/jira/board-issues"):
            if not JIRA_AUTH:
                self._send(503, json.dumps({"error": "Jira not configured"}))
                return
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            board_id  = qs.get("boardId",  [""])[0]
            sprint_id = qs.get("sprintId", [""])[0]
            is_backlog = qs.get("backlog", [""])[0] == "true"
            try:
                fields = "summary,issuetype,status,priority,assignee,customfield_10016,description"
                if sprint_id:
                    data = jira_agile_get(f"sprint/{sprint_id}/issue?maxResults=50&fields={fields}")
                elif is_backlog and board_id:
                    data = jira_agile_get(f"board/{board_id}/backlog?maxResults=50&fields={fields}")
                else:
                    self._send(400, json.dumps({"error": "Need boardId+backlog=true or sprintId"}))
                    return
                issues = []
                for iss in data.get("issues", []):
                    f = iss.get("fields", {})
                    issues.append({
                        "key":      iss["key"],
                        "summary":  f.get("summary", ""),
                        "type":     (f.get("issuetype") or {}).get("name", ""),
                        "status":   (f.get("status")    or {}).get("name", ""),
                        "priority": (f.get("priority")  or {}).get("name", ""),
                        "points":   f.get("customfield_10016"),
                        "assignee": ((f.get("assignee") or {}).get("displayName") or ""),
                        "text":     jira_issue_to_text(iss),
                    })
                self._send(200, json.dumps({"issues": issues, "total": data.get("total", 0)}))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return

        if self.path in ("/status", "/status.html"):
            try:
                with open(os.path.join(ROOT, "status.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "status.html not found", "text/plain")
            return

        # ── History endpoints ─────────────────────────────────────────────────
        if self.path == "/api/history":
            self._send(200, json.dumps(load_history()))
            return
        if self.path == "/api/history/clear":
            save_history([])
            self._send(200, json.dumps({"ok": True}))
            return
        if self.path.startswith("/api/history/delete/"):
            entry_id = self.path.split("/")[-1]
            entries = [e for e in load_history() if str(e.get("id")) != entry_id]
            save_history(entries)
            self._send(200, json.dumps({"ok": True}))
            return
        if self.path in ("/", "/landing.html"):
            try:
                with open(os.path.join(ROOT, "landing.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "landing.html not found", "text/plain")
            return
        if self.path in ("/checker", "/index.html"):
            try:
                with open(os.path.join(ROOT, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "index.html not found", "text/plain")
        elif self.path.startswith("/vendor/"):
            # Serve locally vendored JS libs (no CDN dependency in the browser)
            rel = self.path.lstrip("/").split("?", 1)[0]
            safe = os.path.normpath(rel).replace("\\", "/")
            if not safe.startswith("vendor/"):
                self._send(403, "Forbidden", "text/plain")
                return
            path = os.path.join(ROOT, safe)
            try:
                with open(path, "rb") as f:
                    self._send(200, f.read(), "application/javascript; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "Not found", "text/plain")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/status-deck":
            if not GROQ_API_KEY:
                self._send(500, json.dumps({"error": "GROQ_API_KEY not found in .env"})); return
            if not JIRA_AUTH:
                self._send(503, json.dumps({"error": "Jira not configured"})); return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or "{}")
            except Exception as e:
                self._send(400, json.dumps({"error": f"Bad request: {e}"})); return

            board_id    = payload.get("boardId")
            sprint_id   = payload.get("sprintId")
            is_backlog  = payload.get("isBacklog", False)
            project_name = payload.get("projectName", "Project")
            sprint_name  = payload.get("sprintName", "Sprint")

            try:
                fields = "summary,issuetype,status,priority,assignee,customfield_10016,labels"
                if sprint_id:
                    data = jira_agile_get(f"sprint/{sprint_id}/issue?maxResults=100&fields={fields}")
                elif is_backlog and board_id:
                    data = jira_agile_get(f"board/{board_id}/backlog?maxResults=50&fields={fields}")
                else:
                    self._send(400, json.dumps({"error": "Need sprintId or boardId+isBacklog=true"})); return

                issues = data.get("issues", [])
                done_issues, in_progress_issues, todo_issues = [], [], []
                sp_planned, sp_done = 0, 0

                for iss in issues:
                    f = iss.get("fields", {})
                    status = f.get("status") or {}
                    cat = (status.get("statusCategory") or {}).get("key", "new")
                    points = f.get("customfield_10016") or 0
                    sp_planned += points or 0
                    desc_raw = f.get("description") or {}
                    def _txt(n):
                        if not n: return ""
                        if isinstance(n, str): return n
                        parts = [_txt(c) for c in n.get("content", [])]
                        return ("; ".join(p for p in parts if p.strip()) if parts else n.get("text", ""))
                    desc_text = _txt(desc_raw).strip()[:300] if isinstance(desc_raw, dict) else str(desc_raw or "").strip()[:300]
                    entry = {
                        "key": iss["key"],
                        "summary": f.get("summary", ""),
                        "type": (f.get("issuetype") or {}).get("name", ""),
                        "status": status.get("name", ""),
                        "priority": (f.get("priority") or {}).get("name", ""),
                        "assignee": ((f.get("assignee") or {}).get("displayName") or "Unassigned"),
                        "points": points,
                        "labels": f.get("labels") or [],
                        "description": desc_text,
                    }
                    if cat == "done":
                        sp_done += points or 0
                        done_issues.append(entry)
                    elif cat == "indeterminate":
                        in_progress_issues.append(entry)
                    else:
                        todo_issues.append(entry)

                total = len(issues)
                done_count = len(done_issues)
                completion_pct = round((done_count / total * 100) if total else 0)

                today = datetime.date.today()
                week_start = today - datetime.timedelta(days=today.weekday())
                week_end   = week_start + datetime.timedelta(days=6)
                week_of    = f"{week_start.strftime('%B %d')}–{week_end.strftime('%d, %Y')}"

                # Fetch comments in parallel for done + in-progress issues
                issues_needing_comments = done_issues + in_progress_issues
                comments_map = {}
                if issues_needing_comments:
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        futures = {pool.submit(fetch_issue_comments, iss["key"]): iss["key"]
                                   for iss in issues_needing_comments}
                        for fut in as_completed(futures):
                            key = futures[fut]
                            try:
                                comments_map[key] = fut.result()
                            except Exception:
                                comments_map[key] = []

                def fmt(iss, include_desc=False):
                    pts = f"| Points: {iss['points']}" if iss['points'] else ""
                    lbl = f"| Labels: {', '.join(iss['labels'])}" if iss['labels'] else ""
                    desc = ("\n  Description: " + iss['description'] if include_desc and iss.get('description') else "")
                    line = f"- [{iss['key']}] {iss['summary']} | Type: {iss['type']} | Assignee: {iss['assignee']} | Priority: {iss['priority']} {pts} {lbl}{desc}"
                    coms = comments_map.get(iss["key"], [])
                    if coms:
                        line += "\n  Comments:\n" + "\n".join(f"    > {c}" for c in coms)
                    return line

                lines = [
                    f"PROJECT: {project_name}", f"SPRINT: {sprint_name}", f"DATE: {week_of}",
                    f"TOTAL: {total} | DONE: {done_count} | IN PROGRESS: {len(in_progress_issues)} | TO DO: {len(todo_issues)}",
                    f"STORY POINTS: Planned={int(sp_planned)} Done={int(sp_done)}",
                    "", "COMPLETED (Done):",
                ] + ([fmt(i, include_desc=True) for i in done_issues] if done_issues else ["(none)"]) + [
                    "", "IN PROGRESS:",
                ] + ([fmt(i) for i in in_progress_issues] if in_progress_issues else ["(none)"]) + [
                    "", "TO DO / NOT STARTED:",
                ] + ([fmt(i) for i in todo_issues] if todo_issues else ["(none)"])

                groq_body = json.dumps({
                    "model": MODEL, "max_tokens": 2000, "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": STATUS_DECK_SYSTEM_PROMPT},
                        {"role": "user",   "content": "\n".join(lines)},
                    ],
                }).encode("utf-8")

                req = urllib.request.Request(GROQ_URL, data=groq_body, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) story-readiness-checker/1.0",
                    "Accept": "application/json",
                }, method="POST")

                with urllib.request.urlopen(req, timeout=90) as resp:
                    groq_data = json.loads(resp.read().decode("utf-8"))

                text = groq_data["choices"][0]["message"]["content"]
                text = text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                parsed["metrics"] = {
                    "totalIssues": total, "doneIssues": done_count,
                    "inProgressIssues": len(in_progress_issues), "toDoIssues": len(todo_issues),
                    "completionPct": completion_pct,
                    "storyPointsPlanned": int(sp_planned), "storyPointsDone": int(sp_done),
                }
                self._send(200, json.dumps(parsed))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return

        if self.path == "/api/export-pptx":
            try:
                length = int(self.headers.get("Content-Length", 0))
                deck = json.loads(self.rfile.read(length) or "{}")
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)})); return
            try:
                pptx_bytes = build_pptx(deck)
                safe_name = (deck.get("projectName") or "status").replace(" ", "-")
                fn = safe_name + "-status-deck.pptx"
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.presentationml.presentation")
                self.send_header("Content-Disposition", f'attachment; filename="{fn}"')
                self.send_header("Content-Length", str(len(pptx_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(pptx_bytes)
            except Exception as e:
                import traceback
                self._send(500, json.dumps({"error": str(e), "trace": traceback.format_exc()}))
            return

        if self.path == "/api/jira/update-description":
            if not JIRA_AUTH:
                self._send(503, json.dumps({"error": "Jira not configured"})); return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or "{}")
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)})); return

            key  = (payload.get("key") or "").strip().upper()
            text = (payload.get("description") or "").strip()
            if not key or not text:
                self._send(400, json.dumps({"error": "key and description are required"})); return

            # Convert plain text to Atlassian Document Format (ADF)
            paragraphs = []
            for line in text.split("\n"):
                if line.strip():
                    paragraphs.append({
                        "type": "paragraph",
                        "content": [{"type": "text", "text": line}]
                    })
                else:
                    paragraphs.append({"type": "paragraph", "content": []})
            adf = {"version": 1, "type": "doc", "content": paragraphs or [{"type": "paragraph", "content": []}]}

            body = json.dumps({"fields": {"description": adf}}).encode("utf-8")
            url  = f"{JIRA_BASE_URL}/rest/api/3/issue/{key}"
            req  = urllib.request.Request(url, data=body, method="PUT",
                       headers={"Authorization": f"Basic {JIRA_AUTH}",
                                "Content-Type": "application/json",
                                "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    _ = resp.read()
                self._send(200, json.dumps({"ok": True, "key": key}))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                self._send(e.code, json.dumps({"error": f"Jira {e.code}", "detail": detail}))
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return

        if self.path != "/api/analyze":
            self._send(404, json.dumps({"error": "Not found"}))
            return

        if not GROQ_API_KEY:
            self._send(500, json.dumps({"error": "GROQ_API_KEY not found in .env"}))
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or "{}")
        except Exception as e:
            self._send(400, json.dumps({"error": f"Bad request: {e}"}))
            return

        story = (payload.get("story") or "").strip()
        epic = (payload.get("epic") or "").strip()
        dor = (payload.get("dor") or "").strip()
        mode = (payload.get("mode") or "story").strip()  # "story" | "epic"

        active_prompt = EPIC_SYSTEM_PROMPT if mode == "epic" else SYSTEM_PROMPT

        context_parts = []
        if epic:
            context_parts.append(f"Parent epic: {epic}")
        if dor:
            context_parts.append(f"Team Definition of Ready: {dor}")
        context = "\n".join(context_parts)
        user_content = f"User story:\n{story}" + (f"\n\n{context}" if context else "")

        groq_body = json.dumps({
            "model": MODEL,
            "max_tokens": 4096,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": active_prompt},
                {"role": "user", "content": user_content},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            GROQ_URL,
            data=groq_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) story-readiness-checker/1.0",
                "Accept": "application/json",
            },
            method="POST",
        )

        # Retry up to 3 times on 429 rate-limit with backoff
        groq_data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    groq_data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                print(f"  Groq HTTP {e.code} (attempt {attempt+1}): {body}")
                if e.code == 429 and attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s before retry")
                    time.sleep(wait)
                    continue
                self._send(502, json.dumps({"error": f"Groq API {e.code}", "detail": body}))
                return
            except Exception as e:
                self._send(502, json.dumps({"error": f"Upstream error: {e}"}))
                return

        try:
            text = groq_data["choices"][0]["message"]["content"]
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
        except Exception as e:
            self._send(502, json.dumps({"error": f"Could not parse model output: {e}",
                                        "raw": groq_data}))
            return

        # Enforce consistent readiness_level based on total score
        total = parsed.get("total", 0)
        if total >= 80:
            parsed["readiness_level"] = "Sprint Ready"
        elif total >= 60:
            parsed["readiness_level"] = "Almost Ready"
        elif total >= 40:
            parsed["readiness_level"] = "Needs Work"
        else:
            parsed["readiness_level"] = "Not Ready"

        # Save to history (full result stored for panel view)
        entry = {
            "id": int(time.time() * 1000),
            "title": story.split("\n")[0][:80],
            "story_text": story,
            "mode": mode,
            "epic": epic,
            "score": parsed.get("total", 0),
            "readiness_level": parsed.get("readiness_level", ""),
            "checked_at": time.strftime("%Y-%m-%d %H:%M"),
            "result": parsed,
        }
        history = load_history()
        history.insert(0, entry)
        save_history(history)

        self._send(200, json.dumps(parsed))


if __name__ == "__main__":
    port = 8000
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not found in .env next to this script.")
    print(f"Model: {MODEL}")
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
