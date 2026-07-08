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


def load_env_key():
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "GROQ_API_KEY":
                return v.strip().strip('"').strip("'")
    return None


GROQ_API_KEY = load_env_key()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("  " + (fmt % args))

    def _send(self, code, body, content_type="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
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
            "max_tokens": 1500,
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
                # Cloudflare in front of Groq blocks the default Python-urllib UA (err 1010)
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) story-readiness-checker/1.0",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                groq_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            print(f"  Groq HTTP {e.code}: {detail}")
            self._send(502, json.dumps({"error": f"Groq API {e.code}", "detail": detail}))
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
