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
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert agile coach and product manager specializing in user story quality assessment. Your job is to review user stories or epics before sprint planning and return a structured readiness report.

Evaluate the story across these 5 dimensions, each scored 0-20 (total: 0-100):

1. COMPLETENESS (0-50): Is the "As a [persona]... I want... so that..." format present? Are acceptance criteria defined? Are edge cases covered?
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
        if self.path in ("/", "/index.html"):
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
                {"role": "system", "content": SYSTEM_PROMPT},
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

        self._send(200, json.dumps(parsed))


if __name__ == "__main__":
    port = 8000
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not found in .env next to this script.")
    print(f"Model: {MODEL}")
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
