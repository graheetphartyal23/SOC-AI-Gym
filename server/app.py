"""FastAPI server: SOC AI Gym HTTP API with docs, landing page, and evaluation helpers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .actions import Action, ActionType
from .constants import MAX_STEPS
from .environment import Observation, SOCEnvironment
from .grader import grade_episode

app = FastAPI(
    title="🛡️ SOC AI Gym",
    description=(
        "AI-powered SOC simulation for training autonomous incident response agents using RL "
        "and LLM policies. OpenEnv-style ``/reset``, ``/step``, and ``/state`` with deterministic "
        "graders and multi-layer rewards."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_env = SOCEnvironment(max_steps=MAX_STEPS, log_window=40)


class ResetBody(BaseModel):
    """Payload for starting an episode."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"task": "easy"}]},
    )

    task: str = Field(description="Task id: easy | medium | hard")


class GraderResponse(BaseModel):
    """Current episode grader output."""

    task: str
    score: float


TASKS_RESPONSE: dict[str, Any] = {
    "tasks": [
        {
            "id": "easy",
            "description": "Detect and block brute-force attack",
        },
        {
            "id": "medium",
            "description": "Investigate suspicious login and disable compromised user",
        },
        {
            "id": "hard",
            "description": "Stop multi-stage attack including data exfiltration",
        },
    ],
    "actions": [
        "investigate",
        "block_ip",
        "disable_user",
        "stop_exfiltration",
        "acknowledge_alert",
    ],
}

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SOC AI Gym</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #0c0f14;
      --surface: #141a22;
      --border: #243044;
      --text: #e8edf4;
      --muted: #8b9bb4;
      --accent: #3d8bfd;
      --accent-dim: #2563c9;
      --glow: rgba(61, 139, 253, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "DM Sans", system-ui, sans-serif;
      background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a2840 0%, var(--bg) 55%);
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .card {
      width: 100%;
      max-width: 520px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 2.5rem 2.25rem;
      box-shadow: 0 24px 64px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.03) inset;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-family: "JetBrains Mono", monospace;
      font-size: 0.72rem;
      font-weight: 500;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 1rem;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin: 0 0 0.5rem;
      line-height: 1.15;
    }
    p.lead {
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.55;
      margin: 0 0 2rem;
    }
    .actions {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    a.btn {
      display: block;
      text-align: center;
      text-decoration: none;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 0.85rem 1.25rem;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      color: var(--text);
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }
    a.btn:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
      background: rgba(61, 139, 253, 0.12);
      box-shadow: 0 8px 28px var(--glow);
    }
    a.btn.primary {
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%);
      border-color: transparent;
      color: #fff;
    }
    a.btn.primary:hover {
      box-shadow: 0 10px 32px var(--glow);
    }
    footer {
      margin-top: 2rem;
      padding-top: 1.25rem;
      border-top: 1px solid var(--border);
      font-size: 0.8rem;
      color: var(--muted);
      font-family: "JetBrains Mono", monospace;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">🛡️ OpenEnv · RL · SOC</div>
    <h1>SOC AI Gym</h1>
    <p class="lead">
      Train and evaluate agents on realistic incident-response tasks—brute force, suspicious login,
      and multi-stage attacks—with deterministic grading and a polished HTTP API.
    </p>
    <div class="actions">
      <a class="btn primary" href="/docs">Open API docs</a>
      <a class="btn" href="/baseline">Run baseline (grader scores)</a>
      <a class="btn" href="/tasks">View tasks &amp; actions</a>
    </div>
    <footer>POST /reset · POST /step · GET /state · GET /grader</footer>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> str:
    """Serve the marketing / demo landing page."""
    return LANDING_HTML


@app.post(
    "/reset",
    response_model=Observation,
    tags=["Environment"],
    summary="Start episode",
)
def reset(
    body: Annotated[
        ResetBody,
        Body(
            openapi_examples={
                "easy": {
                    "summary": "Brute-force scenario",
                    "description": "Reset the gym into the easy (brute force) task.",
                    "value": {"task": "easy"},
                },
                "medium": {"summary": "Suspicious login", "value": {"task": "medium"}},
                "hard": {"summary": "Multi-stage attack", "value": {"task": "hard"}},
            },
        ),
    ],
) -> Observation:
    """Initialize a new episode for the given task id."""
    t = body.task.lower().strip()
    if t not in ("easy", "medium", "hard"):
        raise HTTPException(400, "task must be easy, medium, or hard")
    return _env.reset(t)


@app.post(
    "/step",
    tags=["Actions"],
    summary="Apply one action",
)
def step(
    action: Annotated[
        Action,
        Body(
            openapi_examples={
                "investigate_default": {
                    "summary": "Investigate (smart default)",
                    "description": "Uses the primary alert when target is omitted.",
                    "value": {"action_type": "investigate"},
                },
                "block_default": {
                    "summary": "Block attacker IP (default)",
                    "value": {"action_type": "block_ip"},
                },
                "explicit": {
                    "summary": "Explicit alert + IP",
                    "value": {
                        "action_type": "investigate",
                        "target": "ALT-BF-001",
                        "note": "triage",
                    },
                },
            },
        ),
    ],
) -> dict[str, Any]:
    """Execute one analyst action and return observation, reward, terminal flag, and info."""
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["Environment"], summary="Internal state snapshot")
def state_endpoint() -> dict[str, Any]:
    """Return the full mutable simulation state as JSON (debugging / tooling)."""
    return _env.state()


@app.get("/tasks", tags=["Environment"], summary="Tasks and action names")
def tasks() -> dict[str, Any]:
    """List task ids with short descriptions plus recommended action type names."""
    return TASKS_RESPONSE


@app.get("/grader", response_model=GraderResponse, tags=["Evaluation"], summary="Grader (GET)")
def grader_get() -> GraderResponse:
    """Return the deterministic grader score for the current episode state."""
    s = _env.soc_state
    return GraderResponse(task=s.task.value, score=grade_episode(s))


@app.post("/grader", response_model=GraderResponse, tags=["Evaluation"], summary="Grader (POST)")
def grader_post() -> GraderResponse:
    """Same as GET /grader (for clients that prefer POST)."""
    return grader_get()


def _run_rule_policy(task: str) -> float:
    """
    Run the built-in scripted policy for one task and return the final grader score.

    Uses the same ``SOCEnvironment`` defaults as the live server (including ``MAX_STEPS``).
    """
    local = SOCEnvironment(max_steps=MAX_STEPS, log_window=40)
    local.reset(task)
    done = False
    s = local.soc_state

    def act(a: Action) -> None:
        nonlocal done
        _obs, _r, d, _info = local.step(a)
        done = d

    if task == "easy":
        act(Action(action_type=ActionType.INVESTIGATE, target=""))
        if not done:
            act(Action(action_type=ActionType.BLOCK_IP, target=""))
    elif task == "medium":
        act(Action(action_type=ActionType.INVESTIGATE, target=""))
        if not done:
            act(Action(action_type=ActionType.DISABLE_USER, target=""))
    else:
        act(Action(action_type=ActionType.INVESTIGATE, target=""))
        if not done:
            act(Action(action_type=ActionType.BLOCK_IP, target=""))
        if not done:
            act(Action(action_type=ActionType.DISABLE_USER, target=""))
        if not done:
            act(Action(action_type=ActionType.STOP_EXFILTRATION, target=""))

    return float(grade_episode(local.soc_state))


@app.get(
    "/baseline",
    tags=["Evaluation"],
    summary="Rule-based grader scores",
)
def baseline() -> dict[str, float]:
    """
    Run the deterministic rule-based policy on all three tasks.

    Returns only final grader scores keyed by task id (each in ``[0, 1]``).
    """
    return {
        "easy": _run_rule_policy("easy"),
        "medium": _run_rule_policy("medium"),
        "hard": _run_rule_policy("hard"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
