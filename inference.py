#!/usr/bin/env python3
"""
LLM-driven evaluation: reads observations from the SOC gym HTTP API and queries an
OpenAI-compatible chat model (OpenAI, Hugging Face router, etc.).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import httpx
from openai import OpenAI

from server.actions import Action, ActionType
from server.constants import MAX_STEPS

GYM_URL = os.environ.get("API_BASE_URL")
if not GYM_URL:
    raise ValueError("API_BASE_URL must be set")
GYM_URL = GYM_URL.rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# Client-side cap on LLM turns (aligns with server ``MAX_STEPS`` by default).
INFERENCE_MAX_STEPS = MAX_STEPS


def _client() -> OpenAI:
    """Build an OpenAI-compatible client from environment variables."""
    key = HF_TOKEN or OPENAI_API_KEY
    if not key:
        print(
            "Warning: HF_TOKEN and OPENAI_API_KEY are empty; set one for live LLM calls.",
            file=sys.stderr,
        )
        key = "dummy-key"
    return OpenAI(api_key=key, base_url=OPENAI_BASE_URL)


def _parse_action(text: str) -> Action:
    """Parse a JSON action from raw model output."""
    text = text.strip()
    try:
        data = json.loads(text)
        return Action.model_validate(data)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            return Action.model_validate_json(m.group(0))
    raise ValueError(f"Could not parse action JSON from model output: {text[:500]}")


def _system_prompt() -> str:
    return """You are a SOC analyst agent in a simulated environment.
You must respond with a single JSON object only (no markdown), matching this schema:
{
  "action_type": one of [
    "no_op", "acknowledge_alert", "investigate", "block_ip", "unblock_ip",
    "disable_user", "enable_user", "stop_exfiltration"
  ],
  "target": "string — alert_id (e.g. ALT-BF-001), user_id, or IP address as required",
  "note": "optional short note"
}
You may omit \"target\" for investigate, block_ip, and disable_user when appropriate (server applies smart defaults).
For hard tasks, you must also stop_exfiltration (target can be empty string).
Minimize no_op."""


def run_episode(task: str, client: OpenAI) -> dict[str, Any]:
    """Run one episode over HTTP until ``done`` or step budget is exhausted."""
    with httpx.Client(timeout=60.0) as http:
        r = http.post(f"{GYM_URL}/reset", json={"task": task})
        r.raise_for_status()
        obs = r.json()

    total_reward = 0.0
    done = False
    last_info: dict[str, Any] = {}

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": "Current observation JSON:\n"
            + json.dumps(obs, indent=2)
            + "\nChoose the next action JSON.",
        },
    ]

    steps = 0
    while not done and steps < INFERENCE_MAX_STEPS:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            content = (resp.choices[0].message.content or "").strip()
            action = _parse_action(content)
        except Exception as exc:  # noqa: BLE001
            print(f"[{task}] LLM/step error: {exc}", file=sys.stderr)
            action = Action(action_type=ActionType.INVESTIGATE, target="")

        with httpx.Client(timeout=60.0) as http:
            r = http.post(f"{GYM_URL}/step", json=action.model_dump(mode="json"))
            r.raise_for_status()
            out = r.json()

        rew = out["reward"]["total"]
        total_reward += float(rew)
        done = bool(out["done"])
        last_info = out.get("info") or {}
        obs = out["observation"]
        steps += 1

        messages.append({"role": "assistant", "content": action.model_dump_json()})
        messages.append(
            {
                "role": "user",
                "content": "Step result:\n"
                + json.dumps(
                    {"reward": out["reward"], "done": done, "info": last_info},
                    indent=2,
                )
                + "\nNext observation:\n"
                + json.dumps(obs, indent=2)
                + "\nNext action JSON only.",
            }
        )

    with httpx.Client(timeout=30.0) as http:
        gr = http.post(f"{GYM_URL}/grader")
        gr.raise_for_status()
        grader = gr.json()

    score = float(grader.get("score", 0.0))
    return {
        "task": task,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_grader_score": score,
        "success": last_info.get("success"),
    }


def main() -> None:
    """Run all three tasks and print a judge-friendly score summary."""
    client = _client()
    print(f"Gym API: {GYM_URL}")
    print(f"Model: {MODEL_NAME} @ {OPENAI_BASE_URL}")
    print()
    print("=== SOC AI Evaluation ===")
    for task in ("easy", "medium", "hard"):
        summary = run_episode(task, client)
        sc = summary.get("final_grader_score")
        print(f"Task {task} → Score: {sc}")
    print()


if __name__ == "__main__":
    main()
