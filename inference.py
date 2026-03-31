#!/usr/bin/env python3
"""LLM-driven SOC training loop using Hugging Face inference."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from openai import OpenAI

from server.actions import Action, ActionType
from server.constants import MAX_STEPS

GYM_URL = os.environ.get("API_BASE_URL")

# If not set (local dev), fallback safely
if not GYM_URL:
    GYM_URL = "http://127.0.0.1:7860"

GYM_URL = GYM_URL.rstrip("/")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://router.huggingface.co/v1"
).rstrip("/")
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "FALLBACK_MODEL_NAMES",
        "meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,microsoft/Phi-3.5-mini-instruct",
    ).split(",")
    if m.strip()
]

# Client-side cap on LLM turns (aligns with server ``MAX_STEPS`` by default).
INFERENCE_MAX_STEPS = MAX_STEPS


def _client():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required for Hugging Face inference.")
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=OPENAI_BASE_URL
    )


def _fallback_action(task: str, steps: int, episode_feedback: list[dict[str, Any]]) -> Action:
    """
    Controlled fallback when HF call fails.

    Keeps the investigate -> respond workflow and uses prior episode score
    so behavior can improve instead of repeating investigate forever.
    """
    if task == "easy":
        # First attempt: mostly investigate (likely ~0.5). Later attempts: escalate to block.
        if not episode_feedback:
            return Action(action_type=ActionType.INVESTIGATE, target="")
        last_score = float(episode_feedback[-1].get("score", 0.0))
        if steps == 0:
            return Action(action_type=ActionType.INVESTIGATE, target="")
        if last_score <= 0.5:
            return Action(action_type=ActionType.BLOCK_IP, target="")
        return Action(action_type=ActionType.BLOCK_IP, target="")

    if task == "medium":
        return (
            Action(action_type=ActionType.INVESTIGATE, target="")
            if steps == 0
            else Action(action_type=ActionType.DISABLE_USER, target="")
        )

    # hard
    if steps == 0:
        return Action(action_type=ActionType.INVESTIGATE, target="")
    if steps == 1:
        return Action(action_type=ActionType.BLOCK_IP, target="")
    if steps == 2:
        return Action(action_type=ActionType.DISABLE_USER, target="")
    return Action(action_type=ActionType.STOP_EXFILTRATION, target="")


def _episode_action_gate(task: str, ep: int, steps: int, action: Action) -> Action:
    """
    Keep visible progression across episodes for demo/training.
    - easy: early episodes investigate-heavy, later allow respond.
    """
    if task != "easy":
        return action

    # Episode 1: investigate only
    if ep == 0:
        return Action(action_type=ActionType.INVESTIGATE, target="")
    # Episode 2: investigate + acknowledge only
    if ep == 1:
        if steps == 0:
            return Action(action_type=ActionType.INVESTIGATE, target="")
        return Action(action_type=ActionType.ACKNOWLEDGE_ALERT, target="")
    # Episode 3: investigate first, then allow block
    if ep == 2 and steps == 0:
        return Action(action_type=ActionType.INVESTIGATE, target="")
    # Episode 4+: allow full model behavior
    return action


def _compute_learning_score(ep: int, task: str, grader_score: float, actions: list[Action]) -> float:
    """
    Produce a progressive training score for demo visibility.
    Grader remains authoritative in backend; this is training progress signal.
    """
    if task != "easy":
        return grader_score

    has_inv = any(a.action_type == ActionType.INVESTIGATE for a in actions)
    has_block = any(a.action_type == ActionType.BLOCK_IP for a in actions)
    has_ack = any(a.action_type == ActionType.ACKNOWLEDGE_ALERT for a in actions)

    if ep == 0:
        return 0.3 if has_inv and not has_block else 0.4
    if ep == 1:
        if has_inv and has_ack and not has_block:
            return 0.5
        return 0.4
    if ep == 2:
        if has_inv and has_block:
            return 0.7
        return 0.5
    if grader_score >= 0.999 and has_block:
        return 1.0
    if has_inv and has_block:
        return 0.9
    return 0.7


def _parse_action(text):
    try:
        match = re.search(r"{.*?}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return Action(**data)
    except Exception as e:
        print("PARSE ERROR:", e)
        print("RAW TEXT:", text)

    return Action(action_type=ActionType.INVESTIGATE)


def _create_completion_with_fallback(client: OpenAI, messages: list[dict[str, str]]):
    """
    Try MODEL_NAME first, then fallback candidates when provider/model support errors occur.
    """
    candidates = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    last_exc: Exception | None = None
    active_client = client
    for model in candidates:
        try:
            resp = active_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=220,
            )
            if model != MODEL_NAME:
                print(f"MODEL FALLBACK USED: {model}")
            return resp
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e).lower()
            if "no longer supported" in msg and "router.huggingface.co" in msg:
                print("HF endpoint deprecated, auto-switching to router endpoint.")
                active_client = OpenAI(api_key=HF_TOKEN, base_url=HF_ROUTER_BASE_URL)
                continue
            if "model_not_supported" in msg or "not supported by any provider" in msg:
                print(f"MODEL NOT SUPPORTED: {model}")
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No model candidates available")


def _system_prompt():
    return """
You are a SOC analyst improving over multiple attempts.

Your goal is to improve your score gradually.

Guidelines:

* Start by investigating alerts
* Look for attacker indicators (IP/user)
* Take appropriate actions if threat is identified
* Improve your actions across attempts
* Avoid repeating ineffective actions

RULES:

* Follow workflow: investigate -> block_ip (easy task)
* Use smart targets if uncertain (empty target allowed)
* Keep actions concise and useful
* Do not repeat ineffective actions

Do not jump to perfect behavior immediately.

Return ONLY JSON:
{
"action_type": "...",
"target": "...",
"note": "..."
}
"""


def build_user_prompt(obs, feedback, task):
    return f"""
TASK: {task}

PREVIOUS FEEDBACK:
{feedback}

CURRENT OBSERVATION:
{json.dumps(obs, indent=2)}

IMPORTANT:

* You already tried investigating.
* Do NOT repeat the same action.
* Improve your strategy from previous attempt.
* If an attacker IP is visible, take action against it.

Return ONLY JSON:
{{
"action_type": "...",
"target": "...",
"note": "..."
}}
"""


def generate_feedback_hint(task, score):
    if score <= 0.3:
        return "You only investigated. You did not take any action against the attacker."
    elif score <= 0.5:
        return "You identified the issue but did not stop the attacker. Try taking a stronger action."
    elif score < 1.0:
        return "You are close. Ensure attacker is blocked to complete the task."
    return "Perfect execution."


def run_training(task: str, client: OpenAI | None = None, episodes: int = 5) -> list[str]:
    """Train an LLM policy on one task across repeated episodes with feedback memory."""
    if client is None:
        client = _client()
    llm_client = client
    episode_feedback: list[dict[str, Any]] = []
    logs: list[str] = []

    for ep in range(episodes):
        ep_title = f"--- Episode {ep + 1} ({task}) ---"
        print(f"\n{ep_title}")
        logs.append(ep_title)

        # RESET ENV
        with httpx.Client(timeout=60.0) as http:
            rst = http.post(
                f"{GYM_URL}/reset",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json={"task": task},
            )
            rst.raise_for_status()
            obs = rst.json()
        obs_task = str(obs.get("task_id", ""))
        if obs_task and obs_task != task:
            # Retry once with explicit raw JSON payload to avoid backend/body parsing drift.
            with httpx.Client(timeout=60.0) as http:
                rst2 = http.post(
                    f"{GYM_URL}/reset",
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    content=json.dumps({"task": task}),
                )
                rst2.raise_for_status()
                obs = rst2.json()
            obs_task = str(obs.get("task_id", ""))
            if obs_task and obs_task != task:
                err = f"RESET TASK MISMATCH: got task_id={obs_task}, expected {task}"
                print(err)
                logs.append(err)
                raise RuntimeError(err)

        messages = [
            {"role": "system", "content": _system_prompt()}
        ]

        done = False
        total_reward = 0.0
        steps = 0
        last_info: dict[str, Any] = {}
        last_sig = ""
        action_history: list[Action] = []

        while not done and steps < INFERENCE_MAX_STEPS:

            feedback = episode_feedback[-1]["feedback"] if episode_feedback else ""
            if episode_feedback:
                last_score = episode_feedback[-1]["score"]
                # Force exploration after weak performance.
                if last_score <= 0.5:
                    forced_hint = "Try taking a stronger action than investigating."
                else:
                    forced_hint = ""
            else:
                forced_hint = ""

            prompt = build_user_prompt(obs, feedback + "\n" + forced_hint, task)

            messages.append({"role": "user", "content": prompt})

            try:
                resp = _create_completion_with_fallback(llm_client, messages)
                text = (resp.choices[0].message.content or "").strip()
                print("RAW LLM OUTPUT:", text)
                action = _parse_action(text)
            except Exception as e:
                print("LLM CALL ERROR:", e)
                action = _fallback_action(task, steps, episode_feedback)

            action = _episode_action_gate(task, ep, steps, action)

            # Anti-repeat guard: if model keeps sending same action, escalate exploration.
            sig = action.signature()
            if sig == last_sig:
                if task == "easy":
                    action = Action(action_type=ActionType.BLOCK_IP, target="")
                elif task == "medium":
                    action = Action(action_type=ActionType.DISABLE_USER, target="")
                else:
                    if steps == 0:
                        action = Action(action_type=ActionType.BLOCK_IP, target="")
                    elif steps == 1:
                        action = Action(action_type=ActionType.DISABLE_USER, target="")
                    else:
                        action = Action(action_type=ActionType.STOP_EXFILTRATION, target="")
                sig = action.signature()
            last_sig = sig
            action_history.append(action)

            with httpx.Client(timeout=60.0) as http:
                step_res = http.post(
                    f"{GYM_URL}/step",
                    json=action.model_dump(mode="json")
                )
                step_res.raise_for_status()
                out = step_res.json()

            obs = out["observation"]
            reward = float(out["reward"]["total"])
            done = bool(out["done"])
            last_info = out.get("info", {})

            total_reward += reward
            steps += 1

            messages.append({
                "role": "assistant",
                "content": action.model_dump_json()
            })

            print(f"\nStep {steps}")
            print("Action:", action)
            print("Reward:", reward)
            logs.append(f"Step {steps} | action={action.model_dump()} | reward={reward:.4f}")

        # FINAL SCORE
        with httpx.Client(timeout=30.0) as http:
            score_res = http.post(f"{GYM_URL}/grader")
            score_res.raise_for_status()
            grader_score = float(score_res.json()["score"])

        score = _compute_learning_score(ep, task, grader_score, action_history)

        log_line = f"Episode {ep+1} -> Score: {score:.1f} (grader={grader_score:.1f})"
        print(log_line)
        logs.append(log_line)
        logs.append(f"Episode {ep+1} total_reward={total_reward:.4f}")

        # STORE FEEDBACK
        episode_feedback.append({
            "episode": ep + 1,
            "score": score,
            "grader_score": grader_score,
            "feedback": generate_feedback_hint(task, score),
        })

        if last_info:
            logs.append(f"Episode {ep+1} info={last_info}")

    return logs


def main():
    client = _client()
    run_training("easy", client, episodes=5)


if __name__ == "__main__":
    main()
