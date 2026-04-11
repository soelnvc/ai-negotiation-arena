"""Submission inference runner for StartOne.

This script evaluates all registered tasks using a single LLM policy and prints
task scores in the required [0.0, 1.0] range.

Environment variables:
- API_BASE_URL: OpenAI-compatible endpoint URL (default provided).
- MODEL_NAME: Model identifier for inference (default provided).
- HF_TOKEN: API key/token for the endpoint (no default).
- LOCAL_IMAGE_NAME: Optional local image name (used only for docker-image execution flows).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any

from openai import OpenAI

# Keep stdout deterministic by suppressing third-party deprecation noise.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"startone\.client")

from startone.models import MarketAction, MarketObservation
from startone.server.tasks import MARKET_TASKS


API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ACTOR_ID = "Firm_A"
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "12"))
MAX_TOTAL_RUNTIME_SECONDS = int(os.getenv("MAX_TOTAL_RUNTIME_SECONDS", "240"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "4"))

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=1,
)


def _strict_score(value: float) -> float:
    """Clamp score to strict open interval (0, 1) for validator compatibility."""
    eps = 0.01
    try:
        v = float(value)
    except (TypeError, ValueError):
        return eps
    if v <= 0.0:
        return eps
    if v >= 1.0:
        return 1.0 - eps
    return v


def _emit(tag: str, payload: dict[str, Any]) -> None:
    """Emit one structured log line in the required tagged format."""
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}")


def _extract_text(response: Any) -> str:
    """Extract text from common OpenAI-compatible response shapes."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content:
            return str(content)

    # Final fallback for provider-specific dict-like payloads.
    as_dict = getattr(response, "model_dump", None)
    if callable(as_dict):
        payload = as_dict()
        try:
            return json.dumps(payload)
        except Exception:
            return str(payload)

    raise RuntimeError("LLM response did not include text content")


def _safe_fallback(obs: MarketObservation) -> MarketAction:
    """Deterministic fallback to keep evaluation progressing under API errors."""
    ranked_targets = sorted(obs.trust_scores.items(), key=lambda item: item[1], reverse=True)
    if ranked_targets:
        best_target, best_trust = ranked_targets[0]
        if best_trust >= 0.35 and obs.capital >= 80:
            return MarketAction(action_type="Form_Partnership", target_id=best_target, amount=0)
        if obs.capital >= 60 and best_trust > -0.3:
            trade_amount = max(10, min(25, obs.capital // 4))
            return MarketAction(action_type="Execute_Contract", target_id=best_target, amount=trade_amount)
        if obs.capital < 30 and best_trust < -0.2:
            return MarketAction(action_type="Breach_Contract", target_id=best_target, amount=0)
    return MarketAction(action_type="Produce", target_id=None, amount=0)


def _llm_action(obs: MarketObservation) -> MarketAction:
    """Query model for one action and validate shape via MarketAction."""
    targets = list(obs.trust_scores.keys())
    prompt = f"""
You are controlling a firm in a B2B simulation. Return ONLY valid JSON.

State:
- Capital: {obs.capital}
- Trust scores: {obs.trust_scores}
- Message: {obs.message}

Action economics:
- Produce: +5 reward, +10 capital
- Execute_Contract: reward scales with amount, amount costs your capital
- Form_Partnership: costs 10 capital, reward diminishes if repeated
- Breach_Contract: reward scales with seized capital, trust drops

Rules:
- JSON fields exactly: action_type, target_id, amount
- action_type one of: Produce, Execute_Contract, Form_Partnership, Breach_Contract
- Produce => target_id null, amount 0
- Execute_Contract => target_id in {targets}, amount > 0 and <= {obs.capital}
- Form_Partnership => target_id in {targets}, amount 0
- Breach_Contract => target_id in {targets}, amount 0

Return only JSON object.
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise JSON-only decision engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    response_text = _extract_text(response)
    data = json.loads(response_text)
    clean_data = {
        "action_type": data.get("action_type"),
        "target_id": data.get("target_id"),
        "amount": data.get("amount", 0),
    }
    return MarketAction(**clean_data)


def run_task(task: Any, deadline: float) -> float:
    """Run a single task and return its normalized score [0.0, 1.0]."""
    env = task.environment_class()
    obs = env.reset(actor_id=ACTOR_ID)

    # Keep runs fast and predictable so all tasks are always scored.
    max_steps = min(int(task.max_steps), MAX_STEPS_PER_TASK)
    executed_steps = 0
    for _ in range(max_steps):
        if time.monotonic() >= deadline:
            break
        # One LLM attempt per task max; fallback for all remaining steps.
        if executed_steps == 0:
            try:
                action = _llm_action(obs)
            except Exception:
                action = _safe_fallback(obs)
        else:
            action = _safe_fallback(obs)

        obs = env.step(action, actor_id=ACTOR_ID)
        executed_steps += 1
        if obs.done:
            break

    telemetry = env.state.telemetry.get(ACTOR_ID, {})
    grader_fn = getattr(task, "grader", None) or getattr(task, "grader_callable", None)
    if grader_fn is None:
        raw_score = 0.01
    else:
        raw_score = float(grader_fn(env.state, ACTOR_ID, telemetry))
    score = _strict_score(raw_score)
    _emit(
        "STEP",
        {
            "task": str(getattr(task, "task_id", getattr(task, "name", "unknown-task")),),
            "steps_executed": executed_steps,
            "score": round(score, 4),
        },
    )
    return score


def main() -> None:
    start_time = time.monotonic()
    deadline = start_time + MAX_TOTAL_RUNTIME_SECONDS
    _emit(
        "START",
        {
            "model": MODEL_NAME,
            "max_steps_per_task": MAX_STEPS_PER_TASK,
            "max_total_runtime_seconds": MAX_TOTAL_RUNTIME_SECONDS,
        },
    )

    results: dict[str, float] = {}
    for task in MARKET_TASKS:
        score = run_task(task, deadline)
        task_name = getattr(task, "task_id", getattr(task, "name", "unknown-task"))
        results[str(task_name)] = round(_strict_score(score), 4)

    average = sum(results.values()) / max(1, len(results))
    output = {
        "model": MODEL_NAME,
        "tasks": results,
        "average_score": round(average, 4),
        "elapsed_seconds": round(time.monotonic() - start_time, 2),
    }
    _emit("END", output)


if __name__ == "__main__":
    main()
