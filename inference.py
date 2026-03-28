"""Submission inference runner for StartOne.

This script evaluates all registered tasks using a single LLM policy and prints
task scores in the required [0.0, 1.0] range.

Required environment variables:
- API_BASE_URL: OpenAI-compatible endpoint URL.
- MODEL_NAME: Model identifier for inference.
- HF_TOKEN: API key/token for the endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from startone.models import MarketAction, MarketObservation
from startone.server.tasks import MARKET_TASKS


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_BASE_URL = _require_env("API_BASE_URL")
MODEL_NAME = _require_env("MODEL_NAME")
HF_TOKEN = _require_env("HF_TOKEN")

ACTOR_ID = "Firm_A"
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "100"))

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


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


def run_task(task: Any) -> float:
    """Run a single task and return its normalized score [0.0, 1.0]."""
    env = task.environment_class()
    obs = env.reset(actor_id=ACTOR_ID)

    max_steps = min(int(task.max_steps), MAX_STEPS_PER_TASK)
    for _ in range(max_steps):
        try:
            action = _llm_action(obs)
        except Exception:
            action = _safe_fallback(obs)

        obs = env.step(action, actor_id=ACTOR_ID)
        if obs.done:
            break

    telemetry = env.state.telemetry.get(ACTOR_ID, {})
    raw_score = float(task.grader_callable(env.state, ACTOR_ID, telemetry))
    return max(0.0, min(1.0, raw_score))


def main() -> None:
    results: dict[str, float] = {}
    for task in MARKET_TASKS:
        score = run_task(task)
        task_name = getattr(task, "task_id", getattr(task, "name", "unknown-task"))
        results[str(task_name)] = round(score, 4)

    average = sum(results.values()) / max(1, len(results))
    output = {
        "model": MODEL_NAME,
        "tasks": results,
        "average_score": round(average, 4),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
