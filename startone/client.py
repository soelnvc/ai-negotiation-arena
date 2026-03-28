"""Baseline corporate agent client for the StartOne B2B market simulation.

This module runs an evaluation loop where one firm (Firm_A) decides actions
quarter-by-quarter using a Gemini model, with robust safeguards:

- Safe mode to bypass live model calls.
- Retry + backoff for transient failures.
- Validation of model JSON into `MarketAction`.
- Deterministic fallback strategy when model calls fail.
- Runtime telemetry for reliability and decision quality.

The default model and runtime behavior can be tuned via constants and env vars.
"""

import json
import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from .models import MarketAction, MarketObservation
from .server.startone_environment import MarketEnvironment

logger = logging.getLogger(__name__)

# === SDK Configuration ===
try:
    import google.generativeai as gemini_sdk  # type: ignore
    SDK_MODE = "legacy"
except ImportError:
    raise ImportError("google.generativeai SDK required. Install: pip install google-generativeai")

# === Decision Strategy Constants ===
# Thresholds that guide deterministic fallback behavior.
# They are intentionally conservative to prioritize stability under API failures.
TRUST_THRESHOLD_PARTNERSHIP = 0.35
CAPITAL_THRESHOLD_CONTRACT = 60
CAPITAL_THRESHOLD_BREACH = 30
TRUST_THRESHOLD_HOSTILE = -0.2

# === API & Retry Configuration ===
# API_CALL_DELAY_SECONDS helps keep request rate within free-tier limits.
# MAX_RETRIES controls model-call attempts before falling back.
API_CALL_DELAY_SECONDS = 10
MAX_RETRIES = 2
BACKOFF_BASE = 0.2
BACKOFF_MULTIPLIER = 2
RUN_QUERIES = 100

# Default model can be overridden via GEMINI_MODEL in .env
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# Environment variables used by this module:
# - GEMINI_API_KEY (required for live calls)
# - GEMINI_MODEL (optional, defaults to DEFAULT_MODEL_NAME)
# - STARTONE_SAFE_MODE (optional: 1/true/yes/on)

# === System Prompt Template ===
SYSTEM_PROMPT = """You are a Chief Strategy Officer optimizing for maximum Capital over {quarters} quarters.

CRITICAL: Preserve Capital. Never let it hit zero or the firm collapses.

Available Competitors/Partners: {competitors}

Current State:
- Capital: {capital} (YOU NEED THIS TO SURVIVE - keep above 30 if possible)
- Trust Scores: {trust_scores} (positive = ally, negative = enemy)
- News: {message}

Action Economics:
- Produce: +5 reward, +10 capital (safe, builds reserves)
- Execute_Contract: +20 reward, -20 capital (risky, costs resources)
- Form_Partnership: +30 reward, 0 capital (builds trust, high reward but target must be trusted)
- Breach_Contract: +40 reward, -30 capital cost BUT only if target trust < -0.2 (exploit enemies, huge cost)

Strategy Tips:
- When Capital < 50: Use Produce more to rebuild reserves
- When Capital >= 60 and target trust > 0: Execute_Contract selectively
- When Capital >= 60 and target trust >= 0.35: Form_Partnership (best ROI)
- AVOID breaching unless capital is comfortable and target is hostile

Respond with ONLY valid JSON (no markdown, no extra text). The JSON must have exactly these fields:
{{
  "action_type": "Produce" OR "Execute_Contract" OR "Form_Partnership" OR "Breach_Contract",
  "target_id": null OR one of {targets},
  "amount": 0 (integer)
}}

Rules:
- Produce: target_id must be null, amount must be 0
- Execute_Contract: target_id must be a competitor name, amount must be positive integer
- Form_Partnership: target_id must be a competitor name, amount must be 0
- Breach_Contract: target_id must be a competitor name, amount must be 0

Return ONLY the JSON object, nothing else."""

# --- 1. Lazy Client Initialization ---
_client: Optional[Any] = None

# Lightweight runtime telemetry for decision quality and reliability.
# - attempts/successes/fallbacks: overall decision pipeline outcomes
# - *_errors: categorized failure reasons
# - total_latency_ms: cumulative latency across all decisions
DECISION_METRICS: dict[str, float] = {
    "attempts": 0.0,
    "successes": 0.0,
    "fallbacks": 0.0,
    "api_errors": 0.0,
    "network_errors": 0.0,
    "auth_errors": 0.0,
    "quota_errors": 0.0,
    "config_errors": 0.0,
    "sdk_errors": 0.0,
    "json_errors": 0.0,
    "validation_errors": 0.0,
    "total_latency_ms": 0.0,
}


def is_safe_mode_enabled() -> bool:
    """Return whether safe mode is enabled.

    Safe mode skips live Gemini calls and always uses deterministic fallback
    decisions, which is useful for offline testing and quota-free runs.
    """
    raw = os.getenv("STARTONE_SAFE_MODE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_client() -> Any:
    """Initialize and cache the Gemini client lazily.

    Reads credentials and model configuration from environment:
    - GEMINI_API_KEY (required)
    - GEMINI_MODEL (optional, falls back to DEFAULT_MODEL_NAME)

    Returns:
        The initialized Gemini model client instance.

    Raises:
        ValueError: If GEMINI_API_KEY is missing.
    """
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing in .env")

        gemini_sdk.configure(api_key=api_key)  # type: ignore[attr-defined]
        _client = gemini_sdk.GenerativeModel(model_name)  # type: ignore[attr-defined]
    return _client


def get_startup_health() -> dict[str, Any]:
    """Return startup diagnostics for quick visibility before execution.

    Includes selected SDK mode, active model name, API key presence,
    safe-mode state, and whether the client has already been initialized.
    """
    load_dotenv()
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)
    return {
        "sdk_mode": SDK_MODE,
        "safe_mode": is_safe_mode_enabled(),
        "has_api_key": has_api_key,
        "model_name": model_name,
        "client_initialized": _client is not None,
    }


def _call_model(client: Any, prompt: str) -> str:
    """Call the model once and return response text.

    Args:
        client: Gemini model client from `get_client()`.
        prompt: Fully formatted instruction prompt.

    Returns:
        Raw text payload returned by the model.

    Raises:
        RuntimeError: If the model response has no text payload.
    """
    response = client.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )

    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError("Model response missing text payload")
    return response_text


def _emit(message: str) -> None:
    """Log and print a user-visible message in one place."""
    logger.info(message)
    print(message)


def _build_decision_prompt(observation: MarketObservation) -> str:
    """Build the model prompt from the current observation."""
    competitors = ", ".join(observation.trust_scores.keys())
    targets = list(observation.trust_scores.keys())
    return SYSTEM_PROMPT.format(
        competitors=competitors,
        capital=observation.capital,
        trust_scores=observation.trust_scores,
        message=observation.message,
        targets=targets,
        quarters=RUN_QUERIES,
    )


def _extract_action_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields required by MarketAction and ignore extras."""
    return {
        "action_type": data.get("action_type"),
        "target_id": data.get("target_id"),
        "amount": data.get("amount", 0),
    }


def _format_metrics_line(metrics: dict[str, float]) -> str:
    """Render final metrics in a compact, readable single line."""
    return (
        f"Decision Metrics | "
        f"Attempts: {int(metrics['attempts'])}, "
        f"Successes: {int(metrics['successes'])}, "
        f"Fallbacks: {int(metrics['fallbacks'])}, "
        f"API Errors: {int(metrics['api_errors'])}, "
        f"Network Errors: {int(metrics['network_errors'])}, "
        f"Auth Errors: {int(metrics['auth_errors'])}, "
        f"Quota Errors: {int(metrics['quota_errors'])}, "
        f"Config Errors: {int(metrics['config_errors'])}, "
        f"SDK Errors: {int(metrics['sdk_errors'])}, "
        f"JSON Errors: {int(metrics['json_errors'])}, "
        f"Validation Errors: {int(metrics['validation_errors'])}, "
        f"Avg Latency: {metrics['avg_latency_ms']:.2f} ms"
    )


def get_decision_metrics() -> dict[str, float]:
    """Return a metrics snapshot with computed average latency.

    `avg_latency_ms` is derived from `total_latency_ms / max(attempts, 1)`
    to avoid divide-by-zero during early runs.
    """
    snapshot = dict(DECISION_METRICS)
    attempts = max(1.0, snapshot["attempts"])
    snapshot["avg_latency_ms"] = snapshot["total_latency_ms"] / attempts
    return snapshot


def reset_decision_metrics() -> None:
    """Reset all decision telemetry counters to zero."""
    for key in DECISION_METRICS:
        DECISION_METRICS[key] = 0.0


def _categorize_api_error(exc: Exception) -> str:
    """Map an exception to one telemetry error bucket.

    Classification is best-effort: type checks first, then message pattern
    matching for SDK/API-specific failures.
    """
    message = str(exc).lower()
    exc_type = type(exc).__name__

    # Check exception types first (more reliable than string matching)
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "network_errors"
    if isinstance(exc, ValueError):
        if "GEMINI_API_KEY" in str(exc) or "missing" in message:
            return "config_errors"
    
    # Fall back to string pattern matching for API-specific errors
    if any(x in message for x in ["timeout", "timed out", "connection", "connection reset"]):
        return "network_errors"
    if any(x in message for x in ["api key", "unauthorized", "forbidden", "401", "403"]):
        return "auth_errors"
    if any(x in message for x in ["quota", "rate limit", "429", "resource exhausted"]):
        return "quota_errors"
    if any(x in message for x in ["attribute", "import", "module", "sdk"]):
        return "sdk_errors"
    if "error" in exc_type.lower():
        return "api_errors"
    
    logger.debug(f"Uncategorized error: {exc_type} - {message}")
    return "api_errors"

# --- 2. Smart Fallback Logic ---
def get_safe_fallback(obs: MarketObservation) -> MarketAction:
    """Return a deterministic fallback action from observation state.

    Strategy order:
    1. Prefer partnerships when trust is strong.
    2. Execute contracts when capital and trust are healthy.
    3. Breach only under low-capital + hostile-trust conditions.
    4. Otherwise produce as the safest default.
    """
    # Pick best-trust target deterministically
    ranked_targets = sorted(obs.trust_scores.items(), key=lambda item: item[1], reverse=True)
    if ranked_targets:
        best_target, best_trust = ranked_targets[0]

        # Partner first when trust is strong
        if best_trust >= TRUST_THRESHOLD_PARTNERSHIP:
            return MarketAction(action_type="Form_Partnership", target_id=best_target, amount=0)

        # Contract when capital and trust allow it
        if obs.capital >= CAPITAL_THRESHOLD_CONTRACT and best_trust >= 0.0:
            trade_amount = max(10, min(25, obs.capital // 4))
            return MarketAction(
                action_type="Execute_Contract",
                target_id=best_target,
                amount=trade_amount,
            )

        # Breach only with low capital and hostile targets
        if obs.capital < CAPITAL_THRESHOLD_BREACH and best_trust < TRUST_THRESHOLD_HOSTILE:
            return MarketAction(action_type="Breach_Contract", target_id=best_target, amount=0)

    # Default safe action
    return MarketAction(action_type="Produce")

# --- 3. Dynamic Decision Making ---
def get_corporate_decision(observation: MarketObservation) -> MarketAction:
    """Compute one action using Gemini with retries and guarded fallback.

    Flow:
    - In safe mode: immediately return deterministic fallback.
    - Otherwise: build prompt, call model, parse JSON, validate MarketAction.
    - On failures: retry with exponential backoff.
    - If retries exhausted: log and return deterministic fallback.
    """
    started_at = time.perf_counter()
    DECISION_METRICS["attempts"] += 1.0

    if is_safe_mode_enabled():
        DECISION_METRICS["fallbacks"] += 1.0
        DECISION_METRICS["total_latency_ms"] += (time.perf_counter() - started_at) * 1000.0
        return get_safe_fallback(observation)
    
    prompt = _build_decision_prompt(observation)

    max_retries = MAX_RETRIES
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            client = get_client()
            response_text = _call_model(client, prompt)
            data = json.loads(response_text)
            
            clean_data = _extract_action_payload(data)
            
            action = MarketAction(**clean_data)
            DECISION_METRICS["successes"] += 1.0
            DECISION_METRICS["total_latency_ms"] += (time.perf_counter() - started_at) * 1000.0
            
            # Respect API rate limits
            time.sleep(API_CALL_DELAY_SECONDS)
            
            return action
        except json.JSONDecodeError as exc:
            last_error = exc
            DECISION_METRICS["json_errors"] += 1.0
            logger.debug(f"Attempt {attempt + 1}: JSON decode error - {exc}")
        except ValidationError as exc:
            last_error = exc
            DECISION_METRICS["validation_errors"] += 1.0
            logger.debug(f"Attempt {attempt + 1}: Validation error - {exc}")
        except Exception as exc:
            last_error = exc
            metric_key = _categorize_api_error(exc)
            DECISION_METRICS[metric_key] += 1.0
            logger.debug(f"Attempt {attempt + 1}: {metric_key} - {type(exc).__name__}")

        if attempt < max_retries:
            # Exponential backoff for transient failures
            backoff_delay = BACKOFF_BASE * (BACKOFF_MULTIPLIER ** attempt)
            time.sleep(backoff_delay)

    # All retries exhausted, use fallback
    DECISION_METRICS["fallbacks"] += 1.0
    DECISION_METRICS["total_latency_ms"] += (time.perf_counter() - started_at) * 1000.0
    error_name = type(last_error).__name__ if last_error else "Unknown"
    logger.warning(f"Decision fallback activated after {max_retries} retries. Last error: {error_name}")
    _emit(f"Decision fallback activated after retries. Last error: {error_name}")
    return get_safe_fallback(observation)

# --- 4. The Runtime Loop ---
def run_simulation():
    """Run the full market simulation loop and print summary metrics.

    Uses `RUN_QUERIES` to control total quarters and prints per-quarter
    outcomes plus a final metrics summary.
    """
    reset_decision_metrics()
    health = get_startup_health()
    if not health["safe_mode"] and not health["has_api_key"]:
        raise ValueError("GEMINI_API_KEY missing in .env and STARTONE_SAFE_MODE is disabled")

    env = MarketEnvironment()
    obs = env.reset(actor_id="Firm_A")
    
    _emit("--- EVALUATION STARTED ---")
    health_msg = (
        f"Startup Health | SDK: {health['sdk_mode']} | "
        f"Model: {health['model_name']} | "
        f"Safe Mode: {health['safe_mode']} | "
        f"API Key: {health['has_api_key']}"
    )
    _emit(health_msg)
    
    for q in range(1, RUN_QUERIES + 1):
        time.sleep(5)  # Buffer between quarters
        action = get_corporate_decision(obs)
        obs = env.step(action, actor_id="Firm_A")
        quarter_msg = f"Q{q} | {action.action_type} -> Reward: {obs.reward} | Capital: {obs.capital}"
        _emit(quarter_msg)
        if obs.done:
            break

    metrics = get_decision_metrics()
    _emit(_format_metrics_line(metrics))
    _emit("--- EVALUATION ENDED ---")

if __name__ == "__main__":
    run_simulation()