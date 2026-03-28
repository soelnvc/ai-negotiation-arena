from types import SimpleNamespace

import startone.client as client
from startone.models import MarketObservation


def _sample_obs() -> MarketObservation:
    return MarketObservation(
        done=False,
        reward=0.0,
        capital=100,
        trust_scores={"Firm_B": 0.4, "Firm_C": -0.1},
        active_partnerships=[],
        message="ok",
    )


def test_safe_mode_forces_fallback(monkeypatch) -> None:
    monkeypatch.setenv("STARTONE_SAFE_MODE", "1")
    client.reset_decision_metrics()

    action = client.get_corporate_decision(_sample_obs())

    metrics = client.get_decision_metrics()
    assert action.action_type in {"Produce", "Execute_Contract", "Form_Partnership", "Breach_Contract"}
    assert metrics["attempts"] == 1.0
    assert metrics["fallbacks"] == 1.0


def test_retry_then_fallback_on_json_errors(monkeypatch) -> None:
    monkeypatch.delenv("STARTONE_SAFE_MODE", raising=False)
    monkeypatch.setattr(client, "get_client", lambda: object())
    monkeypatch.setattr(client, "_call_model", lambda _c, _p: "{invalid json")

    client.reset_decision_metrics()
    action = client.get_corporate_decision(_sample_obs())

    metrics = client.get_decision_metrics()
    assert action.action_type in {"Produce", "Execute_Contract", "Form_Partnership", "Breach_Contract"}
    # max_retries=2 means 3 total attempts of model parse
    assert metrics["json_errors"] == 3.0
    assert metrics["fallbacks"] == 1.0


def test_metrics_reset_clears_previous_state() -> None:
    client.DECISION_METRICS["attempts"] = 5.0
    client.DECISION_METRICS["api_errors"] = 2.0

    client.reset_decision_metrics()

    metrics = client.get_decision_metrics()
    assert metrics["attempts"] == 0.0
    assert metrics["api_errors"] == 0.0


def test_startup_health_reports_safe_mode(monkeypatch) -> None:
    monkeypatch.setenv("STARTONE_SAFE_MODE", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    health = client.get_startup_health()

    assert health["safe_mode"] is True
    assert health["has_api_key"] is False
    assert health["sdk_mode"] in {"modern", "legacy"}


def test_success_path_updates_success_metric(monkeypatch) -> None:
    monkeypatch.delenv("STARTONE_SAFE_MODE", raising=False)
    monkeypatch.setattr(client, "get_client", lambda: object())
    monkeypatch.setattr(
        client,
        "_call_model",
        lambda _c, _p: '{"action_type":"Produce","target_id":null,"amount":0}',
    )

    client.reset_decision_metrics()
    action = client.get_corporate_decision(_sample_obs())

    metrics = client.get_decision_metrics()
    assert action.action_type == "Produce"
    assert metrics["successes"] == 1.0
    assert metrics["fallbacks"] == 0.0
