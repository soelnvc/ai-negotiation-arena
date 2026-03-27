from typing import Any

from startone.models import MarketAction, StartoneAction
from startone.server.startone_environment import MarketEnvironment


def test_reset_initializes_trust_matrix_and_observation() -> None:
    env = MarketEnvironment()
    obs = env.reset(seed=123)

    assert obs.capital == 100
    assert obs.done is False
    assert obs.trust_scores == {"Firm_B": 0.0, "Firm_C": 0.0}

    matrix = env.state.trust_matrix
    assert set(matrix.keys()) == {"Firm_A", "Firm_B", "Firm_C"}
    assert matrix["Firm_A"] == {"Firm_B": 0.0, "Firm_C": 0.0}


def test_execute_contract_updates_capital_and_trust() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    obs = env.step(MarketAction(action_type="Execute_Contract", target_id="Firm_B", amount=10))

    assert obs.reward == 20.0
    assert obs.capital == 90
    assert env.state.firm_capital["Firm_B"] == 112
    assert obs.trust_scores["Firm_B"] == 0.05


def test_breach_contract_updates_capital_and_trust() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    obs = env.step(MarketAction(action_type="Breach_Contract", target_id="Firm_B", amount=0))

    assert obs.reward == 40.0
    assert obs.capital == 115
    assert env.state.firm_capital["Firm_B"] == 85
    assert obs.trust_scores["Firm_B"] == -0.15


def test_form_partnership_updates_trust_and_consumes_turn() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    before = env.state.step_count
    obs = env.step(MarketAction(action_type="Form_Partnership", target_id="Firm_B", amount=0))
    after = env.state.step_count

    assert obs.reward == 30.0
    assert obs.trust_scores["Firm_B"] == 0.1
    assert env.state.trust_matrix["Firm_B"]["Firm_A"] == 0.1
    assert before == 0
    assert after == 1


def test_unsupported_action_does_not_consume_turn() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    before = env.state.step_count
    obs = env.step(MarketAction(action_type="Hold", amount=0))
    after = env.state.step_count

    assert obs.reward == -10.0
    assert before == 0
    assert after == 0


def test_unknown_firm_returns_safe_error_observation() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    obs = env.step(MarketAction(action_type="Produce", amount=0), actor_id="Ghost")

    assert obs.reward == -10.0
    assert obs.capital == 0
    assert set(obs.trust_scores.keys()) == {"Firm_A", "Firm_B", "Firm_C"}


def test_terminal_guard_returns_neutral_reward_and_no_mutation() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)
    env.state.step_count = env.state.max_rounds

    obs = env.step(MarketAction(action_type="Produce", amount=0))

    assert obs.done is True
    assert obs.reward == 0.0
    assert env.state.step_count == env.state.max_rounds


def test_legacy_action_payload_is_handled_without_crash() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    before = env.state.step_count
    legacy_action: Any = StartoneAction(action_type="Execute_Contract", target_id=None, amount=0)
    obs = env.step(legacy_action)
    after = env.state.step_count

    assert obs.reward == -10.0
    assert "missing action_type" in obs.message
    assert before == after


def test_telemetry_defaults_initialized_on_reset() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    telemetry = env.state.telemetry["Firm_A"]
    assert telemetry["initial_capital"] == 100.0
    assert telemetry["successful_contracts"] == 0.0
    assert telemetry["contracts_breached"] == 0.0
    assert telemetry["partnership_streak_steps"] == 0.0
    assert telemetry["market_decline_ratio"] == 0.0


def test_telemetry_contract_counter_increments() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    env.step(MarketAction(action_type="Execute_Contract", target_id="Firm_B", amount=10))

    telemetry = env.state.telemetry["Firm_A"]
    assert telemetry["successful_contracts"] == 1.0
    assert telemetry["contracts_breached"] == 0.0
    assert telemetry["partnership_streak_steps"] == 0.0


def test_telemetry_breach_counter_increments() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    env.step(MarketAction(action_type="Breach_Contract", target_id="Firm_B", amount=0))

    telemetry = env.state.telemetry["Firm_A"]
    assert telemetry["successful_contracts"] == 0.0
    assert telemetry["contracts_breached"] == 1.0
    assert telemetry["partnership_streak_steps"] == 0.0


def test_telemetry_partnership_counter_increments() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    env.step(MarketAction(action_type="Form_Partnership", target_id="Firm_B", amount=0))

    telemetry = env.state.telemetry["Firm_A"]
    assert telemetry["successful_contracts"] == 0.0
    assert telemetry["contracts_breached"] == 0.0
    assert telemetry["partnership_streak_steps"] == 1.0


def test_partnership_streak_resets_after_non_partnership_action() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    env.step(MarketAction(action_type="Form_Partnership", target_id="Firm_B", amount=0))
    env.step(MarketAction(action_type="Form_Partnership", target_id="Firm_B", amount=0))
    assert env.state.telemetry["Firm_A"]["partnership_streak_steps"] == 2.0

    env.step(MarketAction(action_type="Produce", amount=0))
    assert env.state.telemetry["Firm_A"]["partnership_streak_steps"] == 0.0


def test_partnership_streak_resets_on_failed_partnership_attempt() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    env.step(MarketAction(action_type="Form_Partnership", target_id="Firm_B", amount=0))
    assert env.state.telemetry["Firm_A"]["partnership_streak_steps"] == 1.0

    env.step(MarketAction(action_type="Form_Partnership", target_id="Firm_A", amount=0))
    assert env.state.telemetry["Firm_A"]["partnership_streak_steps"] == 0.0


def test_telemetry_produce_does_not_increment_social_counters() -> None:
    env = MarketEnvironment()
    env.reset(seed=1)

    env.step(MarketAction(action_type="Produce", amount=0))

    telemetry = env.state.telemetry["Firm_A"]
    assert telemetry["successful_contracts"] == 0.0
    assert telemetry["contracts_breached"] == 0.0
    assert telemetry["partnership_streak_steps"] == 0.0
