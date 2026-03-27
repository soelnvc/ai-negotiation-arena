from typing import Any

from startone.models import ArenaAction, StartoneAction
from startone.server.startone_environment import ArenaEnvironment


def test_reset_initializes_reputation_matrix_and_observation() -> None:
    env = ArenaEnvironment()
    obs = env.reset(seed=123)

    assert obs.resources == 100
    assert obs.done is False
    assert obs.reputation_scores == {"Agent_B": 0.0, "Agent_C": 0.0}

    matrix = env.state.reputation_matrix
    assert set(matrix.keys()) == {"Agent_A", "Agent_B", "Agent_C"}
    assert matrix["Agent_A"] == {"Agent_B": 0.0, "Agent_C": 0.0}


def test_trade_updates_resources_and_reputation() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    obs = env.step(ArenaAction(action_type="Trade", target_id="Agent_B", amount=10))

    assert obs.reward == 20.0
    assert obs.resources == 90
    assert env.state.agent_resources["Agent_B"] == 112
    assert obs.reputation_scores["Agent_B"] == 0.05


def test_betrayal_updates_resources_and_reputation() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    obs = env.step(ArenaAction(action_type="Betray", target_id="Agent_B", amount=0))

    assert obs.reward == 40.0
    assert obs.resources == 115
    assert env.state.agent_resources["Agent_B"] == 85
    assert obs.reputation_scores["Agent_B"] == -0.15


def test_ally_updates_reputation_and_consumes_turn() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    before = env.state.step_count
    obs = env.step(ArenaAction(action_type="Ally", target_id="Agent_B", amount=0))
    after = env.state.step_count

    assert obs.reward == 30.0
    assert obs.reputation_scores["Agent_B"] == 0.1
    assert env.state.reputation_matrix["Agent_B"]["Agent_A"] == 0.1
    assert before == 0
    assert after == 1


def test_unsupported_action_does_not_consume_turn() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    before = env.state.step_count
    obs = env.step(ArenaAction(action_type="Idle", amount=0))
    after = env.state.step_count

    assert obs.reward == -10.0
    assert before == 0
    assert after == 0


def test_unknown_player_returns_safe_error_observation() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    obs = env.step(ArenaAction(action_type="Gather", amount=0), player_id="Ghost")

    assert obs.reward == -10.0
    assert obs.resources == 0
    assert set(obs.reputation_scores.keys()) == {"Agent_A", "Agent_B", "Agent_C"}


def test_terminal_guard_returns_neutral_reward_and_no_mutation() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)
    env.state.step_count = env.state.max_rounds

    obs = env.step(ArenaAction(action_type="Gather", amount=0))

    assert obs.done is True
    assert obs.reward == 0.0
    assert env.state.step_count == env.state.max_rounds


def test_legacy_action_payload_is_handled_without_crash() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    before = env.state.step_count
    legacy_action: Any = StartoneAction(message="legacy")
    obs = env.step(legacy_action)
    after = env.state.step_count

    assert obs.reward == -10.0
    assert "missing action_type" in obs.message
    assert before == after


def test_telemetry_defaults_initialized_on_reset() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    telemetry = env.state.telemetry["Agent_A"]
    assert telemetry["initial_resources"] == 100.0
    assert telemetry["successful_trades"] == 0.0
    assert telemetry["betrayals_initiated"] == 0.0
    assert telemetry["alliance_streak_steps"] == 0.0
    assert telemetry["global_decline_ratio"] == 0.0


def test_telemetry_trade_counter_increments() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    env.step(ArenaAction(action_type="Trade", target_id="Agent_B", amount=10))

    telemetry = env.state.telemetry["Agent_A"]
    assert telemetry["successful_trades"] == 1.0
    assert telemetry["betrayals_initiated"] == 0.0
    assert telemetry["alliance_streak_steps"] == 0.0


def test_telemetry_betrayal_counter_increments() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    env.step(ArenaAction(action_type="Betray", target_id="Agent_B", amount=0))

    telemetry = env.state.telemetry["Agent_A"]
    assert telemetry["successful_trades"] == 0.0
    assert telemetry["betrayals_initiated"] == 1.0
    assert telemetry["alliance_streak_steps"] == 0.0


def test_telemetry_ally_counter_increments() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    env.step(ArenaAction(action_type="Ally", target_id="Agent_B", amount=0))

    telemetry = env.state.telemetry["Agent_A"]
    assert telemetry["successful_trades"] == 0.0
    assert telemetry["betrayals_initiated"] == 0.0
    assert telemetry["alliance_streak_steps"] == 1.0


def test_alliance_streak_resets_after_non_ally_action() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    env.step(ArenaAction(action_type="Ally", target_id="Agent_B", amount=0))
    env.step(ArenaAction(action_type="Ally", target_id="Agent_B", amount=0))
    assert env.state.telemetry["Agent_A"]["alliance_streak_steps"] == 2.0

    env.step(ArenaAction(action_type="Gather", amount=0))
    assert env.state.telemetry["Agent_A"]["alliance_streak_steps"] == 0.0


def test_alliance_streak_resets_on_failed_ally_attempt() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    env.step(ArenaAction(action_type="Ally", target_id="Agent_B", amount=0))
    assert env.state.telemetry["Agent_A"]["alliance_streak_steps"] == 1.0

    env.step(ArenaAction(action_type="Ally", target_id="Agent_A", amount=0))
    assert env.state.telemetry["Agent_A"]["alliance_streak_steps"] == 0.0


def test_telemetry_gather_does_not_increment_social_counters() -> None:
    env = ArenaEnvironment()
    env.reset(seed=1)

    env.step(ArenaAction(action_type="Gather", amount=0))

    telemetry = env.state.telemetry["Agent_A"]
    assert telemetry["successful_trades"] == 0.0
    assert telemetry["betrayals_initiated"] == 0.0
    assert telemetry["alliance_streak_steps"] == 0.0
