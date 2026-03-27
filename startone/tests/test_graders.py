from startone.models import ArenaState
from startone.server.graders import ArenaGraders


def _base_state() -> ArenaState:
    return ArenaState(
        episode_id="ep-1",
        step_count=10,
        agent_resources={"Agent_A": 100, "Agent_B": 100, "Agent_C": 100},
        agent_personalities={"Agent_A": "Rational", "Agent_B": "Cooperative", "Agent_C": "Aggressive"},
        reputation_matrix={
            "Agent_A": {"Agent_B": 0.0, "Agent_C": 0.0},
            "Agent_B": {"Agent_A": 0.0, "Agent_C": 0.0},
            "Agent_C": {"Agent_A": 0.0, "Agent_B": 0.0},
        },
        max_rounds=100,
    )


def test_grade_resource_scavenger_uses_survival_and_gain() -> None:
    state = _base_state()
    state.step_count = 50
    state.agent_resources["Agent_A"] = 150

    score = ArenaGraders.grade_resource_scavenger(
        state,
        "Agent_A",
        telemetry={"initial_resources": 100},
    )

    assert score == 1.0


def test_grade_resource_scavenger_strict_without_initial_resources() -> None:
    state = _base_state()
    state.step_count = 50
    state.agent_resources["Agent_A"] = 150

    score = ArenaGraders.grade_resource_scavenger(state, "Agent_A")

    assert score == 0.0


def test_grade_honest_trader_uses_telemetry_when_present() -> None:
    state = _base_state()

    score = ArenaGraders.grade_honest_trader(
        state,
        "Agent_A",
        telemetry={"successful_trades": 3, "betrayals_initiated": 0},
    )

    assert score == 1.0


def test_grade_honest_trader_fails_if_betrayal_initiated() -> None:
    state = _base_state()

    score = ArenaGraders.grade_honest_trader(
        state,
        "Agent_A",
        telemetry={"successful_trades": 3, "betrayals_initiated": 1},
    )

    assert score == 0.0


def test_grade_diplomat_alias_matches_honest_trader() -> None:
    state = _base_state()
    telemetry = {"successful_trades": 2, "betrayals_initiated": 0}

    a = ArenaGraders.grade_honest_trader(state, "Agent_A", telemetry)
    b = ArenaGraders.grade_diplomat(state, "Agent_A", telemetry)

    assert a == b


def test_grade_master_negotiator_telemetry_path() -> None:
    state = _base_state()

    score = ArenaGraders.grade_master_negotiator(
        state,
        "Agent_A",
        telemetry={"alliance_streak_steps": 30, "global_decline_ratio": 1.0},
    )

    assert score == 1.0


def test_grade_master_negotiator_fallback_is_clamped() -> None:
    state = _base_state()
    state.reputation_matrix["Agent_A"] = {"Agent_B": -1.0, "Agent_C": -1.0}

    score = ArenaGraders.grade_master_negotiator(state, "Agent_A")

    assert 0.0 <= score <= 1.0
    assert score == 0.0
