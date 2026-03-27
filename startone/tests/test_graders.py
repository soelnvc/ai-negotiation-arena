from startone.models import MarketState
from startone.server.graders import MarketGraders


def _base_state() -> MarketState:
    return MarketState(
        episode_id="ep-1",
        step_count=10,
        firm_capital={"Firm_A": 100, "Firm_B": 100, "Firm_C": 100},
        firm_strategies={"Firm_A": "Rational", "Firm_B": "Cooperative", "Firm_C": "Aggressive"},
        trust_matrix={
            "Firm_A": {"Firm_B": 0.0, "Firm_C": 0.0},
            "Firm_B": {"Firm_A": 0.0, "Firm_C": 0.0},
            "Firm_C": {"Firm_A": 0.0, "Firm_B": 0.0},
        },
        max_rounds=100,
    )


def test_grade_capital_accumulator_uses_survival_and_gain() -> None:
    state = _base_state()
    state.step_count = 50
    state.firm_capital["Firm_A"] = 150

    score = MarketGraders.grade_capital_accumulator(
        state,
        "Firm_A",
        telemetry={"initial_capital": 100},
    )

    assert score == 1.0


def test_grade_capital_accumulator_strict_without_initial_capital() -> None:
    state = _base_state()
    state.step_count = 50
    state.firm_capital["Firm_A"] = 150

    score = MarketGraders.grade_capital_accumulator(state, "Firm_A")

    assert score == 0.0


def test_grade_reliable_partner_uses_telemetry_when_present() -> None:
    state = _base_state()

    score = MarketGraders.grade_reliable_partner(
        state,
        "Firm_A",
        telemetry={"successful_contracts": 3, "contracts_breached": 0},
    )

    assert score == 1.0


def test_grade_reliable_partner_fails_if_breach_initiated() -> None:
    state = _base_state()

    score = MarketGraders.grade_reliable_partner(
        state,
        "Firm_A",
        telemetry={"successful_contracts": 3, "contracts_breached": 1},
    )

    assert score == 0.0


def test_grade_diplomat_alias_matches_reliable_partner() -> None:
    state = _base_state()
    telemetry = {"successful_contracts": 2, "contracts_breached": 0}

    a = MarketGraders.grade_reliable_partner(state, "Firm_A", telemetry)
    b = MarketGraders.grade_diplomat(state, "Firm_A", telemetry)

    assert a == b


def test_grade_strategic_alliance_master_telemetry_path() -> None:
    state = _base_state()

    score = MarketGraders.grade_strategic_alliance_master(
        state,
        "Firm_A",
        telemetry={"partnership_streak_steps": 30, "market_decline_ratio": 1.0},
    )

    assert score == 1.0


def test_grade_strategic_alliance_master_fallback_is_clamped() -> None:
    state = _base_state()
    state.trust_matrix["Firm_A"] = {"Firm_B": -1.0, "Firm_C": -1.0}

    score = MarketGraders.grade_strategic_alliance_master(state, "Firm_A")

    assert 0.0 <= score <= 1.0
    assert score == 0.0
