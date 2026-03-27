"""Regression checks for market model contracts and compatibility aliases."""

from startone.models import (
    ArenaAction,
    ArenaObservation,
    ArenaState,
    MarketAction,
    MarketObservation,
    MarketState,
)


def test_market_action_execute_contract_shape() -> None:
    action = MarketAction(action_type="Execute_Contract", target_id="Firm_A", amount=10)
    assert action.action_type == "Execute_Contract"
    assert action.target_id == "Firm_A"
    assert action.amount == 10


def test_market_observation_fields() -> None:
    obs = MarketObservation(
        done=False,
        reward=10.0,
        capital=100,
        trust_scores={"Firm_B": 0.1},
        active_partnerships=[],
        message="Test",
    )
    assert obs.capital == 100
    assert obs.trust_scores == {"Firm_B": 0.1}
    assert obs.done is False
    assert obs.reward == 10.0


def test_market_state_fields() -> None:
    state = MarketState(
        episode_id="test",
        step_count=5,
        firm_capital={"Firm_A": 100, "Firm_B": 100},
        firm_strategies={"Firm_A": "test", "Firm_B": "test"},
        trust_matrix={"Firm_A": {"Firm_B": 0.0}, "Firm_B": {"Firm_A": 0.0}},
        telemetry={},
        max_rounds=100,
    )
    assert state.firm_capital["Firm_A"] == 100
    assert state.firm_strategies["Firm_A"] == "test"
    assert state.step_count == 5


def test_arena_aliases_map_to_market_models() -> None:
    arena_action = ArenaAction(action_type="Execute_Contract", target_id="Firm_A", amount=10)
    arena_obs = ArenaObservation(
        done=False,
        reward=0.0,
        capital=100,
        trust_scores={"Firm_B": 0.0},
        active_partnerships=[],
        message="ok",
    )
    arena_state = ArenaState(
        episode_id="alias-test",
        step_count=0,
        firm_capital={"Firm_A": 100},
        firm_strategies={"Firm_A": "Cooperative"},
        trust_matrix={"Firm_A": {}},
        telemetry={},
        max_rounds=100,
    )

    assert isinstance(arena_action, MarketAction)
    assert isinstance(arena_obs, MarketObservation)
    assert isinstance(arena_state, MarketState)
