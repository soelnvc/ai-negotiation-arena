"""
Evaluation graders for the B2B Market Economic Simulation.
Translates raw environment state and telemetry into normalized 0.0 - 1.0 scores.

Grading Philosophy:
- Each task measures a specific capability: capital accumulation, cooperation, negotiation.
- All scores are normalized to [0.0, 1.0], where 1.0 = perfect task execution.
- Scores are deterministic given state and telemetry (reproducible evaluation).

Telemetry Policy:
- Task 1 (Capital Accumulator) and Task 2 (Reliable Partner) are TELEMETRY-FIRST.
  If required telemetry is missing, scores degrade to strict defaults (often 0.0).
  This prevents hidden heuristic assumptions and keeps grading transparent.
- Task 3 (Strategic Alliance Master) supports TELEMETRY-FIRST scoring with a bounded
  TRUST FALLBACK when telemetry is unavailable (graceful degradation).

Scoring Targets:
- Capital Accumulator: 50 steps survival + 50 capital gain (70% survival, 30% wealth).
- Reliable Partner: 3 successful contracts with zero breaches.
- Strategic Alliance Master: 30-step partnership streak + scarcity resilience (60/40 split).
"""

import math
from typing import Mapping, Optional
from ..models import MarketState

class MarketGraders:
    """Deterministic task graders returning strict scores in the range [0.0, 1.0].

    Static methods implement grading logic for 3 difficulty tiers:
    - grade_capital_accumulator: EASY - Test basic survival and capital accumulation.
    - grade_reliable_partner: MEDIUM - Test cooperation without breach temptation.
    - grade_strategic_alliance_master: HARD - Test partnership maintenance under scarcity.
    
    All graders take MarketState (full episode record) and optional telemetry dict,
    returning a normalized score. Scores reflect task progress toward defined targets.
    
    Helper methods:
    - _clamp01(value): Safely normalize any float to [0.0, 1.0], guarding against NaN/Inf.
    - _telemetry_value(telemetry, key, default): Safe extraction with graceful fallback.
    
    Telemetry policy:
    - `grade_capital_accumulator` requires `initial_capital` telemetry.
    - `grade_reliable_partner` expects explicit contract/breach telemetry.
    - `grade_strategic_alliance_master` supports telemetry-first scoring with a bounded
      trust fallback when telemetry is unavailable.
    """

    TARGET_SURVIVAL_STEPS = 50.0
    TARGET_RESOURCE_GAIN = 50.0
    TARGET_SUCCESSFUL_TRADES = 3.0
    TARGET_ALLIANCE_STREAK = 30.0
    TARGET_WEALTH = 300.0

    @staticmethod
    def _clamp01(value: float) -> float:
        """Ensure score is strictly [0.0, 1.0] and guard against NaN/Inf.
        
        Behavior:
        - None, NaN, Inf, or negative values -> 0.0 (strict).
        - Values > 1.0 -> 1.0 (clamp).
        - Regular floats -> returned as-is if in [0.0, 1.0].
        
        Args:
            value: Any float value.
            
        Returns:
            float: Normalized value in [0.0, 1.0].
        """
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp01_strict(value: float) -> float:
        """Ensure score is strictly (0.0, 1.0) - open interval, never boundary values.
        
        Required by OpenEnv Phase 2 validation: scores must be strictly between 0 and 1,
        never exactly 0.0 or 1.0.
        
        Behavior:
        - None, NaN, Inf, or negative values -> 0.01 (epsilon buffer).
        - Values > 1.0 -> 0.99 (epsilon buffer).
        - Value == 0.0 -> 0.01
        - Value == 1.0 -> 0.99
        - Otherwise -> returned as-is if in (0.0, 1.0)
        
        Args:
            value: Any float value.
            
        Returns:
            float: Normalized value strictly within (0.0, 1.0).
        """
        EPSILON = 0.01
        if value is None or math.isnan(value) or math.isinf(value):
            return EPSILON
        clamped = max(0.0, min(1.0, value))
        if clamped <= 0.0:
            return EPSILON
        if clamped >= 1.0:
            return 1.0 - EPSILON
        return clamped

    @staticmethod
    def _telemetry_value(
        telemetry: Optional[Mapping[str, float]],
        key: str,
        default: float,
    ) -> float:
        """Safely extract and cast telemetry data with graceful fallback.
        
        Behavior:
        1. If telemetry is None, return default.
        2. If key missing in telemetry, return default.
        3. Try to cast value to float; if fails, return default.
        4. Otherwise return float value.
        
        Args:
            telemetry: Optional dict of telemetry counters.
            key: Telemetry key to extract (e.g., 'successful_trades').
            default: Fallback value if key missing or conversion fails.
            
        Returns:
            float: Extracted value or default.
        """
        if telemetry is None:
            return default
        raw = telemetry.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def grade_capital_accumulator(
        state: MarketState,
        actor_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """TASK 1 (EASY): Capital Accumulator - Survive & Accumulate.
        
        Objective:
        Survive for 50 steps without elimination and increase capital by 50+ assets.
        
        Scoring Formula:
        - survival_ratio = min(current_steps / 50, 1.0) [0-50 steps -> 0.0-1.0]
        - capital_gain = max(0, current_capital - initial_capital)
        - gain_ratio = min(capital_gain / 50, 1.0) [0-50 gain -> 0.0-1.0]
        - final_score = (0.70 * survival_ratio) + (0.30 * gain_ratio)
        
        Telemetry Requirement (STRICT):
        - Requires `initial_capital` telemetry key.
        - Missing initial_capital -> score = 0.0 (fails hard, no fallback).
        - This enforces deterministic grading (transparent dependencies).
        
        Args:
            state: MarketState with firm_capital and step_count.
            actor_id: Firm ID to grade.
            telemetry: Optional per-firm telemetry dict with 'initial_capital'.
            
        Returns:
            float: Normalized score [0.0, 1.0].
        """
        survival_steps = float(state.step_count)
        survival_ratio = MarketGraders._clamp01(
            survival_steps / MarketGraders.TARGET_SURVIVAL_STEPS
        )

        initial_capital = MarketGraders._telemetry_value(
            telemetry,
            "initial_capital",
            -1.0,
        )
        if initial_capital < 0:
            return 0.0

        current_capital = float(state.firm_capital.get(actor_id, 0))
        capital_gain = max(0.0, current_capital - initial_capital)
        gain_ratio = MarketGraders._clamp01(
            capital_gain / MarketGraders.TARGET_RESOURCE_GAIN
        )

        return MarketGraders._clamp01_strict(
            (0.7 * survival_ratio) + (0.3 * gain_ratio)
        )

    @staticmethod
    def grade_reliable_partner(
        state: MarketState,
        actor_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """TASK 2 (MEDIUM): Reliable Partner - Cooperate Without Breach.
        
        Objective:
        Complete 3 successful contracts with ZERO breaches initiated throughout episode.
        
        Scoring Formula:
        - successful_contracts_ratio = successful_contracts / 3.0
        - If contracts_breached > 0: score = 0.0 (instant fail, non-recoverable).
        - Otherwise: score = min(successful_contracts_ratio, 1.0)
        
        Telemetry Requirement (STRICT):
        - Requires `successful_contracts` and `contracts_breached` counters.
        - Missing/invalid telemetry defaults to 0 contracts (fails task).
        - Single breach anywhere in episode -> 0.0 (hard constraint).
        
        Semantics:
        This task tests whether firms can resist short-term temptation (breach reward)
        to achieve long-term cooperative goals. The zero-tolerance breach policy
        mirrors real-world contract violations.
        
        Args:
            state: MarketState (unused, kept for API consistency).
            actor_id: Firm ID to grade (unused, kept for API consistency).
            telemetry: Optional per-firm telemetry dict with contract/breach counters.
            
        Returns:
            float: Normalized score [0.0, 1.0]. Zero if any breach detected.
        """
        successful_contracts = MarketGraders._telemetry_value(
            telemetry,
            "successful_contracts",
            0.0,
        )
        contracts_breached = MarketGraders._telemetry_value(
            telemetry,
            "contracts_breached",
            0.0,
        )

        if contracts_breached > 0:
            return MarketGraders._clamp01_strict(0.0)

        return MarketGraders._clamp01_strict(
            successful_contracts / MarketGraders.TARGET_SUCCESSFUL_TRADES
        )

    @staticmethod
    def grade_diplomat(
        state: MarketState,
        actor_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Backward-compatible alias for grade_reliable_partner.
        
        Retained for legacy task naming during migration.
        Delegates directly to grade_reliable_partner with same args/returns.
        """
        return MarketGraders.grade_reliable_partner(state, actor_id, telemetry)

    @staticmethod
    def grade_strategic_alliance_master(
        state: MarketState,
        actor_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """TASK 3 (HARD): Strategic Alliance Master - Partnership Under Pressure.
        
        Objective:
        Maintain partnership stability (30+ contiguous partnership turns) while the economy
        deteriorates due to breaches by other firms.
        
        Scoring Formula (Telemetry-First Path):
        IF partnership_streak_steps > 0 OR market_decline_ratio > 0:
            - partnership_score = 0.60 * min(partnership_streak / 30, 1.0)
            - scarcity_score = 0.40 * market_decline_ratio
            - final_score = clamp(partnership_score + scarcity_score, 0.0, 1.0)
        
        Scoring Formula (Trust Fallback Path):
        ELSE (no telemetry):
            - avg_trust = mean(trust_scores[*])
            - trust_ratio = (avg_trust + 1.0) / 2.0  [maps [-1, 1] -> [0, 1]]
            - final_score = 0.60 * clamp(trust_ratio, 0.0, 1.0)
        
        Telemetry Requirement (GRACEFUL):
        - Prefers `partnership_streak_steps` and `market_decline_ratio`.
        - If missing, falls back to trust matrix (bounded trust metric).
        - Never returns 0.0 unless trust is catastrophically negative.
        
        Semantics:
        This task tests high-level negotiation skills:
        1. Building partnership alliances early (partnership_streak).
        2. Adapting cooperation strategies to scarcity (market_decline_ratio).
        3. Using trust as a strategic signal (fallback path).
        
        Args:
            state: MarketState with trust_matrix and telemetry.
            actor_id: Firm ID to grade.
            telemetry: Optional per-firm telemetry dict with partnership/scarcity metrics.
            
        Returns:
            float: Normalized score [0.0, 1.0].
        """
        partnership_streak = MarketGraders._telemetry_value(
            telemetry,
            "partnership_streak_steps",
            0.0,
        )
        market_decline_ratio = MarketGraders._telemetry_value(
            telemetry,
            "market_decline_ratio",
            0.0,
        )
        if partnership_streak > 0 or market_decline_ratio > 0:
            partnership_score = 0.6 * MarketGraders._clamp01(
                partnership_streak / MarketGraders.TARGET_ALLIANCE_STREAK
            )
            scarcity_score = 0.4 * MarketGraders._clamp01(market_decline_ratio)
            return MarketGraders._clamp01_strict(partnership_score + scarcity_score)

        player_trust = state.trust_matrix.get(actor_id, {})
        if not player_trust:
            trust_ratio = 0.0
        else:
            avg_trust = sum(player_trust.values()) / len(player_trust)
            trust_ratio = MarketGraders._clamp01((avg_trust + 1.0) / 2.0)

        return MarketGraders._clamp01_strict(0.6 * trust_ratio)

    # Legacy method aliases for backward compatibility
    @staticmethod
    def grade_resource_scavenger(state: MarketState, actor_id: str, telemetry: Optional[Mapping[str, float]] = None) -> float:
        """Backward-compatible alias for grade_capital_accumulator."""
        return MarketGraders.grade_capital_accumulator(state, actor_id, telemetry)

    @staticmethod
    def grade_honest_trader(state: MarketState, actor_id: str, telemetry: Optional[Mapping[str, float]] = None) -> float:
        """Backward-compatible alias for grade_reliable_partner."""
        return MarketGraders.grade_reliable_partner(state, actor_id, telemetry)

    @staticmethod
    def grade_master_negotiator(state: MarketState, actor_id: str, telemetry: Optional[Mapping[str, float]] = None) -> float:
        """Backward-compatible alias for grade_strategic_alliance_master."""
        return MarketGraders.grade_strategic_alliance_master(state, actor_id, telemetry)


# Legacy class alias for backward compatibility
class ArenaGraders(MarketGraders):
    """Backward-compatible alias for legacy code using arena terminology.
    
    Retained for compatibility during transition from ArenaGraders naming
    to MarketGraders. Direct subclass with no overrides.
    """
    pass