"""
Evaluation graders for the AI Negotiation Arena.
Translates raw environment state and telemetry into normalized 0.0 - 1.0 scores.

Grading Philosophy:
- Each task measures a specific capability: resource gathering, cooperation, negotiation.
- All scores are normalized to [0.0, 1.0], where 1.0 = perfect task execution.
- Scores are deterministic given state and telemetry (reproducible evaluation).

Telemetry Policy:
- Task 1 (Resource Scavenger) and Task 2 (Honest Trader) are TELEMETRY-FIRST.
  If required telemetry is missing, scores degrade to strict defaults (often 0.0).
  This prevents hidden heuristic assumptions and keeps grading transparent.
- Task 3 (Master Negotiator) supports TELEMETRY-FIRST scoring with a bounded
  REPUTATION FALLBACK when telemetry is unavailable (graceful degradation).

Scoring Targets:
- Resource Scavenger: 50 steps survival + 50 resource gain (70% survival, 30% wealth).
- Honest Trader: 3 successful trades with zero betrayals.
- Master Negotiator: 30-step alliance streak + scarcity resilience (60/40 split).
"""

import math
from typing import Mapping, Optional
from ..models import ArenaState

class ArenaGraders:
    """Deterministic task graders returning strict scores in the range [0.0, 1.0].

    Static methods implement grading logic for 3 difficulty tiers:
    - grade_resource_scavenger: EASY - Test basic survival and resource gathering.
    - grade_honest_trader: MEDIUM - Test cooperation without betrayal temptation.
    - grade_master_negotiator: HARD - Test alliance maintenance under scarcity.
    
    All graders take ArenaState (full episode record) and optional telemetry dict,
    returning a normalized score. Scores reflect task progress toward defined targets.
    
    Helper methods:
    - _clamp01(value): Safely normalize any float to [0.0, 1.0], guarding against NaN/Inf.
    - _telemetry_value(telemetry, key, default): Safe extraction with graceful fallback.
    
    Telemetry policy:
    - `grade_resource_scavenger` requires `initial_resources` telemetry.
    - `grade_honest_trader` expects explicit trade/betrayal telemetry.
    - `grade_master_negotiator` supports telemetry-first scoring with a bounded
      reputation fallback when telemetry is unavailable.
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
    def grade_resource_scavenger(
        state: ArenaState,
        player_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """TASK 1 (EASY): Resource Scavenger - Survive & Gather.
        
        Objective:
        Survive for 50 steps without elimination and increase wealth by 50+ resources.
        
        Scoring Formula:
        - survival_ratio = min(current_steps / 50, 1.0) [0-50 steps -> 0.0-1.0]
        - resource_gain = max(0, current_resources - initial_resources)
        - gain_ratio = min(resource_gain / 50, 1.0) [0-50 gain -> 0.0-1.0]
        - final_score = (0.70 * survival_ratio) + (0.30 * gain_ratio)
        
        Telemetry Requirement (STRICT):
        - Requires `initial_resources` telemetry key.
        - Missing initial_resources -> score = 0.0 (fails hard, no fallback).
        - This enforces deterministic grading (transparent dependencies).
        
        Args:
            state: ArenaState with agent_resources and step_count.
            player_id: Agent ID to grade.
            telemetry: Optional per-agent telemetry dict with 'initial_resources'.
            
        Returns:
            float: Normalized score [0.0, 1.0].
        """
        survival_steps = float(state.step_count)
        survival_ratio = ArenaGraders._clamp01(
            survival_steps / ArenaGraders.TARGET_SURVIVAL_STEPS
        )

        initial_resources = ArenaGraders._telemetry_value(
            telemetry,
            "initial_resources",
            -1.0,
        )
        if initial_resources < 0:
            return 0.0

        current_resources = float(state.agent_resources.get(player_id, 0))
        resource_gain = max(0.0, current_resources - initial_resources)
        gain_ratio = ArenaGraders._clamp01(
            resource_gain / ArenaGraders.TARGET_RESOURCE_GAIN
        )

        return (0.7 * survival_ratio) + (0.3 * gain_ratio)

    @staticmethod
    def grade_honest_trader(
        state: ArenaState,
        player_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """TASK 2 (MEDIUM): Honest Trader - Cooperate Without Betrayal.
        
        Objective:
        Complete 3 successful trades with ZERO betrayals initiated throughout episode.
        
        Scoring Formula:
        - successful_trades_ratio = successful_trades / 3.0
        - If betrayals_initiated > 0: score = 0.0 (instant fail, non-recoverable).
        - Otherwise: score = min(successful_trades_ratio, 1.0)
        
        Telemetry Requirement (STRICT):
        - Requires `successful_trades` and `betrayals_initiated` counters.
        - Missing/invalid telemetry defaults to 0 trades (fails task).
        - Single betrayal anywhere in episode -> 0.0 (hard constraint).
        
        Semantics:
        This task tests whether agents can resist short-term temptation (betrayal reward)
        to achieve long-term cooperative goals. The zero-tolerance betrayal policy
        mirrors real-world contract violations.
        
        Args:
            state: ArenaState (unused, kept for API consistency).
            player_id: Agent ID to grade (unused, kept for API consistency).
            telemetry: Optional per-agent telemetry dict with trade/betrayal counters.
            
        Returns:
            float: Normalized score [0.0, 1.0]. Zero if any betrayal detected.
        """
        successful_trades = ArenaGraders._telemetry_value(
            telemetry,
            "successful_trades",
            0.0,
        )
        betrayals_initiated = ArenaGraders._telemetry_value(
            telemetry,
            "betrayals_initiated",
            0.0,
        )

        if betrayals_initiated > 0:
            return 0.0

        return ArenaGraders._clamp01(
            successful_trades / ArenaGraders.TARGET_SUCCESSFUL_TRADES
        )

    @staticmethod
    def grade_diplomat(
        state: ArenaState,
        player_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Backward-compatible alias for grade_honest_trader.
        
        Retained for legacy task naming during migration.
        Delegates directly to grade_honest_trader with same args/returns.
        """
        return ArenaGraders.grade_honest_trader(state, player_id, telemetry)

    @staticmethod
    def grade_master_negotiator(
        state: ArenaState,
        player_id: str,
        telemetry: Optional[Mapping[str, float]] = None,
    ) -> float:
        """TASK 3 (HARD): Master Negotiator - Alliance Under Pressure.
        
        Objective:
        Maintain alliance stability (30+ contiguous ally turns) while the economy
        deteriorates due to betrayals by other agents.
        
        Scoring Formula (Telemetry-First Path):
        IF alliance_streak_steps > 0 OR global_decline_ratio > 0:
            - alliance_score = 0.60 * min(alliance_streak / 30, 1.0)
            - scarcity_score = 0.40 * decline_ratio
            - final_score = clamp(alliance_score + scarcity_score, 0.0, 1.0)
        
        Scoring Formula (Reputation Fallback Path):
        ELSE (no telemetry):
            - avg_reputation = mean(reputation_scores[*])
            - trust_ratio = (avg_reputation + 1.0) / 2.0  [maps [-1, 1] -> [0, 1]]
            - final_score = 0.60 * clamp(trust_ratio, 0.0, 1.0)
        
        Telemetry Requirement (GRACEFUL):
        - Prefers `alliance_streak_steps` and `global_decline_ratio`.
        - If missing, falls back to reputation matrix (bounded trust metric).
        - Never returns 0.0 unless reputation is catastrophically negative.
        
        Semantics:
        This task tests high-level negotiation skills:
        1. Building alliance partnerships early (alliance_streak).
        2. Adapting cooperation strategies to scarcity (decline_ratio).
        3. Using reputation as a strategic signal (fallback path).
        
        Args:
            state: ArenaState with reputation_matrix and telemetry.
            player_id: Agent ID to grade.
            telemetry: Optional per-agent telemetry dict with alliance/decline metrics.
            
        Returns:
            float: Normalized score [0.0, 1.0].
        """
        alliance_streak = ArenaGraders._telemetry_value(
            telemetry,
            "alliance_streak_steps",
            0.0,
        )
        decline_ratio = ArenaGraders._telemetry_value(
            telemetry,
            "global_decline_ratio",
            0.0,
        )
        if alliance_streak > 0 or decline_ratio > 0:
            alliance_score = 0.6 * ArenaGraders._clamp01(
                alliance_streak / ArenaGraders.TARGET_ALLIANCE_STREAK
            )
            scarcity_score = 0.4 * ArenaGraders._clamp01(decline_ratio)
            return ArenaGraders._clamp01(alliance_score + scarcity_score)

        player_reps = state.reputation_matrix.get(player_id, {})
        if not player_reps:
            trust_ratio = 0.0
        else:
            avg_rep = sum(player_reps.values()) / len(player_reps)
            trust_ratio = ArenaGraders._clamp01((avg_rep + 1.0) / 2.0)

        return ArenaGraders._clamp01(0.6 * trust_ratio)