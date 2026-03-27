"""
Data models for the AI Negotiation Arena.
Defines the Pydantic contracts for actions, observations, and environment state.
"""

import math
from typing import Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator
from openenv.core.env_server.types import Action, Observation, State

# --- ARENA CONSTANTS & TYPES ---

# Strict definition of allowed strategic moves
ActionType = Literal["Gather", "Trade", "Ally", "Betray", "Attack", "Idle"]

# --- CORE ARENA MODELS ---

class ArenaAction(Action):
    """Strategic move issued by an arena agent.
    
    Represents a single decision in the negotiation arena. The environment routes
    this to the appropriate handler (Trade, Ally, Betray) or rejects it with a
    penalty observation if the action is malformed.
    
    Valid action combinations:
    - Gather, Idle: No target or amount.
    - Trade, Ally, Betray, Attack: Require target_id; Trade also requires amount > 0.
    """

    model_config = ConfigDict(extra="forbid")
    
    action_type: ActionType = Field(
        ..., 
        description="One of: 'Gather' (collect resources), 'Trade' (exchange), 'Ally' (cooperate), "
                    "'Betray' (steal), 'Attack' (conflict), 'Idle' (wait). Only Gather/Trade/Ally/Betray are routed."
    )
    target_id: Optional[str] = Field(
        default=None, 
        description="Target agent ID for social or conflict actions (Trade, Ally, Betray, Attack). "
                    "Must be None for Gather/Idle."
    )
    amount: int = Field(
        default=0, 
        ge=0, 
        description="Resource quantity for Trade. Must be 0 for Ally/Betray/Idle. Unused for Gather/Attack."
    )

    @model_validator(mode="after")
    def validate_action_consistency(self) -> "ArenaAction":
        """Enforce action-specific rules so invalid combinations are rejected early.
        
        This validator ensures:
        1. Social/conflict actions (Trade, Ally, Betray, Attack) have a target_id.
        2. Passive/solo actions (Gather, Idle) have no target_id.
        3. Trade requires a positive amount.
        4. Social actions (Ally, Betray, Idle) have amount = 0.
        
        Returns:
            Self, or raises ValueError if constraints violated.
        """
        requires_target = {"Trade", "Ally", "Betray", "Attack"}
        no_target = {"Gather", "Idle"}

        if self.action_type in requires_target and not self.target_id:
            raise ValueError(f"{self.action_type} requires a target_id")

        if self.action_type in no_target and self.target_id is not None:
            raise ValueError(f"{self.action_type} cannot include target_id")

        if self.action_type == "Trade" and self.amount <= 0:
            raise ValueError("Trade requires amount > 0")

        if self.action_type in {"Ally", "Betray", "Idle"} and self.amount != 0:
            raise ValueError(f"{self.action_type} must use amount = 0")

        return self


class ArenaObservation(Observation):
    """Per-turn perception packet for an arena agent.
    
    This is the observation returned by step() and reset(). It includes the agent's
    local view of resources, trust scores, and episode status (done, reward).
    The message field provides human-readable feedback (e.g., error descriptions,
    turn completion info).
    """

    model_config = ConfigDict(extra="forbid")
    
    resources: int = Field(
        ..., 
        ge=0, 
        description="Agent's current resource balance (non-negative integer)."
    )
    reputation_scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Per-agent trust/reputation scores normalized to [-1.0, 1.0]. "
                    "Positive = trust, Negative = distrust. Excludes self-trust."
    )
    active_alliances: List[str] = Field(
        default_factory=list, 
        description="List of agent IDs currently in alliance with this agent. Reserved for future use."
    )
    message: str = Field(
        default="", 
        description="Server feedback: Turn confirmation, error descriptions, or game state updates."
    )

    @field_validator("reputation_scores")
    @classmethod
    def validate_reputation_scores(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Validate reputation values are numeric and finite.
        
        Checks that each reputation score:
        1. Is finite (not NaN or Inf).
        2. Falls within the normalized range [-1.0, 1.0].
        
        This ensures graders and comparisons work without special cases.
        Returns the validated dict or raises ValueError.
        """
        for agent_id, score in value.items():
            if not math.isfinite(score):
                raise ValueError(f"reputation score for '{agent_id}' must be finite")
            if score < -1.0 or score > 1.0:
                raise ValueError(
                    f"reputation score for '{agent_id}' must be between -1.0 and 1.0"
                )
        return value


class ArenaState(State):
    """Hidden server-side metadata for episode management.
    
    This is the internal persistent state maintained by the environment throughout
    an episode. It is NOT exposed to agents (they only see Observations) but is
    used by graders to compute task scores. Accessible via the .state property
    of ArenaEnvironment.
    
    Fields:
    - agent_resources: Current wealth per agent.
    - agent_personalities: Bot personality traits (e.g., 'Cooperative', 'Aggressive').
    - reputation_matrix: Directed trust graph (agent -> {other_agent -> score}).
    - telemetry: Raw event counters per agent (trades, betrayals, alliance streaks).
    - step_count: Elapsed turns in current episode.
    - max_rounds: Episode termination threshold (default 100 steps).
    """

    model_config = ConfigDict(extra="forbid")
    
    agent_resources: Dict[str, int] = Field(
        default_factory=dict,
        description="Current resource balance for each agent. Used to validate Trade/Betray feasibility."
    )
    agent_personalities: Dict[str, str] = Field(
        default_factory=dict, 
        description="Hidden personality type per agent (e.g., 'Aggressive', 'Cooperative'). "
                    "Guides bot behavior in future versions."
    )
    reputation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Directed trust graph: agent -> {target -> trust_score in [-1.0, 1.0]}. "
                    "Used for alliance decisions and fallback scoring."
    )
    telemetry: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-agent event counters: successful_trades, betrayals_initiated, alliance_streak_steps, "
                    "initial_resources, global_decline_ratio. Primary input to graders."
    )
    max_rounds: int = Field(
        default=100, 
        gt=0,
        description="Maximum steps before episode termination. Matches MAX_EPISODE_STEPS in environment."
    )

    @field_validator("agent_resources")
    @classmethod
    def validate_agent_resources(cls, value: Dict[str, int]) -> Dict[str, int]:
        """Verify all agent resource balances are non-negative.
        
        Prevents bookkeeping errors where an agent's wealth could fall below 0.
        This is a defensive check; handlers should enforce this before mutation.
        Returns the validated dict or raises ValueError.
        """
        for agent_id, resources in value.items():
            if resources < 0:
                raise ValueError(f"agent_resources['{agent_id}'] cannot be negative")
        return value


# --- LEGACY COMPATIBILITY MODELS ---
# Note: These are retained to prevent 'client.py' and 'app.py' from breaking 
# during the initial refactor phase.

class StartoneAction(Action):
    """Temporary echo action for template compatibility."""
    message: str = Field(..., description="Message to echo back")


class StartoneObservation(Observation):
    """Temporary echo observation for template compatibility."""
    echoed_message: str = Field(default="")
    message_length: int = Field(default=0, ge=0)