"""
Data models for the B2B Market Economic Simulation.
Defines the Pydantic contracts for corporate actions, observations, and market state.
"""

import math
import uuid
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Try to import openenv types, fallback to Pydantic BaseModel if not available
try:
    from openenv.core.env_server.types import Action, Observation, State  # type: ignore
except ImportError:
    # Fallback base classes when openenv is not installed
    # Define as empty classes inheriting from BaseModel
    class Action(BaseModel):  # type: ignore
        """Fallback Action base class when openenv is not available."""
        pass
    
    class Observation(BaseModel):  # type: ignore
        """Fallback Observation base class when openenv is not available."""
        pass
    
    class State(BaseModel):  # type: ignore
        """Fallback State base class when openenv is not available."""
        pass

# --- MARKET CONSTANTS & TYPES ---

# Strict definition of allowed corporate strategic moves
ActionType = Literal["Produce", "Execute_Contract", "Form_Partnership", "Breach_Contract", "Hostile_Acquisition", "Hold"]

# --- CORE MARKET MODELS ---

class MarketAction(Action):
    """Strategic move issued by a corporate entity in the B2B market.
    
    Represents a single business decision. The environment routes this to the
    appropriate handler (Execute_Contract, Form_Partnership, Breach_Contract) or
    rejects it with a penalty observation if the action is malformed.
    
    Valid action combinations:
    - Produce, Hold: No target or amount.
    - Execute_Contract, Form_Partnership, Breach_Contract, Hostile_Acquisition: Require target_id;
      Execute_Contract also requires amount > 0.
    """

    model_config = ConfigDict(extra="forbid")
    
    action_type: ActionType = Field(
        ..., 
        description="One of: 'Produce' (generate capital internally), 'Execute_Contract' (B2B exchange), "
                    "'Form_Partnership' (strategic alliance), 'Breach_Contract' (contract violation), "
                    "'Hostile_Acquisition' (acquire competitor), 'Hold' (wait). "
                    "Only Produce/Execute_Contract/Form_Partnership/Breach_Contract are routed."
    )
    target_id: Optional[str] = Field(
        default=None, 
        description="Target firm ID for contract/partnership/acquisition actions. "
                    "Must be None for Produce/Hold."
    )
    amount: int = Field(
        default=0, 
        ge=0, 
        description="Capital quantity for Execute_Contract. Must be 0 for partnerships/breach/hold. "
                    "Unused for acquisitions."
    )

    @model_validator(mode="after")
    def validate_action_consistency(self) -> "MarketAction":
        """Enforce action-specific rules so invalid combinations are rejected early.
        
        NOTE: This validator is lenient to allow for legacy/malformed action payloads
        to reach step() for graceful error handling. Strict validation should occur
        in the environment's step() method.
        
        This validator ensures:
        1. Social/hostile actions require a target_id (when target_id is provided).
        2. Solo actions (Produce, Hold) should not have a target_id.
        3. Execute_Contract requires positive amount (when amount is positive).
        4. Partnership/Breach actions should use amount = 0.
        
        Returns:
            Self, or raises ValueError only for clearly malformed data.
        """
        requires_target = {"Execute_Contract", "Form_Partnership", "Breach_Contract", "Hostile_Acquisition"}
        no_target = {"Produce", "Hold"}

        # Only enforce no_target rule for Produce/Hold (these should never have a target)
        if self.action_type in no_target and self.target_id is not None:
            raise ValueError(f"{self.action_type} cannot include target_id")

        # Only enforce amount > 0 for Execute_Contract if amount is explicitly positive
        if self.action_type == "Execute_Contract" and self.amount > 0 and not self.target_id:
            raise ValueError("Execute_Contract requires amount > 0 and a target_id")

        # Partnership/Breach must use amount = 0
        if self.action_type in {"Form_Partnership", "Breach_Contract"} and self.amount != 0:
            raise ValueError(f"{self.action_type} must use amount = 0")

        return self


class MarketObservation(Observation):
    """Per-turn perception packet for a corporate entity in the market.
    
    This is the observation returned by step() and reset(). It includes the firm's
    local view of capital, trust scores, and episode status (done, reward).
    The message field provides human-readable feedback (e.g., contract execution, breaches).
    """

    model_config = ConfigDict(extra="forbid")
    
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated (max steps reached or terminal condition met)."
    )
    reward: float = Field(
        default=0.0,
        description="Reward signal for the current step. Used for reinforcement learning evaluation."
    )
    capital: int = Field(
        ..., 
        ge=0, 
        description="Firm's current capital balance (non-negative integer)."
    )
    trust_scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Per-firm market trust scores normalized to [-1.0, 1.0]. "
                    "Positive = trusted partner, Negative = unreliable. Excludes self-trust."
    )
    active_partnerships: List[str] = Field(
        default_factory=list, 
        description="List of firm IDs currently in strategic partnership with this firm."
    )
    message: str = Field(
        default="", 
        description="Market feedback: Contract confirmation, breach notifications, partnership updates."
    )

    @field_validator("trust_scores")
    @classmethod
    def validate_trust_scores(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Validate trust scores are numeric and finite.
        
        Checks that each trust score:
        1. Is finite (not NaN or Inf).
        2. Falls within the normalized range [-1.0, 1.0].
        
        Returns the validated dict or raises ValueError.
        """
        for firm_id, score in value.items():
            if not math.isfinite(score):
                raise ValueError(f"trust score for '{firm_id}' must be finite")
            if score < -1.0 or score > 1.0:
                raise ValueError(
                    f"trust score for '{firm_id}' must be between -1.0 and 1.0"
                )
        return value


class MarketState(State):
    """Hidden server-side metadata for market episode management.
    
    This is the internal persistent state maintained by the market throughout
    an episode. It is NOT exposed to firms (they only see Observations) but is
    used by evaluators to compute task scores. Accessible via the .state property
    of MarketEnvironment.
    
    Fields:
    - episode_id: Unique identifier for this market episode.
    - step_count: Elapsed turns in current market episode.
    - firm_capital: Current capital per firm.
    - firm_strategies: Bot strategy types (e.g., 'Aggressive', 'Conservative', 'Cooperative').
    - trust_matrix: Directed trust graph (firm -> {other_firm -> score}).
    - telemetry: Raw event counters per firm (contracts, breaches, partnership streaks).
    - max_rounds: Episode termination threshold (default 100 steps).
    """

    model_config = ConfigDict(extra="forbid")
    
    episode_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this market episode. Used to track multiple independent runs."
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of elapsed steps in the current episode. Incremented after each action execution."
    )
    firm_capital: Dict[str, int] = Field(
        default_factory=dict,
        description="Current capital balance for each firm. Used to validate contract/breach feasibility."
    )
    firm_strategies: Dict[str, str] = Field(
        default_factory=dict, 
        description="Hidden strategy type per firm (e.g., 'Aggressive', 'Conservative'). "
                    "Guides bot behavior in future versions."
    )
    trust_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Directed trust graph: firm -> {target_firm -> trust_score in [-1.0, 1.0]}. "
                    "Used for partnership decisions and fallback scoring."
    )
    telemetry: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-firm event counters: successful_contracts, contracts_breached, partnership_streak_steps, "
                    "initial_capital, market_decline_ratio. Primary input to evaluators."
    )
    max_rounds: int = Field(
        default=100, 
        gt=0,
        description="Maximum steps before market termination. Matches MAX_EPISODE_STEPS in market environment."
    )

    @field_validator("firm_capital")
    @classmethod
    def validate_firm_capital(cls, value: Dict[str, int]) -> Dict[str, int]:
        """Verify all firm capital balances are non-negative.
        
        Prevents accounting errors where a firm's balance could fall below 0.
        This is a defensive check; handlers should enforce this before mutation.
        Returns the validated dict or raises ValueError.
        """
        for firm_id, capital in value.items():
            if capital < 0:
                raise ValueError(f"firm_capital['{firm_id}'] cannot be negative")
        return value


# --- LEGACY COMPATIBILITY MODELS ---
# Note: These are retained to prevent legacy code from breaking during migration.

class ArenaAction(MarketAction):
    """Backward-compatible alias for MarketAction."""
    pass

class ArenaObservation(MarketObservation):
    """Backward-compatible alias for MarketObservation."""
    
    @property
    def resources(self) -> int:
        """Legacy field: resources → capital."""
        return self.capital
    
    @property
    def reputation_scores(self) -> Dict[str, float]:
        """Legacy field: reputation_scores → trust_scores."""
        return self.trust_scores
    
    @property
    def active_alliances(self) -> List[str]:
        """Legacy field: active_alliances → active_partnerships."""
        return self.active_partnerships

class ArenaState(MarketState):
    """Backward-compatible alias for MarketState."""
    
    @property
    def agent_resources(self) -> Dict[str, int]:
        """Legacy field: agent_resources → firm_capital."""
        return self.firm_capital
    
    @property
    def agent_personalities(self) -> Dict[str, str]:
        """Legacy field: agent_personalities → firm_strategies."""
        return self.firm_strategies
    
    @property
    def reputation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Legacy field: reputation_matrix → trust_matrix."""
        return self.trust_matrix

class StartoneAction(MarketAction):
    """Backward-compatible alias for MarketAction."""
    pass

class StartoneObservation(MarketObservation):
    """Backward-compatible alias for MarketObservation."""
    pass