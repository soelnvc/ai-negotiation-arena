import uuid
import random
from typing import Dict, List, Optional

# Try to import Environment from openenv, fallback to plain object if not available
try:
    from openenv.core.env_server import Environment  # type: ignore
except ImportError:
    # Fallback: simple base class when openenv is not available
    class Environment:  # type: ignore
        """Fallback Environment base class when openenv is not available."""
        pass

from ..models import MarketAction, MarketObservation, MarketState

class MarketEnvironment(Environment):
    """B2B market economic simulation for testing corporate intelligence.
    
    Overview:
    The Market is a 3-firm economic game where firms manage capital, form
    partnerships, and navigate contract compliance/violation tradeoffs. The
    environment is deterministic given a seed, supporting reproducible evaluation
    of corporate agent strategies.
    
    Corporate Actions (routed):
    - Produce: Generate 10 capital internally, +5 reward.
    - Execute_Contract: Exchange capital for 20% systemic profit, +20 reward.
    - Form_Partnership: Establish strategic alliance, +30 reward, increments partnership_streak_steps.
    - Breach_Contract: Seize up to 15 capital from target, +40 reward, marks contracts_breached.
    
    Unsupported Actions (return error, no turn consumed):
    - Hostile_Acquisition, Hold: Reserved for future expansion.
    
    Episode Flow:
    1. reset(seed, episode_id, actor_id) initializes state and returns first observation.
    2. step(action, actor_id) validates, routes, and returns outcome observation.
    3. Episode terminates when step_count >= max_rounds (100).
    
    Scoring:
    Evaluators inspect state.telemetry and trust_matrix to compute task scores
    on a normalized [0.0, 1.0] scale.
    """

    # Global Configuration Constants
    SUPPORTS_CONCURRENT_SESSIONS = True
    DEFAULT_FIRM_IDS = ["Firm_A", "Firm_B", "Firm_C"]
    DEFAULT_ACTOR_ID = "Firm_A"
    DEFAULT_STRATEGIES = ["Cooperative", "Aggressive", "Rational"]
    STARTING_CAPITAL = 100
    MAX_EPISODE_STEPS = 100
    BREACH_GAIN = 15
    PARTNERSHIP_REWARD = 30.0

    def __init__(self):
        """Initializes the market environment and default firm list."""
        self._state = MarketState()
        self._firm_ids = list(self.DEFAULT_FIRM_IDS)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> MarketObservation:
        """Initialize the market to a clean state for a new episode.
        
        Lifecycle:
        1. Seed the RNG for reproducible randomness.
        2. Identify the active actor (observer) perspective.
        3. Randomly assign strategies to firms.
        4. Initialize all firms with STARTING_CAPITAL (100).
        5. Zero out trust matrix and telemetry counters.
        6. Create persistent MarketState for this episode.
        7. Return initial observation from the actor's viewpoint.
        
        Args:
            seed: Optional integer for reproducible randomness. If None, uses real random.
            episode_id: Optional unique identifier for logging/tracking. Auto-generates if None.
            **kwargs: Supports 'actor_id' (firm to observe from). Defaults to Firm_A.
            
        Returns:
            MarketObservation: Initial perception (done=False, reward=0.0, capital=100, etc.).
            
        Raises:
            ValueError: If actor_id not in DEFAULT_FIRM_IDS.
        """
        # 1. Setup Deterministic Randomness
        rng = random.Random(seed)

        # 2. Identify the Active Actor
        actor_id = kwargs.get("actor_id", self.DEFAULT_ACTOR_ID)
        if actor_id not in self._firm_ids:
            raise ValueError(f"Invalid actor_id '{actor_id}'. Must be in {self._firm_ids}")

        # 3. Initialize Firm Strategies
        # Uses rng.choices to handle any number of firms vs strategies
        firm_strategies = {
            fid: rng.choice(self.DEFAULT_STRATEGIES) for fid in self._firm_ids
        }

        # 4. Initialize Capital
        firm_capital = {fid: self.STARTING_CAPITAL for fid in self._firm_ids}
        trust_matrix = {
            fid: {other: 0.0 for other in self._firm_ids if other != fid}
            for fid in self._firm_ids
        }

        # 5. Initialize Telemetry Counters
        # These per-firm metrics are incremented by handlers and read by evaluators.
        telemetry = {
            fid: {
                "initial_capital": float(self.STARTING_CAPITAL),
                "successful_contracts": 0.0,
                "contracts_breached": 0.0,
                "partnership_streak_steps": 0.0,
                "market_decline_ratio": 0.0
            } for fid in self._firm_ids
        }

        # 6. Seed the Persistent Global State
        self._state = MarketState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            firm_capital=firm_capital,
            firm_strategies=firm_strategies,
            trust_matrix=trust_matrix,
            telemetry=telemetry,
            max_rounds=self.MAX_EPISODE_STEPS,
        )

        # 7. Return Initial Observation
        return MarketObservation(
            done=False,
            reward=0.0,
            capital=firm_capital[actor_id],
            trust_scores={fid: 0.0 for fid in self._firm_ids if fid != actor_id},
            active_partnerships=[],
            message="Market Initialized. Begin contract negotiations."
        )
    
    def step(self, action: MarketAction, **kwargs) -> MarketObservation:
        """Execute one market action and return the outcome.
        
        Lifecycle:
        1. Identify actor (actor_id) from kwargs; default to Firm_A.
        2. Validate actor exists in current state.
        3. Guard against terminal state (episode already finished).
        4. Route action to appropriate handler (Execute_Contract, Form_Partnership, Breach_Contract, Produce) or reject.
        5. Increment step counter and update global metrics (market_decline_ratio).
        6. Check termination condition (step_count >= max_rounds).
        7. Return outcome observation with reward and new market state view.
        
        Args:
            action: MarketAction with action_type, optional target_id, optional amount.
            **kwargs: Supports 'actor_id' (firm actor); defaults to Firm_A.
            
        Returns:
            MarketObservation: Outcome with reward, capital, trust_scores, done flag.
            
        Behavior:
        - Invalid/unsupported actions return error observation with penalty, no turn consumed.
        - Valid actions consume a turn, may mutate state, return reward.
        """
        actor_id = kwargs.get("actor_id", self.DEFAULT_ACTOR_ID)
        action_type = getattr(action, "action_type", None)

        # ===== Validation Phase =====
        # 1. Validate session actor before any state mutation.
        if actor_id not in self._state.firm_capital:
            return self._generate_error_obs(actor_id, "Unknown firm session.")

        # 2. Early terminal guard: no mutation and no penalty after done.
        if self._state.step_count >= self._state.max_rounds:
            return self._generate_error_obs(
                actor_id,
                "Episode already finished.",
                penalty=0.0,
            )

        # 3. Validate action payload shape before routing.
        if action_type is None:
            return self._generate_error_obs(actor_id, "Invalid action payload: missing action_type")

        # 4. Validate action parameters based on type BEFORE routing
        # Execute_Contract requires target_id and positive amount
        if action_type == "Execute_Contract":
            target_id = getattr(action, "target_id", None)
            amount = getattr(action, "amount", 0)
            if target_id is None or amount <= 0:
                return self._generate_error_obs(actor_id, "Invalid action payload: missing action_type")
        
        # ===== Routing & Action Execution =====
        # Route to appropriate action handler.
        if action_type == "Execute_Contract":
            reward = self._handle_execute_contract(actor_id, action)
        elif action_type == "Form_Partnership":
            reward = self._handle_form_partnership(actor_id, action)
        elif action_type == "Breach_Contract":
            reward = self._handle_breach_contract(actor_id, action)
        elif action_type == "Produce":
            reward = 5.0
            self._state.firm_capital[actor_id] += 10
        else:
            # Unsupported action does not consume a turn.
            return self._generate_error_obs(actor_id, f"Unsupported action: {action_type}")

        # ===== Telemetry & Termination =====
        # Reset partnership streak if not a Form_Partnership action (only contiguous Form_Partnership turns count).
        if not (action_type == "Form_Partnership" and reward > 0):
            self._set_telemetry(actor_id, "partnership_streak_steps", 0.0)

        # Only valid routed actions consume a turn.
        self._state.step_count += 1
        self._update_global_decline_ratio()
        done = self._state.step_count >= self._state.max_rounds
        return MarketObservation(
            done=done,
            reward=reward,
            capital=self._state.firm_capital[actor_id],
            trust_scores=self._get_trust_for(actor_id),
            active_partnerships=[],
            message=f"Contract cycle {self._state.step_count} complete."
        )

    def _handle_execute_contract(self, actor_id: str, action: MarketAction) -> float:
        """Execute a contract: exchange capital for higher reciprocal gain.
        
        Contract mechanics:
        1. Validate action is a proper Execute_Contract (has target, no self-contract).
        2. Check actor has enough capital to exchange.
        3. Actor loses 'amount'; target gains (amount * 1.2), rounded down.
        4. Both parties gain +0.05 trust (market cooperation).
        5. Increment successful_contracts counter.
        
        Args:
            actor_id: Firm executing the contract.
            action: MarketAction with target_id and amount.
            
        Returns:
            float: Reward (+20.0 on success, -10.0/-5.0 on failure).
        """
        target_id = action.target_id
        amount = action.amount

        # Strict Type & Target Guard
        if action.action_type != "Execute_Contract" or target_id == actor_id or target_id not in self._firm_ids:
            return -10.0

        if self._state.firm_capital[actor_id] >= amount:
            self._state.firm_capital[actor_id] -= amount
            # Deterministic Economy: 20% gain on contract
            gain = (amount * 12) // 10 
            self._state.firm_capital[target_id] += gain
            
            # Update Trust
            self._update_trust(actor_id, target_id, 0.05) 
            self._increment_telemetry(actor_id, "successful_contracts", 1.0)
            return 20.0 
            
        return -5.0

    def _update_trust(self, actor: str, target: str, change: float):
        """Update directed trust score between two firms.
        
        Trust mechanics:
        1. Access or create the actor's trust row in the matrix.
        2. Retrieve current trust score for target (default 0.0).
        3. Apply change and clamp to [-1.0, 1.0] (soft bounds for trust/distrust).
        4. Store updated score.
        
        Args:
            actor: Firm changing their trust in another.
            target: Firm being re-evaluated.
            change: Trust delta (typically +0.05 for contract, -0.15 for breach, +0.10 for partnership).
        """
        actor_row = self._state.trust_matrix.setdefault(actor, {})
        current = actor_row.get(target, 0.0)
        # Keep trust between -1.0 and 1.0
        actor_row[target] = max(-1.0, min(1.0, current + change))

    def _handle_breach_contract(self, actor_id: str, action: MarketAction) -> float:
        """Execute a contract breach: seize assets and damage trust.
        
        Breach mechanics:
        1. Validate action is a proper Breach_Contract (has target, no self-breach).
        2. Check target has positive capital.
        3. Actor seizes min(BREACH_GAIN, target_balance).
        4. Trust with target drops -0.15 (severe damage).
        5. Increment contracts_breached counter.
        
        Args:
            actor_id: Firm executing the breach.
            action: MarketAction with target_id.
            
        Returns:
            float: Reward (+40.0 on success, -10.0/-5.0 on failure).
        """
        target_id = action.target_id
        if action.action_type != "Breach_Contract" or target_id is None or target_id == actor_id or target_id not in self._firm_ids:
            return -10.0

        target_balance = self._state.firm_capital.get(target_id, 0)
        if target_balance <= 0:
            return -5.0

        seize_amount = min(self.BREACH_GAIN, target_balance)
        self._state.firm_capital[target_id] -= seize_amount
        self._state.firm_capital[actor_id] += seize_amount
        self._update_trust(actor_id, target_id, -0.15)
        self._increment_telemetry(actor_id, "contracts_breached", 1.0)
        return 40.0

    def _handle_form_partnership(self, actor_id: str, action: MarketAction) -> float:
        """Execute a partnership: strengthen reciprocal trust and cooperation.
        
        Partnership mechanics:
        1. Validate action is a proper Form_Partnership (has target, no self-partnership).
        2. Actor's trust in target increases +0.10 (mutual cooperation).
        3. Target's trust in actor also increases +0.10 (symmetric).
        4. Increment partnership_streak_steps counter (only for contiguous Form_Partnership turns).
        5. Return PARTNERSHIP_REWARD (+30.0).
        
        Args:
            actor_id: Firm initiating the partnership request.
            action: MarketAction with target_id.
            
        Returns:
            float: Reward (+30.0 on success, -10.0 on failure).
        """
        target_id = action.target_id
        if action.action_type != "Form_Partnership" or target_id is None or target_id == actor_id or target_id not in self._firm_ids:
            return -10.0

        # Partnership is symmetric strategic cooperation.
        self._update_trust(actor_id, target_id, 0.10)
        self._update_trust(target_id, actor_id, 0.10)
        self._increment_telemetry(actor_id, "partnership_streak_steps", 1.0)
        return self.PARTNERSHIP_REWARD

    def _increment_telemetry(self, actor_id: str, key: str, delta: float) -> None:
        """Increment a per-firm telemetry counter safely.
        
        Telemetry safety:
        1. Ensure firm row exists in telemetry dict (create with defaults if missing).
        2. Retrieve current value for key.
        3. Add delta and cast to float.
        4. Store updated value.
        
        Prevents KeyError at runtime even if state was partially initialized.
        
        Args:
            actor_id: Firm whose telemetry to update.
            key: Telemetry key (e.g., 'successful_contracts').
            delta: Amount to add (typically +1.0 for each event).
        """
        defaults = {
            "initial_capital": float(self.STARTING_CAPITAL),
            "successful_contracts": 0.0,
            "contracts_breached": 0.0,
            "partnership_streak_steps": 0.0,
            "market_decline_ratio": 0.0,
        }
        row = self._state.telemetry.setdefault(actor_id, defaults.copy())
        current = row.get(key, 0.0)
        row[key] = float(current) + float(delta)

    def _set_telemetry(self, actor_id: str, key: str, value: float) -> None:
        """Set a per-firm telemetry counter to a specific value safely.
        
        Telemetry safety:
        1. Ensure firm row exists in telemetry dict (create with defaults if missing).
        2. Cast value to float.
        3. Store updated value in row.
        
        Used for resetting counters (e.g., partnership_streak_steps = 0.0) between turns.
        
        Args:
            actor_id: Firm whose telemetry to update.
            key: Telemetry key (e.g., 'partnership_streak_steps').
            value: Exact value to set.
        """
        defaults = {
            "initial_capital": float(self.STARTING_CAPITAL),
            "successful_contracts": 0.0,
            "contracts_breached": 0.0,
            "partnership_streak_steps": 0.0,
            "market_decline_ratio": 0.0,
        }
        row = self._state.telemetry.setdefault(actor_id, defaults.copy())
        row[key] = float(value)

    def _update_global_decline_ratio(self) -> None:
        """Update the global scarcity metric: how much total capital has been lost.
        
        Scarcity computation:
        1. Sum initial_capital across all firms (shared baseline).
        2. Sum current capital across all firms.
        3. Compute market_decline_ratio = (initial_total - current_total) / initial_total.
        4. Store in each firm's telemetry (global metric).
        
        Semantics:
        - 0.0 = no loss, abundant economy.
        - 1.0 = total collapse, all capital gone (rare).
        - Mid-range (0.2 - 0.5) = typical scarcity under breach pressure.
        
        Used by Strategic Alliance Master grader to reward partnership maintenance under pressure.
        Called once per step after action execution.
        """
        if not self._state.telemetry:
            return

        initial_total = 0.0
        for firm_id in self._firm_ids:
            row = self._state.telemetry.setdefault(firm_id, {})
            initial_total += float(row.get("initial_capital", self.STARTING_CAPITAL))

        if initial_total <= 0:
            market_decline_ratio = 0.0
        else:
            current_total = float(sum(self._state.firm_capital.get(fid, 0) for fid in self._firm_ids))
            market_decline_ratio = max(0.0, min(1.0, (initial_total - current_total) / initial_total))

        for firm_id in self._firm_ids:
            row = self._state.telemetry.setdefault(firm_id, {})
            row["market_decline_ratio"] = market_decline_ratio

    def _get_trust_for(self, actor_id: str) -> Dict[str, float]:
        """Extract the firm's local view of trust scores for all other firms.
        
        This is what the firm observes as their trust_scores in the Observation.
        It's a copy of their row in the trust_matrix, excluding self-trust.
        
        Args:
            actor_id: Firm whose trust view to return.
            
        Returns:
            Dict[firm -> score] for all other firms. Defaults to 0.0 per firm.
        """
        if actor_id in self._state.trust_matrix:
            return dict(self._state.trust_matrix[actor_id])
        return {fid: 0.0 for fid in self._firm_ids if fid != actor_id}

    def _generate_error_obs(self, actor_id: str, message: str, penalty: float = -10.0) -> MarketObservation:
        """Create a safe observation payload when step inputs are invalid.
        
        Error handling:
        1. Retrieve firm's current capital.
        2. Determine done flag (already terminal?).
        3. Build trust_scores view (or fallback if firm unknown).
        4. Return observation with negative reward (penalty) and error message.
        
        Used for validation failures (unknown firm, terminal episode, malformed action).
        Does not consume a turn; error handling is \"free\" (no step_count increment).
        
        Args:
            actor_id: Firm issuing the bad action.
            message: Human-readable error description.
            penalty: Reward penalty (default -10.0). Can be 0.0 for soft errors.
            
        Returns:
            MarketObservation: Safe payload with all required fields; done may be True.
        """
        capital = self._state.firm_capital.get(actor_id, 0)
        done = self._state.step_count >= self._state.max_rounds
        if actor_id in self._firm_ids:
            trust_scores = self._get_trust_for(actor_id)
        else:
            # Stable fallback shape for unknown sessions.
            trust_scores = {fid: 0.0 for fid in self._firm_ids}

        return MarketObservation(
            done=done,
            reward=penalty,
            capital=capital,
            trust_scores=trust_scores,
            active_partnerships=[],
            message=message,
        )

    @property
    def state(self) -> MarketState:
        """Expose current server-side episode state.
        
        This provides read-only access to the hidden state for graders and inspection.
        Contains firm capital, trust matrix, telemetry counters, and step count.
        
        Returns:
            MarketState: Current persistent state object for this episode.
        """
        return self._state


class ArenaEnvironment(MarketEnvironment):
    """Backward-compatible alias for legacy code using arena terminology.
    
    Retained for compatibility during transition from ArenaEnvironment naming
    to MarketEnvironment. Direct subclass with no overrides.
    """
    pass


class StartoneEnvironment(MarketEnvironment):
    """Backward-compatible alias while migrating template wiring.
    
    Retained for legacy compatibility during transition from StartoneEnvironment naming
    to MarketEnvironment. Direct subclass with no overrides.
    """
    pass