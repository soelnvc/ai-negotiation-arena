import uuid
import random
from typing import Dict, List, Optional
from openenv.core.env_server import Environment
from ..models import ArenaAction, ArenaObservation, ArenaState

class ArenaEnvironment(Environment):
    """Multi-agent negotiation simulation for testing social intelligence.
    
    Overview:
    The Arena is a 3-agent negotiation game where agents manage resources, form
    alliances, and navigate trust/betrayal tradeoffs. The environment is deterministic
    given a seed, supporting reproducible evaluation of agent strategies.
    
    Agent Actions (routed):
    - Gather: Collect 10 resources, +5 reward.
    - Trade: Exchange resources for higher-value reciprocal gain, +20 reward.
    - Ally: Form/strengthen alliance, +30 reward, increments alliance_streak_steps.
    - Betray: Steal up to 15 resources from target, +40 reward, marks betrayals_initiated.
    
    Unsupported Actions (return error, no turn consumed):
    - Attack, Idle: Reserved for future expansion.
    
    Episode Flow:
    1. reset(seed, episode_id, player_id) initializes state and returns first observation.
    2. step(action, player_id) validates, routes, and returns outcome observation.
    3. Episode terminates when step_count >= max_rounds (100).
    
    Scoring:
    Graders inspect state.telemetry and reputation_matrix to compute task scores
    on a normalized [0.0, 1.0] scale.
    """

    # Global Configuration Constants
    SUPPORTS_CONCURRENT_SESSIONS = True
    DEFAULT_AGENT_IDS = ["Agent_A", "Agent_B", "Agent_C"]
    DEFAULT_PLAYER_ID = "Agent_A"
    DEFAULT_PERSONALITIES = ["Cooperative", "Aggressive", "Rational"]
    STARTING_RESOURCES = 100
    MAX_EPISODE_STEPS = 100
    BETRAYAL_STEAL = 15
    ALLY_REWARD = 30.0

    def __init__(self):
        """Initializes the environment class and default agent list."""
        self._state = ArenaState()
        self._agent_ids = list(self.DEFAULT_AGENT_IDS)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ArenaObservation:
        """Initialize the arena to a clean state for a new episode.
        
        Lifecycle:
        1. Seed the RNG for reproducible randomness.
        2. Identify the active player (observer) perspective.
        3. Randomly assign personalities to agents.
        4. Initialize all agents with STARTING_RESOURCES (100).
        5. Zero out reputation matrix and telemetry counters.
        6. Create persistent ArenaState for this episode.
        7. Return initial observation from the player's viewpoint.
        
        Args:
            seed: Optional integer for reproducible randomness. If None, uses real random.
            episode_id: Optional unique identifier for logging/tracking. Auto-generates if None.
            **kwargs: Supports 'player_id' (agent to observe from). Defaults to Agent_A.
            
        Returns:
            ArenaObservation: Initial perception (done=False, reward=0.0, resources=100, etc.).
            
        Raises:
            ValueError: If player_id not in DEFAULT_AGENT_IDS.
        """
        # 1. Setup Deterministic Randomness
        rng = random.Random(seed)

        # 2. Identify the Active Player
        player_id = kwargs.get("player_id", self.DEFAULT_PLAYER_ID)
        if player_id not in self._agent_ids:
            raise ValueError(f"Invalid player_id '{player_id}'. Must be in {self._agent_ids}")

        # 3. Initialize Agent Personalities (Fixed Risk)
        # Uses rng.choices to handle any number of agents vs personalities
        agent_personalities = {
            aid: rng.choice(self.DEFAULT_PERSONALITIES) for aid in self._agent_ids
        }

        # 4. Initialize Resources
        agent_resources = {aid: self.STARTING_RESOURCES for aid in self._agent_ids}
        reputation_matrix = {
            aid: {other: 0.0 for other in self._agent_ids if other != aid}
            for aid in self._agent_ids
        }

        telemetry = {
            aid: {
                "initial_resources": float(self.STARTING_RESOURCES),
                "successful_trades": 0.0,
                "betrayals_initiated": 0.0,
                "alliance_streak_steps": 0.0,
                "global_decline_ratio": 0.0
            } for aid in self._agent_ids
        }

        # 5. Seed the Persistent Global State
        self._state = ArenaState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            agent_resources=agent_resources,
            agent_personalities=agent_personalities,
            reputation_matrix=reputation_matrix,
            telemetry=telemetry,
            max_rounds=self.MAX_EPISODE_STEPS,
        )

        # 6. Return Initial Observation
        return ArenaObservation(
            done=False,
            reward=0.0,
            resources=agent_resources[player_id],
            reputation_scores={aid: 0.0 for aid in self._agent_ids if aid != player_id},
            active_alliances=[],
            message="Arena Initialized. Begin negotiations."
        )
    
    def step(self, action: ArenaAction, **kwargs) -> ArenaObservation:
        """Execute one action in the arena and return the outcome.
        
        Lifecycle:
        1. Identify actor (player_id) from kwargs; default to Agent_A.
        2. Validate actor exists in current state.
        3. Guard against terminal state (episode already finished).
        4. Route action to appropriate handler (Trade, Ally, Betray, Gather) or reject.
        5. Increment step counter and update global metrics (decline_ratio).
        6. Check termination condition (step_count >= max_rounds).
        7. Return outcome observation with reward and new state view.
        
        Args:
            action: ArenaAction with action_type, optional target_id, optional amount.
            **kwargs: Supports 'player_id' (agent actor); defaults to Agent_A.
            
        Returns:
            ArenaObservation: Outcome with reward, resources, reputation, done flag.
            
        Behavior:
        - Invalid/unsupported actions return error observation with penalty, no turn consumed.
        - Valid actions consume a turn, may mutate state, return reward.
        """
        player_id = kwargs.get("player_id", self.DEFAULT_PLAYER_ID)
        action_type = getattr(action, "action_type", None)

        # ===== Validation Phase =====
        # 1. Validate session actor before any state mutation.
        if player_id not in self._state.agent_resources:
            return self._generate_error_obs(player_id, "Unknown agent session.")

        # 2. Early terminal guard: no mutation and no penalty after done.
        if self._state.step_count >= self._state.max_rounds:
            return self._generate_error_obs(
                player_id,
                "Episode already finished.",
                penalty=0.0,
            )

        # 3. Validate action payload shape before routing.
        if action_type is None:
            return self._generate_error_obs(player_id, "Invalid action payload: missing action_type")

        # ===== Routing & Action Execution =====
        # 4. Route to appropriate action handler.
        if action_type == "Trade":
            reward = self._handle_trade(player_id, action)
        elif action_type == "Ally":
            reward = self._handle_ally(player_id, action)
        elif action_type == "Betray":
            reward = self._handle_betrayal(player_id, action)
        elif action_type == "Gather":
            reward = 5.0
            self._state.agent_resources[player_id] += 10
        else:
            # Unsupported action does not consume a turn.
            return self._generate_error_obs(player_id, f"Unsupported action: {action_type}")

        # ===== Telemetry & Termination =====
        # Reset alliance streak if not an Ally action (only contiguous Ally turns count).
        if not (action_type == "Ally" and reward > 0):
            self._set_telemetry(player_id, "alliance_streak_steps", 0.0)

        # Only valid routed actions consume a turn.
        self._state.step_count += 1
        self._update_global_decline_ratio()
        done = self._state.step_count >= self._state.max_rounds
        return ArenaObservation(
            done=done,
            reward=reward,
            resources=self._state.agent_resources[player_id],
            reputation_scores=self._get_reputation_for(player_id),
            active_alliances=[],
            message=f"Turn {self._state.step_count} complete."
        )

    def _handle_trade(self, player_id: str, action: ArenaAction) -> float:
        """Execute a trade: exchange resources for higher reciprocal gain.
        
        Trade mechanics:
        1. Validate action is a proper Trade (has target, no self-trade).
        2. Check actor has enough resources to give.
        3. Actor loses 'amount'; target gains (amount * 1.2), rounded down.
        4. Both parties gain +0.05 reputation (trust).
        5. Increment successful_trades counter.
        
        Args:
            player_id: Actor initiating the trade.
            action: ArenaAction with target_id and amount.
            
        Returns:
            float: Reward (+20.0 on success, -10.0/-5.0 on failure).
        """
        target_id = action.target_id
        amount = action.amount

        # Strict Type & Target Guard
        if action.action_type != "Trade" or target_id == player_id or target_id not in self._agent_ids:
            return -10.0

        if self._state.agent_resources[player_id] >= amount:
            self._state.agent_resources[player_id] -= amount
            # Deterministic Economy: 20% gain on trade
            gain = (amount * 12) // 10 
            self._state.agent_resources[target_id] += gain
            
            # Real Reputation Update
            self._update_reputation(player_id, target_id, 0.05) 
            self._increment_telemetry(player_id, "successful_trades", 1.0)
            return 20.0 
            
        return -5.0

    def _update_reputation(self, actor: str, target: str, change: float):
        """Update directed trust score between two agents.
        
        Reputation mechanics:
        1. Access or create the actor's trust row in the matrix.
        2. Retrieve current trust score for target (default 0.0).
        3. Apply change and clamp to [-1.0, 1.0] (soft bounds for trust/distrust).
        4. Store updated score.
        
        Args:
            actor: Agent changing their trust in another.
            target: Agent being re-evaluated.
            change: Reputation delta (typically +0.05 for trade, -0.15 for betray, +0.10 for ally).
        """
        actor_row = self._state.reputation_matrix.setdefault(actor, {})
        current = actor_row.get(target, 0.0)
        # Keep trust between -1.0 and 1.0
        actor_row[target] = max(-1.0, min(1.0, current + change))

    def _handle_betrayal(self, player_id: str, action: ArenaAction) -> float:
        """Execute a betrayal: steal resources and damage trust.
        
        Betrayal mechanics:
        1. Validate action is a proper Betray (has target, no self-betrayal).
        2. Check target has positive resources.
        3. Actor steals min(BETRAYAL_STEAL, target_balance).
        4. Reputation with target drops -0.15 (severe damage).
        5. Increment betrayals_initiated counter.
        
        Args:
            player_id: Actor executing the betrayal.
            action: ArenaAction with target_id.
            
        Returns:
            float: Reward (+40.0 on success, -10.0/-5.0 on failure).
        """
        target_id = action.target_id
        if action.action_type != "Betray" or target_id is None or target_id == player_id or target_id not in self._agent_ids:
            return -10.0

        target_balance = self._state.agent_resources.get(target_id, 0)
        if target_balance <= 0:
            return -5.0

        steal_amount = min(self.BETRAYAL_STEAL, target_balance)
        self._state.agent_resources[target_id] -= steal_amount
        self._state.agent_resources[player_id] += steal_amount
        self._update_reputation(player_id, target_id, -0.15)
        self._increment_telemetry(player_id, "betrayals_initiated", 1.0)
        return 40.0

    def _handle_ally(self, player_id: str, action: ArenaAction) -> float:
        """Execute an alliance: strengthen reciprocal trust and cooperation.
        
        Alliance mechanics:
        1. Validate action is a proper Ally (has target, no self-alliance).
        2. Actor's trust in target increases +0.10 (mutual cooperation).
        3. Target's trust in actor also increases +0.10 (symmetric).
        4. Increment alliance_streak_steps counter (only for contiguous Ally turns).
        5. Return ALLY_REWARD (+30.0).
        
        Args:
            player_id: Agent initiating the alliance request.
            action: ArenaAction with target_id.
            
        Returns:
            float: Reward (+30.0 on success, -10.0 on failure).
        """
        target_id = action.target_id
        if action.action_type != "Ally" or target_id is None or target_id == player_id or target_id not in self._agent_ids:
            return -10.0

        # Alliance is symmetric social cooperation.
        self._update_reputation(player_id, target_id, 0.10)
        self._update_reputation(target_id, player_id, 0.10)
        self._increment_telemetry(player_id, "alliance_streak_steps", 1.0)
        return self.ALLY_REWARD

    def _increment_telemetry(self, player_id: str, key: str, delta: float) -> None:
        """Increment a per-player telemetry counter safely.
        
        Telemetry safety:
        1. Ensure player row exists in telemetry dict (create with defaults if missing).
        2. Retrieve current value for key.
        3. Add delta and cast to float.
        4. Store updated value.
        
        Prevents KeyError at runtime even if state was partially initialized.
        
        Args:
            player_id: Agent whose telemetry to update.
            key: Telemetry key (e.g., 'successful_trades').
            delta: Amount to add (typically +1.0 for each event).
        """
        defaults = {
            "initial_resources": float(self.STARTING_RESOURCES),
            "successful_trades": 0.0,
            "betrayals_initiated": 0.0,
            "alliance_streak_steps": 0.0,
            "global_decline_ratio": 0.0,
        }
        row = self._state.telemetry.setdefault(player_id, defaults.copy())
        current = row.get(key, 0.0)
        row[key] = float(current) + float(delta)

    def _set_telemetry(self, player_id: str, key: str, value: float) -> None:
        """Set a per-player telemetry counter to a specific value safely.
        
        Telemetry safety:
        1. Ensure player row exists in telemetry dict (create with defaults if missing).
        2. Cast value to float.
        3. Store updated value in row.
        
        Used for resetting counters (e.g., alliance_streak_steps = 0.0) between turns.
        
        Args:
            player_id: Agent whose telemetry to update.
            key: Telemetry key (e.g., 'alliance_streak_steps').
            value: Exact value to set.
        """
        defaults = {
            "initial_resources": float(self.STARTING_RESOURCES),
            "successful_trades": 0.0,
            "betrayals_initiated": 0.0,
            "alliance_streak_steps": 0.0,
            "global_decline_ratio": 0.0,
        }
        row = self._state.telemetry.setdefault(player_id, defaults.copy())
        row[key] = float(value)

    def _update_global_decline_ratio(self) -> None:
        """Update the global scarcity metric: how much total wealth has been lost.
        
        Scarcity computation:
        1. Sum initial_resources across all agents (shared baseline).
        2. Sum current resources across all agents.
        3. Compute decline_ratio = (initial_total - current_total) / initial_total.
        4. Store in each agent's telemetry (global metric).
        
        Semantics:
        - 0.0 = no loss, abundant economy.
        - 1.0 = total collapse, all resources gone (rare).
        - Mid-range (0.2 - 0.5) = typical scarcity under betrayal pressure.
        
        Used by Master Negotiator grader to reward alliance maintenance under pressure.
        Called once per step after action execution.
        """
        if not self._state.telemetry:
            return

        initial_total = 0.0
        for agent_id in self._agent_ids:
            row = self._state.telemetry.setdefault(agent_id, {})
            initial_total += float(row.get("initial_resources", self.STARTING_RESOURCES))

        if initial_total <= 0:
            decline_ratio = 0.0
        else:
            current_total = float(sum(self._state.agent_resources.get(aid, 0) for aid in self._agent_ids))
            decline_ratio = max(0.0, min(1.0, (initial_total - current_total) / initial_total))

        for agent_id in self._agent_ids:
            row = self._state.telemetry.setdefault(agent_id, {})
            row["global_decline_ratio"] = decline_ratio

    def _get_reputation_for(self, player_id: str) -> Dict[str, float]:
        """Extract the agent's local view of trust scores for all other agents.
        
        This is what the agent observes as their reputation_scores in the Observation.
        It's a copy of their row in the reputation_matrix, excluding self-trust.
        
        Args:
            player_id: Agent whose reputation view to return.
            
        Returns:
            Dict[agent -> score] for all other agents. Defaults to 0.0 per agent.
        """
        if player_id in self._state.reputation_matrix:
            return dict(self._state.reputation_matrix[player_id])
        return {aid: 0.0 for aid in self._agent_ids if aid != player_id}

    def _generate_error_obs(self, player_id: str, message: str, penalty: float = -10.0) -> ArenaObservation:
        """Create a safe observation payload when step inputs are invalid.
        
        Error handling:
        1. Retrieve agent's current resources.
        2. Determine done flag (already terminal?).
        3. Build reputation_scores view (or fallback if agent unknown).
        4. Return observation with negative reward (penalty) and error message.
        
        Used for validation failures (unknown agent, terminal episode, malformed action).
        Does not consume a turn; error handling is \"free\" (no step_count increment).
        
        Args:
            player_id: Agent issuing the bad action.
            message: Human-readable error description.
            penalty: Reward penalty (default -10.0). Can be 0.0 for soft errors.
            
        Returns:
            ArenaObservation: Safe payload with all required fields; done may be True.
        """
        resources = self._state.agent_resources.get(player_id, 0)
        done = self._state.step_count >= self._state.max_rounds
        if player_id in self._agent_ids:
            reputation_scores = self._get_reputation_for(player_id)
        else:
            # Stable fallback shape for unknown sessions.
            reputation_scores = {aid: 0.0 for aid in self._agent_ids}

        return ArenaObservation(
            done=done,
            reward=penalty,
            resources=resources,
            reputation_scores=reputation_scores,
            active_alliances=[],
            message=message,
        )

    @property
    def state(self) -> ArenaState:
        """Expose current server-side episode state.
        
        This provides read-only access to the hidden state for graders and inspection.
        Contains agent resources, reputation matrix, telemetry counters, and step count.
        
        Returns:
            ArenaState: Current persistent state object for this episode.
        """
        return self._state


class StartoneEnvironment(ArenaEnvironment):
    """Backward-compatible alias while migrating template wiring.
    
    Retained for legacy compatibility during transition from StartoneEnvironment naming
    to ArenaEnvironment. Direct subclass with no overrides.
    """