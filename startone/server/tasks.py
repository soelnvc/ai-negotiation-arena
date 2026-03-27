"""
Task Registry for the AI Negotiation Arena.
Binds the environment engine to the specific grading logic for OpenEnv evaluation.

Architecture:
- Each task wraps an Environment class and a Grader callable.
- TaskDefinition is either imported from OpenEnv or fallback to local dataclass.
- ARENA_TASKS list is exported to OpenEnv platform for benchmark registration.

Task Design:
- All tasks use the same ArenaEnvironment (multiplayer negotiation simulator).
- Each task differs only in grader_callable and description (goal statement).
- max_steps=100 is consistent across all tasks (single episode horizon).

Compatibility:
- TaskDefinition import uses dynamic getattr() fallback to handle OpenEnv version variance.
- If OpenEnv exports TaskDefinition, use that; else use local frozen dataclass.
- This ensures robustness across OpenEnv upgrades.
"""

from dataclasses import dataclass
from typing import Any, Callable, Type

try:
    from openenv.core import env_server as _env_server
except ImportError:
    _env_server = None


# Compatibility Layer: Resolve TaskDefinition from OpenEnv or fallback to local.
# This handles version variance where TaskDefinition export may differ across OpenEnv versions.
_task_definition_type = (
    getattr(_env_server, "TaskDefinition", None) if _env_server is not None else None
)

if _task_definition_type is None:
    @dataclass(frozen=True)
    class TaskDefinition:
        """Compatibility fallback when OpenEnv doesn't export TaskDefinition.
        
        Defines a single benchmark task: environment, grader, and metadata.
        Frozen dataclass ensures immutability in the task registry.
        """

        task_id: str  # Unique task identifier (e.g., "arena-resource-scavenger-v1")
        name: str  # Human-readable task name (e.g., "Resource Scavenger")
        description: str  # Full task objective and complexity statement
        environment_class: Type[Any]  # Environment class to instantiate for this task
        grader_callable: Callable[..., float]  # Grader function returning [0.0, 1.0] score
        max_steps: int  # Episode termination threshold (max turns per episode)
else:
    TaskDefinition = _task_definition_type

from .startone_environment import ArenaEnvironment
from .graders import ArenaGraders

# ---------------------------------------------------------
# TASK 1: EASY - Resource Scavenger
# ---------------------------------------------------------
# Measures basic survival and resource gathering capability.
task_resource_scavenger = TaskDefinition(
    task_id="arena-resource-scavenger-v1",
    name="Resource Scavenger",
    description=(
        "Survive for 50 steps and gather as many resources as possible. "
        "Tests the agent's ability to understand basic environment mechanics."
    ),
    environment_class=ArenaEnvironment,
    grader_callable=ArenaGraders.grade_resource_scavenger,
    max_steps=100,
)

# ---------------------------------------------------------
# TASK 2: MEDIUM - Honest Trader
# ---------------------------------------------------------
# Measures cooperation under temptation to defect.
task_honest_trader = TaskDefinition(
    task_id="arena-honest-trader-v1",
    name="Honest Trader",
    description=(
        "Successfully complete 3 trades without ever initiating a betrayal. "
        "Tests the agent's ability to cooperate and avoid destructive short-term gains."
    ),
    environment_class=ArenaEnvironment,
    grader_callable=ArenaGraders.grade_honest_trader,
    max_steps=100,
)

# ---------------------------------------------------------
# TASK 3: HARD - Master Negotiator
# ---------------------------------------------------------
# Measures advanced alliance maintenance and strategic adaptation.
task_master_negotiator = TaskDefinition(
    task_id="arena-master-negotiator-v1",
    name="Master Negotiator",
    description=(
        "Maintain alliance stability under resource pressure while accumulating wealth. "
        "Tests advanced social intelligence, long-term planning, and multi-variable optimization."
    ),
    environment_class=ArenaEnvironment,
    grader_callable=ArenaGraders.grade_master_negotiator,
    max_steps=100,
)

# Official registry list exported to the OpenEnv platform for benchmark runs.
# All tasks share the same environment but differ in grader (objective) and difficulty.
ARENA_TASKS = [
    task_resource_scavenger,
    task_honest_trader,
    task_master_negotiator,
]