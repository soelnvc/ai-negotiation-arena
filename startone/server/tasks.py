"""
Task Registry for the B2B Market Economic Simulation.
Binds the environment engine to the specific grading logic for OpenEnv evaluation.

Architecture:
- Each task wraps an Environment class and a Grader callable.
- TaskDefinition is either imported from OpenEnv or fallback to local dataclass.
- MARKET_TASKS list is exported to OpenEnv platform for benchmark registration.

Task Design:
- All tasks use the same MarketEnvironment (multiplayer market negotiation simulator).
- Each task differs only in grader_callable and description (goal statement).
- max_steps=100 is consistent across all tasks (single episode horizon).

Compatibility:
- TaskDefinition import uses dynamic getattr() fallback to handle OpenEnv version variance.
- If OpenEnv exports TaskDefinition, use that; else use local frozen dataclass.
- This ensures robustness across OpenEnv upgrades.
"""

from dataclasses import dataclass
import inspect
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


def _make_task_definition(
    task_id: str,
    name: str,
    description: str,
    environment_class: Type[Any],
    grader_callable: Callable[..., float],
    max_steps: int,
) -> Any:
    """Build a TaskDefinition across OpenEnv API variants.

    Some OpenEnv versions expect `grader`, others expect `grader_callable`.
    This adapter ensures the grader is always wired, avoiding false
    "not enough tasks with graders" validation failures.
    """
    try:
        params = inspect.signature(TaskDefinition).parameters
    except (TypeError, ValueError):
        params = {}

    payload: dict[str, Any] = {
        "task_id": task_id,
        "name": name,
        "description": description,
        "environment_class": environment_class,
        "max_steps": max_steps,
    }

    if "grader" in params:
        payload["grader"] = grader_callable
    else:
        payload["grader_callable"] = grader_callable

    return TaskDefinition(**payload)

from .startone_environment import MarketEnvironment
from .graders import MarketGraders

# ---------------------------------------------------------
# TASK 1: EASY - Capital Accumulator
# ---------------------------------------------------------
# Measures basic survival and capital accumulation capability.
task_capital_accumulator = _make_task_definition(
    task_id="independent_producer",
    name="Independent Producer",
    description=(
        "Survive for 50 steps and accumulate as much capital as possible. "
        "Tests the firm's ability to understand basic market mechanics."
    ),
    environment_class=MarketEnvironment,
    grader_callable=MarketGraders.grade_capital_accumulator,
    max_steps=100,
)

# ---------------------------------------------------------
# TASK 2: MEDIUM - Reliable Partner
# ---------------------------------------------------------
# Measures cooperation under temptation to breach contracts.
task_reliable_partner = _make_task_definition(
    task_id="ethical_contractor",
    name="Ethical Contractor",
    description=(
        "Successfully complete 3 contracts without ever initiating a breach. "
        "Tests the firm's ability to cooperate and avoid destructive short-term gains."
    ),
    environment_class=MarketEnvironment,
    grader_callable=MarketGraders.grade_reliable_partner,
    max_steps=100,
)

# ---------------------------------------------------------
# TASK 3: HARD - Strategic Alliance Master
# ---------------------------------------------------------
# Measures advanced partnership maintenance and strategic adaptation.
task_strategic_alliance_master = _make_task_definition(
    task_id="enterprise_stabilizer",
    name="Enterprise Stabilizer",
    description=(
        "Maintain partnership stability under capital pressure while accumulating wealth. "
        "Tests advanced negotiation skills, long-term planning, and multi-variable optimization."
    ),
    environment_class=MarketEnvironment,
    grader_callable=MarketGraders.grade_strategic_alliance_master,
    max_steps=100,
)

# Official registry list exported to the OpenEnv platform for benchmark runs.
# All tasks share the same environment but differ in grader (objective) and difficulty.
MARKET_TASKS = [
    task_capital_accumulator,
    task_reliable_partner,
    task_strategic_alliance_master,
]

# Legacy aliases for backward compatibility
ARENA_TASKS = MARKET_TASKS
task_resource_scavenger = task_capital_accumulator
task_honest_trader = task_reliable_partner
task_master_negotiator = task_strategic_alliance_master

# OpenEnv YAML entrypoint compatibility aliases.
IndependentProducerTask = task_capital_accumulator
EthicalContractorTask = task_reliable_partner
EnterpriseStabilizerTask = task_strategic_alliance_master