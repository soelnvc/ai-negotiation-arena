"""Compatibility task exports for OpenEnv YAML import paths.

This module forwards task entrypoints to the canonical registry in
startone.server.tasks so validators can resolve either `server.*` or
`startone.server.*` paths.
"""

from startone.server.tasks import (
    ARENA_TASKS,
    MARKET_TASKS,
    EthicalContractorTask,
    EnterpriseStabilizerTask,
    IndependentProducerTask,
)

__all__ = [
    "MARKET_TASKS",
    "ARENA_TASKS",
    "IndependentProducerTask",
    "EthicalContractorTask",
    "EnterpriseStabilizerTask",
]
