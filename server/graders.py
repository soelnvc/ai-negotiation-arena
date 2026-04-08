"""Compatibility grader exports for OpenEnv YAML import paths.

This module forwards grader callables to the canonical implementations in
startone.server.graders so validators can resolve either `server.*` or
`startone.server.*` paths.
"""

from startone.server.graders import (
    grade_capital_accumulator,
    grade_reliable_partner,
    grade_strategic_alliance_master,
)

__all__ = [
    "grade_capital_accumulator",
    "grade_reliable_partner",
    "grade_strategic_alliance_master",
]
