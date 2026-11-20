"""skillrot: measure whether Agent Skills still help on each model.

Library entry points::

    from skillrot import Condition, ModelRef, Store, analyze, build_plan, execute, load_suite
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("skillrot")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

from skillrot.analysis import ModelResult, ResultBundle, analyze
from skillrot.errors import (
    BudgetExceededError,
    ConfigError,
    GraderError,
    HarnessError,
    SkillrotError,
    SuiteError,
)
from skillrot.harness import get_harness
from skillrot.models import CaseKind, Condition, ExitReason, Flag, ModelRef, Tier, Verdict
from skillrot.runner import Plan, RunSummary, build_plan, execute
from skillrot.store import Store
from skillrot.suite import Case, Skill, Suite, load_suite

__all__ = [
    "BudgetExceededError",
    "Case",
    "CaseKind",
    "Condition",
    "ConfigError",
    "ExitReason",
    "Flag",
    "GraderError",
    "HarnessError",
    "ModelRef",
    "ModelResult",
    "Plan",
    "ResultBundle",
    "RunSummary",
    "Skill",
    "SkillrotError",
    "Store",
    "Suite",
    "SuiteError",
    "Tier",
    "Verdict",
    "__version__",
    "analyze",
    "build_plan",
    "execute",
    "get_harness",
    "load_suite",
]
