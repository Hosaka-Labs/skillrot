"""The harness port: anything that can run one agent trial in a prepared working directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from skillrot.models import Condition, ExitReason, ModelRef, ToolCall, Usage
    from skillrot.suite import Case, Skill


@dataclass(frozen=True, slots=True)
class TrialRequest:
    """Everything a harness needs for one trial.

    `skill` is the version to install for SKILL, FORCED and PRIOR, and None for NONE.
    `skill_name` is always the skill under test, so harnesses can detect it loading
    even in the NONE condition (a sign that a user-level copy leaked in).
    """

    case: Case
    condition: Condition
    model: ModelRef
    workspace: Path
    skill_name: str
    skill: Skill | None
    timeout_s: float
    max_turns: int


@dataclass(frozen=True, slots=True)
class Transcript:
    """What happened in one trial."""

    final_text: str
    exit_reason: ExitReason
    skill_loaded: bool
    usage: Usage
    harness_version: str
    tool_calls: tuple[ToolCall, ...] = ()
    error: str = ""
    events: tuple[dict[str, Any], ...] = field(default=(), repr=False)


@runtime_checkable
class Harness(Protocol):
    """A way of running an agent: Claude Code, a bare API loop, a test double."""

    name: str

    async def version(self) -> str:
        """Version string recorded with every trial; changing it invalidates the cache."""
        ...

    async def check(self) -> None:
        """Raise `HarnessError` with a fix-it message if the harness cannot run here."""
        ...

    async def run(self, request: TrialRequest) -> Transcript:
        """Run one trial in `request.workspace` and return its transcript."""
        ...
