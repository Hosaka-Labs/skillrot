"""Core value types shared by every layer. Plain frozen dataclasses; no I/O here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class Condition(StrEnum):
    """How the skill is made available to the agent for one trial."""

    NONE = "none"  # skill not installed: the baseline
    SKILL = "skill"  # installed normally; the agent decides whether to load it
    FORCED = "forced"  # skill body placed straight into context
    PRIOR = "prior"  # an earlier version of the skill, installed normally


class CaseKind(StrEnum):
    """What a case measures."""

    TASK = "task"  # does the outcome get better with the skill?
    TRIGGER = "trigger"  # should load the skill
    NO_TRIGGER = "no-trigger"  # should not load the skill


class Split(StrEnum):
    """Dev cases may be used while editing a skill; holdout cases only for version comparisons."""

    DEV = "dev"
    HOLDOUT = "holdout"


class SkillType(StrEnum):
    """Capability skills can go stale; convention skills encode choices a model can't guess."""

    CAPABILITY = "capability"
    CONVENTION = "convention"


class ExitReason(StrEnum):
    """Why a trial ended. Only OK and TASK_ERROR count as data; the rest are infrastructure."""

    OK = "ok"
    TASK_ERROR = "task_error"  # the agent ran and failed on its own terms
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    REFUSED = "refused"
    INFRA = "infra"

    @property
    def counts_as_data(self) -> bool:
        """Whether a trial with this exit reason enters the statistics."""
        return self in {ExitReason.OK, ExitReason.TASK_ERROR, ExitReason.TIMEOUT}


class Verdict(StrEnum):
    """One verdict per skill, model and harness. Decided by `verdicts.decide`."""

    UNTESTED = "untested"
    HARMFUL = "harmful"
    ESSENTIAL = "essential"
    REDUNDANT = "redundant"
    NOT_TRIGGERING = "not-triggering"
    INCONCLUSIVE = "inconclusive"


class Flag(StrEnum):
    """Notes shown next to a verdict. They never change it."""

    EFFICIENCY = "efficiency"  # cheaper per passing run with the skill
    CONVENTION = "convention"  # convention skill: never suggest deleting on accuracy alone
    CONTAMINATED = "contaminated"  # the skill loaded in a run where it was not installed


class Tier(StrEnum):
    """Preset run sizes."""

    SMOKE = "smoke"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A model as one provider names it. Prefer exact IDs over moving aliases."""

    provider: str
    model_id: str

    @classmethod
    def parse(cls, text: str, default_provider: str = "anthropic") -> ModelRef:
        """Parse `provider/model` or a bare model ID."""
        provider, sep, model_id = text.partition("/")
        return cls(provider, model_id) if sep else cls(default_provider, text)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass(frozen=True, slots=True)
class Usage:
    """Token and cost accounting for one trial."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    wall_s: float = 0.0
    turns: int = 0


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call the agent made, as recorded in the transcript."""

    name: str
    input: dict[str, Any] = field(default_factory=dict[str, Any])


def content_hash(*parts: bytes | str) -> str:
    """Stable short hash of some content; used for skills, cases and cache keys."""
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode() if isinstance(part, str) else part
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()[:16]


def hash_json(value: object) -> str:
    """Hash a JSON-serialisable value with sorted keys."""
    return content_hash(json.dumps(value, sort_keys=True, default=str))


def hash_tree(root: Path) -> str:
    """Hash every file under `root` by relative path and content."""
    parts: list[bytes | str] = []
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            parts.extend((path.relative_to(root).as_posix(), path.read_bytes()))
    return content_hash(*parts)
