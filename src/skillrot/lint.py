"""Static checks on a suite. No model calls; runs in milliseconds."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from skillrot.graders import DETERMINISTIC_TYPES
from skillrot.models import CaseKind, Split

if TYPE_CHECKING:
    from skillrot.suite import Case, GraderSpec, Suite

RECOMMENDED_TASK_CASES = 15
TODO_MARKER = "TODO(skillrot)"
_SHINGLE = 8


class Level(StrEnum):
    """Errors block `check`; warnings are shown."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class Finding:
    """One lint result."""

    level: Level
    code: str
    message: str
    case_id: str | None = None

    def __str__(self) -> str:
        where = f"{self.case_id}: " if self.case_id else ""
        return f"{self.level} {self.code} {where}{self.message}"


def lint(suite: Suite) -> list[Finding]:
    """Every finding for a suite, errors first."""
    findings = [*_suite_level(suite)]
    shingles = _shingles(suite.skill.body)
    for case in suite.cases:
        findings.extend(_case_level(case, suite, shingles))
    order = list(Level)
    return sorted(findings, key=lambda f: (order.index(f.level), f.case_id or "", f.code))


def _suite_level(suite: Suite) -> list[Finding]:
    out: list[Finding] = []
    n_task = len(suite.task_cases)
    kinds = {c.kind for c in suite.cases}
    if n_task < suite.config.min_cases:
        out.append(
            Finding(
                Level.ERROR,
                "L001",
                f"{n_task} task cases; verdicts need at least {suite.config.min_cases}",
            )
        )
    elif n_task < RECOMMENDED_TASK_CASES:
        out.append(
            Finding(
                Level.WARNING,
                "L001",
                f"{n_task} task cases; {RECOMMENDED_TASK_CASES}+ gives usable intervals",
            )
        )
    if CaseKind.TRIGGER not in kinds:
        out.append(
            Finding(
                Level.WARNING,
                "L002",
                "no trigger cases: cannot tell 'not loading' from 'not helping'",
            )
        )
    if CaseKind.NO_TRIGGER not in kinds:
        out.append(
            Finding(
                Level.WARNING,
                "L002",
                "no no-trigger cases: cannot catch a skill that loads too eagerly",
            )
        )
    if all(c.split is Split.DEV for c in suite.cases):
        out.append(
            Finding(
                Level.INFO, "L009", "no holdout cases; add some before comparing skill versions"
            )
        )
    uses_judge = any(g.type == "judge" for c in suite.cases for g in c.graders)
    if uses_judge and importlib.util.find_spec("anthropic") is None:
        out.append(
            Finding(Level.ERROR, "L010", "judge graders need: pip install 'skillrot[anthropic]'")
        )
    return out


def _case_level(case: Case, suite: Suite, shingles: set[str]) -> list[Finding]:
    out: list[Finding] = []
    if TODO_MARKER in case.prompt or any(TODO_MARKER in g.body for g in case.graders):
        out.append(Finding(Level.ERROR, "L008", "still has scaffold TODOs", case.id))
    if case.kind is CaseKind.TASK and not case.graders:
        out.append(Finding(Level.ERROR, "L003", "task case has no graders", case.id))
    if re.search(rf"\b{re.escape(suite.skill.name)}\b", case.prompt, re.IGNORECASE):
        out.append(
            Finding(Level.WARNING, "L006", "prompt names the skill; real users rarely do", case.id)
        )
    if shingles & _shingles(case.prompt):
        out.append(
            Finding(
                Level.WARNING,
                "L005",
                "prompt quotes the skill body; write cases from the task, not the skill",
                case.id,
            )
        )
    for grader in case.graders:
        out.extend(_grader_findings(grader, case))
    return out


def _grader_findings(grader: GraderSpec, case: Case) -> list[Finding]:
    if grader.type != "judge" and grader.type not in DETERMINISTIC_TYPES:
        return [
            Finding(
                Level.ERROR, "L007", f"grader {grader.id}: unknown type {grader.type!r}", case.id
            )
        ]
    reason = _cannot_fail(grader, case)
    if reason:
        return [
            Finding(Level.ERROR, "L004", f"grader {grader.id} can never fail: {reason}", case.id)
        ]
    return []


def _cannot_fail(grader: GraderSpec, case: Case) -> str | None:
    params = grader.params
    match grader.type:
        case "regex" if params.get("must_match", True):
            pattern = str(params.get("pattern", grader.body))
            try:
                if re.fullmatch(pattern, "") is not None or re.search(pattern, "") is not None:
                    return "pattern matches the empty string"
            except re.error:
                return None
        case "file_exists" if not params.get("absent") and case.files_dir is not None:
            if (case.files_dir / str(params.get("path", grader.body)).strip()).exists():
                return "the file is already in files/"
        case "command":
            if (grader.body or str(params.get("run", ""))).strip() in {"true", ":", "exit 0"}:
                return "the command always succeeds"
        case _:
            return None
    return None


def _shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}
