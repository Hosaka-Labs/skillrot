"""Graders decide pass or fail for one trial.

Deterministic graders run first and are preferred. The `judge` grader asks a pinned model
to check one statement against a redacted view of the outcome: it never sees the condition,
the skill's name or the skill's text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import anyio

from skillrot.errors import GraderError
from skillrot.models import CaseKind, ToolCall

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from skillrot.suite import Case, GraderSpec

MAX_EVIDENCE = 400
MAX_JUDGE_FILE_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class Outcome:
    """The finished state of a trial, as graders see it."""

    prompt: str
    final_text: str
    workspace: Path
    tool_calls: tuple[ToolCall, ...]
    skill_loaded: bool


@dataclass(frozen=True, slots=True)
class GradeResult:
    """One grader's decision."""

    grader_id: str
    grader_hash: str
    passed: bool
    evidence: str = ""
    judge: str = ""


class Judge(Protocol):
    """Model-backed check of one statement. See `AnthropicJudge`."""

    model: str

    async def __call__(self, statement: str, view: JudgeView) -> tuple[bool, str]:
        """Return (passed, quoted evidence)."""
        ...


@dataclass(frozen=True, slots=True)
class JudgeView:
    """What a judge may see. Built by `JudgeView.build`, which redacts the skill."""

    prompt: str
    final_text: str
    files: Mapping[str, str]

    @classmethod
    def build(cls, outcome: Outcome, skill_name: str) -> JudgeView:
        """Collect text files and redact anything that would reveal the condition."""
        files: dict[str, str] = {}
        for path in sorted(outcome.workspace.rglob("*")):
            rel = path.relative_to(outcome.workspace)
            if not path.is_file() or rel.parts[0].startswith("."):
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            files[rel.as_posix()] = _redact(text[:MAX_JUDGE_FILE_CHARS], skill_name)
        return cls(
            prompt=_redact(outcome.prompt, skill_name),
            final_text=_redact(outcome.final_text, skill_name),
            files=files,
        )

    def render(self) -> str:
        """Plain-text rendering for the judge prompt."""
        parts = [
            f"<task>\n{self.prompt}\n</task>",
            f"<final_message>\n{self.final_text}\n</final_message>",
        ]
        parts.extend(f'<file path="{p}">\n{t}\n</file>' for p, t in self.files.items())
        return "\n\n".join(parts)


def _redact(text: str, skill_name: str) -> str:
    text = re.sub(re.escape(skill_name), "[redacted]", text, flags=re.IGNORECASE)
    return re.sub(r"\S*SKILL\.md", "[redacted]", text)


async def grade_trial(
    case: Case, outcome: Outcome, *, skill_name: str, judge: Judge | None = None
) -> list[GradeResult]:
    """Run every grader for a case. Trigger cases are graded on whether the skill loaded."""
    if case.kind is CaseKind.TRIGGER:
        return [_trigger_result(outcome, should_load=True)]
    if case.kind is CaseKind.NO_TRIGGER:
        return [_trigger_result(outcome, should_load=False)]
    if not case.graders:
        msg = f"{case.id}: task case has no graders"
        raise GraderError(msg)
    results: list[GradeResult] = []
    for spec in case.graders:
        if spec.type == "judge":
            results.append(await _judge(spec, outcome, skill_name, judge))
        else:
            passed, evidence = await anyio.to_thread.run_sync(run_deterministic, spec, outcome)
            results.append(GradeResult(spec.id, spec.hash, passed, evidence[:MAX_EVIDENCE]))
    return results


def _trigger_result(outcome: Outcome, *, should_load: bool) -> GradeResult:
    loaded = outcome.skill_loaded
    return GradeResult(
        grader_id="trigger" if should_load else "no-trigger",
        grader_hash="builtin",
        passed=loaded is should_load,
        evidence=f"skill {'loaded' if loaded else 'not loaded'}",
    )


async def _judge(
    spec: GraderSpec, outcome: Outcome, skill_name: str, judge: Judge | None
) -> GradeResult:
    if judge is None:
        msg = (
            f"grader {spec.id} needs a judge model: install `skillrot[anthropic]` "
            "and set ANTHROPIC_API_KEY, or replace it with a deterministic grader"
        )
        raise GraderError(msg)
    statement = spec.body or str(spec.params.get("statement", ""))
    if not statement:
        msg = f"judge grader {spec.id} has no statement"
        raise GraderError(msg)
    passed, evidence = await judge(statement, JudgeView.build(outcome, skill_name))
    return GradeResult(spec.id, spec.hash, passed, evidence[:MAX_EVIDENCE], judge=judge.model)


# --- deterministic graders -------------------------------------------------------------


def run_deterministic(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    """Dispatch a non-judge grader. Raises GraderError on bad config."""
    fn = _DETERMINISTIC.get(spec.type)
    if fn is None:
        known = ", ".join(sorted([*_DETERMINISTIC, "judge"]))
        msg = f"grader {spec.id}: unknown type {spec.type!r} (known: {known})"
        raise GraderError(msg)
    return fn(spec, outcome)


def _target_text(spec: GraderSpec, outcome: Outcome) -> tuple[str | None, str]:
    """Text a grader checks: the agent's final message, or a file it wrote."""
    file = spec.params.get("file")
    if file is None:
        return outcome.final_text, "final message"
    path = _safe_path(outcome.workspace, str(file))
    if not path.is_file():
        return None, f"{file} does not exist"
    return path.read_text(errors="replace"), str(file)


def _safe_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        msg = f"path {rel!r} escapes the workspace"
        raise GraderError(msg)
    return path


def _regex(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    pattern = str(spec.params.get("pattern", spec.body))
    if not pattern:
        msg = f"regex grader {spec.id} has no pattern"
        raise GraderError(msg)
    flags = re.IGNORECASE if spec.params.get("ignore_case") else 0
    try:
        compiled = re.compile(pattern, flags | re.MULTILINE)
    except re.error as exc:
        msg = f"regex grader {spec.id}: {exc}"
        raise GraderError(msg) from exc
    text, where = _target_text(spec, outcome)
    if text is None:
        return False, where
    match = compiled.search(text)
    want = bool(spec.params.get("must_match", True))
    found = f"matched {match.group(0)!r} in {where}" if match else f"no match in {where}"
    return (match is not None) is want, found


def _contains(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    needle = str(spec.params.get("text", spec.body))
    if not needle:
        msg = f"contains grader {spec.id} has no text"
        raise GraderError(msg)
    text, where = _target_text(spec, outcome)
    if text is None:
        return False, where
    if spec.params.get("ignore_case"):
        text, needle = text.lower(), needle.lower()
    present = needle in text
    want = bool(spec.params.get("must_contain", True))
    return present is want, f"{'found' if present else 'did not find'} text in {where}"


def _file_exists(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    rel = str(spec.params.get("path", spec.body)).strip()
    if not rel:
        msg = f"file_exists grader {spec.id} has no path"
        raise GraderError(msg)
    exists = _safe_path(outcome.workspace, rel).exists()
    want = not spec.params.get("absent", False)
    return exists is want, f"{rel} {'exists' if exists else 'missing'}"


_MISSING = object()


def _json_field(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    if "file" not in spec.params:
        msg = f"json_field grader {spec.id} needs `file`"
        raise GraderError(msg)
    text, where = _target_text(spec, outcome)
    if text is None:
        return False, where
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"{where} is not valid JSON: {exc}"
    path = str(spec.params.get("path", ""))
    value = _dig(data, path)
    if value is _MISSING:
        return False, f"{path or '<root>'} not found in {where}"
    shown = f"{path or '<root>'} = {json.dumps(value)[:120]}"
    if "equals" in spec.params:
        return value == spec.params["equals"], shown
    if "approx" in spec.params:
        tolerance = float(spec.params.get("tolerance", 1e-6))
        ok = (
            isinstance(value, int | float)
            and abs(float(value) - float(spec.params["approx"])) <= tolerance
        )
        return ok, shown
    if "length" in spec.params:
        ok = (
            isinstance(value, list | dict | str)
            and len(cast("list[Any]", value)) == spec.params["length"]
        )
        return ok, shown
    return True, shown  # `path` alone checks that the field exists


def _dig(data: object, path: str) -> object:
    current = data
    for part in filter(None, re.split(r"\.|\[(\d+)\]", path)):
        if isinstance(current, dict):
            current = cast("dict[str, Any]", current).get(part, _MISSING)
        elif isinstance(current, list) and part.isdigit():
            items = cast("list[Any]", current)
            current = items[int(part)] if int(part) < len(items) else _MISSING
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _command(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    import subprocess  # noqa: PLC0415 - only this grader needs it

    command = spec.body or str(spec.params.get("run", ""))
    if not command:
        msg = f"command grader {spec.id} has no command"
        raise GraderError(msg)
    timeout = float(spec.params.get("timeout_s", 60))
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=outcome.workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"command timed out after {timeout:g}s"
    want = int(spec.params.get("expect_exit", 0))
    tail = (proc.stdout + proc.stderr).strip()[-MAX_EVIDENCE:]
    return proc.returncode == want, f"exit {proc.returncode}: {tail}"


def _tool_used(spec: GraderSpec, outcome: Outcome) -> tuple[bool, str]:
    name = str(spec.params.get("name", spec.body)).strip()
    if not name:
        msg = f"tool_used grader {spec.id} has no tool name"
        raise GraderError(msg)
    used = any(call.name == name for call in outcome.tool_calls)
    want = not spec.params.get("absent", False)
    return used is want, f"{name} {'was' if used else 'was not'} called"


_DETERMINISTIC = {
    "regex": _regex,
    "contains": _contains,
    "file_exists": _file_exists,
    "json_field": _json_field,
    "command": _command,
    "tool_used": _tool_used,
}

DETERMINISTIC_TYPES = frozenset(_DETERMINISTIC)
