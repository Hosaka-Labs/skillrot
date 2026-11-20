from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

from skillrot.errors import GraderError
from skillrot.graders import JudgeView, Outcome, grade_trial, run_deterministic
from skillrot.models import CaseKind, Split, ToolCall
from skillrot.suite import Case, GraderSpec


def outcome(
    ws: Path, text: str = "All done.", calls: tuple[ToolCall, ...] = (), loaded: bool = False
) -> Outcome:
    return Outcome(
        prompt="Summarize data.csv",
        final_text=text,
        workspace=ws,
        tool_calls=calls,
        skill_loaded=loaded,
    )


def check(spec: GraderSpec, out: Outcome) -> bool:
    return run_deterministic(spec, out)[0]


def g(type_: str, body: str = "", **params: object) -> GraderSpec:
    return GraderSpec(id="g", type=type_, params=dict(params), body=body)


def test_regex(tmp_path: Path):
    out = outcome(tmp_path, "Total: 42")
    assert check(g("regex", r"Total: \d+"), out)
    assert not check(g("regex", r"Total: 43"), out)
    assert check(g("regex", r"error", must_match=False), out)


def test_regex_on_missing_file_fails(tmp_path: Path):
    assert not check(g("regex", "x", file="nope.txt"), outcome(tmp_path))


def test_contains_and_file_exists(tmp_path: Path):
    (tmp_path / "out.md").write_text("Hello World")
    out = outcome(tmp_path)
    assert check(g("contains", "hello world", file="out.md", ignore_case=True), out)
    assert check(g("file_exists", path="out.md"), out)
    assert check(g("file_exists", path="gone.md", absent=True), out)


def test_path_escape_is_rejected(tmp_path: Path):
    with pytest.raises(GraderError, match="escapes"):
        check(g("file_exists", path="../../etc/passwd"), outcome(tmp_path))


def test_json_field(tmp_path: Path):
    (tmp_path / "s.json").write_text(
        json.dumps({"rows": 3, "totals": {"qty": 6.004}, "cols": ["a", "b"]})
    )
    out = outcome(tmp_path)
    assert check(g("json_field", file="s.json", path="rows", equals=3), out)
    assert check(g("json_field", file="s.json", path="totals.qty", approx=6.0, tolerance=0.01), out)
    assert check(g("json_field", file="s.json", path="cols[1]", equals="b"), out)
    assert check(g("json_field", file="s.json", path="cols", length=2), out)
    assert not check(g("json_field", file="s.json", path="totals.price"), out)
    (tmp_path / "bad.json").write_text("{nope")
    assert not check(g("json_field", file="bad.json", path="rows"), out)


def test_command(tmp_path: Path):
    out = outcome(tmp_path)
    assert check(g("command", "test -d ."), out)
    passed, evidence = run_deterministic(g("command", "echo boom; exit 3"), out)
    assert not passed
    assert "boom" in evidence


def test_tool_used(tmp_path: Path):
    out = outcome(tmp_path, calls=(ToolCall("Bash", {"command": "ls"}),))
    assert check(g("tool_used", name="Bash"), out)
    assert check(g("tool_used", name="Write", absent=True), out)


def test_unknown_type(tmp_path: Path):
    with pytest.raises(GraderError, match="unknown type"):
        check(g("telepathy"), outcome(tmp_path))


def _case(kind: CaseKind, graders: tuple[GraderSpec, ...] = ()) -> Case:
    return Case(id="c", prompt="p", kind=kind, split=Split.DEV, graders=graders, hash="h")


def test_trigger_cases_grade_on_loading(tmp_path: Path):
    hit = anyio.run(
        lambda: grade_trial(_case(CaseKind.TRIGGER), outcome(tmp_path, loaded=True), skill_name="s")
    )
    miss = anyio.run(
        lambda: grade_trial(
            _case(CaseKind.NO_TRIGGER), outcome(tmp_path, loaded=True), skill_name="s"
        )
    )
    assert hit[0].passed
    assert not miss[0].passed


def test_judge_grader_needs_a_judge(tmp_path: Path):
    case = _case(CaseKind.TASK, (g("judge", "It is correct"),))
    with pytest.raises(GraderError, match="judge model"):
        anyio.run(lambda: grade_trial(case, outcome(tmp_path), skill_name="s"))


def test_judge_sees_redacted_view(tmp_path: Path):
    (tmp_path / "notes.md").write_text("I used the csv-summary skill from SKILL.md")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "secret.md").write_text("hidden")
    seen: list[JudgeView] = []

    class FakeJudge:
        model = "fake"

        async def __call__(self, statement: str, view: JudgeView) -> tuple[bool, str]:
            seen.append(view)
            return True, "quote"

    case = _case(CaseKind.TASK, (g("judge", "It is correct"),))
    out = outcome(tmp_path, text="Loaded CSV-SUMMARY.")
    (result,) = anyio.run(
        lambda: grade_trial(case, out, skill_name="csv-summary", judge=FakeJudge())
    )
    assert result.passed
    assert result.judge == "fake"
    view = seen[0]
    assert "csv-summary" not in view.render().lower()
    assert "SKILL.md" not in view.render()
    assert ".claude/secret.md" not in view.files
