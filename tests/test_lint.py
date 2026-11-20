from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from skillrot.errors import SuiteError
from skillrot.lint import Level, lint
from skillrot.scaffold import scaffold
from skillrot.suite import load_suite


def codes(path: Path, level: Level | None = None) -> set[str]:
    return {f.code for f in lint(load_suite(path)) if level is None or f.level is level}


def test_sample_is_clean(sample_skill: Path):
    assert not codes(sample_skill, Level.ERROR)


def test_scaffold_then_lint_flags_todos(tmp_path: Path, sample_skill: Path):
    shutil.rmtree(sample_skill / "evals")
    written = scaffold(sample_skill, task_brief="monthly sales exports")
    assert any(p.name == "skillrot.toml" for p in written)
    errors = codes(sample_skill, Level.ERROR)
    assert {"L001", "L008"} <= errors


def test_scaffold_refuses_existing(sample_skill: Path):
    with pytest.raises(SuiteError, match="already exists"):
        scaffold(sample_skill)


def test_cannot_fail_graders(sample_skill: Path):
    graders = sample_skill / "evals" / "plain-totals" / "graders"
    (graders / "09-empty.md").write_text("---\ntype: regex\npattern: '.*'\n---\n")
    (graders / "10-true.md").write_text("---\ntype: command\n---\ntrue\n")
    (graders / "11-present.md").write_text("---\ntype: file_exists\npath: orders.csv\n---\n")
    findings = [f for f in lint(load_suite(sample_skill)) if f.code == "L004"]
    assert len(findings) == 3


def test_prompt_naming_or_quoting_skill(sample_skill: Path):
    prompt = sample_skill / "evals" / "plain-totals" / "prompt.md"
    prompt.write_text(
        "Use csv-summary. Drop rows whose status column (any case) is void or cancelled please."
    )
    assert {"L005", "L006"} <= codes(sample_skill, Level.WARNING)
