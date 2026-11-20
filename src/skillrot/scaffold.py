"""`skillrot new`: create an evals/ folder to fill in. Never reads the skill body."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from skillrot.errors import SuiteError
from skillrot.lint import TODO_MARKER
from skillrot.suite import load_skill

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = """\
# skillrot settings for this skill. All keys are optional.
type = "capability"        # or "convention": encodes team choices a model can't infer
epsilon = 0.05             # smallest uplift worth keeping the skill for
min_cases = 10             # fewer paired task cases than this -> Untested
models = []                # e.g. ["anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5"]
harness = "claude-code"    # or "anthropic"
budget_usd = 20
"""

_TASK_PROMPT = f"""\
{TODO_MARKER}: write a real request a user would make, in their words.
Describe the task, not the skill. Put input files in files/.
"""

_TASK_CASE = """\
kind: task
split: dev
expected_outcome: "TODO: one sentence on what a correct result looks like"
"""

_TASK_GRADER = f"""\
---
type: command
---
{TODO_MARKER}: replace with a check that fails on a wrong result, e.g.
python3 -c "import json; assert json.load(open('out.json'))['total'] == 42"
"""


def scaffold(skill_path: Path, *, task_brief: str = "") -> list[Path]:
    """Create evals/ with settings, one task case and one trigger/no-trigger pair."""
    skill = load_skill(skill_path)
    evals = skill.path / "evals"
    if evals.exists() and any(evals.iterdir()):
        msg = f"{evals} already exists and is not empty"
        raise SuiteError(msg)
    brief = f"\n\nTask brief:\n{task_brief}" if task_brief else ""
    files = {
        "skillrot.toml": _CONFIG,
        "task-01/prompt.md": _TASK_PROMPT,
        "task-01/case.yaml": _TASK_CASE,
        "task-01/graders/01-correct-result.md": _TASK_GRADER,
        "task-01/files/.gitkeep": "",
        "trigger-01/prompt.md": f"{TODO_MARKER}: a request where this skill should load.\n"
        f"Skill description, for reference:\n{skill.description}{brief}\n",
        "trigger-01/case.yaml": "kind: trigger\n",
        "no-trigger-01/prompt.md": f"{TODO_MARKER}: a nearby request where this skill should NOT load.\n",
        "no-trigger-01/case.yaml": "kind: no-trigger\n",
    }
    written: list[Path] = []
    for rel, text in files.items():
        path = evals / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text))
        written.append(path)
    return written
