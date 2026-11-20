"""Load a skill folder and its eval cases.

Layout (compatible with the `claude plugin eval` case-folder style)::

    my-skill/
    ├── SKILL.md
    └── evals/
        ├── skillrot.toml                # optional settings
        └── <case-id>/
            ├── prompt.md                # the user prompt
            ├── case.yaml                # kind, split, expected_outcome, graders (optional)
            ├── files/                   # copied into the agent's working directory
            └── graders/*.md             # one grader per file: YAML front matter + body

Older skill-creator suites (`evals/evals.json`) are read through the same types.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from skillrot.config import SuiteConfig, load_config
from skillrot.errors import SuiteError
from skillrot.models import CaseKind, Split, content_hash, hash_json, hash_tree

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Skill:
    """A SKILL.md folder."""

    path: Path
    name: str
    description: str
    body: str
    hash: str

    @property
    def sections(self) -> tuple[str, ...]:
        """Second-level headings in the body, in order."""
        return tuple(_SECTION.findall(self.body))


@dataclass(frozen=True, slots=True)
class GraderSpec:
    """Declarative grader config. `type` picks the implementation in `skillrot.graders`."""

    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    body: str = ""

    @property
    def hash(self) -> str:
        """Changes whenever the grader's behaviour could change."""
        return hash_json({"type": self.type, "params": self.params, "body": self.body})


@dataclass(frozen=True, slots=True)
class Case:
    """One frozen test case."""

    id: str
    prompt: str
    kind: CaseKind
    split: Split
    graders: tuple[GraderSpec, ...]
    hash: str
    expected_outcome: str = ""
    files_dir: Path | None = None
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class Suite:
    """A skill, its cases and its settings."""

    skill: Skill
    cases: tuple[Case, ...]
    config: SuiteConfig

    @property
    def task_cases(self) -> tuple[Case, ...]:
        """Cases that measure outcome quality."""
        return tuple(c for c in self.cases if c.kind is CaseKind.TASK)

    def case(self, case_id: str) -> Case:
        """Look up a case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        msg = f"no case {case_id!r} in {self.skill.name}"
        raise SuiteError(msg)


def _yaml(text: str, where: str) -> object:
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        msg = f"{where}: invalid YAML: {exc}"
        raise SuiteError(msg) from exc


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown file into its YAML front matter and body."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta = _yaml(match.group(1), "front matter")
    if not isinstance(meta, dict):
        msg = "front matter must be a mapping"
        raise SuiteError(msg)
    return cast("dict[str, Any]", meta), match.group(2)


def load_skill(path: Path) -> Skill:
    """Load a skill from its folder (or from a SKILL.md path)."""
    folder = path.parent if path.name == "SKILL.md" else path
    skill_md = folder / "SKILL.md"
    if not skill_md.is_file():
        msg = f"{folder}: no SKILL.md found"
        raise SuiteError(msg)
    meta, body = split_front_matter(skill_md.read_text())
    name, description = meta.get("name"), meta.get("description")
    if not name or not description:
        msg = f"{skill_md}: front matter needs `name` and `description`"
        raise SuiteError(msg)
    return Skill(
        path=folder.resolve(),
        name=str(name),
        description=str(description).strip(),
        body=body.strip(),
        hash=_skill_hash(folder),
    )


def _skill_hash(folder: Path) -> str:
    parts: list[bytes | str] = []
    for file in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = file.relative_to(folder)
        if rel.parts[0] in {"evals", ".skillrot"} or rel.name == ".DS_Store":
            continue
        parts.extend((rel.as_posix(), file.read_bytes()))
    return content_hash(*parts)


def load_suite(path: str | Path) -> Suite:
    """Load a skill and every case under its `evals/` folder."""
    skill = load_skill(Path(path))
    evals = skill.path / "evals"
    config = load_config(evals)
    cases: list[Case] = []
    if (evals / "evals.json").is_file():
        cases.extend(_load_skill_creator(evals / "evals.json", skill.path))
    if evals.is_dir():
        cases.extend(
            _load_case_dir(d) for d in sorted(evals.iterdir()) if (d / "prompt.md").is_file()
        )
    ids = [c.id for c in cases]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        msg = f"{evals}: duplicate case ids {sorted(dupes)}"
        raise SuiteError(msg)
    return Suite(skill=skill, cases=tuple(cases), config=config)


def _load_case_dir(folder: Path) -> Case:
    meta: dict[str, Any] = {}
    if (folder / "case.yaml").is_file():
        loaded = _yaml((folder / "case.yaml").read_text(), str(folder / "case.yaml"))
        if not isinstance(loaded, dict):
            msg = f"{folder / 'case.yaml'}: must be a mapping"
            raise SuiteError(msg)
        meta = cast("dict[str, Any]", loaded)
    graders = [_inline_grader(folder.name, i, g) for i, g in enumerate(meta.get("graders", []))]
    graders_dir = folder / "graders"
    if graders_dir.is_dir():
        graders.extend(_file_grader(p) for p in sorted(graders_dir.glob("*.md")))
    files_dir = folder / "files"
    try:
        kind = CaseKind(meta.get("kind", CaseKind.TASK))
        split = Split(meta.get("split", Split.DEV))
    except ValueError as exc:
        msg = f"{folder}: {exc}"
        raise SuiteError(msg) from exc
    timeout = meta.get("timeout_s")
    return Case(
        id=folder.name,
        prompt=(folder / "prompt.md").read_text().strip(),
        kind=kind,
        split=split,
        graders=tuple(graders),
        hash=hash_tree(folder),
        expected_outcome=str(meta.get("expected_outcome", "")),
        files_dir=files_dir if files_dir.is_dir() else None,
        timeout_s=float(timeout) if timeout is not None else None,
    )


def _inline_grader(case_id: str, index: int, raw: object) -> GraderSpec:
    if not isinstance(raw, dict) or "type" not in raw:
        msg = f"{case_id}: each inline grader needs a `type`"
        raise SuiteError(msg)
    params = dict(cast("dict[str, Any]", raw))
    kind = str(params.pop("type"))
    body = str(params.pop("body", ""))
    grader_id = str(params.pop("id", f"{index:02d}-{kind}"))
    return GraderSpec(id=grader_id, type=kind, params=params, body=body)


def _file_grader(path: Path) -> GraderSpec:
    try:
        meta, body = split_front_matter(path.read_text())
    except SuiteError as exc:
        msg = f"{path}: {exc}"
        raise SuiteError(msg) from exc
    if "type" not in meta:
        msg = f"{path}: front matter needs a `type`"
        raise SuiteError(msg)
    kind = str(meta.pop("type"))
    return GraderSpec(id=path.stem, type=kind, params=meta, body=body.strip())


def _load_skill_creator(path: Path, skill_dir: Path) -> list[Case]:
    """Map skill-creator's evals.json; `expectations` become rubric-judge graders."""
    data = json.loads(path.read_text())
    cases: list[Case] = []
    for item in data.get("evals", []):
        expectations = [str(e) for e in item.get("expectations", [])]
        graders = tuple(
            GraderSpec(id=f"expectation-{i:02d}", type="judge", body=text)
            for i, text in enumerate(expectations)
        )
        files = [skill_dir / f for f in item.get("files", [])]
        cases.append(
            Case(
                id=f"evals-json-{item['id']}",
                prompt=str(item["prompt"]),
                kind=CaseKind.TASK,
                split=Split.DEV,
                graders=graders,
                hash=hash_json({"item": item, "files": [hash_tree(f) for f in files]}),
                expected_outcome=str(item.get("expected_output", "")),
                files_dir=_common_dir(files),
            )
        )
    return cases


def _common_dir(files: list[Path]) -> Path | None:
    dirs = {f.parent for f in files}
    return dirs.pop() if len(dirs) == 1 else None


def discover(paths: list[Path], *, max_depth: int = 4) -> list[Path]:
    """Skill folders at or under the given paths (any folder holding a SKILL.md)."""
    found: list[Path] = []
    for root in paths:
        if (root / "SKILL.md").is_file() or root.name == "SKILL.md":
            found.append(root.parent if root.name == "SKILL.md" else root)
            continue
        for skill_md in sorted(root.rglob("SKILL.md")):
            rel = skill_md.relative_to(root)
            if len(rel.parts) <= max_depth + 1 and "evals" not in rel.parts:
                found.append(skill_md.parent)
    unique = dict.fromkeys(p.resolve() for p in found)
    return list(unique)
