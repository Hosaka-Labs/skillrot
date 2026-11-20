from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillrot.errors import ConfigError, SuiteError
from skillrot.models import CaseKind, SkillType, Split
from skillrot.suite import discover, load_suite


def test_loads_sample(sample_skill: Path):
    suite = load_suite(sample_skill)
    assert suite.skill.name == "csv-summary"
    assert len(suite.task_cases) == 16
    assert {c.kind for c in suite.cases} == set(CaseKind)
    assert any(c.split is Split.HOLDOUT for c in suite.cases)
    assert suite.config.models == ("anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5")
    assert "Numbers" in suite.skill.sections


def test_skill_hash_ignores_evals(sample_skill: Path):
    before = load_suite(sample_skill).skill.hash
    (sample_skill / "evals" / "plain-totals" / "prompt.md").write_text("changed")
    assert load_suite(sample_skill).skill.hash == before


def test_case_hash_changes_with_content(sample_skill: Path):
    h1 = load_suite(sample_skill).case("plain-totals").hash
    (sample_skill / "evals" / "plain-totals" / "prompt.md").write_text("changed")
    assert load_suite(sample_skill).case("plain-totals").hash != h1


def test_missing_front_matter(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("# no front matter")
    with pytest.raises(SuiteError, match="name"):
        load_suite(tmp_path)


def test_bad_case_yaml_is_a_suite_error(sample_skill: Path):
    (sample_skill / "evals" / "plain-totals" / "case.yaml").write_text("kind: [unclosed")
    with pytest.raises(SuiteError, match="invalid YAML"):
        load_suite(sample_skill)


def test_unknown_config_key(sample_skill: Path):
    (sample_skill / "evals" / "skillrot.toml").write_text("colour = 'blue'\n")
    with pytest.raises(ConfigError, match="colour"):
        load_suite(sample_skill)


def test_config_type_and_prices(sample_skill: Path):
    (sample_skill / "evals" / "skillrot.toml").write_text(
        'type = "convention"\n[prices."my-model"]\ninput = 1.0\noutput = 2.0\n'
    )
    cfg = load_suite(sample_skill).config
    assert cfg.skill_type is SkillType.CONVENTION
    assert cfg.cost("my-model", 1_000_000, 1_000_000) == 3.0


def test_skill_creator_evals_json(tmp_path: Path):
    (tmp_path / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill.\n---\nbody")
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "evals": [
                    {
                        "id": 1,
                        "prompt": "do it",
                        "expected_output": "done",
                        "expectations": ["It is done"],
                    }
                ],
            }
        )
    )
    suite = load_suite(tmp_path)
    (case,) = suite.cases
    assert case.id == "evals-json-1"
    assert case.graders[0].type == "judge"
    assert case.graders[0].body == "It is done"


def test_discover(tmp_path: Path, sample_skill: Path):
    assert discover([tmp_path]) == [sample_skill.resolve()]
    assert discover([sample_skill / "SKILL.md"]) == [sample_skill]
