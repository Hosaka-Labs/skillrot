from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from skillrot.store import Store

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "skills" / "csv-summary"
FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"


@pytest.fixture
def sample_skill(tmp_path: Path) -> Path:
    """A private copy of the sample skill, safe to edit."""
    target = tmp_path / "skills" / "csv-summary"
    shutil.copytree(SAMPLE, target)
    return target


@pytest.fixture
def store(tmp_path: Path):
    with Store.open(tmp_path / "store") as db:
        yield db


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SKILLROT_CLAUDE", str(FAKE_CLAUDE))
    return FAKE_CLAUDE
