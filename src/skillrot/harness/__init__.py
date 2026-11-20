"""Harness registry. Third-party harnesses register under the `skillrot.harnesses` entry point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from skillrot.errors import ConfigError
from skillrot.harness.base import Harness, Transcript, TrialRequest

ENTRY_POINT_GROUP = "skillrot.harnesses"

__all__ = ["ENTRY_POINT_GROUP", "Harness", "Transcript", "TrialRequest", "available", "get_harness"]


def available() -> dict[str, str]:
    """Return registered harness names and their import paths."""
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def get_harness(name: str, **options: Any) -> Harness:  # noqa: ANN401
    """Instantiate a harness by registered name."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            factory = ep.load()
            harness = factory(**options)
            if not isinstance(harness, Harness):
                msg = f"{ep.value} does not implement the Harness protocol"
                raise ConfigError(msg)
            return harness
    known = ", ".join(sorted(available())) or "none installed"
    msg = f"unknown harness {name!r} (available: {known})"
    raise ConfigError(msg)
