"""Per-skill settings, read from `evals/skillrot.toml`. Every key is optional."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

from skillrot.errors import ConfigError
from skillrot.models import SkillType

if TYPE_CHECKING:
    from pathlib import Path

CONFIG_FILENAME = "skillrot.toml"

# USD per million tokens, Anthropic first-party rates as of 2026-11-03.
# Used only for the API harness and for estimates; override under [prices] in skillrot.toml.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    """Settings that shape how a suite is run and judged."""

    skill_type: SkillType = SkillType.CAPABILITY
    epsilon: float = 0.05  # smallest uplift worth keeping a skill for (5 points)
    min_cases: int = 10  # fewer paired cases than this -> Untested
    trigger_threshold: float = 0.8  # skill must load in this share of runs to be judged Redundant
    efficiency_ratio: float = 0.8  # skill cost per pass <= 80% of baseline -> Efficiency flag
    harness: str = "claude-code"
    models: tuple[str, ...] = ()
    budget_usd: float = 20.0
    timeout_s: float = 600.0
    max_turns: int = 30
    judge_model: str = "claude-haiku-5-5"
    estimate_run_usd: float = 0.05  # used for cost estimates until real costs are in the store
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    def price_for(self, model_id: str) -> tuple[float, float] | None:
        """Return (input, output) USD per million tokens, if known."""
        return self.prices.get(model_id)

    def cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float | None:
        """Cost in USD for a token count, or None when the model has no known price."""
        price = self.price_for(model_id)
        if price is None:
            return None
        return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def load_config(evals_dir: Path) -> SuiteConfig:
    """Load `skillrot.toml` from an evals folder, or the defaults if there is none."""
    path = evals_dir / CONFIG_FILENAME
    if not path.exists():
        return SuiteConfig()
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc
    return _from_mapping(raw, path)


def _from_mapping(raw: dict[str, Any], path: Path) -> SuiteConfig:
    base = SuiteConfig()
    known = {f.name for f in fields(SuiteConfig)}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        match key:
            case "type":
                try:
                    updates["skill_type"] = SkillType(value)
                except ValueError as exc:
                    msg = f"{path}: type must be one of {[t.value for t in SkillType]}"
                    raise ConfigError(msg) from exc
            case "models":
                updates["models"] = tuple(str(m) for m in value)
            case "judge":
                updates["judge_model"] = str(value.get("model", base.judge_model))
            case "prices":
                prices = dict(base.prices)
                for model_id, spec in value.items():
                    prices[model_id] = (float(spec["input"]), float(spec["output"]))
                updates["prices"] = prices
            case _ if key in known:
                updates[key] = value
            case _:
                msg = f"{path}: unknown key {key!r}"
                raise ConfigError(msg)
    return replace(base, **updates)
