"""The verdict decision table. Ordered and total: the first matching row wins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillrot.models import Flag, SkillType, Verdict

if TYPE_CHECKING:
    from skillrot.config import SuiteConfig
    from skillrot.stats import Estimate

ADVICE: dict[Verdict, str] = {
    Verdict.UNTESTED: "Not enough data. Add cases or run more trials.",
    Verdict.HARMFUL: "The skill makes results worse on this model. Remove or rewrite it.",
    Verdict.ESSENTIAL: "Keep it. The skill clearly helps on this model.",
    Verdict.REDUNDANT: "The model does as well without it. Candidate for deletion.",
    Verdict.NOT_TRIGGERING: "The skill rarely loads. Fix its description, then re-check.",
    Verdict.INCONCLUSIVE: "Effect unclear. Re-run with --escalate for more trials.",
}


@dataclass(frozen=True, slots=True)
class Evidence:
    """Inputs to the decision table for one skill on one model."""

    uplift: Estimate | None
    trigger_rate: float | None  # share of SKILL-condition runs where the skill loaded
    content_uplift: Estimate | None = None  # FORCED vs NONE
    cost_per_pass_skill: float | None = None
    cost_per_pass_none: float | None = None
    contaminated: bool = False


def decide(evidence: Evidence, config: SuiteConfig) -> tuple[Verdict, frozenset[Flag]]:
    """Apply the ordered rules. See the README for the table."""
    return _verdict(evidence, config), _flags(evidence, config)


def _verdict(ev: Evidence, cfg: SuiteConfig) -> Verdict:
    u = ev.uplift
    eps = cfg.epsilon
    triggered = ev.trigger_rate is not None and ev.trigger_rate >= cfg.trigger_threshold
    if u is None or u.n < cfg.min_cases:
        return Verdict.UNTESTED
    if u.high < 0:
        return Verdict.HARMFUL
    if u.low > 0 and u.mean >= eps:
        return Verdict.ESSENTIAL
    if not triggered and _content_helps(ev.content_uplift, eps):
        return Verdict.NOT_TRIGGERING
    if u.low >= -eps and u.high <= eps:
        return Verdict.REDUNDANT if triggered else Verdict.NOT_TRIGGERING
    return Verdict.INCONCLUSIVE


def _content_helps(content: Estimate | None, eps: float) -> bool:
    return content is not None and content.low > 0 and content.mean >= eps


def _flags(ev: Evidence, cfg: SuiteConfig) -> frozenset[Flag]:
    flags: set[Flag] = set()
    if cfg.skill_type is SkillType.CONVENTION:
        flags.add(Flag.CONVENTION)
    skill_cost, none_cost = ev.cost_per_pass_skill, ev.cost_per_pass_none
    if skill_cost is not None and none_cost and skill_cost <= cfg.efficiency_ratio * none_cost:
        flags.add(Flag.EFFICIENCY)
    if ev.contaminated:
        flags.add(Flag.CONTAMINATED)
    return frozenset(flags)


def advice(verdict: Verdict, flags: frozenset[Flag]) -> str:
    """One-line next step for a verdict."""
    if verdict is Verdict.REDUNDANT and Flag.CONVENTION in flags:
        return "Same accuracy without it, but it encodes a team convention. Keep unless the convention changed."
    if verdict is Verdict.REDUNDANT and Flag.EFFICIENCY in flags:
        return "Same accuracy without it, but cheaper per passing run with it. Keep for cost."
    return ADVICE[verdict]
