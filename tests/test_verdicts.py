from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from skillrot.config import SuiteConfig
from skillrot.models import Flag, SkillType, Verdict
from skillrot.stats import Estimate
from skillrot.verdicts import Evidence, advice, decide

CFG = SuiteConfig()


def est(mean: float, low: float, high: float, n: int = 20) -> Estimate:
    return Estimate(mean, low, high, n, 0.1)


def verdict(**kwargs: object) -> Verdict:
    ev = Evidence(**{"trigger_rate": 1.0, **kwargs})  # type: ignore[arg-type]
    return decide(ev, CFG)[0]


def test_untested_without_data():
    assert verdict(uplift=None) is Verdict.UNTESTED


def test_untested_with_too_few_cases():
    assert verdict(uplift=est(0.5, 0.3, 0.7, n=5)) is Verdict.UNTESTED


def test_harmful_when_interval_below_zero():
    assert verdict(uplift=est(-0.2, -0.3, -0.05)) is Verdict.HARMFUL


def test_essential_when_clearly_positive():
    assert verdict(uplift=est(0.3, 0.1, 0.5)) is Verdict.ESSENTIAL


def test_positive_but_below_epsilon_is_not_essential():
    assert verdict(uplift=est(0.02, 0.01, 0.04)) is Verdict.REDUNDANT


def test_redundant_needs_triggering():
    assert verdict(uplift=est(0.0, -0.03, 0.03), trigger_rate=0.3) is Verdict.NOT_TRIGGERING


def test_not_triggering_when_forced_content_helps():
    v = verdict(uplift=est(0.02, -0.1, 0.15), trigger_rate=0.2, content_uplift=est(0.4, 0.2, 0.6))
    assert v is Verdict.NOT_TRIGGERING


def test_inconclusive_when_wide():
    assert verdict(uplift=est(0.03, -0.2, 0.25)) is Verdict.INCONCLUSIVE


@given(
    mean=st.floats(-1, 1),
    spread=st.floats(0, 1),
    n=st.integers(0, 40),
    trig=st.none() | st.floats(0, 1),
)
def test_table_is_total(mean: float, spread: float, n: int, trig: float | None):
    ev = Evidence(uplift=est(mean, mean - spread, mean + spread, n), trigger_rate=trig)
    v, _ = decide(ev, CFG)
    assert isinstance(v, Verdict)


def test_flags():
    cfg = SuiteConfig(skill_type=SkillType.CONVENTION)
    ev = Evidence(
        uplift=est(0, -0.01, 0.01),
        trigger_rate=1.0,
        cost_per_pass_skill=0.5,
        cost_per_pass_none=1.0,
        contaminated=True,
    )
    v, flags = decide(ev, cfg)
    assert v is Verdict.REDUNDANT
    assert flags == {Flag.CONVENTION, Flag.EFFICIENCY, Flag.CONTAMINATED}
    assert "convention" in advice(v, flags)
