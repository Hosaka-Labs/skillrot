"""Turn stored trials into verdicts, and verdicts into a versioned result bundle.

The bundle (`ResultBundle`) is the public contract for other programs, such as a web app
that charts skills over time. Its schema is versioned; the store's is not.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from statistics import fmean
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from skillrot.models import CaseKind, Condition, Flag, Verdict
from skillrot.stats import Estimate, case_means, paired_drift, paired_uplift, rate
from skillrot.verdicts import Evidence, advice, decide

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skillrot.store import Store, TrialRecord
    from skillrot.suite import Suite

BUNDLE_SCHEMA_VERSION = "1"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Interval(_Frozen):
    """A difference in pass rate with its 90% bootstrap interval, in [-1, 1]."""

    mean: float
    low: float
    high: float
    n_cases: int
    mde: float = Field(description="Smallest effect detectable with 80% power")

    @classmethod
    def of(cls, est: Estimate | None) -> Interval | None:
        """Convert a stats estimate."""
        if est is None:
            return None
        return cls(mean=est.mean, low=est.low, high=est.high, n_cases=est.n, mde=est.mde)


class ConditionStats(_Frozen):
    """Per-condition totals on one model."""

    condition: Condition
    trials: int
    pass_rate: float | None
    cost_usd: float | None
    cost_per_pass_usd: float | None
    tokens_per_pass: float | None
    mean_wall_s: float | None


class ModelResult(_Frozen):
    """The verdict for one skill on one model and harness."""

    model: str
    harness: str
    verdict: Verdict
    flags: list[Flag]
    advice: str
    uplift: Interval | None = Field(description="SKILL minus NONE pass rate, paired by case")
    content_uplift: Interval | None = Field(
        description="FORCED minus NONE: value if it always loaded"
    )
    vs_prior: Interval | None = Field(
        default=None, description="SKILL minus PRIOR: did the edit help"
    )
    trigger_rate: float | None
    trigger_accuracy: float | None = Field(description="Share of trigger/no-trigger cases right")
    conditions: list[ConditionStats]
    missing_trials: int = Field(description="Infrastructure failures excluded from the statistics")
    last_run: str | None


class Drift(_Frozen):
    """Change in uplift between two models on the same cases."""

    from_model: str
    to_model: str
    change: Interval | None
    verdict_from: Verdict
    verdict_to: Verdict
    significant: bool


class SkillInfo(_Frozen):
    """Identity of the skill version that was evaluated."""

    name: str
    hash: str
    type: str
    description: str
    n_cases: int
    n_task_cases: int


class ResultBundle(_Frozen):
    """Everything known about one skill, ready to serialise."""

    schema_version: Literal["1"] = BUNDLE_SCHEMA_VERSION
    generated_at: str
    skillrot_version: str
    skill: SkillInfo
    epsilon: float
    results: list[ModelResult]
    drift: list[Drift] = []


def _latest(records: Iterable[TrialRecord]) -> list[TrialRecord]:
    """Keep the newest record per (case, condition, trial index), across harness versions."""
    seen: dict[tuple[str, Condition, int], TrialRecord] = {}
    for r in records:  # store returns newest first
        seen.setdefault((r.case_id, r.condition, r.trial_index), r)
    return list(seen.values())


def analyze_model(suite: Suite, store: Store, model: str, harness: str) -> ModelResult | None:
    """Compute the verdict for one model from what is in the store."""
    records = _records(suite, store, model, harness)
    if not records:
        return None
    task_ids = {c.id for c in suite.task_cases}
    data = [r for r in records if r.exit_reason.counts_as_data and r.passed is not None]
    passes: dict[Condition, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in data:
        if r.case_id in task_ids:
            passes[r.condition][r.case_id].append(bool(r.passed))
    means = {cond: case_means(cases) for cond, cases in passes.items()}
    uplift = _pair(means, Condition.SKILL, Condition.NONE)
    content = _pair(means, Condition.FORCED, Condition.NONE)
    vs_prior = _pair(means, Condition.SKILL, Condition.PRIOR)

    skill_runs = [r for r in data if r.condition is Condition.SKILL]
    trigger_rate = rate(
        [r.skill_loaded for r in skill_runs if r.case_id in task_ids or _is_trigger(suite, r)]
    )
    trig = [r for r in data if r.case_id not in task_ids]
    trigger_accuracy = rate([bool(r.passed) for r in trig])
    contaminated = any(r.skill_loaded for r in data if r.condition is Condition.NONE)

    stats = [
        _condition_stats(cond, [r for r in data if r.condition is cond and r.case_id in task_ids])
        for cond in Condition
    ]
    stats = [s for s in stats if s.trials]
    cost = {s.condition: s.cost_per_pass_usd for s in stats}
    evidence = Evidence(
        uplift=uplift,
        trigger_rate=trigger_rate,
        content_uplift=content,
        cost_per_pass_skill=cost.get(Condition.SKILL),
        cost_per_pass_none=cost.get(Condition.NONE),
        contaminated=contaminated,
    )
    verdict, flags = decide(evidence, suite.config)
    return ModelResult(
        model=model,
        harness=harness,
        verdict=verdict,
        flags=sorted(flags),
        advice=advice(verdict, flags),
        uplift=Interval.of(uplift),
        content_uplift=Interval.of(content),
        vs_prior=Interval.of(vs_prior),
        trigger_rate=trigger_rate,
        trigger_accuracy=trigger_accuracy,
        conditions=stats,
        missing_trials=sum(1 for r in records if not r.exit_reason.counts_as_data),
        last_run=max((r.created_at for r in records), default=None),
    )


def _records(suite: Suite, store: Store, model: str, harness: str) -> list[TrialRecord]:
    """Newest trials for this suite's cases and this skill version (NONE and PRIOR don't depend on it)."""
    rows = store.trials(
        suite.skill.name, model=model, harness=harness, case_hashes={c.hash for c in suite.cases}
    )
    current = suite.skill.hash
    keep = [
        r
        for r in rows
        if r.condition in {Condition.NONE, Condition.PRIOR} or r.skill_hash == current
    ]
    return _latest(keep)


def _is_trigger(suite: Suite, record: TrialRecord) -> bool:
    return any(c.id == record.case_id and c.kind is CaseKind.TRIGGER for c in suite.cases)


def _pair(means: dict[Condition, dict[str, float]], a: Condition, b: Condition) -> Estimate | None:
    if a not in means or b not in means:
        return None
    return paired_uplift(means[a], means[b])


def _condition_stats(condition: Condition, records: list[TrialRecord]) -> ConditionStats:
    passed = [r for r in records if r.passed]
    costs = [r.usage.cost_usd for r in records if r.usage.cost_usd is not None]
    total_cost = sum(costs) if costs and len(costs) == len(records) else None
    tokens = sum(r.usage.input_tokens + r.usage.output_tokens for r in records)
    return ConditionStats(
        condition=condition,
        trials=len(records),
        pass_rate=rate([bool(r.passed) for r in records]),
        cost_usd=total_cost,
        cost_per_pass_usd=total_cost / len(passed) if total_cost is not None and passed else None,
        tokens_per_pass=tokens / len(passed) if passed else None,
        mean_wall_s=fmean(r.usage.wall_s for r in records) if records else None,
    )


def per_case_uplift(suite: Suite, store: Store, model: str, harness: str) -> dict[str, float]:
    """SKILL minus NONE pass rate for each task case, used for drift."""
    records = _records(suite, store, model, harness)
    task_ids = {c.id for c in suite.task_cases}
    by: dict[Condition, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.case_id in task_ids and r.exit_reason.counts_as_data and r.passed is not None:
            by[r.condition][r.case_id].append(bool(r.passed))
    skill, none = case_means(by[Condition.SKILL]), case_means(by[Condition.NONE])
    return {c: skill[c] - none[c] for c in skill.keys() & none.keys()}


def drift(suite: Suite, store: Store, from_result: ModelResult, to_result: ModelResult) -> Drift:
    """Paired change in uplift between two models."""
    old = per_case_uplift(suite, store, from_result.model, from_result.harness)
    new = per_case_uplift(suite, store, to_result.model, to_result.harness)
    change = paired_drift(old, new)
    eps = suite.config.epsilon
    significant = change is not None and (change.low > eps or change.high < -eps)
    return Drift(
        from_model=from_result.model,
        to_model=to_result.model,
        change=Interval.of(change),
        verdict_from=from_result.verdict,
        verdict_to=to_result.verdict,
        significant=significant or from_result.verdict is not to_result.verdict,
    )


def analyze(
    suite: Suite, store: Store, *, models: Iterable[str] | None = None, harness: str | None = None
) -> ResultBundle:
    """Build the result bundle for a suite from every stored model (or the ones given)."""
    pairs = [(m, h) for m, h, _, _ in store.models(suite.skill.name) if harness in {None, h}]
    if models is not None:
        wanted = list(models)
        pairs = [p for p in pairs if p[0] in wanted]
        pairs.sort(key=lambda p: wanted.index(p[0]))
    results = [r for m, h in pairs if (r := analyze_model(suite, store, m, h))]
    drifts = [
        drift(suite, store, a, b) for a, b in itertools.pairwise(results) if a.harness == b.harness
    ]
    skill = suite.skill
    return ResultBundle(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        skillrot_version=_version(),
        skill=SkillInfo(
            name=skill.name,
            hash=skill.hash,
            type=suite.config.skill_type.value,
            description=skill.description,
            n_cases=len(suite.cases),
            n_task_cases=len(suite.task_cases),
        ),
        epsilon=suite.config.epsilon,
        results=results,
        drift=drifts,
    )


def _version() -> str:
    try:
        return version("skillrot")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0+unknown"
