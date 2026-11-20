"""Plan a run matrix, then execute it: concurrent, resumable, cached and budget-capped."""

from __future__ import annotations

import json
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import anyio

from skillrot.errors import BudgetExceededError, ConfigError
from skillrot.graders import Judge, Outcome, grade_trial
from skillrot.harness.base import Harness, TrialRequest
from skillrot.models import CaseKind, Condition, ExitReason, ModelRef, Tier, Usage, hash_json
from skillrot.sandbox import scrub_text, workspace
from skillrot.store import Store, TrialRecord

if TYPE_CHECKING:
    from skillrot.config import SuiteConfig
    from skillrot.suite import Case, Skill, Suite

MAX_ATTEMPTS = 3
TIER_TRIALS: dict[Tier, int] = {Tier.SMOKE: 1, Tier.STANDARD: 3, Tier.FULL: 5}
TIER_CONDITIONS: dict[Tier, tuple[Condition, ...]] = {
    Tier.SMOKE: (Condition.NONE, Condition.SKILL),
    Tier.STANDARD: (Condition.NONE, Condition.SKILL, Condition.FORCED),
    Tier.FULL: (Condition.NONE, Condition.SKILL, Condition.FORCED),
}


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One cell of the run matrix."""

    case: Case
    condition: Condition
    model: ModelRef
    trial_index: int
    skill: Skill | None  # the version installed for this condition
    key: str


@dataclass(frozen=True, slots=True)
class Plan:
    """The full matrix for one suite, plus what is already cached."""

    suite: Suite
    harness: str
    harness_version: str
    specs: tuple[TrialSpec, ...]
    cached: frozenset[str] = frozenset()

    @property
    def pending(self) -> tuple[TrialSpec, ...]:
        """Specs that still need running."""
        return tuple(s for s in self.specs if s.key not in self.cached)

    def estimate_usd(self, store: Store | None = None) -> float:
        """Rough cost of the pending trials, from past runs or the configured default."""
        default = self.suite.config.estimate_run_usd
        per_model: dict[ModelRef, float] = {}
        total = 0.0
        for spec in self.pending:
            if spec.model not in per_model:
                past = store.mean_cost(str(spec.model), self.harness) if store else None
                per_model[spec.model] = past if past is not None else default
            total += per_model[spec.model]
        return total


def trial_key(
    case: Case,
    *, condition: Condition, skill: Skill | None, model: ModelRef,
    harness: str, harness_version: str, trial_index: int,
) -> str:  # fmt: skip
    """Cache key. NONE trials don't depend on the skill, so they survive skill edits."""
    return hash_json(
        {
            "case": case.hash,
            "condition": condition.value,
            "skill": skill.hash if skill and condition is not Condition.NONE else "",
            "model": str(model),
            "harness": harness,
            "harness_version": harness_version,
            "trial": trial_index,
        }
    )


def build_plan(  # noqa: PLR0913 - a plan has this many knobs
    suite: Suite,
    *,
    models: Sequence[ModelRef],
    harness: str,
    harness_version: str,
    conditions: Sequence[Condition] = TIER_CONDITIONS[Tier.SMOKE],
    trials: int = 1,
    prior: Skill | None = None,
    store: Store | None = None,
    case_ids: Sequence[str] | None = None,
) -> Plan:
    """Expand suite x model x condition x trial into specs. Trigger cases run only with the skill."""
    if not models:
        msg = "no models given: pass --models or set `models` in evals/skillrot.toml"
        raise ConfigError(msg)
    if Condition.PRIOR in conditions and prior is None:
        msg = "the prior condition needs --prior PATH to an earlier version of the skill"
        raise ConfigError(msg)
    cases = [suite.case(i) for i in case_ids] if case_ids else list(suite.cases)
    specs: list[TrialSpec] = []
    for model in models:
        for case in cases:
            for condition in _conditions_for(case, conditions):
                skill = _skill_for(condition, suite.skill, prior)
                for index in range(trials):
                    key = trial_key(
                        case,
                        condition=condition,
                        skill=skill,
                        model=model,
                        harness=harness,
                        harness_version=harness_version,
                        trial_index=index,
                    )
                    specs.append(TrialSpec(case, condition, model, index, skill, key))
    cached = frozenset(
        s.key
        for s in specs
        if (r := store.get(s.key) if store else None) and r.exit_reason.counts_as_data
    )
    return Plan(suite, harness, harness_version, tuple(specs), cached)


def _conditions_for(case: Case, conditions: Sequence[Condition]) -> list[Condition]:
    if case.kind is CaseKind.TASK:
        return list(conditions)
    return [c for c in conditions if c in {Condition.SKILL, Condition.PRIOR}]


def _skill_for(condition: Condition, current: Skill, prior: Skill | None) -> Skill | None:
    match condition:
        case Condition.NONE:
            return None
        case Condition.PRIOR:
            return prior
        case _:
            return current


@dataclass(slots=True)
class RunSummary:
    """Totals after a run."""

    completed: int = 0
    cached: int = 0
    infra_failures: int = 0
    spent_usd: float = 0.0
    stopped_for_budget: bool = False
    errors: list[str] = field(default_factory=list[str])


OnTrial = Callable[[TrialSpec, TrialRecord], Awaitable[None] | None]


async def execute(
    plan: Plan,
    *,
    harness: Harness,
    store: Store,
    judge: Judge | None = None,
    max_concurrency: int = 4,
    budget_usd: float | None = None,
    on_trial: OnTrial | None = None,
    seed: int = 0,
) -> RunSummary:
    """Run every pending spec.

    Conditions are interleaved randomly so drift over time (rate limits, provider load)
    does not line up with one condition.
    """
    summary = RunSummary(cached=len(plan.cached))
    pending = list(plan.pending)
    random.Random(seed).shuffle(pending)  # noqa: S311 - ordering, not security
    limiter = anyio.CapacityLimiter(max_concurrency)
    budget = budget_usd if budget_usd is not None else plan.suite.config.budget_usd
    estimate = plan.estimate_usd(store) / max(len(pending), 1)

    async def one(spec: TrialSpec) -> None:
        async with limiter:
            if summary.spent_usd + estimate > budget:
                summary.stopped_for_budget = True
                return
            record = await _run_spec(spec, plan, harness, store, judge)
            summary.spent_usd += record.usage.cost_usd or 0.0
            if record.exit_reason.counts_as_data:
                summary.completed += 1
            else:
                summary.infra_failures += 1
                summary.errors.append(f"{spec.case.id}/{spec.condition}: {record.error}")
            if on_trial is not None:
                result = on_trial(spec, record)
                if result is not None:
                    await result

    async with anyio.create_task_group() as tg:
        for spec in pending:
            tg.start_soon(one, spec)
    return summary


async def _run_spec(
    spec: TrialSpec, plan: Plan, harness: Harness, store: Store, judge: Judge | None
) -> TrialRecord:
    suite = plan.suite
    timeout = spec.case.timeout_s or suite.config.timeout_s
    previous = store.get(spec.key)
    attempts = previous.attempts if previous else 0
    for attempt in range(MAX_ATTEMPTS):
        attempts += 1
        with workspace(spec.case) as root:
            request = TrialRequest(
                case=spec.case,
                condition=spec.condition,
                model=spec.model,
                workspace=root,
                skill_name=suite.skill.name,
                skill=spec.skill,
                timeout_s=timeout,
                max_turns=suite.config.max_turns,
            )
            try:
                with anyio.fail_after(timeout + 30):
                    transcript = await harness.run(request)
            except TimeoutError:
                record = _infra_record(spec, plan, attempts, ExitReason.TIMEOUT, "harness hung")
                store.put(record)
                return record
            grades = []
            passed: bool | None = None
            if transcript.exit_reason.counts_as_data:
                outcome = Outcome(
                    prompt=spec.case.prompt,
                    final_text=transcript.final_text,
                    workspace=root,
                    tool_calls=transcript.tool_calls,
                    skill_loaded=transcript.skill_loaded,
                )
                grades = await grade_trial(
                    spec.case, outcome, skill_name=suite.skill.name, judge=judge
                )
                passed = transcript.exit_reason is not ExitReason.TIMEOUT and all(
                    g.passed for g in grades
                )
        record = TrialRecord(
            key=spec.key,
            skill_name=suite.skill.name,
            skill_hash=spec.skill.hash if spec.skill else "",
            case_id=spec.case.id,
            case_hash=spec.case.hash,
            condition=spec.condition,
            model=str(spec.model),
            harness=plan.harness,
            harness_version=plan.harness_version,
            trial_index=spec.trial_index,
            attempts=attempts,
            exit_reason=transcript.exit_reason,
            skill_loaded=transcript.skill_loaded,
            passed=passed,
            usage=_priced(transcript.usage, spec.model, suite.config),
            error=scrub_text(transcript.error)[:2000],
        )
        events = [json.loads(scrub_text(json.dumps(e, default=str))) for e in transcript.events]
        store.put(record, grades, events)
        if transcript.exit_reason.counts_as_data:
            return record
        if attempt < MAX_ATTEMPTS - 1:
            await anyio.sleep(
                2 ** (attempt + 2) if transcript.exit_reason is ExitReason.RATE_LIMITED else 1
            )
    return record  # pyright: ignore[reportPossiblyUnboundVariable] - loop runs at least once


def _priced(usage: Usage, model: ModelRef, config: SuiteConfig) -> Usage:
    """Fill in cost from the price table when the harness did not report one."""
    if usage.cost_usd is not None:
        return usage
    return replace(
        usage, cost_usd=config.cost(model.model_id, usage.input_tokens, usage.output_tokens)
    )


def _infra_record(
    spec: TrialSpec, plan: Plan, attempts: int, reason: ExitReason, error: str
) -> TrialRecord:
    return TrialRecord(
        key=spec.key,
        skill_name=plan.suite.skill.name,
        skill_hash=spec.skill.hash if spec.skill else "",
        case_id=spec.case.id,
        case_hash=spec.case.hash,
        condition=spec.condition,
        model=str(spec.model),
        harness=plan.harness,
        harness_version=plan.harness_version,
        trial_index=spec.trial_index,
        attempts=attempts,
        exit_reason=reason,
        skill_loaded=False,
        passed=None,
        usage=Usage(),
        error=error,
    )


def ensure_budget(estimate: float, budget: float) -> None:
    """Raise if a plan's estimate is over the budget."""
    if estimate > budget:
        msg = f"estimated ${estimate:.2f} is over the ${budget:.2f} budget (raise --budget-usd)"
        raise BudgetExceededError(msg)
