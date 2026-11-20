"""Command-line interface. `skillrot check` is the one command most people need."""

from __future__ import annotations

import json
import os
import shutil
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import anyio
from cyclopts import App, Parameter
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm

from skillrot import __version__
from skillrot.analysis import ResultBundle, analyze, drift
from skillrot.errors import BudgetExceededError, ConfigError, SkillrotError
from skillrot.harness import available, get_harness
from skillrot.judge import AnthropicJudge
from skillrot.lint import Level, lint
from skillrot.models import Condition, ModelRef, Tier, Verdict
from skillrot.report import detail, drift_lines, markdown, results_table
from skillrot.runner import TIER_CONDITIONS, TIER_TRIALS, Plan, RunSummary, build_plan, execute
from skillrot.scaffold import scaffold
from skillrot.store import DEFAULT_STORE, Store
from skillrot.suite import Suite, discover, load_skill, load_suite

if TYPE_CHECKING:
    from skillrot.harness.base import Harness

console = Console()
err = Console(stderr=True)
app = App(
    name="skillrot",
    help="Check whether your Agent Skills still help on each model.",
    version=__version__,
)

INFRA_FAILURE_LIMIT = 0.2
ESCALATE_ON = frozenset({Verdict.INCONCLUSIVE, Verdict.UNTESTED})


class Exit(IntEnum):
    """Process exit codes."""

    OK = 0
    FINDINGS = 1  # a result matched --fail-on
    USAGE = 2  # bad arguments, config or suite
    BUDGET = 3  # stopped at the spending limit
    INFRA = 4  # too many infrastructure failures to trust the run


StorePath = Annotated[Path, Parameter(help="Result store folder.", env_var="SKILLROT_STORE")]


def _models(values: list[str] | None, suite: Suite) -> list[ModelRef]:
    raw = [m for v in (values or []) for m in v.split(",") if m] or list(suite.config.models)
    return [ModelRef.parse(m) for m in raw]


def _load_all(paths: list[Path]) -> list[Suite]:
    skills = discover(paths)
    if not skills:
        msg = f"no SKILL.md found under {', '.join(map(str, paths))}"
        raise ConfigError(msg)
    return [load_suite(p) for p in skills]


@app.command
def check(  # noqa: C901, PLR0913 - the main workflow, kept in one readable place
    paths: list[Path],
    /,
    *,
    models: Annotated[
        list[str] | None, Parameter(help="Model IDs, comma-separated or repeated.")
    ] = None,
    tier: Tier = Tier.SMOKE,
    trials: Annotated[int | None, Parameter(help="Override trials per case and condition.")] = None,
    conditions: Annotated[list[Condition] | None, Parameter(help="Override conditions.")] = None,
    escalate: Annotated[
        bool, Parameter(help="Re-run unclear results at the standard tier.")
    ] = False,
    harness: Annotated[str | None, Parameter(help="claude-code or anthropic.")] = None,
    no_isolate: Annotated[
        bool, Parameter(help="claude-code: use your own Claude Code config/login.")
    ] = False,
    prior: Annotated[
        Path | None, Parameter(help="Earlier version of the skill, for the prior condition.")
    ] = None,
    budget_usd: Annotated[
        float | None, Parameter(help="Stop spending past this many dollars.")
    ] = None,
    concurrency: int = 4,
    case: Annotated[list[str] | None, Parameter(help="Only run these case IDs.")] = None,
    fail_on: Annotated[
        list[str] | None,
        Parameter(help="Exit 1 on these: harmful, redundant, not-triggering, inconclusive, drift."),
    ] = None,
    json_out: Annotated[
        Path | None, Parameter(name="--json", help="Write the result bundle here ('-' for stdout).")
    ] = None,
    markdown_out: Annotated[
        Path | None, Parameter(name="--markdown", help="Write a Markdown summary here.")
    ] = None,
    yes: Annotated[
        bool, Parameter(name=["--yes", "-y"], help="Don't ask before spending.")
    ] = False,
    dry_run: Annotated[bool, Parameter(help="Show the plan and cost, run nothing.")] = False,
    store: StorePath = DEFAULT_STORE,
) -> int:
    """Run a skill's cases with and without it on each model, then print a verdict per model.

    PATHS are skill folders, or folders that contain skills.
    """
    suites = _load_all(paths)
    blocked = False
    for suite in suites:
        for finding in lint(suite):
            if finding.level is Level.ERROR:
                blocked = True
                err.print(f"[red]{suite.skill.name}[/red] {finding}")
    if blocked:
        err.print("Fix the lint errors above (see `skillrot lint`), then re-run.")
        return Exit.USAGE
    prior_skill = load_skill(prior) if prior else None
    conds = list(conditions) if conditions else list(TIER_CONDITIONS[tier])
    if prior_skill and Condition.PRIOR not in conds:
        conds.append(Condition.PRIOR)
    n_trials = trials or TIER_TRIALS[tier]
    fail_set = {f for v in (fail_on or []) for f in v.split(",") if f}
    exit_code = Exit.OK
    with Store.open(store) as db:
        for suite in suites:
            harness_name = harness or suite.config.harness
            agent = _make_harness(harness_name, no_isolate=no_isolate)
            anyio.run(agent.check)
            version = anyio.run(agent.version)
            model_refs = _models(models, suite)
            plan = build_plan(
                suite, models=model_refs, harness=harness_name, harness_version=version,
                conditions=conds, trials=n_trials, prior=prior_skill, store=db, case_ids=case,
            )  # fmt: skip
            budget = budget_usd if budget_usd is not None else suite.config.budget_usd
            estimate = plan.estimate_usd(db)
            console.print(
                f"[bold]{suite.skill.name}[/bold]: {len(plan.specs)} trials "
                f"({len(plan.cached)} cached) on {', '.join(map(str, model_refs))} via {harness_name} "
                f"{version}; est. ${estimate:.2f}, budget ${budget:.2f}"
            )
            if dry_run:
                continue
            if plan.pending and not _confirm(estimate, budget, yes=yes):
                return Exit.USAGE
            judge = AnthropicJudge(suite.config.judge_model) if _needs_judge(suite) else None
            summary = _execute(plan, agent, db, judge, concurrency, budget)
            if escalate and not summary.stopped_for_budget:
                bundle = analyze(
                    suite, db, models=[str(m) for m in model_refs], harness=harness_name
                )
                unclear = [
                    ModelRef.parse(r.model) for r in bundle.results if r.verdict in ESCALATE_ON
                ]
                if unclear:
                    console.print(f"Escalating {', '.join(map(str, unclear))} to the standard tier")
                    plan = build_plan(
                        suite, models=unclear, harness=harness_name, harness_version=version,
                        conditions=sorted({*conds, *TIER_CONDITIONS[Tier.STANDARD]}),
                        trials=max(n_trials, TIER_TRIALS[Tier.STANDARD]), prior=prior_skill, store=db, case_ids=case,
                    )  # fmt: skip
                    more = _execute(plan, agent, db, judge, concurrency, budget - summary.spent_usd)
                    summary = _merge(summary, more)
            bundle = analyze(suite, db, models=[str(m) for m in model_refs], harness=harness_name)
            _print_bundle(bundle)
            console.print(
                f"Spent ${summary.spent_usd:.2f}; {summary.completed} run, {summary.cached} cached."
            )
            _write_outputs(bundle, json_out, markdown_out)
            exit_code = max(exit_code, _exit_for(bundle, summary, fail_set))
    return exit_code


def _make_harness(name: str, *, no_isolate: bool) -> Harness:
    if name == "claude-code":
        return get_harness(name, isolate_config=not no_isolate)
    return get_harness(name)


def _needs_judge(suite: Suite) -> bool:
    return any(g.type == "judge" for c in suite.cases for g in c.graders)


def _confirm(estimate: float, budget: float, *, yes: bool) -> bool:
    if estimate > budget:
        msg = f"estimated ${estimate:.2f} is over the ${budget:.2f} budget; raise --budget-usd or narrow the run"
        raise BudgetExceededError(msg)
    if yes or not sys.stdin.isatty():
        return True
    return Confirm.ask(f"Spend up to ${estimate:.2f}?", default=True)


def _execute(
    plan: Plan,
    agent: Harness,
    db: Store,
    judge: AnthropicJudge | None,
    concurrency: int,
    budget: float,
) -> RunSummary:
    progress = Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    )  # fmt: skip
    with progress:
        task = progress.add_task(plan.suite.skill.name, total=len(plan.pending))

        def advance(*_: object) -> None:
            progress.advance(task)

        async def go() -> RunSummary:
            return await execute(
                plan, harness=agent, store=db, judge=judge,
                max_concurrency=concurrency, budget_usd=budget, on_trial=advance,
            )  # fmt: skip

        summary = anyio.run(go)
    for line in summary.errors[:5]:
        err.print(f"[yellow]infra[/yellow] {line}")
    return summary


def _merge(a: RunSummary, b: RunSummary) -> RunSummary:
    return RunSummary(
        completed=a.completed + b.completed,
        cached=a.cached,
        infra_failures=a.infra_failures + b.infra_failures,
        spent_usd=a.spent_usd + b.spent_usd,
        stopped_for_budget=b.stopped_for_budget,
        errors=a.errors + b.errors,
    )


def _exit_for(bundle: ResultBundle, summary: RunSummary, fail_set: set[str]) -> Exit:
    attempted = summary.completed + summary.infra_failures
    if attempted and summary.infra_failures / attempted > INFRA_FAILURE_LIMIT:
        err.print(
            f"[red]{summary.infra_failures} of {attempted} trials failed for infrastructure reasons[/red]"
        )
        return Exit.INFRA
    if summary.stopped_for_budget:
        err.print("[yellow]Stopped at the budget; results are partial. Re-run to resume.[/yellow]")
        return Exit.BUDGET
    hit = {r.verdict.value for r in bundle.results} & fail_set
    if "drift" in fail_set and any(d.significant for d in bundle.drift):
        hit.add("drift")
    if hit:
        err.print(f"[red]--fail-on matched: {', '.join(sorted(hit))}[/red]")
        return Exit.FINDINGS
    return Exit.OK


def _print_bundle(bundle: ResultBundle) -> None:
    if not bundle.results:
        console.print(f"{bundle.skill.name}: no results stored yet. Run `skillrot check` first.")
        return
    console.print(results_table(bundle))
    for line in drift_lines(bundle.drift):
        console.print(f"  {line}")


def _write_outputs(bundle: ResultBundle, json_out: Path | None, markdown_out: Path | None) -> None:
    if json_out is not None:
        text = bundle.model_dump_json(indent=2)
        if str(json_out) == "-":
            sys.stdout.write(text + "\n")
        else:
            json_out.write_text(text + "\n")
    if markdown_out is not None:
        md = markdown(bundle)
        with markdown_out.open("a") as fh:
            fh.write(md)


@app.command
def show(
    paths: list[Path],
    /,
    *,
    models: list[str] | None = None,
    details: Annotated[bool, Parameter(help="Per-condition breakdown.")] = False,
    json_out: Annotated[Path | None, Parameter(name="--json")] = None,
    markdown_out: Annotated[Path | None, Parameter(name="--markdown")] = None,
    store: StorePath = DEFAULT_STORE,
) -> int:
    """Show stored verdicts without running anything."""
    with Store.open(store) as db:
        for suite in _load_all(paths):
            wanted = [str(m) for m in _models(models, suite)] if models else None
            bundle = analyze(suite, db, models=wanted)
            if json_out is None or str(json_out) != "-":
                _print_bundle(bundle)
                if details:
                    for result in bundle.results:
                        console.print(detail(result))
            _write_outputs(bundle, json_out, markdown_out)
    return Exit.OK


@app.command(name="diff")
def diff_cmd(
    path: Path, from_model: str, to_model: str, /, *, store: StorePath = DEFAULT_STORE
) -> int:
    """Compare a skill's uplift between two models, from stored results."""
    suite = load_suite(path)
    with Store.open(store) as db:
        bundle = analyze(
            suite, db, models=[str(ModelRef.parse(from_model)), str(ModelRef.parse(to_model))]
        )
        if len(bundle.results) < 2:  # noqa: PLR2004
            err.print(
                "Need stored results for both models. Run `skillrot check --models A,B` first."
            )
            return Exit.USAGE
        a, b = bundle.results[:2]
        console.print(results_table(bundle))
        for line in drift_lines([drift(suite, db, a, b)]):
            console.print(line)
    return Exit.OK


@app.command(name="lint")
def lint_cmd(paths: list[Path], /) -> int:
    """Check cases for problems without calling any model."""
    worst = Exit.OK
    for suite in _load_all(paths):
        findings = lint(suite)
        style = {Level.ERROR: "red", Level.WARNING: "yellow", Level.INFO: "dim"}
        console.print(
            f"[bold]{suite.skill.name}[/bold]: {len(suite.cases)} cases ({len(suite.task_cases)} task)"
        )
        for f in findings:
            console.print(f"  [{style[f.level]}]{f}[/{style[f.level]}]")
        if any(f.level is Level.ERROR for f in findings):
            worst = Exit.FINDINGS
        elif not findings:
            console.print("  [green]ok[/green]")
    return worst


@app.command
def new(
    path: Path,
    /,
    *,
    brief: Annotated[str, Parameter(help="One-line task brief for the trigger case.")] = "",
) -> int:
    """Create an evals/ folder for a skill, with TODOs that `lint` will flag until filled in."""
    for written in scaffold(path, task_brief=brief):
        console.print(f"created {written}")
    console.print("Write cases from the task, not from SKILL.md. Then run `skillrot lint`.")
    return Exit.OK


@app.command
def models(path: Path | None = None, /, *, store: StorePath = DEFAULT_STORE) -> int:
    """List models with stored results."""
    name = load_skill(path).name if path else None
    with Store.open(store) as db:
        rows = db.models(name)
    if not rows:
        console.print("No stored results.")
    for model, harness, count, last in rows:
        console.print(f"{model:40} {harness:12} {count:6} trials  last {last}")
    return Exit.OK


@app.command
def doctor() -> int:
    """Check this machine can run skillrot."""
    ok = True

    def line(*, good: bool, text: str) -> None:
        nonlocal ok
        ok = ok and good
        console.print(f"[{'green' if good else 'red'}]{'✓' if good else '✗'}[/] {text}")

    line(good=sys.version_info >= (3, 12), text=f"Python {sys.version.split()[0]}")
    console.print(f"  harnesses: {', '.join(sorted(available()))}")
    claude = shutil.which(os.environ.get("SKILLROT_CLAUDE", "claude"))
    line(
        good=claude is not None,
        text=f"Claude Code CLI: {claude or 'not found (npm install -g @anthropic-ai/claude-code)'}",
    )
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    line(
        good=has_key,
        text="ANTHROPIC_API_KEY set"
        if has_key
        else "ANTHROPIC_API_KEY not set (needed for isolated runs and judges)",
    )
    try:
        import anthropic  # noqa: PLC0415

        line(good=True, text=f"anthropic SDK {anthropic.__version__}")
    except ImportError:
        console.print(
            "[yellow]-[/yellow] anthropic SDK not installed (pip install 'skillrot[anthropic]')"
        )
    console.print(
        "[yellow]![/yellow] Trials run on this machine in temp folders. Run untrusted skills in a container."
    )
    return Exit.OK if ok else Exit.FINDINGS


@app.command
def schema() -> int:
    """Print the JSON Schema of the result bundle."""
    sys.stdout.write(json.dumps(ResultBundle.model_json_schema(), indent=2) + "\n")
    return Exit.OK


def main() -> None:
    """Console entry point."""
    try:
        code = app(result_action="return_value")
    except BudgetExceededError as exc:
        err.print(f"[red]error:[/red] {exc}")
        sys.exit(Exit.BUDGET)
    except SkillrotError as exc:
        err.print(f"[red]error:[/red] {exc}")
        sys.exit(Exit.USAGE)
    sys.exit(int(code) if isinstance(code, int) else Exit.OK)
