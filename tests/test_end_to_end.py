"""Full pipeline: plan -> execute -> store -> analyze -> CLI, with no model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest

from skillrot import Condition, ModelRef, Verdict, analyze, build_plan, execute, load_suite
from skillrot.cli import Exit, app
from skillrot.harness.base import TrialRequest
from skillrot.harness.scripted import ScriptedHarness, ScriptedReply
from skillrot.models import ExitReason

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from make_csv_summary_cases import reference

HAIKU = ModelRef("anthropic", "claude-haiku-5-5")
OPUS = ModelRef("anthropic", "claude-opus-5-5")


def solver(*, skill_helps: bool, loads: bool = True):
    def script(req: TrialRequest) -> ScriptedReply:
        csvs = [p for p in req.workspace.iterdir() if p.suffix in {".csv", ".tsv"}]
        available = req.condition is not Condition.NONE
        loaded = available and loads and bool(csvs or "csv" in req.case.prompt.lower())
        if csvs:
            text = csvs[0].read_text()
            good = reference(text)
            if not ((loaded or req.condition is Condition.FORCED) and skill_helps):
                good = {**good, "rows": -1}  # wrong unless the skill is used
            (req.workspace / "summary.json").write_text(json.dumps(good))
        return ScriptedReply(
            skill_loaded=loaded or req.condition is Condition.FORCED, cost_usd=0.01
        )

    return script


def run(suite, store, harness, models, conditions=(Condition.NONE, Condition.SKILL), trials=1):
    plan = build_plan(
        suite,
        models=models,
        harness=harness.name,
        harness_version="1",
        conditions=conditions,
        trials=trials,
        store=store,
    )
    summary = anyio.run(lambda: execute(plan, harness=harness, store=store, budget_usd=100))
    return plan, summary


def test_essential_when_skill_helps(sample_skill, store):
    suite = load_suite(sample_skill)
    _, summary = run(suite, store, ScriptedHarness(solver(skill_helps=True)), [HAIKU])
    assert summary.completed == 16 * 2 + 5
    (result,) = analyze(suite, store).results
    assert result.verdict is Verdict.ESSENTIAL
    assert result.uplift is not None
    assert result.uplift.mean == 1.0
    assert result.trigger_accuracy is not None


def test_redundant_and_drift_across_models(sample_skill, store):
    suite = load_suite(sample_skill)
    run(suite, store, ScriptedHarness(solver(skill_helps=True)), [HAIKU])

    def everyone_wins(req: TrialRequest) -> ScriptedReply:
        reply = solver(skill_helps=True)(req)
        csvs = [p for p in req.workspace.iterdir() if p.suffix in {".csv", ".tsv"}]
        if csvs:
            (req.workspace / "summary.json").write_text(json.dumps(reference(csvs[0].read_text())))
        return reply

    run(suite, store, ScriptedHarness(everyone_wins), [OPUS])
    bundle = analyze(suite, store, models=[str(HAIKU), str(OPUS)])
    assert [r.verdict for r in bundle.results] == [Verdict.ESSENTIAL, Verdict.REDUNDANT]
    (d,) = bundle.drift
    assert d.significant
    assert d.change is not None
    assert d.change.mean == -1.0


def test_not_triggering(sample_skill, store):
    suite = load_suite(sample_skill)
    run(
        suite,
        store,
        ScriptedHarness(solver(skill_helps=True, loads=False)),
        [HAIKU],
        conditions=(Condition.NONE, Condition.SKILL, Condition.FORCED),
    )
    (result,) = analyze(suite, store).results
    assert result.verdict is Verdict.NOT_TRIGGERING
    assert result.content_uplift is not None
    assert result.content_uplift.mean == 1.0


def test_cache_and_resume(sample_skill, store):
    suite = load_suite(sample_skill)
    harness = ScriptedHarness(solver(skill_helps=True))
    run(suite, store, harness, [HAIKU])
    calls = len(harness.calls)
    plan, summary = run(suite, store, harness, [HAIKU])
    assert not plan.pending
    assert len(harness.calls) == calls
    # Editing the skill re-runs SKILL trials but keeps the baseline.
    (sample_skill / "SKILL.md").write_text((sample_skill / "SKILL.md").read_text() + "\nMore.\n")
    plan = build_plan(
        load_suite(sample_skill),
        models=[HAIKU],
        harness="scripted",
        harness_version="1",
        store=store,
    )
    pending = {s.condition for s in plan.pending}
    assert pending == {Condition.SKILL}


def test_infra_failures_are_retried_and_excluded(sample_skill, store, monkeypatch):
    monkeypatch.setattr("skillrot.runner.anyio.sleep", lambda _s: anyio.lowlevel.checkpoint())
    suite = load_suite(sample_skill)
    harness = ScriptedHarness(lambda req: ScriptedReply(exit_reason=ExitReason.INFRA))
    _, summary = run(suite, store, harness, [HAIKU])
    assert summary.completed == 0
    assert summary.infra_failures == 37
    assert len(harness.calls) == 37 * 3
    (result,) = analyze(suite, store).results
    assert result.verdict is Verdict.UNTESTED
    assert result.missing_trials == 37


def test_budget_stops_the_run(sample_skill, store):
    suite = load_suite(sample_skill)
    harness = ScriptedHarness(solver(skill_helps=True))
    plan = build_plan(suite, models=[HAIKU], harness="scripted", harness_version="1", store=store)
    summary = anyio.run(
        lambda: execute(plan, harness=harness, store=store, budget_usd=0.2, max_concurrency=1)
    )
    assert summary.stopped_for_budget
    assert summary.completed < len(plan.specs)


def test_bundle_schema_round_trips(sample_skill, store):
    suite = load_suite(sample_skill)
    run(suite, store, ScriptedHarness(solver(skill_helps=True)), [HAIKU])
    bundle = analyze(suite, store)
    data = json.loads(bundle.model_dump_json())
    assert data["schema_version"] == "1"
    assert type(bundle).model_validate(data) == bundle


# --- CLI, through the real Claude Code harness and the fake `claude` executable --------


def cli(*args: str) -> int:
    return app(list(args), result_action="return_value")


def test_cli_check_show_diff(sample_skill, fake_claude, tmp_path, capsys):
    store = str(tmp_path / "store")
    out = tmp_path / "bundle.json"
    md = tmp_path / "summary.md"
    code = cli(
        "check",
        str(sample_skill),
        "--models",
        "claude-haiku-5-5,claude-sonnet-5-5",
        "--no-isolate",
        "--yes",
        "--store",
        store,
        "--json",
        str(out),
        "--markdown",
        str(md),
        "--fail-on",
        "harmful",
    )
    assert code == Exit.OK
    bundle = json.loads(out.read_text())
    verdicts = {r["model"]: r["verdict"] for r in bundle["results"]}
    assert verdicts == {
        "anthropic/claude-haiku-5-5": "essential",
        "anthropic/claude-sonnet-5-5": "essential",
    }
    assert "| `anthropic/claude-haiku-5-5` | **essential**" in md.read_text()
    assert cli("show", str(sample_skill), "--store", store, "--details") == Exit.OK
    assert (
        cli("diff", str(sample_skill), "claude-haiku-5-5", "claude-sonnet-5-5", "--store", store)
        == Exit.OK
    )
    assert cli("models", "--store", store) == Exit.OK
    printed = capsys.readouterr().out
    assert "essential" in printed
    assert "claude-sonnet-5-5" in printed
    # Second run is fully cached.
    assert (
        cli(
            "check",
            str(sample_skill),
            "--no-isolate",
            "--yes",
            "--store",
            store,
            "--fail-on",
            "essential",
        )
        == Exit.FINDINGS
    )
    assert "(74 cached)" in " ".join(capsys.readouterr().out.split())


def test_cli_dry_run_and_budget(sample_skill, fake_claude, tmp_path, capsys):
    store = str(tmp_path / "store")
    assert cli("check", str(sample_skill), "--no-isolate", "--dry-run", "--store", store) == Exit.OK
    assert "74 trials" in capsys.readouterr().out
    from skillrot.errors import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        cli(
            "check",
            str(sample_skill),
            "--no-isolate",
            "--yes",
            "--budget-usd",
            "0.5",
            "--store",
            store,
        )


def test_cli_lint_blocks_check(sample_skill, fake_claude, tmp_path):
    (sample_skill / "evals" / "plain-totals" / "graders" / "09.md").write_text(
        "---\ntype: command\n---\ntrue\n"
    )
    assert cli("lint", str(sample_skill)) == Exit.FINDINGS
    assert (
        cli("check", str(sample_skill), "--no-isolate", "--yes", "--store", str(tmp_path / "s"))
        == Exit.USAGE
    )


def test_cli_schema(capsys):
    assert cli("schema") == Exit.OK
    assert '"ResultBundle"' in capsys.readouterr().out
