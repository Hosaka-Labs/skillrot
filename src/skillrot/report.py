"""Render result bundles for the terminal (rich) and for Markdown (CI summaries)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from skillrot.models import Condition, Verdict

if TYPE_CHECKING:
    from skillrot.analysis import ConditionStats, Drift, Interval, ModelResult, ResultBundle

VERDICT_STYLE: dict[Verdict, str] = {
    Verdict.ESSENTIAL: "bold green",
    Verdict.REDUNDANT: "bold yellow",
    Verdict.HARMFUL: "bold red",
    Verdict.NOT_TRIGGERING: "bold magenta",
    Verdict.INCONCLUSIVE: "cyan",
    Verdict.UNTESTED: "dim",
}


def fmt_interval(iv: Interval | None) -> str:
    """`+12 pts [+4, +20]`."""
    if iv is None:
        return "-"
    return f"{iv.mean * 100:+.0f} pts [{iv.low * 100:+.0f}, {iv.high * 100:+.0f}]"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _cost(stats: list[ConditionStats], condition: Condition) -> str:
    for s in stats:
        if s.condition is condition:
            if s.cost_per_pass_usd is not None:
                return f"${s.cost_per_pass_usd:.3f}"
            if s.tokens_per_pass is not None:
                return f"{s.tokens_per_pass / 1000:.0f}k tok"
    return "-"


def results_table(bundle: ResultBundle) -> Table:
    """One row per model."""
    table = Table(
        title=f"{bundle.skill.name}  ({bundle.skill.n_task_cases} task cases, ε = {bundle.epsilon:.0%})"
    )
    for col in (
        "Model",
        "Verdict",
        "Uplift (90% CI)",
        "Loaded",
        "Cost/pass skill → none",
        "Next step",
    ):
        table.add_column(col, overflow="fold")
    for r in bundle.results:
        flags = f" [{', '.join(r.flags)}]" if r.flags else ""
        table.add_row(
            r.model,
            Text(r.verdict.value + flags, style=VERDICT_STYLE[r.verdict]),
            fmt_interval(r.uplift),
            _pct(r.trigger_rate),
            f"{_cost(r.conditions, Condition.SKILL)} → {_cost(r.conditions, Condition.NONE)}",
            r.advice,
        )
    return table


def drift_lines(drifts: list[Drift]) -> list[str]:
    """Plain lines describing model-to-model changes."""
    lines: list[str] = []
    for d in drifts:
        mark = "changed" if d.significant else "no significant change"
        lines.append(
            f"{d.from_model} → {d.to_model}: {d.verdict_from} → {d.verdict_to}, "
            f"uplift change {fmt_interval(d.change)} ({mark})"
        )
    return lines


def markdown(bundle: ResultBundle) -> str:
    """GitHub-flavoured Markdown summary."""
    rows = [
        f"## skillrot: {bundle.skill.name}",
        "",
        "| Model | Verdict | Uplift (90% CI) | Loaded | Next step |",
        "|---|---|---|---|---|",
    ]
    rows.extend(
        f"| `{r.model}` | **{r.verdict}**{' (' + ', '.join(r.flags) + ')' if r.flags else ''} | "
        f"{fmt_interval(r.uplift)} | {_pct(r.trigger_rate)} | {r.advice} |"
        for r in bundle.results
    )
    if bundle.drift:
        rows += ["", *[f"- {line}" for line in drift_lines(bundle.drift)]]
    return "\n".join(rows) + "\n"


def detail(result: ModelResult) -> Table:
    """Per-condition breakdown for one model."""
    table = Table(title=f"{result.model} via {result.harness}", show_header=True)
    for col in ("Condition", "Trials", "Pass rate", "Cost/pass", "Tokens/pass", "Mean time"):
        table.add_column(col)
    for s in result.conditions:
        table.add_row(
            s.condition.value,
            str(s.trials),
            _pct(s.pass_rate),
            "-" if s.cost_per_pass_usd is None else f"${s.cost_per_pass_usd:.3f}",
            "-" if s.tokens_per_pass is None else f"{s.tokens_per_pass:,.0f}",
            "-" if s.mean_wall_s is None else f"{s.mean_wall_s:.0f}s",
        )
    return table
