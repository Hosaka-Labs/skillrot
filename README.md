# skillrot

**Does this skill still help on this model?** skillrot runs your [Agent Skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills)'s
test cases with and without the skill, on each model you care about. For each model it reports the difference,
how sure that difference is, and what it costs. Run it when a new model ships, and it tells you
which skills to keep, which to fix and which you can delete.

```text
$ skillrot check skills/csv-summary --models claude-haiku-5-5,claude-sonnet-5-5
csv-summary: 74 trials (0 cached) on anthropic/claude-haiku-5-5, anthropic/claude-sonnet-5-5 via claude-code 2.1.0; est. $3.70, budget $10.00
Spend up to $3.70? [Y/n]

                          csv-summary  (16 task cases, ε = 5%)
 Model                       Verdict     Uplift (90% CI)     Loaded  Cost/pass skill → none  Next step
 anthropic/claude-haiku-5-5  essential   +44 pts [+28, +59]  100%    $0.031 → $0.072         Keep it. The skill clearly helps on this model.
 anthropic/claude-sonnet-5-5 redundant   +2 pts [-3, +6]     94%     $0.054 → $0.051         The model does as well without it. Candidate for deletion.
  anthropic/claude-haiku-5-5 → anthropic/claude-sonnet-5-5: essential → redundant, uplift change -42 pts [-57, -27] (changed)
```

*(Illustrative output. Your numbers will differ.)*

## Install

```bash
pip install skillrot                # core + Claude Code harness
pip install 'skillrot[anthropic]'   # + Messages API harness and LLM judge
```

Python 3.12+. The default harness drives the [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI, so
install that too (`npm install -g @anthropic-ai/claude-code`). Run `skillrot doctor` to check your setup.

## Quick start

```bash
skillrot new skills/my-skill        # creates skills/my-skill/evals/ with TODOs
# ...write 15+ real task cases (see below)...
skillrot lint skills/my-skill       # free, static checks
skillrot check skills/my-skill --models claude-sonnet-5-5 --tier smoke
skillrot check skills/my-skill --models claude-sonnet-5-5 --escalate   # more trials only where unclear
skillrot show skills/ --details     # stored verdicts, no new runs
skillrot diff skills/my-skill claude-sonnet-5 claude-sonnet-5-5
```

Try it on the bundled example: `skillrot check examples/skills/csv-summary`.

## How it measures

Each case runs under several **conditions**:

| Condition | What the agent gets | Tells you |
|---|---|---|
| `none` | no skill | the baseline |
| `skill` | skill installed normally; the agent decides whether to load it | real-world value, including triggering |
| `forced` | skill text put straight into the system prompt | content value, even if it never triggers |
| `prior` | an earlier version (`--prior PATH`) | whether your edit helped |

**Uplift** is the pass rate with the skill minus without, paired by case. Trials of one case are averaged first,
then cases are resampled (a cluster bootstrap) to get a 90% interval. With 15 to 40 cases that is more honest than a
normal-approximation error bar. skillrot also reports the smallest effect the run could detect (MDE).

**Verdicts** come from one ordered table. The first matching row wins, so every result gets exactly one verdict:

| # | Verdict | Rule |
|---|---|---|
| 1 | Untested | fewer than `min_cases` paired task cases |
| 2 | Harmful | whole interval below 0 |
| 3 | Essential | interval above 0 and uplift ≥ ε (default 5 pts) |
| 4 | Not triggering | skill loaded in < 80% of runs, and `forced` shows the content helps |
| 5 | Redundant | interval inside ±ε and the skill loads (else Not triggering) |
| 6 | Inconclusive | anything else; `--escalate` adds trials |

**Flags** sit beside the verdict without changing it. `efficiency` means the skill is ≥ 20% cheaper per passing
run. `convention` means the skill is marked `type = "convention"`, so it is never suggested for deletion on accuracy
alone. `contaminated` means the skill loaded in a run where it was not installed.

**Tiers** set run size. `smoke` is 1 trial with `none` and `skill`. `standard` is 3 trials and adds `forced`.
`full` is 5 trials. Results are cached by content hash. Editing the skill re-runs only the skill conditions, editing
a case re-runs only that case, and an interrupted run resumes where it stopped.

## Writing cases

```text
my-skill/
├── SKILL.md
└── evals/
    ├── skillrot.toml               # optional settings
    └── parse-negative-qty/
        ├── prompt.md               # what a real user would ask
        ├── case.yaml               # kind: task|trigger|no-trigger, split: dev|holdout
        ├── files/                  # copied into the agent's working directory
        └── graders/
            ├── 01-rows.md          # one grader per file
            └── 02-total.md
```

A grader is a Markdown file with YAML front matter:

```markdown
---
type: json_field
file: summary.json
path: totals.amount
approx: 2648.5
tolerance: 0.01
---
```

| Type | Passes when |
|---|---|
| `json_field` | a JSON path `equals`, is `approx` (± `tolerance`), has `length`, or exists |
| `command` | the shell command in the body exits 0 (or `expect_exit`) in the working directory |
| `regex` / `contains` | the pattern / text is in the final message or `file` (`must_match: false` to invert) |
| `file_exists` | `path` exists (`absent: true` to invert) |
| `tool_used` | the agent called tool `name` (`absent: true` to invert) |
| `judge` | a pinned model agrees the statement in the body is true (needs `skillrot[anthropic]`) |

Trigger and no-trigger cases need no graders: they pass when the skill does or doesn't load.

Rules that keep results meaningful (`skillrot lint` checks several of them):

1. **Test the outcome, not the instructions.** Check the file the user wanted, not whether the agent said the skill's words.
2. **Write cases from the task, never from SKILL.md.** Real, scrubbed user requests are best. Lint flags prompts that quote the skill or name it.
3. **Every grader must be able to fail.** Lint catches patterns that match nothing and files that are already in `files/`.
4. **Include trigger and no-trigger cases**, so "not loading" can be told apart from "not helping".
5. **Use 15+ task cases.** Fewer than `min_cases` gives Untested.
6. **Keep holdout cases** for comparing skill versions (`--prior`). Cross-model verdicts use all cases.

Older skill-creator suites (`evals/evals.json`) load as-is; their `expectations` become `judge` graders.

### `skillrot.toml`

```toml
type = "capability"      # or "convention"
epsilon = 0.05
min_cases = 10
models = ["anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"]
harness = "claude-code"  # or "anthropic"
budget_usd = 20
timeout_s = 600
max_turns = 30

[judge]
model = "claude-haiku-5-5"

[prices."my-model"]      # USD per million tokens, for the anthropic harness
input = 1.0
output = 5.0
```

## Harnesses

Uplift depends on the model and on the agent harness around it, so every result records both.

- **`claude-code`** (default) runs `claude -p --output-format stream-json` once per trial. The skill is installed as a
  project skill in a fresh temp directory. By default each trial gets an empty Claude Code config directory, so your
  personal skills and CLAUDE.md can't leak into the baseline. That needs `ANTHROPIC_API_KEY`. `--no-isolate` uses
  your own login instead, and skillrot flags any leak as `contaminated`.
- **`anthropic`** is a small Messages API agent loop with Anthropic's bash and text-editor tools. Skills are offered
  by name and description, and the body loads only on request, mirroring progressive disclosure. The API key stays in
  the skillrot process and never enters the agent's shell.
- Your own: implement the `skillrot.harness.Harness` protocol and register it under the `skillrot.harnesses` entry
  point group.

> **Security.** v0.1 runs trials on your machine in temporary directories with a scrubbed environment. That keeps
> trials apart but does not contain them: an agent can run any command your user can. Evaluate untrusted skills
> inside a container or VM.

## In CI

```yaml
# .github/workflows/skills.yml
on:
  schedule: [{ cron: "0 6 * * 1" }]
  workflow_dispatch:
jobs:
  skillrot:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v4
        with: { path: .skillrot, key: skillrot-${{ github.run_id }}, restore-keys: skillrot- }
      - uses: Hosaka-Labs/skillrot@v0
        with:
          skills: skills/
          models: claude-sonnet-5-5,claude-opus-5-5
          tier: smoke
          escalate: "true"
          budget-usd: "40"
          fail-on: harmful,drift
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The action writes a Markdown table to the job summary and `skillrot-results.json` to the workspace.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | a result matched `--fail-on` (or `lint` found errors) |
| 2 | usage, config or suite error |
| 3 | stopped at the budget, or the estimate is over it |
| 4 | more than 20% of trials failed for infrastructure reasons |

## Library and result bundles

```python
import anyio
from skillrot import Condition, ModelRef, Store, analyze, build_plan, execute, get_harness, load_suite

async def main() -> None:
    suite = load_suite("skills/csv-summary")
    harness = get_harness("claude-code")
    with Store.open(".skillrot") as store:
        plan = build_plan(
            suite,
            models=[ModelRef.parse("claude-sonnet-5-5")],
            harness=harness.name,
            harness_version=await harness.version(),
            conditions=[Condition.NONE, Condition.SKILL],
            trials=3,
            store=store,
        )
        await execute(plan, harness=harness, store=store, budget_usd=10)
        bundle = analyze(suite, store)
    print(bundle.model_dump_json(indent=2))

anyio.run(main)
```

`ResultBundle` is the stable, versioned output (`skillrot schema` prints its JSON Schema). Build dashboards on
bundles (`skillrot show --json -`), not on `.skillrot/store.db`, whose layout may change.

## Not yet

These are planned and not in v0.1: Docker sandboxing, a LiteLLM harness for non-Anthropic models, section-by-section
ablation (`trim`), re-grading stored transcripts with a new judge, and HTML/Parquet exports.

## License

Apache-2.0
