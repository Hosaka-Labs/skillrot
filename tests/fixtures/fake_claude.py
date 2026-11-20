#!/usr/bin/env python3
"""Stand-in for the `claude` CLI. Emits stream-json like `claude -p --output-format stream-json`.

With the csv-summary skill available (installed or forced) it follows the skill's rules;
without it, it writes a naive summary that misses the edge cases.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from make_csv_summary_cases import reference


def emit(event: dict[str, object]) -> None:
    print(json.dumps(event), flush=True)


def naive(text: str) -> dict[str, object]:
    rows = list(csv.reader(io.StringIO(text)))
    names = [h.strip().lower().replace(" ", "_") for h in rows[0]]
    body = [r for r in rows[1:] if r]
    totals: dict[str, float] = {}
    for i, name in enumerate(names):
        try:
            totals[name] = round(sum(float(r[i]) for r in body), 2)
        except (ValueError, IndexError):
            pass
    return {"rows": len(body), "columns": names, "totals": totals}


def main() -> int:
    args = sys.argv[1:]
    if args == ["--version"]:
        print("9.9.9 (Claude Code)")
        return 0
    if Path("FAKE_FAIL").exists():
        print("API Error: 529 overloaded", file=sys.stderr)
        return 1
    prompt = args[args.index("-p") + 1]
    forced = "--append-system-prompt" in args
    installed = Path(".claude/skills/csv-summary/SKILL.md").exists()
    emit({"type": "system", "subtype": "init", "model": args[args.index("--model") + 1]})
    uses_skill = installed and bool(re.search(r"csv|summar|total", prompt, re.IGNORECASE))
    if uses_skill:
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Skill",
                            "input": {"skill": "csv-summary"},
                        }
                    ]
                },
            }
        )
    files = [p for p in Path().iterdir() if p.suffix in {".csv", ".tsv"}]
    if files:
        text = files[0].read_text()
        summary = reference(text) if (forced or uses_skill) else naive(text)
        Path("summary.json").write_text(json.dumps(summary))
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t2",
                            "name": "Write",
                            "input": {"file_path": "summary.json"},
                        }
                    ]
                },
            }
        )
    emit(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "Wrote summary.json.",
            "total_cost_usd": 0.01 if (forced or uses_skill) else 0.012,
            "num_turns": 3,
            "usage": {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500},
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
