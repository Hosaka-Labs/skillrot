"""Regenerate the csv-summary example cases. Expected values come from `reference()`.

Run from the repo root:  uv run python examples/make_csv_summary_cases.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from pathlib import Path

EVALS = Path(__file__).parent / "skills" / "csv-summary" / "evals"
PROMPT = (
    "Summarize {file} into summary.json: the number of rows, the column names, "
    "and totals for the numeric columns."
)


def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def clean(value: str, delimiter: str) -> float | None:
    v = value.strip()
    if not v:
        return 0.0
    negative = False
    if v.startswith("(") and v.endswith(")"):
        negative, v = True, v[1:-1]
    if v.endswith("-"):
        negative, v = True, v[:-1]
    v = re.sub(r"[$€£\s]", "", v)
    v = re.sub(r",(?=\d{3}(\D|$))", "", v) if delimiter == "," else v.replace(",", "")
    try:
        number = float(v)
    except ValueError:
        return None
    return -number if negative else number


def reference(text: str) -> dict[str, object]:
    text = text.removeprefix("﻿")
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[0]
    delimiter = max([",", ";", "\t"], key=header.count)
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    names = [snake(h) for h in rows[0]]
    body = rows[1:]
    if "status" in names:
        i = names.index("status")
        body = [r for r in body if r[i].strip().lower() not in {"void", "cancelled"}]
    totals: dict[str, float] = {}
    for col, name in enumerate(names):
        values = [clean(r[col], delimiter) for r in body]
        if all(v is not None for v in values):
            totals[name] = round(sum(v for v in values if v is not None), 2)
    return {"rows": len(body), "columns": names, "totals": totals}


CASES: dict[str, tuple[str, str]] = {
    "plain-totals": ("orders.csv", "order_id,qty,price\n1,2,9.99\n2,1,15.00\n3,4,2.50\n"),
    "currency-symbols": (
        "sales.csv",
        "region,revenue\nnorth,$1200.50\nsouth,$980.25\neast,$15.00\n",
    ),
    "thousands-separators": ("big.csv", 'item,amount\nA,"1,250.00"\nB,"12,000.75"\nC,99.25\n'),
    "parenthesis-negatives": (
        "ledger.csv",
        "entry,amount\nrent,(1500.00)\nsales,4200.00\nfees,(35.50)\n",
    ),
    "trailing-minus": ("adjust.csv", "sku,delta\nX1,12.50-\nX2,4.00\nX3,0.75-\n"),
    "void-rows": (
        "orders.csv",
        "order_id,status,total\n1,paid,10.00\n2,void,99.00\n3,paid,5.50\n4,VOID,1.00\n",
    ),
    "cancelled-rows": (
        "bookings.csv",
        "id,status,nights\n10,confirmed,3\n11,cancelled,7\n12,Cancelled,2\n13,confirmed,1\n",
    ),
    "semicolon-delimited": ("export.csv", "name;units;price\nwidget;3;4.50\ngadget;10;1.25\n"),
    "tab-delimited": ("export.tsv", "name\tunits\tweight_kg\nbox\t2\t1.5\ncrate\t1\t12.25\n"),
    "byte-order-mark": ("bom.csv", "﻿customer,spend\nann,20.00\nbo,35.40\n"),
    "blank-lines": ("gappy.csv", "day,visits\n\nmon,120\n\ntue,98\nwed,143\n\n"),
    "empty-cells": ("partial.csv", "store,q1,q2\nA,100,\nB,,250\nC,75,25\n"),
    "messy-headers": (
        "prices.csv",
        "Product Name,Unit Price ($),In Stock?\npen,1.20,40\nink,3.80,12\n",
    ),
    "text-column-not-numeric": ("people.csv", "id,name,age\n1,Ada,36\n2,Lin,29\n3,Sam,n/a\n"),
    "euro-and-pound": ("fx.csv", "line,cost\na,€10.00\nb,£5.50\nc,€0.50\n"),
    "everything-at-once": (
        "messy.csv",
        "﻿Order ID;Status;Net Amount (€);Qty\n\n1;paid;€1 200.00;2\n2;VOID;€50.00;1\n3;paid;(€30.00);1\n4;paid;;3\n",
    ),
}


def main() -> None:
    for old in EVALS.glob("*"):
        if old.is_dir():
            shutil.rmtree(old)
    for case_id, (filename, text) in CASES.items():
        expected = reference(text)
        folder = EVALS / case_id
        (folder / "files").mkdir(parents=True)
        (folder / "files" / filename).write_text(text)
        (folder / "prompt.md").write_text(PROMPT.format(file=filename) + "\n")
        split = (
            "holdout"
            if case_id in {"cancelled-rows", "euro-and-pound", "everything-at-once"}
            else "dev"
        )
        (folder / "case.yaml").write_text(
            f"kind: task\nsplit: {split}\nexpected_outcome: {json.dumps('summary.json matches ' + json.dumps(expected))}\n"
        )
        graders = folder / "graders"
        graders.mkdir()
        (graders / "01-rows.md").write_text(
            f"---\ntype: json_field\nfile: summary.json\npath: rows\nequals: {expected['rows']}\n---\n"
        )
        (graders / "02-columns.md").write_text(
            f"---\ntype: json_field\nfile: summary.json\npath: columns\nequals: {json.dumps(expected['columns'])}\n---\n"
        )
        totals = expected["totals"]
        assert isinstance(totals, dict)
        for i, (name, value) in enumerate(sorted(totals.items()), start=3):
            (graders / f"{i:02d}-total-{name}.md").write_text(
                f"---\ntype: json_field\nfile: summary.json\npath: totals.{name}\n"
                f"approx: {value}\ntolerance: 0.01\n---\n"
            )
        (graders / "99-no-extra-totals.md").write_text(
            f"---\ntype: json_field\nfile: summary.json\npath: totals\nlength: {len(totals)}\n---\n"
        )
    triggers = {
        "trigger-profile": (
            "Can you profile inventory.csv and tell me the totals?",
            "inventory.csv",
            "sku,count\nA,3\nB,4\n",
        ),
        "trigger-total-up": (
            "Total up the numbers in q3.csv for me.",
            "q3.csv",
            "month,spend\njul,10\naug,12\n",
        ),
        "trigger-whats-in": (
            "What's in payments.csv? Give me a quick summary.",
            "payments.csv",
            "id,amount\n1,5\n2,7\n",
        ),
    }
    for case_id, (prompt, filename, text) in triggers.items():
        folder = EVALS / case_id
        (folder / "files").mkdir(parents=True)
        (folder / "files" / filename).write_text(text)
        (folder / "prompt.md").write_text(prompt + "\n")
        (folder / "case.yaml").write_text("kind: trigger\n")
    no_triggers = {
        "no-trigger-json-to-yaml": "Convert config.json in this folder to YAML and save it as config.yaml.",
        "no-trigger-write-parser": "Write a Python function that reverses the words in a sentence, in reverse.py.",
    }
    for case_id, prompt in no_triggers.items():
        folder = EVALS / case_id
        folder.mkdir(parents=True)
        (folder / "prompt.md").write_text(prompt + "\n")
        (folder / "case.yaml").write_text("kind: no-trigger\n")
        if "config.json" in prompt:
            (folder / "files").mkdir()
            (folder / "files" / "config.json").write_text('{"name": "demo", "debug": true}\n')
    print(f"wrote {len(CASES) + len(triggers) + len(no_triggers)} cases to {EVALS}")


if __name__ == "__main__":
    main()
