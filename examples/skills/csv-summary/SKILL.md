---
name: csv-summary
description: Summarize a CSV export into summary.json with the row count, column names and numeric totals. Use when asked to summarize, total or profile a CSV file.
---

# CSV summary

Write `summary.json` in the working directory:

```json
{"rows": 3, "columns": ["order_id", "unit_price"], "totals": {"unit_price": 61.5}}
```

`columns` lists every column in file order. `totals` has one entry per numeric column.

## Reading the file

- Strip a UTF-8 byte-order mark if present.
- Detect the delimiter from the header line: comma, semicolon or tab.
- Skip blank lines.

## Rows

- Drop rows whose `status` column (any case) is `void` or `cancelled`.
- `rows` is the number of rows left, not counting the header.

## Numbers

- Clean each value before parsing: remove currency symbols (`$`, `€`, `£`), thousands
  separators (`,` when the delimiter is not a comma, or `,` followed by three digits), and spaces.
- `(12.50)` means -12.50. A trailing minus (`12.50-`) also means negative.
- A column is numeric if every non-empty cleaned value parses as a number. `id`-like columns
  are still numeric if they parse.
- Empty cells count as 0.
- Round every total to 2 decimals.

## Column names

Convert headers to snake_case: lowercase, replace every run of non-alphanumeric characters
with `_`, then trim `_` from both ends. `Unit Price ($)` becomes `unit_price`.
