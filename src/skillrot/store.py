"""Local result store: SQLite (WAL) for trial rows, gzipped JSONL for transcripts.

The store is an internal detail. Other programs should read result bundles
(`skillrot show --json`), whose schema is versioned; this one is not.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from skillrot.models import Condition, ExitReason, Usage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import TracebackType

    from skillrot.graders import GradeResult

DEFAULT_STORE = Path(".skillrot")
_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trials (
    key TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    skill_hash TEXT NOT NULL,
    case_id TEXT NOT NULL,
    case_hash TEXT NOT NULL,
    condition TEXT NOT NULL,
    model TEXT NOT NULL,
    harness TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    exit_reason TEXT NOT NULL,
    skill_loaded INTEGER NOT NULL,
    passed INTEGER,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL,
    wall_s REAL NOT NULL,
    turns INTEGER NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trials_by_skill ON trials (skill_name, model, harness);
CREATE TABLE IF NOT EXISTS grades (
    trial_key TEXT NOT NULL REFERENCES trials(key) ON DELETE CASCADE,
    grader_id TEXT NOT NULL,
    grader_hash TEXT NOT NULL,
    passed INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    judge TEXT NOT NULL,
    PRIMARY KEY (trial_key, grader_id)
);
"""


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One stored trial."""

    key: str
    skill_name: str
    skill_hash: str
    case_id: str
    case_hash: str
    condition: Condition
    model: str
    harness: str
    harness_version: str
    trial_index: int
    attempts: int
    exit_reason: ExitReason
    skill_loaded: bool
    passed: bool | None
    usage: Usage
    error: str = ""
    created_at: str = ""


class Store:
    """Open with `Store.open(path)` and use as a context manager."""

    def __init__(self, root: Path, conn: sqlite3.Connection) -> None:
        self.root = root
        self._conn = conn

    @classmethod
    def open(cls, root: str | Path = DEFAULT_STORE) -> Store:
        """Open or create a store folder."""
        path = Path(root)
        (path / "transcripts").mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path / "store.db", isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(_SCHEMA_VERSION),)
        )
        return cls(path, conn)

    def close(self) -> None:
        """Close the database."""
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writes ------------------------------------------------------------------------

    def put(
        self,
        record: TrialRecord,
        grades: Sequence[GradeResult] = (),
        events: Iterable[dict[str, Any]] = (),
    ) -> None:
        """Insert or replace a trial with its grades and transcript."""
        created = record.created_at or datetime.now(UTC).isoformat(timespec="seconds")
        u = record.usage
        with self._conn:
            self._conn.execute("BEGIN")
            self._conn.execute("DELETE FROM grades WHERE trial_key = ?", (record.key,))
            self._conn.execute(
                "INSERT OR REPLACE INTO trials VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.key, record.skill_name, record.skill_hash, record.case_id,
                    record.case_hash, record.condition.value, record.model, record.harness,
                    record.harness_version, record.trial_index, record.attempts,
                    record.exit_reason.value, int(record.skill_loaded),
                    None if record.passed is None else int(record.passed),
                    u.input_tokens, u.output_tokens, u.cost_usd, u.wall_s, u.turns,
                    record.error, created,
                ),
            )  # fmt: skip
            self._conn.executemany(
                "INSERT INTO grades VALUES (?,?,?,?,?,?)",
                [
                    (record.key, g.grader_id, g.grader_hash, int(g.passed), g.evidence, g.judge)
                    for g in grades
                ],
            )
        with gzip.open(self.transcript_path(record.key), "wt") as fh:
            for event in events:
                fh.write(json.dumps(event, default=str) + "\n")

    def set_grades(self, key: str, grades: Sequence[GradeResult]) -> None:
        """Replace the grades of an existing trial and update its pass flag."""
        with self._conn:
            self._conn.execute("BEGIN")
            self._conn.execute("DELETE FROM grades WHERE trial_key = ?", (key,))
            self._conn.executemany(
                "INSERT INTO grades VALUES (?,?,?,?,?,?)",
                [
                    (key, g.grader_id, g.grader_hash, int(g.passed), g.evidence, g.judge)
                    for g in grades
                ],
            )
            passed = all(g.passed for g in grades) if grades else None
            self._conn.execute(
                "UPDATE trials SET passed = ? WHERE key = ?",
                (None if passed is None else int(passed), key),
            )

    # --- reads -------------------------------------------------------------------------

    def transcript_path(self, key: str) -> Path:
        """Where a trial's transcript lives."""
        return self.root / "transcripts" / f"{key}.jsonl.gz"

    def events(self, key: str) -> list[dict[str, Any]]:
        """Load a stored transcript."""
        path = self.transcript_path(key)
        if not path.exists():
            return []
        with gzip.open(path, "rt") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def get(self, key: str) -> TrialRecord | None:
        """Fetch one trial by cache key."""
        row = self._conn.execute("SELECT * FROM trials WHERE key = ?", (key,)).fetchone()
        return _record(row) if row else None

    def trials(
        self,
        skill_name: str,
        *,
        model: str | None = None,
        harness: str | None = None,
        case_hashes: Iterable[str] | None = None,
    ) -> list[TrialRecord]:
        """Trials for a skill, newest first, optionally filtered."""
        sql = "SELECT * FROM trials WHERE skill_name = ?"
        args: list[Any] = [skill_name]
        if model is not None:
            sql += " AND model = ?"
            args.append(model)
        if harness is not None:
            sql += " AND harness = ?"
            args.append(harness)
        rows = self._conn.execute(sql + " ORDER BY created_at DESC", args).fetchall()
        records = [_record(r) for r in rows]
        if case_hashes is not None:
            wanted = set(case_hashes)
            records = [r for r in records if r.case_hash in wanted]
        return records

    def grades(self, key: str) -> list[dict[str, Any]]:
        """Return stored grades for a trial."""
        rows = self._conn.execute("SELECT * FROM grades WHERE trial_key = ?", (key,)).fetchall()
        return [dict(r) for r in rows]

    def models(self, skill_name: str | None = None) -> list[tuple[str, str, int, str]]:
        """(model, harness, trial count, last run) for everything stored."""
        sql = "SELECT model, harness, COUNT(*), MAX(created_at) FROM trials"
        args: tuple[str, ...] = ()
        if skill_name:
            sql += " WHERE skill_name = ?"
            args = (skill_name,)
        rows = self._conn.execute(sql + " GROUP BY model, harness ORDER BY 4 DESC", args)
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def mean_cost(self, model: str, harness: str) -> float | None:
        """Average cost of a completed trial on a model, for estimates."""
        row = self._conn.execute(
            "SELECT AVG(cost_usd) FROM trials WHERE model = ? AND harness = ? AND cost_usd IS NOT NULL",
            (model, harness),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None


def _record(row: sqlite3.Row) -> TrialRecord:
    return TrialRecord(
        key=row["key"],
        skill_name=row["skill_name"],
        skill_hash=row["skill_hash"],
        case_id=row["case_id"],
        case_hash=row["case_hash"],
        condition=Condition(row["condition"]),
        model=row["model"],
        harness=row["harness"],
        harness_version=row["harness_version"],
        trial_index=row["trial_index"],
        attempts=row["attempts"],
        exit_reason=ExitReason(row["exit_reason"]),
        skill_loaded=bool(row["skill_loaded"]),
        passed=None if row["passed"] is None else bool(row["passed"]),
        usage=Usage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=row["cost_usd"],
            wall_s=row["wall_s"],
            turns=row["turns"],
        ),
        error=row["error"],
        created_at=row["created_at"],
    )
