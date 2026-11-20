"""Per-trial working directories and environment hygiene.

v0.1 isolates each trial in a fresh temporary directory on the host, with a scrubbed
environment. It does not contain the agent: a trial can run any command your user can.
Run untrusted skills inside a container or VM you control.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from skillrot.suite import Case

_PASS_THROUGH = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR")
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "SESSION")


@contextmanager
def workspace(case: Case, *, keep: bool = False) -> Generator[Path]:
    """Yield a fresh directory holding a copy of the case's `files/`; delete it afterwards."""
    root = Path(tempfile.mkdtemp(prefix=f"skillrot-{case.id[:40]}-"))
    try:
        if case.files_dir is not None:
            shutil.copytree(case.files_dir, root, dirs_exist_ok=True)
        yield root
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


def scrubbed_env(keep: Iterable[str] = (), extra: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal environment: basics plus the variables a harness explicitly needs."""
    names = {*_PASS_THROUGH, *keep}
    env = {k: v for k, v in os.environ.items() if k in names}
    env.update(extra or {})
    return env


def scrub_text(text: str) -> str:
    """Remove values of secret-looking environment variables from text before storing it."""
    for name, value in os.environ.items():
        if len(value) >= 8 and any(h in name.upper() for h in _SECRET_HINTS):  # noqa: PLR2004
            text = text.replace(value, f"[{name}]")
    return text
