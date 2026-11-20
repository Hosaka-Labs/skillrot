"""Run trials through the Claude Code CLI in headless mode (`claude -p`).

The skill is installed as a project skill (`.claude/skills/<name>/`) in the trial's working
directory for SKILL and PRIOR, and appended to the system prompt for FORCED. By default each
trial gets an empty Claude Code config directory, so your personal skills, CLAUDE.md and
settings cannot leak into the baseline. That also hides a subscription login: use an API key
(ANTHROPIC_API_KEY) or pass `isolate_config=False` / `--no-isolate`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Any, cast

import anyio

from skillrot.errors import HarnessError
from skillrot.harness.base import Transcript, TrialRequest
from skillrot.models import Condition, ExitReason, ToolCall, Usage
from skillrot.sandbox import scrubbed_env

if TYPE_CHECKING:
    from pathlib import Path

_AUTH_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "AWS_REGION", "AWS_PROFILE",
    "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION", "GOOGLE_APPLICATION_CREDENTIALS",
)  # fmt: skip
_RATE_LIMIT_HINTS = ("rate limit", "rate_limit", "429", "overloaded")


class ClaudeCodeHarness:
    """Claude Code CLI, one subprocess per trial."""

    name = "claude-code"

    def __init__(
        self,
        executable: str | None = None,
        *,
        isolate_config: bool = True,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.executable = executable or os.environ.get("SKILLROT_CLAUDE", "claude")
        self.isolate_config = isolate_config
        self.extra_args = extra_args
        self._version: str | None = None

    def _resolve(self) -> str:
        path = shutil.which(self.executable)
        if path is None:
            msg = (
                f"Claude Code CLI not found ({self.executable!r}). Install it "
                "(npm install -g @anthropic-ai/claude-code) or set SKILLROT_CLAUDE to its path."
            )
            raise HarnessError(msg)
        return path

    async def version(self) -> str:
        """`claude --version`, cached."""
        if self._version is None:
            result = await anyio.run_process([self._resolve(), "--version"], check=False)
            if result.returncode != 0:
                msg = f"`claude --version` failed: {result.stderr.decode().strip()}"
                raise HarnessError(msg)
            self._version = result.stdout.decode().strip().split()[0]
        return self._version

    async def check(self) -> None:
        """Fail early with a fix-it message."""
        await self.version()
        if self.isolate_config and not any(os.environ.get(k) for k in _AUTH_ENV[:2]):
            msg = (
                "isolated runs need ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN). To use your "
                "Claude Code login instead, pass --no-isolate (your personal skills may then leak "
                "into the baseline; skillrot flags that as `contaminated`)."
            )
            raise HarnessError(msg)

    def command(self, request: TrialRequest) -> list[str]:
        """Build the argv for one trial."""
        args = [
            self._resolve(), "-p", request.case.prompt,
            "--output-format", "stream-json", "--verbose",
            "--model", request.model.model_id,
            "--max-turns", str(request.max_turns),
            "--permission-mode", "bypassPermissions",
        ]  # fmt: skip
        if request.condition is Condition.FORCED and request.skill is not None:
            args += ["--append-system-prompt", forced_prompt(request.skill.body)]
        return [*args, *self.extra_args]

    async def run(self, request: TrialRequest) -> Transcript:
        """Install the skill if needed, run the CLI, parse its stream-json output."""
        if request.condition in {Condition.SKILL, Condition.PRIOR} and request.skill is not None:
            install_skill(
                request.skill.path, request.workspace / ".claude" / "skills" / request.skill_name
            )
        version = await self.version()
        with tempfile.TemporaryDirectory(prefix="skillrot-cc-config-") as config_dir:
            extra = {"CLAUDE_CONFIG_DIR": config_dir} if self.isolate_config else {}
            env = scrubbed_env(keep=_AUTH_ENV, extra=extra)
            started = time.monotonic()
            try:
                with anyio.fail_after(request.timeout_s):
                    proc = await anyio.run_process(
                        self.command(request), cwd=request.workspace, env=env, check=False
                    )
            except TimeoutError:
                return Transcript(
                    final_text="",
                    exit_reason=ExitReason.TIMEOUT,
                    skill_loaded=False,
                    usage=Usage(wall_s=time.monotonic() - started),
                    harness_version=version,
                    error=f"timed out after {request.timeout_s:g}s",
                )
        wall = time.monotonic() - started
        return parse_stream(
            proc.stdout.decode(errors="replace"),
            stderr=proc.stderr.decode(errors="replace"),
            returncode=proc.returncode,
            request=request,
            harness_version=version,
            wall_s=wall,
        )


def forced_prompt(body: str) -> str:
    """System-prompt text for the FORCED condition."""
    return f"Follow these instructions for this task.\n\n<instructions>\n{body}\n</instructions>"


def install_skill(source: Path, target: Path) -> None:
    """Copy a skill folder, without its evals, to where an agent will find it."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("evals", ".skillrot", ".git"))


def parse_stream(
    stdout: str,
    *,
    stderr: str,
    returncode: int,
    request: TrialRequest,
    harness_version: str,
    wall_s: float,
) -> Transcript:
    """Parse `claude -p --output-format stream-json` output into a transcript."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(cast("dict[str, Any]", item))
    calls = [call for event in events for call in _tool_calls(event)]
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    loaded = request.condition is Condition.FORCED or any(
        skill_invoked(c, request.skill_name) for c in calls
    )
    if result is None:
        reason = ExitReason.RATE_LIMITED if _looks_rate_limited(stderr) else ExitReason.INFRA
        return Transcript(
            final_text="",
            exit_reason=reason,
            skill_loaded=loaded,
            usage=Usage(wall_s=wall_s),
            harness_version=harness_version,
            tool_calls=tuple(calls),
            error=(stderr.strip() or f"claude exited {returncode} without a result")[-2000:],
            events=tuple(events),
        )
    text = str(result.get("result", ""))
    usage = cast("dict[str, Any]", result.get("usage") or {})
    input_tokens = sum(
        int(usage.get(k, 0) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    cost = result.get("total_cost_usd")
    return Transcript(
        final_text=text,
        exit_reason=_exit_reason(result, text),
        skill_loaded=loaded,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cost_usd=float(cost) if cost is not None else None,
            wall_s=wall_s,
            turns=int(result.get("num_turns", 0) or 0),
        ),
        harness_version=harness_version,
        tool_calls=tuple(calls),
        error=text if result.get("is_error") else "",
        events=tuple(events),
    )


def _exit_reason(result: dict[str, Any], text: str) -> ExitReason:
    if not result.get("is_error") and result.get("subtype") == "success":
        return ExitReason.OK
    if _looks_rate_limited(text):
        return ExitReason.RATE_LIMITED
    if str(result.get("subtype", "")).startswith("error_max_turns"):
        return ExitReason.TASK_ERROR
    if result.get("subtype") == "success":  # is_error with a success subtype: API-level failure
        return ExitReason.INFRA
    return ExitReason.TASK_ERROR


def _looks_rate_limited(text: str) -> bool:
    lowered = text.lower()
    return any(h in lowered for h in _RATE_LIMIT_HINTS)


def _tool_calls(event: dict[str, Any]) -> list[ToolCall]:
    if event.get("type") != "assistant":
        return []
    message = cast("dict[str, Any]", event.get("message") or {})
    content: object = message.get("content")
    if not isinstance(content, list):
        return []
    calls: list[ToolCall] = []
    for block in cast("list[object]", content):
        if not isinstance(block, dict):
            continue
        b = cast("dict[str, Any]", block)
        if b.get("type") == "tool_use":
            raw_input = b.get("input")
            tool_input = cast("dict[str, Any]", raw_input) if isinstance(raw_input, dict) else {}
            calls.append(ToolCall(name=str(b.get("name", "")), input=tool_input))
    return calls


def skill_invoked(call: ToolCall, skill_name: str) -> bool:
    """Tell whether a tool call loaded the named skill (Skill tool, or reading its SKILL.md)."""
    if call.name == "Skill":
        target = str(
            call.input.get("skill") or call.input.get("command") or call.input.get("name") or ""
        )
        return target.rsplit(":", maxsplit=1)[-1].strip().lstrip("/") == skill_name
    if call.name == "Read":
        path = str(call.input.get("file_path", ""))
        return path.endswith(f"skills/{skill_name}/SKILL.md")
    return False
