"""A small agent loop on the Anthropic Messages API (`pip install skillrot[anthropic]`).

Tools: Anthropic's bash and text-editor tools, executed locally inside the trial's working
directory with a scrubbed environment (the API key stays in this process, never in the
agent's shell). For SKILL and PRIOR the system prompt lists the skill's name and description
and offers a `load_skill` tool, mirroring progressive disclosure: the body enters context only
if the model asks for it. FORCED puts the body in the system prompt.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import anyio

from skillrot.errors import HarnessError
from skillrot.harness.base import Transcript, TrialRequest
from skillrot.harness.claude_code import forced_prompt, install_skill
from skillrot.models import Condition, ExitReason, ToolCall, Usage
from skillrot.sandbox import scrubbed_env

TOOLS_VERSION = "1"
BASH_TIMEOUT_S = 120
MAX_OUTPUT_CHARS = 30_000
SKILLS_DIR = ".skills"
_BASE_SYSTEM = (
    "You are an autonomous agent working in a project directory (your shell's current "
    "directory). Complete the user's task using the tools. Write any requested files inside "
    "that directory. When you are done, reply with a short summary."
)
_LOAD_SKILL: dict[str, Any] = {
    "name": "load_skill",
    "description": "Load the full instructions of one of the available skills by name.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name"}},
        "required": ["name"],
        "additionalProperties": False,
    },
    "strict": True,
}


class AnthropicHarness:
    """Messages API tool loop with bash + text editor."""

    name = "anthropic"

    def __init__(self, client: Any | None = None, *, max_tokens: int = 16_000) -> None:  # noqa: ANN401
        self._client = client
        self.max_tokens = max_tokens

    def _sdk(self) -> Any:  # noqa: ANN401 - optional dependency
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            msg = "the anthropic harness needs the extra: pip install 'skillrot[anthropic]'"
            raise HarnessError(msg) from exc
        return anthropic

    def _ensure_client(self) -> Any:  # noqa: ANN401
        if self._client is None:
            self._client = self._sdk().AsyncAnthropic()
        return self._client

    async def version(self) -> str:
        """SDK version plus this loop's tool version."""
        return f"anthropic-sdk-{self._sdk().__version__}+tools{TOOLS_VERSION}"

    async def check(self) -> None:
        """Make sure the SDK is importable and a client can be built."""
        try:
            self._ensure_client()
        except Exception as exc:
            msg = f"cannot create an Anthropic client: {exc}"
            raise HarnessError(msg) from exc

    async def run(self, request: TrialRequest) -> Transcript:  # noqa: C901 - one loop, many exits
        """Drive the loop until the model stops calling tools or runs out of turns."""
        sdk = self._sdk()
        client = self._ensure_client()
        system, tools = _setup(request)
        messages: list[Any] = [{"role": "user", "content": request.case.prompt}]
        events: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        loaded = request.condition is Condition.FORCED
        in_tok = out_tok = turns = 0
        started = time.monotonic()

        def done(reason: ExitReason, text: str = "", error: str = "") -> Transcript:
            return Transcript(
                final_text=text,
                exit_reason=reason,
                skill_loaded=loaded,
                usage=Usage(in_tok, out_tok, None, time.monotonic() - started, turns),
                harness_version=f"anthropic-sdk-{sdk.__version__}+tools{TOOLS_VERSION}",
                tool_calls=tuple(calls),
                error=error,
                events=tuple(events),
            )

        try:
            with anyio.fail_after(request.timeout_s):
                while turns < request.max_turns:
                    turns += 1
                    response = await client.messages.create(
                        model=request.model.model_id,
                        max_tokens=self.max_tokens,
                        system=system,
                        tools=tools,
                        messages=messages,
                    )
                    events.append(response.model_dump(mode="json"))
                    u = response.usage
                    in_tok += (
                        (u.input_tokens or 0)
                        + (u.cache_creation_input_tokens or 0)
                        + (u.cache_read_input_tokens or 0)
                    )
                    out_tok += u.output_tokens or 0
                    text = "\n".join(b.text for b in response.content if b.type == "text")
                    if response.stop_reason == "refusal":
                        return done(ExitReason.REFUSED, text, "model refused")
                    if response.stop_reason != "tool_use":
                        return done(ExitReason.OK, text)
                    messages.append({"role": "assistant", "content": response.content})
                    results: list[dict[str, Any]] = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        call = ToolCall(block.name, dict(block.input))
                        calls.append(call)
                        output, is_error, hit = await anyio.to_thread.run_sync(
                            _execute, call, request
                        )
                        loaded = loaded or hit
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output,
                                "is_error": is_error,
                            }
                        )
                    events.append({"type": "tool_results", "results": results})
                    messages.append({"role": "user", "content": results})
                return done(ExitReason.TASK_ERROR, error=f"hit max turns ({request.max_turns})")
        except TimeoutError:
            return done(ExitReason.TIMEOUT, error=f"timed out after {request.timeout_s:g}s")
        except sdk.RateLimitError as exc:
            return done(ExitReason.RATE_LIMITED, error=str(exc))
        except (sdk.APIConnectionError, sdk.InternalServerError) as exc:
            return done(ExitReason.INFRA, error=str(exc))
        except sdk.APIStatusError as exc:
            return done(ExitReason.INFRA, error=f"{exc.status_code}: {exc.message}")


def _setup(request: TrialRequest) -> tuple[str, list[dict[str, Any]]]:
    tools: list[dict[str, Any]] = [
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
    ]
    system = _BASE_SYSTEM
    skill = request.skill
    if skill is not None and request.condition in {Condition.SKILL, Condition.PRIOR}:
        install_skill(skill.path, request.workspace / SKILLS_DIR / request.skill_name)
        system += (
            "\n\nAvailable skills (call load_skill to read one when it is relevant):\n"
            f"- {request.skill_name}: {skill.description}"
        )
        tools.append(_LOAD_SKILL)
    elif skill is not None and request.condition is Condition.FORCED:
        system += "\n\n" + forced_prompt(skill.body)
    return system, tools


def _execute(call: ToolCall, request: TrialRequest) -> tuple[str, bool, bool]:
    """Run one tool call. Returns (output, is_error, skill_was_loaded)."""
    try:
        match call.name:
            case "bash":
                return (*_bash(call.input, request.workspace), False)
            case "str_replace_based_edit_tool":
                return (*_edit(call.input, request.workspace), False)
            case "load_skill":
                return _load_skill(call.input, request)
            case _:
                return f"unknown tool {call.name}", True, False
    except Exception as exc:  # noqa: BLE001 - every tool error goes back to the model
        return f"error: {exc}", True, False


def _load_skill(args: dict[str, Any], request: TrialRequest) -> tuple[str, bool, bool]:
    if request.skill is None or args.get("name") != request.skill_name:
        return f"no skill named {args.get('name')!r}", True, False
    folder = f"{SKILLS_DIR}/{request.skill_name}"
    return f"{request.skill.body}\n\n(Skill files are in ./{folder}/)", False, True


def _bash(args: dict[str, Any], cwd: Path) -> tuple[str, bool]:
    if args.get("restart"):
        return "shell restarted", False
    command = str(args.get("command", ""))
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=cwd,
            env=scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"command timed out after {BASH_TIMEOUT_S}s", True
    output = (proc.stdout + proc.stderr)[-MAX_OUTPUT_CHARS:] or "(no output)"
    return output, proc.returncode != 0


def _confined(root: Path, raw: str) -> Path:
    path = Path(raw)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        msg = f"path {raw!r} is outside the working directory"
        raise PermissionError(msg)
    return resolved


def _edit(args: dict[str, Any], root: Path) -> tuple[str, bool]:
    command = args.get("command")
    path = _confined(root, str(args.get("path", "")))
    match command:
        case "view":
            if path.is_dir():
                listing = sorted(
                    p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()
                )
                return "\n".join(listing) or "(empty directory)", False
            lines = path.read_text().splitlines()
            start, end = (args.get("view_range") or [1, len(lines)])[:2]
            end = len(lines) if end == -1 else end
            numbered = [f"{i}\t{line}" for i, line in enumerate(lines[start - 1 : end], start)]
            return "\n".join(numbered)[:MAX_OUTPUT_CHARS], False
        case "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("file_text", "")))
            return f"created {args.get('path')}", False
        case "str_replace":
            text = path.read_text()
            old = str(args.get("old_str", ""))
            count = text.count(old)
            if count != 1:
                return f"old_str must match exactly once, matched {count} times", True
            path.write_text(text.replace(old, str(args.get("new_str", "")), 1))
            return "replaced", False
        case "insert":
            lines = path.read_text().splitlines(keepends=True)
            at = int(args.get("insert_line", 0))
            new = str(args.get("insert_text", ""))
            lines.insert(at, new if new.endswith("\n") else new + "\n")
            path.write_text("".join(lines))
            return "inserted", False
        case _:
            return f"unsupported command {command!r}", True
