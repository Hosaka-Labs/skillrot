"""A harness driven by a Python function. For tests, demos and dry runs: no model, no cost."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from skillrot.harness.base import Transcript, TrialRequest
from skillrot.models import ExitReason, ToolCall, Usage


@dataclass(frozen=True, slots=True)
class ScriptedReply:
    """What the scripted agent did."""

    final_text: str = "done"
    skill_loaded: bool = False
    exit_reason: ExitReason = ExitReason.OK
    tool_calls: tuple[ToolCall, ...] = ()
    cost_usd: float | None = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


Script = Callable[[TrialRequest], ScriptedReply | Awaitable[ScriptedReply]]


class ScriptedHarness:
    """Calls `script(request)`; the script may write files into `request.workspace`."""

    name = "scripted"

    def __init__(self, script: Script, *, version: str = "1") -> None:
        self._script = script
        self._version = version
        self.calls: list[TrialRequest] = []

    async def version(self) -> str:
        """Return the fixed version string."""
        return self._version

    async def check(self) -> None:
        """Do nothing: always available."""

    async def run(self, request: TrialRequest) -> Transcript:
        """Run the script and wrap its reply."""
        self.calls.append(request)
        reply = self._script(request)
        if not isinstance(reply, ScriptedReply):
            reply = await reply
        return Transcript(
            final_text=reply.final_text,
            exit_reason=reply.exit_reason,
            skill_loaded=reply.skill_loaded,
            usage=Usage(reply.input_tokens, reply.output_tokens, reply.cost_usd, 0.0, 1),
            harness_version=self._version,
            tool_calls=reply.tool_calls,
        )
