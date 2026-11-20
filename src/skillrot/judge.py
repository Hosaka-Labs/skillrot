"""Rubric judge backed by the Anthropic Messages API (`pip install skillrot[anthropic]`)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from skillrot.errors import HarnessError

if TYPE_CHECKING:
    from skillrot.graders import JudgeView

JUDGE_PROMPT_VERSION = "1"
_SYSTEM = (
    "You grade the result of a task an AI agent attempted. You are given the task, the agent's "
    "final message and the files in its working directory. Decide whether ONE statement is true "
    "of the result. Judge only that statement. Quote the exact text that decides it as evidence. "
    "If the evidence is missing, the statement is false."
)
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string", "description": "Exact quote that decides the statement"},
        "passed": {"type": "boolean"},
    },
    "required": ["evidence", "passed"],
    "additionalProperties": False,
}


class AnthropicJudge:
    """Checks one statement per call with structured JSON output."""

    def __init__(self, model: str, client: Any | None = None) -> None:  # noqa: ANN401 - SDK client
        self.model = model
        self._client = client

    @property
    def prompt_version(self) -> str:
        """Recorded with each grade so a prompt change starts a new judge series."""
        return f"{self.model}@{JUDGE_PROMPT_VERSION}"

    def _ensure_client(self) -> Any:  # noqa: ANN401
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415 - optional dependency
            except ImportError as exc:
                msg = "the judge needs the anthropic extra: pip install 'skillrot[anthropic]'"
                raise HarnessError(msg) from exc
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def __call__(self, statement: str, view: JudgeView) -> tuple[bool, str]:
        """Return (passed, evidence) for one statement."""
        client = self._ensure_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": f"{view.render()}\n\n<statement>\n{statement}\n</statement>",
                }
            ],
        )
        if response.stop_reason == "refusal":
            return False, "judge refused"
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        try:
            verdict = json.loads(text)
        except json.JSONDecodeError:
            return False, f"judge returned unparseable output: {text[:200]}"
        return bool(verdict.get("passed")), str(verdict.get("evidence", ""))
