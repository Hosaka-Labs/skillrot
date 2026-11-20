from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest

from skillrot.errors import HarnessError
from skillrot.harness import available, get_harness
from skillrot.harness.anthropic_api import AnthropicHarness
from skillrot.harness.base import TrialRequest
from skillrot.harness.claude_code import ClaudeCodeHarness, parse_stream, skill_invoked
from skillrot.models import Condition, ExitReason, ModelRef, ToolCall
from skillrot.sandbox import workspace
from skillrot.suite import Suite, load_suite


def request(suite: Suite, case_id: str, condition: Condition, ws: Path) -> TrialRequest:
    return TrialRequest(
        case=suite.case(case_id),
        condition=condition,
        model=ModelRef("anthropic", "claude-haiku-5-5"),
        workspace=ws,
        skill_name=suite.skill.name,
        skill=None if condition is Condition.NONE else suite.skill,
        timeout_s=30,
        max_turns=5,
    )


def test_registry():
    assert {"claude-code", "anthropic"} <= set(available())
    assert isinstance(get_harness("claude-code"), ClaudeCodeHarness)


def test_skill_invoked():
    assert skill_invoked(ToolCall("Skill", {"skill": "csv-summary"}), "csv-summary")
    assert skill_invoked(ToolCall("Skill", {"command": "plugin:csv-summary"}), "csv-summary")
    assert skill_invoked(
        ToolCall("Read", {"file_path": "/x/.claude/skills/csv-summary/SKILL.md"}), "csv-summary"
    )
    assert not skill_invoked(ToolCall("Skill", {"skill": "other"}), "csv-summary")


@pytest.mark.parametrize(
    ("condition", "loaded", "total"),
    [(Condition.NONE, False, None), (Condition.SKILL, True, 15.5), (Condition.FORCED, True, 15.5)],
)
def test_claude_code_end_to_end(sample_skill, fake_claude, condition, loaded, total):
    suite = load_suite(sample_skill)
    harness = ClaudeCodeHarness(isolate_config=False)
    with workspace(suite.case("void-rows")) as ws:
        transcript = anyio.run(harness.run, request(suite, "void-rows", condition, ws))
        summary = json.loads((ws / "summary.json").read_text())
        installed = (ws / ".claude" / "skills" / "csv-summary" / "SKILL.md").exists()
        assert not (ws / ".claude" / "skills" / "csv-summary" / "evals").exists()
    assert transcript.exit_reason is ExitReason.OK
    assert transcript.skill_loaded is loaded
    assert installed is (condition is Condition.SKILL)
    assert transcript.harness_version == "9.9.9"
    assert transcript.usage.input_tokens == 1500
    assert transcript.usage.cost_usd is not None
    assert summary["totals"].get("total") == (total if total is not None else 115.5)


def test_claude_code_infra_failure(sample_skill, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_FAIL", "1")  # must NOT reach the agent: env is scrubbed
    suite = load_suite(sample_skill)
    with workspace(suite.case("void-rows")) as ws:
        (ws / "FAKE_FAIL").touch()
        transcript = anyio.run(
            ClaudeCodeHarness(isolate_config=False).run,
            request(suite, "void-rows", Condition.NONE, ws),
        )
    assert transcript.exit_reason is ExitReason.RATE_LIMITED
    assert "overloaded" in transcript.error


def test_claude_code_missing_executable(monkeypatch):
    monkeypatch.setenv("SKILLROT_CLAUDE", "definitely-not-claude")
    with pytest.raises(HarnessError, match="not found"):
        anyio.run(ClaudeCodeHarness().check)


def test_isolated_check_requires_api_key(fake_claude, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(HarnessError, match="ANTHROPIC_API_KEY"):
        anyio.run(ClaudeCodeHarness().check)


def test_parse_stream_max_turns(sample_skill):
    suite = load_suite(sample_skill)
    out = json.dumps(
        {"type": "result", "subtype": "error_max_turns", "is_error": True, "result": ""}
    )
    t = parse_stream(
        out,
        stderr="",
        returncode=0,
        request=request(suite, "void-rows", Condition.NONE, Path()),
        harness_version="1",
        wall_s=1,
    )
    assert t.exit_reason is ExitReason.TASK_ERROR


def test_forced_prompt_in_command(sample_skill, fake_claude):
    suite = load_suite(sample_skill)
    argv = ClaudeCodeHarness().command(request(suite, "void-rows", Condition.FORCED, Path()))
    assert "--append-system-prompt" in argv
    assert "Drop rows whose" in argv[argv.index("--append-system-prompt") + 1]


# --- Anthropic API harness with a fake client ----------------------------------------


@dataclass
class Block:
    type: str
    text: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    id: str = "tu"


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 20
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class Response:
    content: list[Block]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"stop_reason": self.stop_reason, "content": [b.__dict__ for b in self.content]}


class FakeMessages:
    def __init__(self, script: list[Response]) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Response:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script: list[Response]) -> None:
        self.messages = FakeMessages(script)


def test_anthropic_harness_tool_loop(sample_skill):
    suite = load_suite(sample_skill)
    script = [
        Response(
            [Block("tool_use", name="load_skill", input={"name": "csv-summary"}, id="a")],
            "tool_use",
        ),
        Response([Block("tool_use", name="bash", input={"command": "ls"}, id="b")], "tool_use"),
        Response(
            [
                Block(
                    "tool_use",
                    name="str_replace_based_edit_tool",
                    input={"command": "create", "path": "summary.json", "file_text": "{}"},
                    id="c",
                )
            ],
            "tool_use",
        ),
        Response([Block("text", text="Done.")], "end_turn"),
    ]
    client = FakeClient(script)
    with workspace(suite.case("void-rows")) as ws:
        t = anyio.run(
            AnthropicHarness(client).run, request(suite, "void-rows", Condition.SKILL, ws)
        )
        assert (ws / "summary.json").read_text() == "{}"
    assert t.exit_reason is ExitReason.OK
    assert t.skill_loaded
    assert t.final_text == "Done."
    assert t.usage.turns == 4
    assert t.usage.input_tokens == 400
    first = client.messages.requests[0]
    assert "csv-summary:" in first["system"]
    assert any(tool["name"] == "load_skill" for tool in first["tools"])
    results = client.messages.requests[1]["messages"][-1]["content"]
    assert "Drop rows whose" in results[0]["content"]
    ls_output = client.messages.requests[2]["messages"][-1]["content"][0]["content"]
    assert "orders.csv" in ls_output


def test_anthropic_harness_baseline_has_no_skill(sample_skill):
    suite = load_suite(sample_skill)
    client = FakeClient([Response([Block("text", text="ok")], "end_turn")])
    with workspace(suite.case("void-rows")) as ws:
        t = anyio.run(AnthropicHarness(client).run, request(suite, "void-rows", Condition.NONE, ws))
    req = client.messages.requests[0]
    assert "csv-summary" not in req["system"]
    assert all(tool["name"] != "load_skill" for tool in req["tools"])
    assert not t.skill_loaded


def test_anthropic_editor_confined(sample_skill):
    suite = load_suite(sample_skill)
    escape = Block(
        "tool_use",
        name="str_replace_based_edit_tool",
        input={"command": "create", "path": "../evil.txt", "file_text": "x"},
        id="e",
    )
    client = FakeClient(
        [Response([escape], "tool_use"), Response([Block("text", text="ok")], "end_turn")]
    )
    with workspace(suite.case("void-rows")) as ws:
        anyio.run(AnthropicHarness(client).run, request(suite, "void-rows", Condition.NONE, ws))
        assert not (ws.parent / "evil.txt").exists()
    result = client.messages.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"]


def test_anthropic_refusal(sample_skill):
    suite = load_suite(sample_skill)
    client = FakeClient([Response([], "refusal")])
    with workspace(suite.case("void-rows")) as ws:
        t = anyio.run(AnthropicHarness(client).run, request(suite, "void-rows", Condition.NONE, ws))
    assert t.exit_reason is ExitReason.REFUSED
