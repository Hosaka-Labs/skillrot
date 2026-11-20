# Changelog

## 0.1.0

First release.

- `skillrot check`: runs cases under none / skill / forced / prior, then gives a verdict per model with a 90% cluster-bootstrap interval. Supports tiers, `--escalate`, budget caps, caching and resume.
- `show`, `diff`, `lint`, `new`, `models`, `doctor`, `schema` commands.
- Claude Code (`claude -p`) and Anthropic Messages API harnesses; harness plugins via entry points.
- Deterministic graders (`json_field`, `command`, `regex`, `contains`, `file_exists`, `tool_used`) and a redacted, pinned `judge`.
- Versioned `ResultBundle` JSON output for dashboards; GitHub composite action.
