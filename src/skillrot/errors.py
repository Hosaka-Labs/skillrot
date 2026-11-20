"""Exception hierarchy. Everything skillrot raises on purpose derives from `SkillrotError`."""

from __future__ import annotations


class SkillrotError(Exception):
    """Base class for all skillrot errors."""


class ConfigError(SkillrotError):
    """Invalid configuration or command-line usage."""


class SuiteError(SkillrotError):
    """A skill or its eval cases could not be loaded."""


class GraderError(SkillrotError):
    """A grader is misconfigured (not: a grader that failed its check)."""


class HarnessError(SkillrotError):
    """A harness could not be started or is not available on this machine."""


class BudgetExceededError(SkillrotError):
    """The run stopped because the spending limit was reached."""
