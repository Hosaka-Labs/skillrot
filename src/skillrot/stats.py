"""Paired, case-clustered statistics for small eval suites.

Trials of one case are not independent, so we average them per case first and resample
cases (a cluster bootstrap). With 15-40 cases the normal approximation is unreliable, so
intervals come from the bootstrap percentiles, not from a standard error.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_RESAMPLES = 4000
DEFAULT_CONFIDENCE = 0.90
_Z_ALPHA = 1.645  # one-sided 95% (two-sided 90%)
_Z_POWER = 0.842  # 80% power


@dataclass(frozen=True, slots=True)
class Estimate:
    """A mean difference with its bootstrap interval."""

    mean: float
    low: float
    high: float
    n: int
    mde: float  # smallest true effect we would detect with 80% power

    @property
    def width(self) -> float:
        """Interval width."""
        return self.high - self.low


def case_means(passes: Mapping[str, Sequence[bool]]) -> dict[str, float]:
    """Pass rate per case, skipping cases with no usable trials."""
    return {case: fmean(map(float, results)) for case, results in passes.items() if results}


def paired_uplift(
    treatment: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate | None:
    """Mean of per-case (treatment - baseline), over cases present in both."""
    shared = sorted(treatment.keys() & baseline.keys())
    diffs = [treatment[c] - baseline[c] for c in shared]
    return bootstrap_mean(diffs, resamples=resamples, confidence=confidence, seed=seed)


def paired_drift(
    uplift_old: Mapping[str, float],
    uplift_new: Mapping[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate | None:
    """Change in per-case uplift between two models, paired by case."""
    return paired_uplift(
        uplift_new, uplift_old, resamples=resamples, confidence=confidence, seed=seed
    )


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate | None:
    """Percentile bootstrap interval for the mean. None when there is no data."""
    n = len(values)
    if n == 0:
        return None
    mean = fmean(values)
    if n == 1 or pstdev(values) == 0:
        se = 0.0 if n > 1 else math.inf
        return Estimate(mean, mean if n > 1 else -1.0, mean if n > 1 else 1.0, n, _mde(se))
    rng = random.Random(seed)  # noqa: S311 - statistics, not crypto
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * (resamples - 1))]
    high = means[math.ceil((1 - tail) * (resamples - 1))]
    se = pstdev(means)
    return Estimate(mean, low, high, n, _mde(se))


def _mde(se: float) -> float:
    return (_Z_ALPHA + _Z_POWER) * se


def rate(values: Sequence[bool]) -> float | None:
    """Share of True values, or None if empty."""
    return fmean(map(float, values)) if values else None
