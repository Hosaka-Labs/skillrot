from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from skillrot.stats import bootstrap_mean, case_means, paired_drift, paired_uplift


def test_empty_is_none():
    assert bootstrap_mean([]) is None


def test_constant_differences_have_zero_width():
    est = bootstrap_mean([1.0] * 12)
    assert est is not None
    assert (est.mean, est.low, est.high) == (1.0, 1.0, 1.0)


def test_single_case_is_maximally_uncertain():
    est = bootstrap_mean([0.5])
    assert est is not None
    assert est.low == -1.0
    assert est.high == 1.0


@given(st.lists(st.floats(-1, 1), min_size=2, max_size=40))
def test_interval_brackets_mean(values: list[float]):
    est = bootstrap_mean(values, resamples=400)
    assert est is not None
    assert est.low - 1e-9 <= est.mean <= est.high + 1e-9
    assert est.mde >= 0


def test_bootstrap_is_deterministic_for_a_seed():
    values = [0.0, 1.0, 0.5, 1.0, 0.0, 0.25]
    assert bootstrap_mean(values, seed=3) == bootstrap_mean(values, seed=3)


def test_paired_uplift_only_uses_shared_cases():
    treatment = {"a": 1.0, "b": 1.0, "c": 0.0}
    baseline = {"a": 0.0, "b": 1.0}
    est = paired_uplift(treatment, baseline)
    assert est is not None
    assert est.n == 2
    assert est.mean == 0.5


def test_case_means_skip_empty():
    assert case_means({"a": [True, False], "b": []}) == {"a": 0.5}


def test_drift_is_new_minus_old():
    old = {f"c{i}": 0.5 for i in range(10)}
    new = {f"c{i}": 0.0 for i in range(10)}
    est = paired_drift(old, new)
    assert est is not None
    assert est.mean == -0.5
