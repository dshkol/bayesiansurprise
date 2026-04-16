from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm


def as_numeric_array(x, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def normalize_prob(x, *, na_rm: bool = False) -> np.ndarray:
    arr = as_numeric_array(x)
    if np.any(arr[~np.isnan(arr)] < 0):
        warnings.warn("Negative values in x set to zero.", RuntimeWarning, stacklevel=2)
        arr = np.maximum(arr, 0)

    total = np.nansum(arr) if na_rm else np.sum(arr)
    if not np.isfinite(total) or total == 0:
        n = np.sum(~np.isnan(arr)) if na_rm else arr.size
        if n == 0:
            raise ValueError("Cannot normalize an empty or all-NA vector.")
        result = np.full(arr.size, 1 / n, dtype=float)
        if not na_rm:
            result[np.isnan(arr)] = np.nan
        return result

    return arr / total


def normalize_rate(count, base, *, per: float = 1.0, na_for_zero: bool = True) -> np.ndarray:
    count_arr = as_numeric_array(count, name="count")
    base_arr = as_numeric_array(base, name="base")
    if count_arr.size != base_arr.size:
        raise ValueError("count and base must have the same length.")

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = count_arr / base_arr * per

    if na_for_zero:
        rate[base_arr == 0] = np.nan
    else:
        rate[~np.isfinite(rate)] = 0
    return rate


def funnel_zscore(observed, expected, sample_size, *, kind: str = "count") -> np.ndarray:
    observed_arr = as_numeric_array(observed, name="observed")
    expected_arr = as_numeric_array(expected, name="expected")
    sample_arr = as_numeric_array(sample_size, name="sample_size")
    if observed_arr.size != expected_arr.size or observed_arr.size != sample_arr.size:
        raise ValueError("observed, expected, and sample_size must have the same length.")

    if kind == "count":
        se = np.sqrt(np.maximum(expected_arr, 0.5))
    elif kind == "proportion":
        p = np.clip(expected_arr / sample_arr, 0.001, 0.999)
        se = np.sqrt(p * (1 - p) / sample_arr)
    else:
        raise ValueError("kind must be 'count' or 'proportion'.")

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed_arr - expected_arr) / se
    z[~np.isfinite(z)] = 0
    return z


def funnel_pvalue(z) -> np.ndarray:
    z_arr = as_numeric_array(z, name="z")
    return 2 * norm.sf(np.abs(z_arr))


def compute_funnel_data(
    observed,
    sample_size,
    *,
    target_rate: float | None = None,
    kind: str = "count",
    limits=(2, 3),
) -> pd.DataFrame:
    """Compute funnel diagnostic data.

    This diagnostic is separate from Bayesian Surprise scoring. With
    ``kind="count"``, ``observed`` is interpreted as event counts and
    ``sample_size`` as the denominator. With ``kind="proportion"``,
    ``observed`` is interpreted as rates or proportions on [0, 1].
    """

    observed_arr = as_numeric_array(observed, name="observed")
    sample_arr = as_numeric_array(sample_size, name="sample_size")
    if observed_arr.size != sample_arr.size:
        raise ValueError("observed and sample_size must have the same length.")
    if kind not in {"count", "proportion"}:
        raise ValueError("kind must be 'count' or 'proportion'.")
    if np.any(sample_arr < 0):
        raise ValueError("sample_size must be non-negative.")

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "count":
            observed_rate = observed_arr / sample_arr
            inferred_rate = np.nansum(observed_arr) / np.nansum(sample_arr)
            expected = sample_arr * (target_rate if target_rate is not None else inferred_rate)
        else:
            observed_rate = observed_arr
            inferred_rate = np.nansum(observed_arr * sample_arr) / np.nansum(sample_arr)
            expected = np.full(observed_arr.size, target_rate if target_rate is not None else inferred_rate)

    rate = float(target_rate if target_rate is not None else inferred_rate)
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("target_rate must be a finite value on [0, 1].")

    p = np.clip(rate, 0.001, 0.999)
    rate_se = np.sqrt(p * (1 - p) / sample_arr)
    count_se = rate_se * sample_arr
    se = count_se if kind == "count" else rate_se

    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = (observed_rate - rate) / rate_se
    z_score[~np.isfinite(z_score)] = 0

    result = pd.DataFrame(
        {
            "observed": observed_arr,
            "sample_size": sample_arr,
            "rate": observed_rate,
            "expected": expected,
            "expected_rate": np.full(observed_arr.size, rate),
            "se": se,
            "rate_se": rate_se,
            "z_score": z_score,
        }
    )
    for limit in limits:
        label = f"{limit:g}sd"
        result[f"lower_{label}"] = expected - float(limit) * se
        result[f"upper_{label}"] = expected + float(limit) * se
        result[f"lower_{label}_rate"] = rate - float(limit) * rate_se
        result[f"upper_{label}_rate"] = rate + float(limit) * rate_se
    return result
