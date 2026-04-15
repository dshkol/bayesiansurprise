from __future__ import annotations

import warnings

import numpy as np
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

