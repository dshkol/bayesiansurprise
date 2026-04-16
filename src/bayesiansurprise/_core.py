from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._models import (
    ModelSpace,
    UniformModel,
    BaseRateModel,
    FunnelModel,
    GaussianModel,
    SampledModel,
    build_model_space_from_spec,
    model_space,
)
from ._utils import as_numeric_array


@dataclass(frozen=True)
class SurpriseResult:
    surprise: np.ndarray
    signed_surprise: np.ndarray | None
    model_space: ModelSpace
    posteriors: np.ndarray | None = None
    model_contributions: np.ndarray | None = None
    data_info: dict | None = None


def kl_divergence(posterior, prior, *, base: float = 2) -> float:
    posterior_arr = as_numeric_array(posterior, name="posterior")
    prior_arr = as_numeric_array(prior, name="prior")
    if posterior_arr.size != prior_arr.size:
        raise ValueError("posterior and prior must have the same length.")
    if np.any((posterior_arr > 0) & (prior_arr == 0)):
        return float("inf")

    idx = (posterior_arr > 0) & (prior_arr > 0)
    if not np.any(idx):
        return 0.0
    return float(np.sum(posterior_arr[idx] * np.log(posterior_arr[idx] / prior_arr[idx]) / np.log(base)))


def log_sum_exp(x) -> float:
    arr = as_numeric_array(x)
    if arr.size == 0:
        return -np.inf
    if np.all(np.isneginf(arr)):
        return -np.inf
    max_x = np.max(arr[np.isfinite(arr)])
    return float(max_x + np.log(np.sum(np.exp(arr - max_x))))


def bayesian_update(space: ModelSpace, observed, *, region_idx: int | None = None) -> ModelSpace:
    observed_arr = as_numeric_array(observed, name="observed")
    log_likelihoods = np.array([model.log_likelihood(observed_arr, region_idx) for model in space.models])
    log_posterior_unnorm = log_likelihoods + np.log(space.prior)
    log_normalizer = log_sum_exp(log_posterior_unnorm)
    posterior = np.exp(log_posterior_unnorm - log_normalizer)
    posterior = posterior / np.sum(posterior)
    return space.with_posterior(posterior)


def compute_surprise(
    space: ModelSpace,
    observed,
    *,
    expected=None,
    return_signed: bool = True,
    return_posteriors: bool = False,
    return_contributions: bool = False,
    normalize_posterior: bool = True,
) -> SurpriseResult:
    observed_arr = as_numeric_array(observed, name="observed")
    n = observed_arr.size
    prior = space.prior

    surprise_values = np.full(n, np.nan, dtype=float)
    signed_values = np.full(n, np.nan, dtype=float) if return_signed else None
    posteriors = np.zeros((n, space.n_models), dtype=float) if return_posteriors else None
    contributions = np.zeros((n, space.n_models), dtype=float) if return_contributions else None

    expected_for_sign = None
    if return_signed:
        if expected is None:
            expected_for_sign = np.full(n, np.nanmean(observed_arr), dtype=float)
        else:
            expected_arr = as_numeric_array(expected, name="expected")
            if expected_arr.size != n:
                raise ValueError("expected must have the same length as observed.")
            if normalize_posterior:
                overall_rate = np.nansum(observed_arr) / np.nansum(expected_arr)
                expected_for_sign = expected_arr * overall_rate
            else:
                rates = observed_arr / expected_arr
                expected_for_sign = expected_arr * np.nanmean(rates)

    for i, obs_i in enumerate(observed_arr):
        if np.isnan(obs_i):
            continue

        log_likelihoods = np.array([model.log_likelihood(observed_arr, i) for model in space.models])
        log_posterior_unnorm = log_likelihoods + np.log(prior)

        if normalize_posterior:
            log_normalizer = log_sum_exp(log_posterior_unnorm)
            region_posterior = np.exp(log_posterior_unnorm - log_normalizer)
            region_posterior = region_posterior / np.sum(region_posterior)
        else:
            region_posterior = np.exp(log_posterior_unnorm)

        kl_value = kl_divergence(region_posterior, prior)
        surprise_values[i] = max(kl_value, 0) if normalize_posterior else abs(kl_value)

        if posteriors is not None:
            posteriors[i, :] = region_posterior

        if contributions is not None:
            idx = (region_posterior > 0) & (prior > 0)
            contributions[i, idx] = region_posterior[idx] * np.log(region_posterior[idx] / prior[idx]) / np.log(2)

        if signed_values is not None and expected_for_sign is not None:
            signed_values[i] = np.sign(obs_i - expected_for_sign[i]) * surprise_values[i]

    data_info = {
        "n": n,
        "observed_range": (float(np.nanmin(observed_arr)), float(np.nanmax(observed_arr))),
        "expected_range": None if expected is None else (
            float(np.nanmin(as_numeric_array(expected))),
            float(np.nanmax(as_numeric_array(expected))),
        ),
    }
    return SurpriseResult(surprise_values, signed_values, space, posteriors, contributions, data_info)


def _extract_vector(data, value, *, name: str):
    if value is None:
        return None
    if isinstance(value, str):
        if not hasattr(data, "__getitem__"):
            raise TypeError(f"{name} can only be a column name when data is table-like.")
        return as_numeric_array(data[value], name=name)
    return as_numeric_array(value, name=name)


def surprise(
    data,
    observed,
    *,
    expected=None,
    sample_size=None,
    models=("uniform", "baserate", "funnel"),
    prior=None,
    signed: bool = True,
    normalize_posterior: bool = True,
):
    obs_vals = _extract_vector(data, observed, name="observed")
    exp_vals = _extract_vector(data, expected, name="expected")
    size_vals = _extract_vector(data, sample_size, name="sample_size")
    if size_vals is None:
        size_vals = exp_vals

    space = build_model_space_from_spec(models, expected=exp_vals, sample_size=size_vals, prior=prior)
    result = compute_surprise(
        space,
        obs_vals,
        expected=exp_vals,
        return_signed=signed,
        normalize_posterior=normalize_posterior,
    )

    if isinstance(data, pd.DataFrame):
        out = data.copy()
        out["surprise"] = result.surprise
        if signed and result.signed_surprise is not None:
            out["signed_surprise"] = result.signed_surprise
        out.attrs["surprise_result"] = result
        return out
    return result


def get_surprise(data, kind: str = "surprise"):
    """Extract surprise values from a result object or augmented DataFrame."""

    if isinstance(data, SurpriseResult):
        if kind == "surprise":
            return data.surprise
        if kind in {"signed", "signed_surprise"}:
            return data.signed_surprise
        raise ValueError("kind must be 'surprise', 'signed', or 'signed_surprise'.")
    if isinstance(data, pd.DataFrame):
        column = "signed_surprise" if kind == "signed" else kind
        if column not in data:
            raise ValueError(f"Column {column!r} not found in data.")
        return data[column].to_numpy()
    raise TypeError("data must be a SurpriseResult or pandas DataFrame.")


def get_signed_surprise(data):
    """Extract signed surprise values from a result object or augmented DataFrame."""

    return get_surprise(data, "signed_surprise")


def get_surprise_result(data) -> SurpriseResult:
    """Return the stored SurpriseResult from an augmented DataFrame."""

    if isinstance(data, SurpriseResult):
        return data
    if isinstance(data, pd.DataFrame) and "surprise_result" in data.attrs:
        return data.attrs["surprise_result"]
    raise ValueError("No SurpriseResult is attached to data.")


def auto_surprise(
    observed,
    expected=None,
    *,
    sample_size=None,
    include_gaussian: bool = False,
    include_sampled: bool = False,
    signed: bool = True,
    normalize_posterior: bool = True,
) -> SurpriseResult:
    models = [UniformModel()]
    if expected is not None:
        models.append(BaseRateModel(expected))
    sample = sample_size if sample_size is not None else expected
    if sample is not None:
        models.append(FunnelModel(sample))
    if include_gaussian:
        models.append(GaussianModel())
    if include_sampled:
        models.append(SampledModel())

    space = model_space(*models)
    return compute_surprise(
        space,
        observed,
        expected=expected,
        return_signed=signed,
        normalize_posterior=normalize_posterior,
    )
