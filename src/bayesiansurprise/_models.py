from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, norm, poisson

from ._utils import as_numeric_array, normalize_prob


@dataclass(frozen=True)
class Model:
    """Base class for Bayesian Surprise likelihood models."""

    name: str
    kind: str

    def log_likelihood(self, observed, region_idx: int | None = None) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class UniformModel(Model):
    n_regions: int | None = None

    def __init__(self, n_regions: int | None = None, name: str = "Uniform"):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "uniform")
        object.__setattr__(self, "n_regions", n_regions)

    def log_likelihood(self, observed, region_idx: int | None = None) -> float:
        observed_arr = as_numeric_array(observed, name="observed")
        n = observed_arr.size
        if region_idx is not None:
            return 0.0

        expected = np.full(n, 1 / n)
        observed_prop = normalize_prob(observed_arr)
        tvd = 0.5 * np.nansum(np.abs(observed_prop - expected))
        return float(np.log(max(1 - tvd, 1e-10)))


@dataclass(frozen=True)
class BaseRateModel(Model):
    expected: np.ndarray
    expected_prop: np.ndarray
    normalize: bool = True

    def __init__(self, expected, *, normalize: bool = True, name: str = "Base Rate"):
        expected_arr = as_numeric_array(expected, name="expected")
        if expected_arr.size == 0:
            raise ValueError("expected cannot be empty.")
        expected_prop = normalize_prob(expected_arr) if normalize and np.nansum(expected_arr) > 0 else expected_arr
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "baserate")
        object.__setattr__(self, "expected", expected_arr)
        object.__setattr__(self, "expected_prop", expected_prop)
        object.__setattr__(self, "normalize", normalize)

    def log_likelihood(self, observed, region_idx: int | None = None) -> float:
        observed_arr = as_numeric_array(observed, name="observed")
        exp_rates = self.expected_prop
        if exp_rates.size != observed_arr.size:
            if exp_rates.size == 1:
                exp_rates = np.repeat(exp_rates, observed_arr.size)
            else:
                raise ValueError("expected length must match observed length.")

        if region_idx is not None:
            total = np.nansum(observed_arr)
            if total == 0:
                return 0.0
            obs_i = observed_arr[region_idx]
            if np.isnan(obs_i):
                return -np.inf
            expected_count = total * exp_rates[region_idx]
            return float(poisson.logpmf(round(obs_i), mu=max(expected_count, 0.5)))

        observed_prop = normalize_prob(observed_arr)
        exp_rates_norm = normalize_prob(exp_rates)
        tvd = 0.5 * np.nansum(np.abs(observed_prop - exp_rates_norm))
        return float(np.log(max(1 - tvd, 1e-10)))


@dataclass(frozen=True)
class GaussianModel(Model):
    mu: float | None = None
    sigma: float | None = None
    fit_from_data: bool = True

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        *,
        fit_from_data: bool = True,
        name: str = "Gaussian",
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "gaussian")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "fit_from_data", fit_from_data)

    def _fit(self, observed) -> tuple[float, float]:
        observed_arr = as_numeric_array(observed, name="observed")
        if self.fit_from_data or self.mu is None or self.sigma is None:
            mu = float(np.nanmean(observed_arr))
            sigma = float(np.nanstd(observed_arr, ddof=1))
            if not np.isfinite(sigma) or sigma == 0:
                sigma = 1.0
            return mu, sigma
        return float(self.mu), float(self.sigma)

    def log_likelihood(self, observed, region_idx: int | None = None) -> float:
        observed_arr = as_numeric_array(observed, name="observed")
        mu, sigma = self._fit(observed_arr)
        if region_idx is not None:
            obs_i = observed_arr[region_idx]
            if np.isnan(obs_i):
                return -np.inf
            return float(norm.logpdf(obs_i, loc=mu, scale=sigma))

        valid = observed_arr[~np.isnan(observed_arr)]
        return float(np.sum(norm.logpdf(valid, loc=mu, scale=sigma)))


@dataclass(frozen=True)
class SampledModel(Model):
    sample_frac: float | None = None
    bandwidth: float | str | None = None
    sample_indices: tuple[int, ...] | None = None

    def __init__(
        self,
        sample_frac: float | None = None,
        *,
        bandwidth: float | str | None = None,
        sample_indices=None,
        name: str = "Sampled (KDE)",
    ):
        if sample_frac is not None and not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")
        indices = None if sample_indices is None else tuple(int(i) for i in sample_indices)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", "sampled")
        object.__setattr__(self, "sample_frac", sample_frac)
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "sample_indices", indices)

    def _training_data(self, observed_arr: np.ndarray) -> np.ndarray:
        n = observed_arr.size
        if self.sample_indices is not None:
            idx = np.asarray(self.sample_indices, dtype=int)
        elif self.sample_frac is not None:
            n_sample = max(3, int(np.floor(n * self.sample_frac)))
            idx = np.arange(n_sample)
        else:
            idx = np.arange(n)
        data = observed_arr[idx]
        return data[~np.isnan(data)]

    def _density_values(self, observed_arr: np.ndarray, values: np.ndarray) -> np.ndarray:
        train = self._training_data(observed_arr)
        if train.size < 2 or np.nanstd(train, ddof=1) == 0:
            return np.full(values.size, 1.0)

        bw_method = None if self.bandwidth in (None, "nrd0", "nrd") else self.bandwidth
        kde = gaussian_kde(train, bw_method=bw_method)
        dens = kde(values)
        return np.maximum(dens, 1e-10)

    def log_likelihood(self, observed, region_idx: int | None = None) -> float:
        observed_arr = as_numeric_array(observed, name="observed")
        if region_idx is not None:
            obs_i = observed_arr[region_idx]
            if np.isnan(obs_i):
                return -np.inf
            return float(np.log(self._density_values(observed_arr, np.array([obs_i]))[0]))

        valid = observed_arr[~np.isnan(observed_arr)]
        return float(np.sum(np.log(self._density_values(observed_arr, valid))))


@dataclass(frozen=True)
class FunnelModel(Model):
    sample_size: np.ndarray
    target_rate: float | None = None
    data_type: str = "count"
    formula: str = "paper"
    control_limits: tuple[float, ...] = (2, 3)

    def __init__(
        self,
        sample_size,
        *,
        target_rate: float | None = None,
        data_type: str = "count",
        formula: str = "paper",
        control_limits=(2, 3),
        name: str | None = None,
    ):
        sample_arr = as_numeric_array(sample_size, name="sample_size")
        if sample_arr.size == 0:
            raise ValueError("sample_size cannot be empty.")
        if data_type not in {"count", "proportion"}:
            raise ValueError("data_type must be 'count' or 'proportion'.")
        if formula not in {"paper", "poisson"}:
            raise ValueError("formula must be 'paper' or 'poisson'.")
        object.__setattr__(self, "name", name or f"de Moivre Funnel ({formula})")
        object.__setattr__(self, "kind", "funnel")
        object.__setattr__(self, "sample_size", sample_arr)
        object.__setattr__(self, "target_rate", target_rate)
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "control_limits", tuple(control_limits))

    def _sample_size_for(self, n_regions: int) -> np.ndarray:
        if self.sample_size.size == 1:
            return np.repeat(self.sample_size, n_regions)
        if self.sample_size.size != n_regions:
            raise ValueError("sample_size length must match observed length.")
        return self.sample_size

    def log_likelihood(self, observed, region_idx: int | None = None) -> float:
        observed_arr = as_numeric_array(observed, name="observed")
        sample_size = self._sample_size_for(observed_arr.size)

        if self.formula == "paper":
            rates = observed_arr / sample_size
            mean_rate = np.nanmean(rates)
            stddev_rate = np.nanstd(rates, ddof=1)
            if not np.isfinite(stddev_rate) or stddev_rate == 0:
                stddev_rate = 1e-10
            z_scores = (rates - mean_rate) / stddev_rate
            pop_frac = sample_size / np.nansum(sample_size)
            dm_scores = z_scores * np.sqrt(pop_frac)
            p_values = 2 * norm.cdf(-np.abs(dm_scores))
        else:
            rate = self.target_rate
            if rate is None:
                rate = float(np.nansum(observed_arr) / np.nansum(sample_size))
            expected = sample_size * rate
            if self.data_type == "count":
                se = np.sqrt(np.maximum(expected, 0.5))
            else:
                p = np.clip(rate, 0.001, 0.999)
                se = np.sqrt(p * (1 - p) * sample_size)
            z = (observed_arr - expected) / se
            p_values = 2 * norm.cdf(-np.abs(z))

        p_values = np.maximum(p_values, 1e-300)
        if region_idx is not None:
            if np.isnan(observed_arr[region_idx]):
                return -np.inf
            return float(np.log(p_values[region_idx]))

        valid = ~np.isnan(observed_arr)
        return float(np.sum(np.log(p_values[valid])))


@dataclass(frozen=True)
class ModelSpace:
    models: tuple[Model, ...]
    prior: np.ndarray
    names: tuple[str, ...]
    posterior: np.ndarray | None = None

    @property
    def n_models(self) -> int:
        return len(self.models)

    def with_posterior(self, posterior) -> "ModelSpace":
        return replace(self, posterior=as_numeric_array(posterior, name="posterior"))

    def with_prior(self, prior) -> "ModelSpace":
        return replace(self, prior=normalize_prob(prior), posterior=None)


def model_space(*models: Model, prior=None, names=None) -> ModelSpace:
    if len(models) == 1 and isinstance(models[0], (list, tuple)):
        models = tuple(models[0])
    if not models:
        raise ValueError("At least one model must be provided.")
    if not all(isinstance(model, Model) for model in models):
        raise TypeError("All models must be Model instances.")

    if prior is None:
        prior_arr = np.full(len(models), 1 / len(models), dtype=float)
    else:
        prior_arr = as_numeric_array(prior, name="prior")
        if prior_arr.size != len(models):
            raise ValueError("prior length must match number of models.")
        if np.any(prior_arr < 0):
            raise ValueError("prior must contain non-negative values.")
        if not np.isclose(np.sum(prior_arr), 1):
            raise ValueError("prior must sum to 1.")

    model_names = tuple(names) if names is not None else tuple(model.name for model in models)
    if len(model_names) != len(models):
        raise ValueError("names length must match number of models.")

    return ModelSpace(tuple(models), prior_arr, model_names)


def default_model_space(expected, *, sample_size=None, include_gaussian: bool = False, prior=None) -> ModelSpace:
    sample = expected if sample_size is None else sample_size
    models: list[Model] = [
        UniformModel(),
        BaseRateModel(expected),
        FunnelModel(sample),
    ]
    if include_gaussian:
        models.append(GaussianModel())
    return model_space(*models, prior=prior)


def build_model_space_from_spec(spec, expected=None, sample_size=None, prior=None) -> ModelSpace:
    if isinstance(spec, ModelSpace):
        return spec
    if isinstance(spec, Model):
        return model_space(spec, prior=prior)
    if isinstance(spec, (list, tuple)) and all(isinstance(item, Model) for item in spec):
        return model_space(*spec, prior=prior)
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, (list, tuple)) and all(isinstance(item, str) for item in spec):
        models: list[Model] = []
        for item in spec:
            key = item.lower()
            if key == "uniform":
                models.append(UniformModel())
            elif key == "baserate":
                if expected is not None:
                    models.append(BaseRateModel(expected))
            elif key == "gaussian":
                models.append(GaussianModel())
            elif key in {"sampled", "kde"}:
                models.append(SampledModel())
            elif key == "funnel":
                sample = sample_size if sample_size is not None else expected
                if sample is not None:
                    models.append(FunnelModel(sample))
            else:
                raise ValueError(f"Unknown model type: {item}")
        if not models:
            raise ValueError("No valid models could be created.")
        return model_space(*models, prior=prior)
    raise TypeError(f"Cannot build model space from {type(spec)!r}.")

