import math

import numpy as np
import pandas as pd

import bayesiansurprise as bs


def test_kl_divergence_equal_distributions_is_zero():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert bs.kl_divergence(p, p) == 0


def test_kl_divergence_known_edge_cases():
    assert bs.kl_divergence([1.0, 0.0], [0.5, 0.5]) == 1
    assert math.isinf(bs.kl_divergence([0.5, 0.5], [1.0, 0.0]))


def test_log_sum_exp_is_stable():
    x = np.array([1000, 1001, 1002])
    expected = 1002 + np.log(1 + np.exp(-1) + np.exp(-2))
    assert bs.log_sum_exp(x) == pytest_approx(expected)


def test_bayesian_update_normalizes_posterior():
    space = bs.model_space(bs.UniformModel(), bs.GaussianModel())
    updated = bs.bayesian_update(space, [10, 20, 30, 40, 50])

    assert updated.posterior is not None
    assert updated.posterior.sum() == pytest_approx(1)
    assert not np.allclose(updated.posterior, updated.prior)


def test_compute_surprise_returns_valid_result():
    space = bs.model_space(bs.UniformModel(), bs.GaussianModel())
    result = bs.compute_surprise(space, [10, 20, 30, 40, 50])

    assert len(result.surprise) == 5
    assert np.all(result.surprise >= 0)
    assert result.signed_surprise is not None


def test_compute_surprise_signed_values_reflect_direction():
    expected = np.array([10, 20, 30, 40, 50])
    observed = np.array([15, 25, 20, 35, 60])
    space = bs.model_space(bs.UniformModel(), bs.BaseRateModel(expected))

    result = bs.compute_surprise(space, observed, expected=expected)

    assert result.signed_surprise[0] >= 0
    assert result.signed_surprise[2] <= 0


def test_auto_surprise_matches_explicit_model_space():
    observed = np.array([50, 100, 150, 200])
    expected = np.array([10000, 50000, 100000, 25000])
    explicit = bs.compute_surprise(
        bs.model_space(
            bs.UniformModel(),
            bs.BaseRateModel(expected),
            bs.FunnelModel(expected),
        ),
        observed,
        expected=expected,
    )

    auto = bs.auto_surprise(observed, expected)

    np.testing.assert_allclose(auto.surprise, explicit.surprise)
    np.testing.assert_allclose(auto.signed_surprise, explicit.signed_surprise)
    assert auto.model_space.posterior is None


def test_surprise_augments_pandas_dataframe():
    data = pd.DataFrame(
        {
            "events": [50, 100, 150, 200],
            "population": [10000, 50000, 100000, 25000],
        }
    )

    out = bs.surprise(data, observed="events", expected="population")

    assert "surprise" in out.columns
    assert "signed_surprise" in out.columns
    assert "surprise_result" in out.attrs


def test_result_accessors_work_for_augmented_dataframe():
    data = pd.DataFrame(
        {
            "events": [50, 100, 150, 200],
            "population": [10000, 50000, 100000, 25000],
        }
    )

    out = bs.surprise(data, observed="events", expected="population")

    np.testing.assert_allclose(bs.get_surprise(out), out["surprise"])
    np.testing.assert_allclose(bs.get_signed_surprise(out), out["signed_surprise"])
    assert isinstance(bs.get_surprise_result(out), bs.SurpriseResult)


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
