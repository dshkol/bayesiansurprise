import numpy as np
import pytest

import bayesiansurprise as bs


def test_normalize_prob_sums_to_one():
    x = np.array([1, 2, 3, 4])
    np.testing.assert_allclose(bs.normalize_prob(x), x / x.sum())


def test_normalize_prob_handles_zero_vector():
    np.testing.assert_allclose(bs.normalize_prob([0, 0, 0, 0]), [0.25, 0.25, 0.25, 0.25])


def test_normalize_prob_warns_on_negative_values():
    with pytest.warns(RuntimeWarning, match="Negative"):
        result = bs.normalize_prob([-1, 2, 3])
    assert result[0] == 0


def test_normalize_rate_computes_rates():
    result = bs.normalize_rate([50, 100, 200], [10000, 50000, 100000], per=100000)
    np.testing.assert_allclose(result, [500, 200, 200])


def test_normalize_rate_handles_zero_base():
    result = bs.normalize_rate([50, 100], [10000, 0])
    assert np.isnan(result[1])

    result_no_na = bs.normalize_rate([50, 100], [10000, 0], na_for_zero=False)
    assert result_no_na[1] == 0


def test_funnel_pvalue_is_symmetric_and_bounded():
    p = bs.funnel_pvalue([-2, -1, 0, 1, 2])
    assert np.all((p >= 0) & (p <= 1))
    assert p[2] == pytest.approx(1)
    assert p[0] == pytest.approx(p[4])


def test_compute_funnel_data_returns_rate_scale_bands_for_counts():
    observed = np.array([50, 100, 150])
    sample_size = np.array([10000, 50000, 100000])

    out = bs.compute_funnel_data(observed, sample_size)

    target_rate = observed.sum() / sample_size.sum()
    expected = sample_size * target_rate
    rate_se = np.sqrt(target_rate * (1 - target_rate) / sample_size)

    np.testing.assert_allclose(out["rate"], observed / sample_size)
    np.testing.assert_allclose(out["expected"], expected)
    np.testing.assert_allclose(out["expected_rate"], target_rate)
    np.testing.assert_allclose(out["rate_se"], rate_se)
    np.testing.assert_allclose(out["lower_2sd_rate"], target_rate - 2 * rate_se)
    np.testing.assert_allclose(out["upper_3sd_rate"], target_rate + 3 * rate_se)


def test_compute_funnel_data_accepts_proportion_observations():
    observed_rate = np.array([0.05, 0.02, 0.015])
    sample_size = np.array([1000, 2000, 4000])

    out = bs.compute_funnel_data(observed_rate, sample_size, kind="proportion")

    target_rate = np.sum(observed_rate * sample_size) / sample_size.sum()
    rate_se = np.sqrt(target_rate * (1 - target_rate) / sample_size)

    np.testing.assert_allclose(out["observed"], observed_rate)
    np.testing.assert_allclose(out["expected"], target_rate)
    np.testing.assert_allclose(out["se"], rate_se)
    np.testing.assert_allclose(out["z_score"], (observed_rate - target_rate) / rate_se)
