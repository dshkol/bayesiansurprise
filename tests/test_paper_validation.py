import numpy as np
from scipy.stats import norm

import bayesiansurprise as bs


def test_funnel_paper_formula_uses_two_tailed_p_value():
    population = np.array([10000, 50000, 100000, 25000, 75000])
    counts = np.array([100, 400, 1000, 250, 600])
    model = bs.FunnelModel(population, formula="paper")

    rates = counts / population
    mean_rate = rates.mean()
    sd_rate = rates.std(ddof=1)
    z_scores = (rates - mean_rate) / sd_rate
    pop_frac = population / population.sum()
    dm_scores = z_scores * np.sqrt(pop_frac)
    expected_pdm = 2 * norm.cdf(-np.abs(dm_scores))

    actual_pdm = np.array([
        np.exp(model.log_likelihood(counts, region_idx=i))
        for i in range(counts.size)
    ])
    np.testing.assert_allclose(actual_pdm, expected_pdm, rtol=1e-10, atol=1e-10)


def test_funnel_model_uses_unweighted_mean_of_rates():
    population = np.array([1000, 100000])
    counts = np.array([100, 5000])
    model = bs.FunnelModel(population, formula="paper")

    assert np.isfinite(model.log_likelihood(counts, region_idx=0))
    assert np.isfinite(model.log_likelihood(counts, region_idx=1))

    rates = counts / population
    mean_rate = rates.mean()
    sd_rate = rates.std(ddof=1)
    z_scores = (rates - mean_rate) / sd_rate

    assert z_scores[0] > 0
    assert z_scores[1] < 0


def test_normalized_surprise_is_default_and_legacy_mode_is_explicit():
    population = np.array([100000, 100000, 100000])
    counts = np.array([1000, 1500, 3000])
    space = bs.model_space(bs.UniformModel(), bs.FunnelModel(population), prior=[0.5, 0.5])

    standard = bs.compute_surprise(space, counts, expected=population)
    legacy = bs.compute_surprise(space, counts, expected=population, normalize_posterior=False)

    assert np.all(standard.surprise >= 0)
    assert np.all(legacy.surprise >= 0)
    assert not np.allclose(standard.surprise, legacy.surprise)


def test_legacy_mode_still_flags_clear_outlier_highest():
    population = np.repeat(50000, 20)
    counts = np.round(population * 0.05).astype(int)
    counts[9] = round(population[9] * 0.05 * 3)

    space = bs.model_space(bs.UniformModel(), bs.FunnelModel(population), prior=[0.5, 0.5])
    result = bs.compute_surprise(
        space,
        counts,
        expected=population,
        normalize_posterior=False,
    )

    assert int(np.argmax(result.surprise)) == 9
    assert result.surprise[9] > np.mean(np.delete(result.surprise, 9)) * 2


def test_paper_dm_score_scales_by_population_fraction():
    population = np.array([10000, 40000, 50000])
    counts = np.array([1000, 3200, 5000])

    rates = counts / population
    z_scores = (rates - rates.mean()) / rates.std(ddof=1)
    expected_dm = z_scores * np.sqrt(population / population.sum())

    np.testing.assert_allclose(expected_dm, z_scores * np.sqrt(population / population.sum()))
    assert abs(expected_dm[2]) > abs(expected_dm[0])

