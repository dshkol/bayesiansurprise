import numpy as np
import pandas as pd
import pytest

import bayesiansurprise as bs


def test_surprise_preserves_geodataframe_metadata():
    gpd = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")

    data = gpd.GeoDataFrame(
        {
            "rate": [0.05, 0.02, 0.08, 0.03],
            "geometry": [
                shapely_geometry.Point(0, 0),
                shapely_geometry.Point(1, 0),
                shapely_geometry.Point(1, 1),
                shapely_geometry.Point(0, 1),
            ],
        },
        crs="EPSG:4326",
    )

    out = bs.surprise(data, observed="rate", models=("uniform", "gaussian", "sampled"))

    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == data.crs
    assert out.geometry.name == data.geometry.name
    assert "surprise" in out
    assert "signed_surprise" in out
    assert isinstance(bs.get_surprise_result(out), bs.SurpriseResult)


def test_plot_surprise_returns_axes_for_dataframe():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    data = pd.DataFrame({"surprise": [0.1, 0.5, 0.2]})
    ax = bs.plot_surprise(data, color="steelblue", alpha=0.8)

    assert ax.get_ylabel() == "surprise"
    assert len(ax.patches) == 3


def test_plot_signed_surprise_uses_symmetric_limits_for_dataframe():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    data = pd.DataFrame({"signed_surprise": [-0.2, 0.5, 0.1]})
    ax = bs.plot_signed_surprise(data, color="gray")

    lower, upper = ax.get_ylim()
    assert lower == pytest.approx(-0.5)
    assert upper == pytest.approx(0.5)


def test_plot_funnel_defaults_to_rate_scale():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    funnel = bs.compute_funnel_data([50, 100, 150], [10000, 50000, 100000])
    ax = bs.plot_funnel(funnel)

    assert ax.get_xlabel() == "sample_size"
    assert ax.get_ylabel() == "rate"
    assert len(ax.collections) == 1
    assert len(ax.lines) >= 5


def test_plot_signed_surprise_accepts_geodataframe():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    gpd = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")

    data = gpd.GeoDataFrame(
        {
            "signed_surprise": np.array([-0.2, 0.1, 0.4]),
            "geometry": [
                shapely_geometry.Point(0, 0),
                shapely_geometry.Point(1, 0),
                shapely_geometry.Point(0, 1),
            ],
        },
        crs="EPSG:4326",
    )

    ax = bs.plot_signed_surprise(data, legend=False)

    assert ax is not None
