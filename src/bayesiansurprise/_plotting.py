from __future__ import annotations

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plot helpers require matplotlib. Install it to use plotting APIs.") from exc
    return plt


def _is_geodataframe(data) -> bool:
    return hasattr(data, "geometry") and hasattr(data, "crs") and callable(getattr(data, "plot", None))


def plot_surprise(
    data,
    *,
    column: str = "surprise",
    ax=None,
    cmap: str = "viridis",
    legend: bool = True,
    symmetric: bool = False,
    **kwargs,
):
    """Plot surprise values for pandas or GeoPandas data.

    GeoDataFrames are plotted through their GeoPandas ``plot`` method. Plain
    DataFrames are plotted as a simple bar chart.
    """

    plt = _require_matplotlib()
    if column not in data:
        raise ValueError(f"Column {column!r} not found in data.")
    if ax is None:
        _, ax = plt.subplots()

    values = np.asarray(data[column], dtype=float)
    plot_kwargs = dict(kwargs)
    if symmetric:
        finite_abs = np.abs(values[np.isfinite(values)])
        max_abs = float(np.max(finite_abs)) if finite_abs.size else 1.0
        if max_abs == 0:
            max_abs = 1.0
        plot_kwargs.setdefault("vmin", -max_abs)
        plot_kwargs.setdefault("vmax", max_abs)

    if _is_geodataframe(data):
        return data.plot(column=column, ax=ax, cmap=cmap, legend=legend, **plot_kwargs)

    if symmetric:
        ax.set_ylim(plot_kwargs.pop("vmin"), plot_kwargs.pop("vmax"))
    bar_kwargs = {}
    for key in ("color", "alpha", "width", "edgecolor", "linewidth"):
        if key in plot_kwargs:
            bar_kwargs[key] = plot_kwargs.pop(key)
    ax.bar(np.arange(len(values)), values, **bar_kwargs)
    ax.set_ylabel(column)
    ax.set_xlabel("observation")
    return ax


def plot_signed_surprise(data, *, column: str = "signed_surprise", ax=None, cmap: str = "RdBu_r", **kwargs):
    """Plot signed surprise with symmetric diverging limits."""

    return plot_surprise(data, column=column, ax=ax, cmap=cmap, symmetric=True, **kwargs)


def plot_funnel(
    funnel_data,
    *,
    ax=None,
    observed_col: str = "observed",
    rate_col: str = "rate",
    sample_size_col: str = "sample_size",
    expected_col: str = "expected",
    expected_rate_col: str = "expected_rate",
    limit_prefixes=("2sd", "3sd"),
    rate: bool | None = None,
    **kwargs,
):
    """Plot observations against funnel diagnostic control limits."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    if rate is None:
        rate = rate_col in funnel_data and expected_rate_col in funnel_data

    y_col = rate_col if rate else observed_col
    center_col = expected_rate_col if rate else expected_col
    limit_suffix = "_rate" if rate else ""

    required = {y_col, sample_size_col, center_col}
    missing = required.difference(funnel_data.columns)
    if missing:
        raise ValueError(f"Missing required funnel columns: {sorted(missing)}")

    sorted_data = funnel_data.sort_values(sample_size_col)
    x = sorted_data[sample_size_col]
    ax.scatter(x, sorted_data[y_col], alpha=0.75, **kwargs)
    ax.plot(x, sorted_data[center_col], color="black", linewidth=1.2, label="expected")

    for prefix in limit_prefixes:
        lower = f"lower_{prefix}{limit_suffix}"
        upper = f"upper_{prefix}{limit_suffix}"
        if lower in sorted_data and upper in sorted_data:
            ax.plot(x, sorted_data[lower], color="gray", linestyle="--", linewidth=0.9, label=lower)
            ax.plot(x, sorted_data[upper], color="gray", linestyle="--", linewidth=0.9, label=upper)

    ax.set_xlabel(sample_size_col)
    ax.set_ylabel(y_col)
    return ax
