"""Plotting bandstructures.

Module modified from https://github.com/chrisjsewell/aiida-qe-demo/blob/main/tutorial/local_module/bandstructure.py

"""
from __future__ import annotations

from typing import TypedDict

from aiida.orm import BandsData
from matplotlib import pyplot as plt


class BandStructureConfig(TypedDict, total=False):
    """Configuration for the bandstructure plot."""

    title: str
    """Title of the plot."""
    legend: None | str
    """The legend for first-type spins."""
    legend2: None | str
    """The legend for second-type spins."""
    y_max_lim: None | float
    """The maximum y-axis limit (if None, put the maximum of the bands)."""
    y_min_lim: None | float
    """The minimum y-axis limit (if None, put the minimum of the bands)."""
    y_origin: float
    """the new origin of the y axis -> all bands are replaced by bands-y_origin."""
    prettify_format: None | str
    """Prettify labels for typesetting in various formats."""

    bands_color: str
    """Color of band lines."""
    bands_linewidth: float
    """linewidth of bands."""
    bands_linestyle: None | str
    """linestyle of bands."""
    bands_marker: None | str
    """marker for bands."""
    bands_markersize: None | float
    """size of the marker of bands."""
    bands_markeredgecolor: None | str
    """marker edge color for bands."""
    bands_markeredgewidth: None | float
    """marker edge width for bands."""
    bands_markerfacecolor: None | str
    """marker face color for bands."""

    bands_color2: str
    """Color of band lines (for other spin, if present)."""
    bands_linewidth2: float
    """linewidth of bands (for other spin, if present)."""
    bands_linestyle2: None | str
    """linestyle of bands (for other spin, if present)."""
    bands_marker2: None | str
    """marker for bands (for other spin, if present)."""
    bands_markersize2: None | float
    """size of the marker of bands (for other spin, if present)."""
    bands_markeredgecolor2: None | str
    """marker edge color for bands (for other spin, if present)."""
    bands_markeredgewidth2: None | float
    """marker edge width for bands (for other spin, if present)."""
    bands_markerfacecolor2: None | str
    """marker face color for bands (for other spin, if present)."""

    plot_zero_axis: bool
    """If true, plot an axis at y=0."""
    zero_axis_color: str
    """Color of the axis at y=0."""
    zero_axis_linestyle: str
    """linestyle of the axis at y=0."""
    zero_axis_linewidth: float
    """linewidth of the axis at y=0."""


def plot_bandstructure(node: BandsData, config: BandStructureConfig = None):
    """Plot a band-structure."""
    config = config or {}
    all_data = node._matplotlib_get_dict(**config)

    # x = all_data['x']
    # bands = all_data['bands']
    paths = all_data["paths"]
    tick_pos = all_data["tick_pos"]
    tick_labels = all_data["tick_labels"]

    # Option for bands (all, or those of type 1 if there are two spins)
    opts1 = {
        "color": all_data.get("bands_color", "k"),
        "linewidth": all_data.get("bands_linewidth", 0.5),
        "linestyle": all_data.get("bands_linestyle", None),
        "marker": all_data.get("bands_marker", None),
        "markersize": all_data.get("bands_markersize", None),
        "markeredgecolor": all_data.get("bands_markeredgecolor", None),
        "markeredgewidth": all_data.get("bands_markeredgewidth", None),
        "markerfacecolor": all_data.get("bands_markerfacecolor", None),
    }

    # Options for second-type of bands if present (e.g. spin up vs. spin down)
    opts2 = {"color": all_data.get("bands_color2", "r")}
    # Use the values of type 1 by default
    for key in (
        "linewidth",
        "linestyle",
        "marker",
        "markersize",
        "markeredgecolor",
        "markeredgewidth",
        "markerfacecolor",
    ):
        opts2[key] = all_data.get(f"bands_{key}2", opts1[key])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    first_band_1 = True
    first_band_2 = True

    for path in paths:
        if path["length"] <= 1:
            # Avoid printing empty lines
            continue
        x = path["x"]
        # for band in bands:
        for band, band_type in zip(path["values"], all_data["band_type_idx"]):

            # For now we support only two colors
            if band_type % 2 == 0:
                further_plot_options = opts1
            else:
                further_plot_options = opts2

            # Put the legend text only once
            label = None
            if first_band_1 and band_type % 2 == 0:
                first_band_1 = False
                label = all_data.get("legend_text", None)
            elif first_band_2 and band_type % 2 == 1:
                first_band_2 = False
                label = all_data.get("legend_text2", None)

            ax.plot(x, band, label=label, **further_plot_options)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim([all_data["x_min_lim"], all_data["x_max_lim"]])
    ax.set_ylim([all_data["y_min_lim"], all_data["y_max_lim"]])
    ax.xaxis.grid(True, which="major", color="#888888", linestyle="-", linewidth=0.5)

    if all_data.get("plot_zero_axis", False):
        ax.axhline(
            0.0,
            color=all_data.get("zero_axis_color", "#888888"),
            linestyle=all_data.get("zero_axis_linestyle", "--"),
            linewidth=all_data.get("zero_axis_linewidth", 0.5),
        )
    if all_data["title"]:
        ax.set_title(all_data["title"])
    if all_data["legend_text"]:
        ax.legend(loc="best")
    ax.set_ylabel(all_data["yaxis_label"])

    return fig, ax
