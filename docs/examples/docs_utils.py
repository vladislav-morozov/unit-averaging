"""
This module provides plotting utilities for the documentation.

It includes functions to visualize data on German Agenturbezirke (AAB) regions
using choropleth maps. The shapefile for AABs is loaded at import time.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Path to the shapefile containing German Agenturbezirke (AAB) boundaries
AAB_SHAPE_FILE_PATH = (
    Path() / "data" / "aab_shapefiles" / "Deutschland_Agenturbezirke.shp"
)

# Load shapefile and keep only relevant columns
german_aab_shapes_df = gpd.read_file(AAB_SHAPE_FILE_PATH)
german_aab_shapes_df = german_aab_shapes_df.loc[:, ["region", "geometry"]]
# Standardize hyphenation to match input data
german_aab_shapes_df["region"] = german_aab_shapes_df["region"].apply(
    lambda x: x.replace("-", " - ")
)


def plot_germany(data_df, title, **kwargs):
    """
    Plot a choropleth map of Germany's Agenturbezirke (AAB) regions.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame with columns 'aab' (region names) and a value column to plot.
    title : str
        Title for the plot.
    **kwargs
        Additional keyword arguments passed to GeoDataFrame.plot().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    # Standardize region names in the input data for merging
    data_df["aab"] = data_df["aab"].apply(
        lambda x: x.replace(" - ", "-").replace("-", " - ")
    )

    # Assume the last column is the value to plot
    series_name = data_df.columns[-1]

    # Merge input data with the shapefile GeoDataFrame
    merged_geo_data_df = german_aab_shapes_df.merge(
        data_df,
        left_on="region",
        right_on="aab",
        how="outer",
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 7))
    merged_geo_data_df.to_crs("EPSG:25832").plot(series_name, ax=ax, **kwargs)
    ax.axis("off")
    ax.set_title(
        title,
        color="white",
        weight="bold",
    )

    # Style the figure background and border
    fig.set_facecolor("#111")
    fig.patch.set_edgecolor("#310c6d")
    fig.patch.set_linewidth(5)

    # Add a horizontal colorbar
    norm = colors.Normalize(vmin=-0.004, vmax=0.1)
    cmap = cm.Purples
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label("a")
    cbar.ax.set_xlim(0, 0.1)
    cbar.ax.tick_params(colors="white")

    return fig, ax
