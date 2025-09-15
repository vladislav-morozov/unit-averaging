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
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from unit_averaging import InlineFocusFunction

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

    Args:
        data_df (pandas.DataFrame): DataFrame with columns 'aab' (region names)
            and a value column to plot.
        title (str):  title for the plot.
        **kwargs: additional keyword arguments passed to GeoDataFrame.plot().

    Returns:
        fig (matplotlib.figure.Figure): the matplotlib Figure object.
        ax (matplotlib.axes.Axes): the axes with German AABs.
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
    norm = colors.Normalize(vmin=-0.004, vmax=data_df[series_name].max())
    cmap = cm.Purples
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label("a")
    cbar.ax.set_xlim(0, data_df[series_name].max())
    cbar.ax.tick_params(colors="white")

    return fig, ax


def prepare_frankfurt_example():
    """
    Compute the data for the Frankfurt prediction example

    Returns:
       ind_estimates (dict): dict of region-level coefficient estimates.
       ind_covar_ests (dict): dict of region-level estimated estimator
            variance-covariance matrices.
       forecast_frankfurt_jan_2020 (InlineFocusFunction): focus function
            describing the unemployment change for Frankfurt in 01.2020.
    """
    # Load and prepare data
    german_data = pd.read_csv(
        "data/tutorial_data.csv", parse_dates=True, index_col="period"
    )
    german_data.index = pd.DatetimeIndex(german_data.index.values, freq="MS")
    german_data = german_data.diff()
    german_data["Germany_lag"] = german_data["Deutschland"].shift(1)
    german_data = german_data.iloc[2:,]
    regions = german_data.columns[:-1].to_numpy()

    # Run individual estimation
    ind_estimates = {}
    ind_covar_ests = {}
    for region in regions:
        # Extract data and add lags
        ind_data = german_data.loc[:, [region, "Germany_lag"]]
        # Run an ARx(1) model
        ar_results = (
            AutoReg(ind_data.loc[:, region], 1, exog=ind_data["Germany_lag"])
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        # Add to dictionaries
        ind_estimates[region] = ar_results.params.to_numpy()
        ind_covar_ests[region] = ar_results.cov_params().to_numpy()

    # Define target data
    target_data = (
        german_data.loc["2019-12", ["Frankfurt", "Germany_lag"]].to_numpy().squeeze()
    )

    # Construct focus function
    forecast_frankfurt_jan_2020 = InlineFocusFunction(
        focus_function=lambda coef: coef[0]
        + coef[1] * target_data[0]
        + coef[2] * target_data[1],
        gradient=lambda coef: np.array([1, target_data[0], target_data[1]]),
    )

    return ind_estimates, ind_covar_ests, forecast_frankfurt_jan_2020
