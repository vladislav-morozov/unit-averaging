"""
This module provides plotting utilities.

"""

from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Shapefile path
AAB_SHAPE_FILE_PATH = (
    Path() / "data" / "aab_shapefiles" / "Deutschland_Agenturbezirke.shp"
)


# Import shapefile for AABs
german_aab_shapes_df = gpd.read_file(AAB_SHAPE_FILE_PATH)
german_aab_shapes_df = german_aab_shapes_df.loc[:, ["region", "geometry"]]
# Use same hyphenation as in the input data
german_aab_shapes_df["region"] = german_aab_shapes_df["region"].apply(
    lambda x: x.replace("-", " - ")
)


def plot_germany(data_df, **kwargs):
    """Plot a choropleth map using data_df with shapes in german_aab_shapes_df"""

    # Clean data_series index
    data_df["aab"] = data_df["aab"].apply(
        lambda x: x.replace(" - ", "-").replace("-", " - ")
    )

    series_name = data_df.columns[-1]

    merged_geo_data_df = german_aab_shapes_df.merge(
        data_df,
        left_on="region",
        right_on="aab",
        how="outer",
    )

    # plot

    fig, ax = plt.subplots(figsize=(8, 7))
    merged_geo_data_df.to_crs("EPSG:25832").plot(series_name, ax=ax, **kwargs)
    ax.axis("off")

    fig.set_facecolor("#111")

    # Add colorbar
    norm = colors.Normalize(vmin=-0.004, vmax=0.1)
    cmap = cm.Purples
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label("a")
    cbar.ax.set_xlim(0, 0.1)
    cbar.ax.tick_params(colors="white")
    # Return plot
    return fig, ax
