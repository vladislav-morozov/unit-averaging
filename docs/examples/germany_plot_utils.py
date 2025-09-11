r"""
TEST
=====
"""

def plot_germany(data_df, aab_shp, **kwargs):
    """Plot a chloropleth map using data_df with shapes in aab_shp"""

    # Clean data_series index
    data_df["aab"] = data_df["aab"].apply(
        lambda x: x.replace(" - ", "-").replace("-", " - ")
    )

    series_name = data_df.columns[-1]

    merged_geo_data_df = aab_shp.merge(
        data_df,
        left_on="region",
        right_on="aab",
        how="outer",
    )

    # plot
    ax = merged_geo_data_df.to_crs("EPSG:25832").plot(series_name, **kwargs)
    ax.axis("off")
    # Return plot
    return ax