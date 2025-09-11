r"""
Getting Started with Unit Averaging
===================================

This is a worked example showcasing how unit averaging can integrated

Goes from raw data, shows the necessary data construction, and showcases
"""

# %%
# Introduction
# ----------------------------------

import pandas as pd
from germany_plot_utils import plot_germany

from unit_averaging import InlineFocusFunction, OptimalUnitAverager

# %%
# We will crea
# We will . Heterogeneous settings. Different mechanics
# Contains data on the 150 employment agent


# %%
# Problem and Data
# ----------------------------------
# A


german_data = pd.read_csv(
    "data/tutorial_data.csv", parse_dates=True, index_col="period"
)
german_data.iloc[-4:, [0, 2, -1]]

# %%
# Constructing Individual Estimates
# ----------------------------------
#
# First step is to prepare the data. Optimal unit averaging


# %%
# The data is difference to ensure stationarity. We will be forecasting changes
# in unemployment

# %%
# All unit averagers expect data in two of two forms: numpy array or dict.

# %%
# Our target

# %%
# Using Optimal Unit Averaging
# ----------------------------------
#

# %%
# Agnostic
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Define focus function. Observe that that in this case using the actual values
# Hannover in 2019, but being able to use different coefficients


# %%
# Our target


1

# %%
# Can also reuse fitted weights

# %%
# Running ``average()`` with no inputs reuses the focus function passed to the constructor


# %%
# With Prior Restrictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# %%
# Visualizing the weights
# ----------------------------------
#
