r"""
Getting Started with Optimal Unit Averaging
============================================

This is a worked example showcasing how unit averaging can integrated.

Goes from raw data, shows the necessary data construction, and showcases things.


This ``unit-averaging`` package is designed to accommodate a variety of approaches
for estimating individual models and transforming their parameters.

This example pays relatively more

.. admonition:: Functionality covered

    ``OptimalUnitAverager`` and ``InlineFocusFunction``




"""

import numpy as np
import pandas as pd
from docs_plot_utils import plot_germany
from statsmodels.tsa.ar_model import AutoReg

from unit_averaging import InlineFocusFunction, OptimalUnitAverager

# %%
# If you want to following this example locally, download the requisite files
#
#

# %%
# Problem, Data, and Key Idea
# ----------------------------------
#
# Introduction
# ^^^^^^^^^^^^^
#
# Suppose that it is December, 2019 and we want to optimally *forecast
# the unemployment rate* in *Cologne* in the next month.
# At our disposal we have a panel (longitudinal) dataset of unemployment rates
# for all the regions of Germany, Cologne included.
#
# What is the key challenge to using all this data efficiently? The regions are
# different in unseen ways (or *heterogeneous*): each of them has its own
# unemployment dynamics, driven by differences in their economies,
# populations, and laws. Throwing the data on all the regions into a single prediction
# model usually may yield a very biased forecast for Cologne (or for any other
# specific region).
#
# Unit averaging is ensemble method specifically designed for efficient estimation
# of such unit-specific parameters (e.g. unemployment in Cologne) for situations
# when you have data on multiple units. It is applicable both with panel data,
# as in this example, and in meta-analysis settings.


# %%
# Data
# ^^^^^
#
# To illustrate optimal unit averaging, we use data on the 150 German employment
# regions. The data records the unemployment rates for each region for each
# month between January 2011 and December 2019. It can be freely obtained from the
# `German Federal Employment Agency <https://statistik.arbeitsagentur.de/>`__.
#
# A quick peek at the data, after reading it in and setting up the time index:

german_data = pd.read_csv(
    "data/tutorial_data.csv", parse_dates=True, index_col="period"
)
german_data.index = pd.DatetimeIndex(german_data.index.values, freq="MS")
print(german_data.iloc[-4:, [0, 2, -1]])

# %%
# The different regions are identified by their string names, which serve as column
# names in the data:

regions = german_data.columns[:-1].to_numpy()
print(regions[:10])


# %%
# Constructing Averaging Inputs
# ----------------------------------
#
# In short, every use of unit averaging involves the following steps:
#
# #. Prepare the models:
#
#    * Define a model for each unit.
#    * Express the parameter of interest in terms of the parameters of the unit-
#      level models (defining a suitable *focus function*)
#
# #. Estimate the model for each unit separately and collect the results.
# #. Pass the unit-level estimates, the focus function, along with any other
#    necessary inputs to a suitable
#    ``Averager`` from this package and ``fit()`` it.
#
#
#
# .. seealso:: See :doc:`this page <../theory/theory>` for a more formal discussion of
#              unit averaging from a mathematical perspective.
#
#
# Region-Specific Models
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# In our case, we will specify that the unemployment in each region :math:`i`
# follows a simple autoregressive (AR) process: it depends on the unemployment
# in the same region in the previous month, along with Germany-wide unemployment
# in the previous month:
#
# .. math::
#
#    U_{i,t} = c_i+\alpha_i U_{i, t-1} + \beta_i U_{Germany, t-1} + \varepsilon_{i, t},
#
# where :math:`U_{i, t}` is the unemployment rate in region :math:`i` in month
# :math:`t`.
# While the general shape of the model is the same for all regions, the coefficients
# are region-specific. That allows different regions to have different unemployment
# dynamics.
#
# Focus Function
# ^^^^^^^^^
#
# With the model in hand, we need to express the target parameter (unemployment
# for Cologne in 01.2020) as a function of the parameters of the unit-level
# models. Mathematically, the model implies the following forecast function:
#
# .. math::
#
#   \mu(c, \alpha, \beta) = c + \alpha U_{Cologne, 12.2019} + \beta U_{Germany, 12.2019}
#
# The function :math:`\mu` is called a *focus function*: it defines how the parameters
# of the underlying models map into the actual final parameter of interest.
#
# All of the averager classes of this package expect as inputs:
#
# #. A focus function along with its gradient with respect to the parameters.
# #. A collection of estimated parameters for each unit (see below).
#
# We start with creating a focus function. In general, the package offers two
# classes for defining one: an ``InlineFocusFunction`` or implementing a concrete
# ``BaseFocusFunction``. The former option is convenient when :math:`\mu`
# and its gradient are already available in form of callables or a simple lambda
# function. These are then simply passed as arguments to the constructor of
# ``InlineFocusFunction``. The latter option is more flexible and requires
# implementing the focus function and its gradient as methods.
#
# In our case, the focus function and its gradient are relatively simple, so we 
# use an ``InlineFocusFunction``. We pass suitable lambda functions as the 
# ``focus_function`` and ``gradient`` arguments:

# Extract data on last month of target region
target_data = german_data.loc["2019-12", ["Köln", "Deutschland"]].to_numpy().squeeze()

# Construct focus function
forecast_cologne_jan_2020 = InlineFocusFunction(
    focus_function=lambda coef: coef[0]
    + coef[1] * target_data[0]
    + coef[2] * target_data[1],
    gradient=lambda coefs: np.array([1, target_data[0], target_data[1]]),
)


# %%
# Estimating Unit Models
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The second input 
# Our model will look like
# The data is difference to ensure stationarity. We will be forecasting changes
# in unemployment

german_data = german_data.diff()
german_data["Germany_lag"] = german_data["Deutschland"].shift(1)
german_data = german_data.iloc[2:,]

# %%
# The key data preparation step consists in preparing data.
#
#
# All unit averagers expect data in two of two forms: numpy array or dict.
# Dicts are appropriate when the individual units have descriptive identifiers.
#
# This is our case.

ind_estimates = {}
ind_covar_ests = {}

# %%
# Now estimate.

for region in regions:
    # Extract data and add lags
    ind_data = german_data.loc[:, [region, "Deutschland"]]
    # Run an ARx(1) model
    ar_results = (
        AutoReg(ind_data.loc[:, region], 1, exog=ind_data["Deutschland"])
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    # Add to dictionary
    ind_estimates[region] = ar_results.params.to_numpy()
    ind_covar_ests[region] = ar_results.cov_params().to_numpy()


# %%
# Using Optimal Unit Averaging
# -------------------------------
#


# %%
# Defining and Fitting the Averager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Fit

averager = OptimalUnitAverager(
    focus_function=forecast_cologne_jan_2020,
    ind_estimates=ind_estimates,
    ind_covar_ests=ind_covar_ests,
)

# %%
# Now fit by specifying target ID. If the things were passed as a dictionary,
# then you need to pass the corresponding key

averager.fit(target_id="Köln")

# %%
#
# .. tip:: The averager classes of this package follow two layer approach  .
#          This allows, which might be more ergonomic.
#          Of course, one can also supply target parameters
#          directly, and pass an identity focus function if that is more convenient
#          in a given context.
#


# %%
# Results
# ----------------------------------
#
# Contains two things: estimated averaging weights and average
#
# Weights
# ^^^^^^^^
#
# NumPy array

print(averager.weights_[:10].round(3))

# %%
# Plot

weight_dict = {}
for key, val in zip(averager.keys, averager.weights_, strict=False):
    weight_dict[key] = val

weight_df = pd.Series(weight_dict).reset_index()
weight_df.columns = ["aab", "weights"]

fig, ax = plot_germany(
    weight_df,
    "Weight in Averaging Combination",
    cmap="Purples",
    vmin=-0.005,
)


# %%
# Unit averaging can also be used to discover patterns in the data
# Stuttgart

# %%
# Averaging Estimate
# ^^^^^^^^^^^^^^^^^^^
# Can also reuse fitted weights


# %%
# ``IndividualUnitAverager``

# %%
# Running ``average()`` with no inputs reuses the focus function passed to the constructor
