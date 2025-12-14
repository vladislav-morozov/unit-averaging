r"""
Using Prior Information as Restrictions
========================================

This tutorial demonstrates how to incorporate prior information about unit
similarities using the unrestricted/restricted unit functionality of
``OptimalUnitAveraging`` —  the large-N regime.

By the end, you should be able to:

#. Understand the motivation for restricted/unrestricted unit classification.
#. Specify unrestricted/restricted unit restrictions in ``OptimalUnitAverager``.
#. Interpret the resulting weight distributions.

.. admonition:: Functionality covered

    :doc:`OptimalUnitAverager <../reference/OptimalUnitAverager>`:
    using the ``unrestricted_unit_bool`` argument for the large-N regime.

"""

import numpy as np
import pandas as pd
from docs_utils import plot_germany, prepare_frankfurt_example

from unit_averaging import IndividualUnitAverager, OptimalUnitAverager

# %%
#
# .. admonition:: Following along
#
#    If you would like to follow along on a local machine, please download the
#    contents of the ``data`` folder
#    `here
#    <https://github.com/vladislav-morozov/unit-averaging/tree/main/docs/examples/data>`__
#    . To recreate the plots, also download the ``docs_utils.py``
#    `file
#    <https://github.com/vladislav-morozov/unit-averaging/blob/main/docs/examples/docs_utils.py>`__
#    .
#
#

# %%
# Some Background
# ----------------------
#
# In our first application of ``OptimalUnitAverager`` in
# :doc:`Getting Started <plot_1_basics>`, the weight of every unit could
# be determined separately. This is an agnostic approach that allows the
# averager to choose which units are important for itself.
#
# However, when dealing with many units, this approach may have two key drawbacks:
#
# - It may overfit: May assign non-zero weights to irrelevant units.
# - It ignores known similarities between units.
#
# ``OptimalUnitAverager`` supports an approach that can reduce the dimensionality
# of the averaging problem by grouping units into two categories:
#
# - "*Unrestricted*" units: units whose weights are still chosen freely (and
#   may be chosen to zero).
# - "*Restricted*" units: the weights of all the restricted units are equal.
#   The algorithm only chooses how much weight overall to give to the average
#   of restricted units.
#
# Using restricted units is called the "large-N" regime for theoretical reasons
# (see the `original paper <https://doi.org/10.1080/07350015.2025.2584579>`__),
# in contrast to the "fixed-N" regime with no restricted units.
#
# In practice, the choice of unrestricted vs. restricted units reflects **prior**
# **information**: units that may be more important for prediction (more similar,
# have tighter economic links, etc.) should be left unrestricted. All other
# units should be restricted.
#
# At the same time, it's important to highlight the following point:
#
# .. admonition:: Choice of unrestricted units is a tuning parameter
#
#   The choice of unrestricted units is not a causal statement about the 
#   underlying reality. ``OptimalUnitAverager`` will optimally adapt to the
#   specified structure to the extent possible.

# %%
# Restricted and Unrestricted Units in Practice
# -----------------------------------------------
#
# We revisit the Frankfurt unemployment forecasting example from
# :doc:`Getting Started <plot_1_basics>`, now incorporating prior information
# about regional similarities. Again, the task is to predict the change in
# unemployment in Frankfurt in January
# 2020 from a panel of 150 German labor market districts, Frankfurt included.
#
# We load our prepared individual estimates, individual covariance matrices,
# and the focus function for forecasting the target unemployment rate:

ind_estimates, ind_covar_ests, forecast_frankfurt_jan_2020 = prepare_frankfurt_example()

# %%
# Specifying Unit Restrictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We assume Hessen regions (Frankfurt's state) are potentially relevant for
# forecasting Frankfurt's unemployment, with Frankfurt influenced by the regions
# in its state. These become **unrestricted units** while all others are restricted.
#
# It is important to stress that this choice means the algorithm is free to choose
# any weight for regions in Hessen. That includes the possibility of assigning
# a zero weights. Hence, setting a unit as unrestricted means that it *may* but
# but *not necessarily will* be included in a greater degree in the average.
#
# Specifically, the regions in Hessen are:

hessen_regions = [
    "Kassel",
    "Korbach",
    "Bad Hersfeld - Fulda",
    "Marburg",
    "Limburg - Wetzlar",
    "Gießen",
    "Hanau",
    "Wiesbaden",
    "Bad Homburg",
    "Frankfurt",
    "Offenbach",
    "Darmstadt",
]

# %%
#
# Information about whether a unit is unrestricted is boolean, with ``True``
# meaning that a unit is unrestricted, and ``False`` that it is restricted.
#

unrestricted_units = {region: (region in hessen_regions) for region in ind_covar_ests}

print(unrestricted_units)


# %%
#
# Since our ``ind_estimates`` and ``ind_covar_ests`` are dictionaries indexed by
# regions, the information on unrestricted units should also be a dictionary
# indexed by the same regions. Otherwise the averager would not be able to match
# the information on unrestricted units to the units themselves.


# %%
# Fitting the Averager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With the information on unrestricted units specified, we can now create
# an instance of ``OptimalUnitAverager`` with these unit restrictions.
#
# We pass ``unrestricted_units`` to the ``unrestricted_units_bool`` argument
# of the averager:


averager = OptimalUnitAverager(
    focus_function=forecast_frankfurt_jan_2020,
    ind_estimates=ind_estimates,
    ind_covar_ests=ind_covar_ests,
    unrestricted_units_bool=unrestricted_units,
)

# %%
# We can now fit our averager and examine the predicted value:


averager.fit(target_id="Frankfurt")
print(averager.estimate.round(3))

# %%
# The predicted value is quite close to the individual-specific estimate:

ind_averager = IndividualUnitAverager(
    focus_function=forecast_frankfurt_jan_2020,
    ind_estimates=ind_estimates,
)
ind_averager.fit(target_id="Frankfurt")
print(ind_averager.estimate.round(3))

# %%
#
# Finally, we can examine the fitted ``weights``. First, we plot them:

weight_df = pd.DataFrame({"aab": averager.keys, "weights": averager.weights})
# sphinx_gallery_thumbnail_number = 1
fig, ax = plot_germany(
    weight_df,
    "Weight in Averaging Combination: Only Units in Same Region Unrestricted",
    cmap="Purples",
    vmin=-0.005,
    vmax=0.3,
)

# %%
#
# We can take a deeper look at the weights of the restricted and the unrestricted
# units by accessing the ``weights`` attribute of the averager.
#
# We first look at the total weight received by the set of restricted units.
# This total is what is chosen by the averager. Then this total is
# equally divided between the restricted units. In our case, we have:

sum(averager.weights[~np.isin(averager.keys, hessen_regions)]).round(3)

# %%
# Even as a group, the restricted units receive effectively no weight.
# All the weight is allocated to Hessen.
#
# For the regions in Hessen, the weights can vary freely, and we examine them
# individually:

print(
    pd.Series(
        {
            reg: weight.round(3)
            for reg, weight in zip(
                averager.keys[np.isin(averager.keys, hessen_regions)],
                averager.weights[np.isin(averager.keys, hessen_regions)],
                strict=True,
            )
        }
    )
)

# %%
# The averager assigns almost all weights to Frankfurt itself, and two bordering
# regions — Offenbach and Bad Homburg. The other regions receive rather small
# weights.


# %%
#
# .. admonition:: Stein unit averaging
#
#   :doc:`SteinUnitAverager <../reference/SteinUnitAverager>` implements a
#   special kind of large-N optimal averaging where all the non-target units
#   are restricted (Stein-like shrinkage). There is no need to specify
#   unrestricted units when using
#   :doc:`SteinUnitAverager <../reference/SteinUnitAverager>`.
#
