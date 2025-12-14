r"""
Using Prior Information as Restrictions
========================================

This tutorial shows how to use prior information on potentially similar units
for optimal unit averaging with ``OptimalUnitAveraging`` using its large-N
regime.

By the end, you should be able to:

#. Understand the difference between restricted and unrestricted units in optimal
   averaging.
#. Create and fit an ``OptimalUnitAverager`` with restricted units.


.. admonition:: Functionality covered

    :doc:`OptimalUnitAverager <../reference/OptimalUnitAverager>`:
    using the ``unrestricted_unit_bool`` argument.

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
# Introduction
# -------------
#
# In the basic ``OptimalUnitAverager`` (see the
# :doc:`Getting Started <plot_1_basics>` page) every
#
# 
# 
# The role of prior information
#
# The other units 
# shrunk
#
#

# %%
# Restricted and Unrestricted Units in Practice
# -----------------------------------------------
#
# To illustrate , we come back to the example
# from :doc:`Getting Started <plot_1_basics>`. In short, the practical problem
# of interest is predicting the change in unemployment in Frankfurt in January
# 2020 using a panel of 150 German labor market districts, Frankfurt included.
#
# We generate the estimates

ind_estimates, ind_covar_ests, forecast_frankfurt_jan_2020 = prepare_frankfurt_example()

# %%
# Specifying Unrestricted Units
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Relevant units
#

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
# Since our ``ind_estimates`` and ``ind_covar_ests`` are dictionaries indexed by
# regions. The information on unrestricted units should also be a region

unrestricted_units = {region: (region in hessen_regions) for region in ind_covar_ests}

# %%
#
# Finally, we can create


averager = OptimalUnitAverager(
    focus_function=forecast_frankfurt_jan_2020,
    ind_estimates=ind_estimates,
    ind_covar_ests=ind_covar_ests,
    unrestricted_units_bool=unrestricted_units,
)

# %%
# Fitting the Averager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now fit our averager and examine the predicted value:

averager.fit(target_id="Frankfurt")
print(averager.estimate.round(3))

# %%
# The predicted value is quite close.

ind_averager = IndividualUnitAverager(
    focus_function=forecast_frankfurt_jan_2020,
    ind_estimates=ind_estimates,
)
ind_averager.fit(target_id="Frankfurt")
print(ind_averager.estimate.round(3))

# %%
#
# Finally, as before, we can examine the fitted ``weights``:

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
# Weights of the unrestricted units:

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
#
#

sum(averager.weights[~np.isin(averager.keys, hessen_regions)]).round(3)

# %%
# The other units receive basically no weight


# %%
#
# .. admonition:: Stein unit averaging
#
#   :doc:`SteinUnitAverager <../reference/SteinUnitAverager>`:
#   shrinkage. Is a variant where every unit.
