r"""
Creating Custom Unit Averagers
====================================




By the end, you should be able to:

#. Understand the anatomy of an averager class.
#. Implement a custom averager.

.. admonition:: Functionality covered

    ``BaseUnitAverager``: implementing ``_compute_weights()``.

"""

import numpy as np
import pandas as pd
from docs_utils import plot_germany
from statsmodels.tsa.ar_model import AutoReg

from unit_averaging import (
    BaseUnitAverager,
    InlineFocusFunction,
)

# %%
# Introduction
# -------------
#
# Different weighting schemes in the weighted averages of unit averaging serve
# different purposes. For example:
#
# - The optimal averaging scheme of ``OptimalUnitAverager`` (see
#   the :doc:`Getting Started <plot_1_basics>` page) is tailored towards
#   estimating *unit-level* parameters with minimum mean squared error.
# - The equal weights scheme of ``MeanGroupUnitAverager`` yields an excellent
#   estimator of the *average* target parameter.
#
# One can also conveniently create custom averaging schemes that target different
# parameters of interest or use different fitting logic. This tutorial shows
# a worked example of how to do so.
#
# A Look at ``BaseUnitAverager``
# -------------------------------
#
# To understand how to define custom averagers, it is useful to take a brief
# look at the ``BaseUnitAverager`` class. ``BaseUnitAverager`` itself is an
# abstract base class, and all the in-built and custom unit averagers should
# derive from it.
#
# ``BaseUnitAverager`` implements several keys pieces of logic, which can be
# inherited from it:
#
# - The ``fit()`` and  ``average()`` methods (see
#   :doc:`Getting Started <plot_1_basics>`).
# - Logic for handling dict and array inputs for individual estimates.
#   Under the hood, all inputs related to individual units are
#   converted to NumPy arrays.
#
#
# The ``fit()`` method itself does the following things
#
# #. Handles the target ID (``target_id`` argument) and identifies the index
#    of the target unit in the processed individual estimates array
#    (creating the ``_target_coord`` attribute).
# #. Calls the ``_compute_weights()`` method.
# #. Computes the averaging estimates.
#
# It is the ``_compute_weights()`` method that captures all the logic specific
# to each approach. ``_compute_weights()`` is an abstract method of
# ``BaseUnitAverager`` that computes the weights and assigns them to the
# corresponding attribute (``weights``).
# This method should be implemented by any concrete averager approach.
#
#

# %%
# Defining an Exponentially Weighted Averager
# --------------------------------------------
#
# To illustrate the workflow, we now define a custom unit averaging schemes
# whose weights decrease exponentially with the distance from the parameter
# estimates for the target unit.
#
# Weight Scheme
# ^^^^^^^^^^^^^^
#
# Formally, let :math:`\hat{\theta}_i` be the estimated coefficient vector
# for unit :math:`i`. We define the weight :math:`w_i` of unit :math:`i`
# in the averaging estimator as
#
# .. math::
#
#   w_i = \dfrac{\exp(-||\hat{\theta}_i - \hat{\theta}_{target}||)}{
#           \sum_{j=1}^N \exp(-||\hat{\theta}_i - \hat{\theta}_{target}||) },
#
# where :math:`||\cdot||` is the Euclidean norm. This scheme gives more
# weights to units whose coefficient estimates are closer to that of the target
# unit.

# %%
# Defining the Averager
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We can now create our ``ExpDistUnitAverager`` by inheriting from
# ``BaseUnitAverager`` and implementing the exponential weights.
#
# Observe that the weights :math:`w_i` can be computed just from the target ID and
# the collection of individual estimates. In such cases, one does not have to
# redefine or expand the constructor obtained from ``BaseUnitAverage``. We only
# need to implement the ``_compute_weights()`` method:

class ExpDistUnitAverager(BaseUnitAverager):
    def _compute_weights(self):
        """Compute unit averaging weights.

        This method implements exponential weights.
        """
        # Extract theta_hat of target unit
        target_params = self.ind_estimates[self._target_coord]
        # Compute weights
        raw_diff = np.linalg.norm(self.ind_estimates - target_params, ord=2, axis=1)
        return raw_diff / sum(raw_diff)

# %%
# .. admonition:: Important
#
#    In the above implementation, we directly work with the internal 
#    representations of data: the ``ind_estimates`` attribute is a NumPy array
#    and the index of the target unit is stored as `_target_coord`. The requisite
#    processing of raw inputs is handled by the ``__init__()`` and ``fit()``
#    methods of ``BaseUnitAverager``.

# %%
# Our Averager in Practice
# -------------------------
#
# To illustrate the 
#
# For convenience,
