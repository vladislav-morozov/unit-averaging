r"""
Creating Custom Unit Averagers
====================================




By the end, you should be able to:

#. Understand the anatomy of an averager class.
#. Implement a custom averager.

.. admonition:: Functionality covered

    ``BaseUnitAverager``: implementing ``_compute_weights()``.

"""

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
# - Logic for handling dict and array inputs for individual estimates, along with
#   appropriate logic in the constructor.
# 
#
# The ``fit()`` method itself does the following things
# 
# #. Appropriately handles the target ID (``target_id`` argument).
# #. Calls the ``_compute_weights()`` method.
# #. Computes the averaging estimates.
#
# It is the ``_compute_weights()`` method that captures all the logic specific
# to each approach. ``_compute_weights()`` is an abstract method of
# ``BaseUnitAverager`` and should be implemented by any concrete averager approach.

# %%
# Defining an Exponentially Weighted Averager
# --------------------------------------------
#
# To illustrate the process,
# This will be a simple
#
# Accordingly, we do not need to redefine the ``_init_`` to add more


# %%
# We will create based on normalized exponential distance to the focus unit
# :class:`BaseFocusFunction`

print(2)
