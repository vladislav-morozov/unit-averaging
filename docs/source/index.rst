.. unit_averaging documentation master file, created by
   sphinx-quickstart on Fri Aug 29 16:49:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Unit Averaging
============================


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents
   
   tutorials/index.rst 
   theory/theory.md
   reference/index.rst
   GitHub Repo <https://github.com/vladislav-morozov/unit-averaging>
 

**Optimal Estimation for Heterogeneous Data**
 
.. image:: _static/logo_transp.png
   :width: 200
   :align: right
   :alt: unit_averaging logo

``unit_averaging`` is a Python package for estimating unit-specific parameters in heterogeneous data settings (panel data and meta-analysis). 
It implements unit averaging: an ensemble method that efficiently combines information across multiple units (e.g., countries, firms, or studies) while accounting for their individual differences.

Key Features
------------

- Optimal weighting: Automatically computes weights that balance bias and variance
- Flexible workflow: Works with panel data, meta-analysis, and other heterogeneous datasets
- Versatility: Can be used on top of various standard estimation packages
- Customizable: Implement your own weighting schemes by subclassing base classes
- Theoretically grounded: Based on statistical theory with proven performance

Installation
---------------------------------

Install the package with ``pip``:

.. code-block:: bash

   pip install unit_averaging

Getting Started
-----------------

New to unit averaging? Take a look at the :doc:`tutorials <tutorials/index>` or check out the :doc:`theory section <theory/theory>` for a background on unit averaging! 


Or take a look at the core workflow in a small synthetic example:

.. code-block:: python

   import numpy as np
   from unit_averaging import OptimalUnitAverager, InlineFocusFunction

   # Example: Forecasting with a linear focus function
   # Replace these with your actual data!
   x_value = 1.0  # Covariate (e.g., lagged value)
   estimates = {   # Dict of unit-specific coefficient estimates
       "unit1": np.array([0.5, 0.3]),
       "unit2": np.array([0.7, 0.1])
   }
   covariances = { # Dict of unit-specific covariance matrices
       "unit1": np.array([[0.1, 0.0], [0.0, 0.1]]),
       "unit2": np.array([[0.1, 0.0], [0.0, 0.1]])
   }

   # Define focus function: e.g., mu(θ) = θ₀ + θ₁ * x
   focus = InlineFocusFunction(
       focus_function=lambda coef: coef[0] + coef[1] * x_value,
       gradient=lambda coef: np.array([1, x_value])
   )

   # Create and fit averager
   averager = OptimalUnitAverager(
       focus_function=focus,
       ind_estimates=estimates,
       ind_covar_ests=covariances
   )
   averager.fit(target_id="unit1")

 
Documentation
-------------

- :doc:`Tutorials <tutorials/index>`: Step-by-step guides
- :doc:`API Reference <reference/index>`: Detailed class and function documentation
- :doc:`Theory <theory/theory>`: Mathematical foundations
- `Original Paper <https://arxiv.org/abs/2210.14205>`__: Complete theoretical treatment

Citation
---------

If you use ``unit_averaging`` in your research, please cite: 

   Brownlees, C. T., & Morozov, V. (2024). *Unit Averaging for Heterogeneous Panels*

.. code-block:: bibtex

   @misc{Brownlees2024UnitAveragingHeterogeneous,
      title = {Unit Averaging for Heterogeneous Panels},
      author = {Brownlees, Christian and Morozov, Vladislav},
      year = {2024},
      month = may,
      number = {arXiv:2210.14205},
      eprint = {2210.14205},
      primaryclass = {econ},
      publisher = {arXiv},
      doi = {10.48550/arXiv.2210.14205},
      archiveprefix = {arXiv},
   }

 
Support
-------

For questions, issues, or contributions:

- Report bugs on our `GitHub issues <https://github.com/vladislav-morozov/unit-averaging/issues>`_
- Contribute via `pull requests <https://github.com/vladislav-morozov/unit-averaging/pulls>`_ 