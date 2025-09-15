Averagers
=================

.. currentmodule:: unit_averaging.averager

   

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   BaseUnitAverager.rst
   IndividualUnitAverager.rst
   MeanGroupUnitAverager.rst
   OptimalUnitAverager.rst
 

This module provides implementations of unit averaging strategies.
 
 

The package implements several specific averaging schemes (optimal, mean group, individual-specific). It also provides the possibility of defining custom averaging scheme by inheriting from ``BaseUnitAverager`` and implementing appropriate weight schemes.


Available averagers: 


.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - :class:`BaseUnitAverager`
     - Abstract base class. Subclass and implement ``_compute_weights()`` for custom strategies.
   * - :class:`IndividualUnitAverager`
     - Assigns all weight to the target unit (``weights = [0, ..., 1, ..., 0]``).
   * - :class:`MeanGroupUnitAverager`
     - Equal weights for all units (``weights = [1/N, ..., 1/N]``).
   * - :class:`OptimalUnitAverager`
     - MSE-optimal weights (agnostic and with prior restrictions).

 
 
All averagers share a common interface:
 
1. Fitting: Call ``fit(target_id)`` to compute weights and compute the unit averaging estimator for the focus function supplied to the class constructor. 
2. Averaging: Use ``average()`` to recompute the averaging estimator 


 

.. seealso::
   
   - :doc:`../tutorials/index` for worked examples
   - :doc:`focus_functions` for information on focus functions.
   - :doc:`../theory/theory` for theoretical background on the implemented schemes.