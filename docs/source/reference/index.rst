API Reference
============================


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Modules
      
   averagers.rst
   focus_functions.rst 

The ``unit_averaging`` package has two principal modules:

- :doc:`unit_averaging.averager <averagers>` that contains averaging strategies.
- :doc:`unit_averaging.focus_function <focus_functions>` that contains focus function classes.

For convenience, all public classes and functions from both modules are exposed at the package level, and you can import everything directly from the package without referencing individual modules:

.. code-block:: python3

   from unit_averaging import OptimalUnitAverager

Averager classes:


.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`BaseUnitAverager`
     - Abstract base class. Subclass and implement ``_compute_weights()`` for custom strategies.
   * - :doc:`IndividualUnitAverager`
     - Assigns all weight to the target unit (``weights = [0, ..., 1, ..., 0]``).
   * - :doc:`MeanGroupUnitAverager`
     - Equal weights for all units (``weights = [1/N, ..., 1/N]``).
   * - :doc:`OptimalUnitAverager`
     - MSE-optimal weights (agnostic and with prior restrictions).

Focus function classes:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`BaseFocusFunction`
     - Abstract base class. Subclass and implement ``focus_function()`` and ``gradient()``.
   * - :doc:`InlineFocusFunction`
     - Convenience class for creating focus functions from callables.
   * - :doc:`IdentityFocusFunction`
     - Convenience class for creating an identity focus function for when the estimates are already the focus parameters.
 
 
 

.. admonition:: See also

    In addition to this API reference, also see:

    - :doc:`Background about unit averaging <../theory/theory>` for a quick reference on the theory of unit averaging.
    - The :doc:`tutorials <../tutorials/index>` to see the code in action.
