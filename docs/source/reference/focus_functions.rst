Focus Functions
=====================


.. currentmodule:: unit_averaging.focus_function

   

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   BaseFocusFunction.rst
   InlineFocusFunction.rst
   IdentityFocusFunction.rst


Focus functions define transformations applied to individual unit estimates.
They consist of two components:

1. Focus Function: Maps estimates to a scalar parameter of interest.
2. Gradient: Computes derivatives for optimization.


Available focus functions classes:  

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - :class:`BaseFocusFunction`
     - Abstract base class. Subclass and implement ``focus_function()`` and ``gradient()``.
   * - :class:`InlineFocusFunction`
     - Convenience class for creating focus functions from callables.
   * - :class:`IdentityFocusFunction`
     - Convenience class for creating an identity focus function for when the estimates are already the focus parameters.
 
 


.. seealso::
   
   - :doc:`../tutorials/index` for worked examples
   - :doc:`averagers` for the different averagers.