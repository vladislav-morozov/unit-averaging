About Unit Averaging
===========
 

This page explains the mathematical foundations of unit averaging.



---

Mathematical Formulation
------------------------
The unit averaging problem solves for weights :math:`w_i` that minimize the objective:

.. math::
   \hat{y} = \argmin_{y} \sum_{i=1}^N w_i \cdot \ell(y, \hat{y}_i),

where:
- :math:`\hat{y}_i` are individual unit estimates,
- :math:`\ell(\cdot)` is a loss function (e.g., squared error),
- :math:`w_i \geq 0` and :math:`\sum_i w_i = 1`.

For the **optimal averager**, the weights are computed as:

.. math::
   w_i \propto \exp\left(-\lambda \cdot \ell(\hat{y}_i, y_{\text{target}})\right),

where :math:`\lambda` controls the focus on the target unit.

---

Key Properties
--------------
1. **Consistency**: If all :math:`\hat{y}_i` are equal, :math:`\hat{y} = \hat{y}_i`.
2. **Robustness**: Outliers (large :math:`\ell(\cdot)`) receive low weights.
3. **Efficiency**: Computation scales as :math:`O(N)` with :math:`N` units.

.. seealso::
   :doc:`../tutorials/running-averaging` for practical examples.

  