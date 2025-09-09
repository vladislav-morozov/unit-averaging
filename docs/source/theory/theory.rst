About Unit Averaging
=======================
 

This page explains the mathematical foundations of unit averaging.



---

Mathematical Formulation
------------------------
The unit averaging problem solves for weights :math:`w_i` that minimize the objective:

.. math::
   \hat{w} = \argmin_{y} something

where:
- :math:`\hat{y}_i` are individual unit estimates,
- :math:`\ell(\cdot)` is a loss function (e.g., squared error),
- :math:`w_i \geq 0` and :math:`\sum_i w_i = 1`.

For the **optimal averager**, the weights are computed as:
 