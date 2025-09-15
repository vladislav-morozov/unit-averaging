"""
Unit Averagers
==============

This module provides classes for unit averaging strategies.
"""

from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np

from unit_averaging.focus_function import BaseFocusFunction


class BaseUnitAverager(ABC):
    """**Base class for unit averaging methods**.

    This abstract base class encapsulates the common behavior for unit averaging
    methods. It provides the basic structure for fitting weights and computing
    averages, and is designed to be subclassed for specific unit averaging strategies.

    Subclasses must implement the ``_compute_weights()`` method to define how the
    weights are computed for the specific averaging strategy.

    Args:
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        ind_estimates (np.ndarray | list | dict[str|int, np.ndarray|list]):
            Individual unit estimates. Can be a list, numpy array, or dictionary.
            Each unit-specific estimate should be a NumPy array or a list.
            The first dimension of `ind_estimates` indexes units (array rows or
            dict entries).

    Attributes:
        ind_estimates (np.ndarray): Array of individual unit estimates.
        keys (np.ndarray): Array of keys corresponding to the units.

            The individual estimates are converted to numpy arrays internally.
            If ``ind_estimates`` is a dictionary, the keys are preserved in the
            ``keys`` attribute. If ``ind_estimates`` is not an array, ``keys``
            defaults to numeric indices (0, 1, 2, ...).
        weights (np.ndarray): The computed weights for each unit.
            Initialized as None, computed by calling ``fit()``
        estimate (float): The computed unit averaging estimate.
            Initialized as None.
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        target_id (int | str): The ID of the target unit. Initialized as None,
            set by calling ``fit()``.

    Example:
        >>> from unit_averaging import BaseUnitAverager, InlineFocusFunction
        >>> import numpy as np
        >>> # Custom averager that uses equal weights
        >>> class CustomUnitAverager(BaseUnitAverager):
        ...     def _compute_weights(self):
        ...         self.weights = (
        ...             np.ones(len(self.ind_estimates)) / len(self.ind_estimates)
        ...            )
        >>> # Using the averager in practice
        >>> focus_fun = lambda x: (x[0])**2
        >>> focus_grad = lambda x: np.array([2*x[0], 0])
        >>> focus_function = InlineFocusFunction(focus_fun, focus_grad)
        >>> ind_estimates = [np.array([4, 2]), np.array([3, 4])]
        >>> averager = CustomUnitAverager(focus_function, ind_estimates)
        >>> averager.fit(target_id=0)
        >>> print(averager.weights)  # [0.5, 0.5]
        >>> print(averager.estimate) # 12.5
    """

    def __init__(
        self,
        focus_function: BaseFocusFunction,
        ind_estimates: np.ndarray | list | dict[str | int, np.ndarray | list],
    ):
        """Initialize the averager with base collection of arguments."""
        # Inputs
        self.focus_function = focus_function
        self.keys, self.ind_estimates = self._convert_inputs_to_array(ind_estimates)
        # For learned parameters
        self.weights = None
        self.estimate = None
        # Related to target unit
        self.target_id = None

    def fit(self, target_id: int | str):
        """Compute the unit averaging weights and the averaging estimator.

        Args:
            target_id (int | str): ID of the target unit. This is specified in
                terms of the keys attribute, which are either numeric indices
                (if ``ind_estimates`` was an array or list) or dictionary keys (if
                ``ind_estimates`` was a dictionary)


        Raises:
            ValueError: If the target unit is not found in the keys.
        """
        self.target_id = target_id
        # Look up index of target ID in the keys array
        target_coord = np.searchsorted(self.keys, target_id)
        if (target_coord == 0 and self.keys[0] != target_id) or (
            target_coord == len(self.keys)
        ):
            raise ValueError("Target unit not in the keys")
        else:
            self._target_coord = target_coord
        # Compute weights
        self._compute_weights()
        # Compute appropriate unit averaging estimate
        self.estimate = self.average(
            self.focus_function,
        )

    def average(self, focus_function: BaseFocusFunction | None = None) -> float:
        """Perform unit averaging with the fitted weights.

        This method computes the unit averaging estimate using the fitted weights.
        It can accept a different focus function and reuse the fitted weights.

        Args:
            focus_function (BaseFocusFunction | None): Focus function to use in
                computing the averaging estimator. Expresses the parameter of
                interest. If None, defaults to the focus function used in fitting.

        Returns:
            float: The unit averaging estimate.

        Raises:
            TypeError: If weights have not been fitted yet by calling ``fit()``
        """
        # Check if weights have been fitted
        if self.weights is None:
            raise TypeError(
                "Weights have not been fitted. Call the 'fit' method first."
            )
        # If no new focus function is supplied, use the base one
        if focus_function is None:
            focus_function = self.focus_function
        # Compute unit weighted average
        weighted_ind_estimates = [
            weight * focus_function.focus_function(ind_est)
            for ind_est, weight in zip(self.ind_estimates, self.weights, strict=True)
        ]
        return sum(weighted_ind_estimates)

    @abstractmethod
    def _compute_weights(self):
        """Compute unit averaging weights.

        This abstract method should be implemented by subclasses to define how
        the weights are computed. The computed weights should be stored in the
        `weights` attribute.
        """
        pass

    def _convert_inputs_to_array(self, input_data: list | np.ndarray | dict):
        """Convert input data (dict, list, or array) into keys and values arrays."""
        if isinstance(input_data, dict):
            # Handle dict inputs
            # Sort to ensure same order of all processed arrays in different calls
            keys = np.fromiter(input_data.keys(), dtype=object)
            keys.sort()
            vals = np.array([input_data[key] for key in keys])
        else:
            # Handle list or array inputs
            keys = np.arange(len(input_data))
            vals = np.array(input_data)

        return keys, vals.astype("float64")


class IndividualUnitAverager(BaseUnitAverager):
    """**Unit averaging scheme that assigns all weight to the target unit.**

    This class implements a unit averaging scheme where all weight is assigned to
    the target unit, effectively ignoring all other units. This is useful when
    the focus is solely on the target unit's estimate and for comparing other
    averaging schemes with no averaging using the same interface.

    Args:
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        ind_estimates (np.ndarray | list | dict[str | int, np.ndarray | list]):
            Individual unit estimates. Can be a list, numpy array, or dictionary.
            Each unit-specific estimate should be a NumPy array or a list.
            The first dimension of `ind_estimates` indexes units (rows or dictionary
            entries).

    Attributes:
        ind_estimates (np.ndarray):
            Array of individual unit estimates.
        keys (np.ndarray):
            Array of keys corresponding to the units. The individual estimates are
            converted to numpy arrays internally. If ``ind_estimates`` is a
            dictionary, the keys are preserved in the ``keys`` attribute. If
            ``ind_estimates`` is a list or array, ``keys`` defaults to numeric
            indices (0, 1, 2, ...).
        weights (np.ndarray):
            The computed weights for each unit. For this scheme, the weight for
            the target unit is 1.0, and the weights for all other units are 0.0.
        estimate (float):
            The computed unit averaging estimate, which is simply the target
            unit's estimate.
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        target_id (int | str):
            The ID of the target unit. Initialized as None, set by calling ``fit()``.

    Example:
        >>> from unit_averaging import IndividualUnitAverager, InlineFocusFunction
        >>> import numpy as np
        >>> # Define a focus function
        >>> focus_function = InlineFocusFunction(lambda x: x[0], lambda x: [1, 0])
        >>> # Define individual unit estimates
        >>> ind_estimates = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        >>> # Create an IndividualUnitAverager instance
        >>> averager = IndividualUnitAverager(focus_function, ind_estimates)
        >>> # Fit the averager to the target unit
        >>> averager.fit(target_id="b")
        >>> # Print the estimate
        >>> print(averager.weights)     # [0., 1.]
        >>> print(averager.estimate)    # 3.0
    """

    def _compute_weights(self):
        """Compute unit averaging weights.

        This method assigns all weight to the target unit.
        """
        num_units = len(self.ind_estimates)
        weights = np.zeros(num_units)
        weights[self._target_coord] = 1.0
        self.weights = weights


class MeanGroupUnitAverager(BaseUnitAverager):
    """**Unit averaging scheme that assigns equal weights to all units.**

    This class implements a unit averaging scheme where equal weights are
    assigned to all units — the mean group estimator. The MG approach typically
    is a good estimator of the expected value of a parameter in a heterogeneous
    setting.

    Args:
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        ind_estimates (np.ndarray | list | dict[str | int, np.ndarray | list]):
            Individual unit estimates. Can be a list, numpy array, or dictionary.
            Each unit-specific estimate should be a NumPy array or a list.
            The first dimension of `ind_estimates` indexes units (rows or
            dictionary entries).

    Attributes:
        ind_estimates (np.ndarray):
            Array of individual unit estimates.
        keys (np.ndarray):
            Array of keys corresponding to the units. The individual estimates are
            converted to numpy arrays internally. If ``ind_estimates`` is a
            dictionary, the keys are preserved in the ``keys`` attribute. If
            ``ind_estimates`` is a list or array, ``keys`` defaults to numeric
            indices (0, 1, 2, ...).
        weights (np.ndarray):
            The computed weights for each unit. For this scheme, all weights
            are equal and sum to 1.
        estimate (float):
            The computed unit averaging estimate. Here a simple average of all
            unit estimates.
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        target_id (int | str):
            The ID of the target unit. Initialized as None, set by calling ``fit()``.

    Example:
        >>> from unit_averaging import MeanGroupUnitAverager, InlineFocusFunction
        >>> import numpy as np
        >>> # Define a focus function
        >>> focus_function = InlineFocusFunction(
        ...     lambda x: x[0],
        ...     lambda x: np.array([1, 0])
        ... )
        >>> # Define individual unit estimates
        >>> ind_estimates = {
        ...     "unit1": np.array([5, 6]),
        ...     "unit2": np.array([7, 8]),
        ...     "unit3": np.array([9, 10])
        ... }
        >>> # Create a MeanGroupUnitAverager instance
        >>> averager = MeanGroupUnitAverager(focus_function, ind_estimates)
        >>> # Fit the averager to the target unit
        >>> averager.fit(target_id="unit1")
        >>> print(averager.weights)   # [0.33333333 0.33333333 0.33333333]
        >>> print(averager.estimate)  # 7.0
    """

    def _compute_weights(self):
        """Compute unit averaging weights.

        This method assigns equal weights to all units.
        """
        num_units = len(self.ind_estimates)
        weights = np.ones(num_units) / num_units
        self.weights = weights


class OptimalUnitAverager(BaseUnitAverager):
    """**Optimal weight scheme that minimizes the plug-in Mean Squared Error (MSE).**

    It supports two regimes: fixed-N and large-N, each with different approaches
    to weight allocation.

    **Fixed-N Regime:**
    In the fixed-N regime, the weights of all units vary independently, subject
    only to the constraints of non-negativity and summing to 1. This is an "agnostic"
    scheme where the algorithm determines the optimal weight for each unit individually,
    without any grouping or restrictions. This regime is suitable when the number of
    units is small or when you want to allow maximum flexibility in weight allocation.

    **Large-N Regime:**
    In the large-N regime, you can specify some units as "unrestricted" (free)
    while the remaining units are considered "restricted." The key idea is that:

    - The weights of unrestricted units vary independently.
    - All restricted units receive equal weights.
    - The algorithm only chooses the weight of the restricted set as a whole.

    This approach is particularly useful when you have a large number of restricted
    units. The average of a large restricted set will closely approximate the true
    average of the parameters. This allows for more efficient and precise shrinkage,
    as the algorithm can focus on optimizing the weights of the unrestricted units
    and the total weight of the restricted set.


    Args:
        focus_function (BaseFocusFunction):
            Focus function expressing the transformation of interest.
        ind_estimates (np.ndarray | list | dict[str | int, np.ndarray | list]):
            Individual unit estimates. Can be a list, numpy array, or dictionary.
            Each unit-specific estimate should be a NumPy array or list.
            The first dimension of ``ind_estimates`` indexes units (rows or
            dictionary entries).
        ind_covar_ests (np.ndarray | list | dict[str | int, np.ndarray | list]):
            Individual unit covariance estimates. Can be a list, numpy array, or
            dictionary. Each unit-specific covariance estimate should be a NumPy
            array or list of lists. The first dimension of ``ind_covar_ests`` indexes
            units (rows or dictionary entries).
        unrestricted_units_bool (np.ndarray | list | dict[str | int, bool] | None):
            Optional. Boolean array indicating which units are unrestricted for
            weight computations, with ``True`` meaning that a unit is unrestricted.
            If a dictionary, keys should match those in `ind_estimates` and
            ``ind_covar_ests``. If None, all units are considered unrestricted.
            Defaults to None.

    Attributes:
        ind_estimates (np.ndarray):
            Array of individual unit estimates.
        ind_covar_ests (np.ndarray):
            Array of individual unit covariance estimates.
        unrestricted_units_bool (np.ndarray):
            Boolean array indicating which units are unrestricted.
        keys (np.ndarray):
            Array of keys corresponding to the units. The individual estimates are
            converted to numpy arrays internally. If ``ind_estimates`` is a
            dictionary, the keys are preserved in the ``keys`` attribute. If
            ``ind_estimates`` is a list or array, ``keys`` defaults to numeric
            indices (0, 1, 2, ...).
        weights (np.ndarray):
            The computed weights for each unit.
        estimate (float):
            The computed unit averaging estimate.
        focus_function (:class:`~unit_averaging.focus_function.BaseFocusFunction`):
            Focus function expressing the transformation of interest.
        target_id (int | str):
            The ID of the target unit. Initialized as None, set by calling ``fit()``.

    Example:
        >>> from unit_averaging import OptimalUnitAverager, InlineFocusFunction
        >>> import numpy as np
        >>> # Define a focus function
        >>> focus_function = InlineFocusFunction(
        ...     lambda x: x[0] * x[1],
        ...     lambda x: np.array([x[1], x[0]]),
        ... )
        >>> # Define individual unit estimates
        >>> ind_estimates = {
        ...     "unit1": np.array([5, 6]),
        ...     "unit2": np.array([7, 8]),
        ...     "unit3": np.array([9, 3]),
        ...     "unit4": np.array([3, 10]),
        ... }
        >>> # Define individual unit covariance estimates
        >>> ind_covar_ests = {
        ...     "unit1": np.array([[3, 0.25], [0.25, 3]]),
        ...     "unit2": np.array([[4, 0.5], [0.5, 5]]),
        ...     "unit3": np.array([[1, -0.25], [-0.25, 1]]),
        ...     "unit4": np.array([[1, 0.5], [0.5, 1]]),
        ... }
        >>> # Define unrestricted units
        >>> unrestricted_units_bool = {
        ...     "unit1": True,
        ...     "unit2": True,
        ...     "unit3": False,
        ...     "unit4": False,
        ... }
        >>> # Create an OptimalUnitAverager instance
        >>> averager = OptimalUnitAverager(
        ...     focus_function, ind_estimates, ind_covar_ests, unrestricted_units_bool
        ... )
        >>> # Fit the averager to the target unit
        >>> averager.fit(target_id="unit1")
        >>> print(averager.weights.round(3))  # [0.324 0.    0.338 0.338]
        >>> print(averager.estimate)  # 28.99
    """

    def __init__(
        self,
        focus_function: BaseFocusFunction,
        ind_estimates: list | np.ndarray | dict[str | int, np.ndarray | list],
        ind_covar_ests: list | np.ndarray | dict[str | int, np.ndarray | list],
        unrestricted_units_bool: np.ndarray
        | list
        | dict[str | int, np.ndarray | list]
        | None = None,
    ):
        super().__init__(focus_function, ind_estimates)

        # Check that all or none of the inputs are dicts
        self._validate_all_dicts_or_none(
            ind_estimates, ind_covar_ests, unrestricted_units_bool
        )

        # Process covariances
        covar_keys, self.ind_covar_ests = self._convert_inputs_to_array(ind_covar_ests)
        if not np.array_equal(self.keys, covar_keys):
            raise ValueError("Keys of estimates and covariances do not match.")

        # Detect fixed-N vs. large-N
        if unrestricted_units_bool is not None:
            unrestr_keys, self.unrestricted_units_bool = self._convert_inputs_to_array(
                unrestricted_units_bool
            )
            self.unrestricted_units_bool = self.unrestricted_units_bool.astype(bool)
            if not np.array_equal(self.keys, unrestr_keys):
                raise ValueError(
                    "Keys of estimates and unrestricted units do not match."
                )

        else:
            self.unrestricted_units_bool = np.full(len(ind_estimates), True)

    def _compute_weights(self):
        # Estimate gradient and ensure it is a 1D numpy array
        gradient_estimatetarget = self._clean_gradient(
            self.focus_function.gradient(self.ind_estimates[self._target_coord])
        )

        # Construct the objective function
        quad_term = self._build_mse_matrix(
            self._target_coord,
            self.ind_estimates,
            self.ind_covar_ests,
            gradient_estimatetarget,
            self.unrestricted_units_bool,
        )
        num_coords = quad_term.shape[0]
        lin_term = np.zeros((num_coords, 1))

        # Specify the constrains
        ineq_lhs = -np.identity(num_coords)
        ineq_rhs = np.zeros(num_coords)
        eq_lhs = np.ones((1, num_coords))
        eq_rhs = np.array([1.0])

        # Minimize the MSE and return the optimal weights
        weights = cp.Variable(num_coords)
        prob = cp.Problem(
            cp.Minimize(
                (1 / 2) * cp.quad_form(weights, quad_term) + lin_term.T @ weights
            ),
            [
                ineq_lhs @ weights <= ineq_rhs,
                eq_lhs @ weights == eq_rhs,
            ],
        )
        # TODO: Resolve raise_error warning after https://github.com/cvxpy/cvxpy/issues/2851
        prob.solve()
        if weights.value is None:
            raise TypeError(
                "Optimizer could not find a feasible solution, returned None."
            )

        # Allocate the weights
        opt_weights = self._allocate_optimal_weights(
            weights.value,
            self.unrestricted_units_bool,
            self.ind_estimates,
        )

        self.weights = opt_weights

    def _clean_gradient(self, gradient) -> np.ndarray:
        """Ensure the gradient is a 1D numpy array."""
        gradient = np.array(gradient)
        if gradient.ndim == 0:
            gradient = np.expand_dims(gradient, axis=0)
        return gradient

    def _build_mse_matrix(
        self,
        target_coord: int | np.intp,
        ind_estimates: np.ndarray,
        ind_covar_ests: np.ndarray,
        gradient_estimatetarget: np.ndarray,
        unrestr_units_bool: np.ndarray,
    ) -> np.ndarray:
        """Build the objective matrix for optimal-weight unit averaging.

        This function accommodates both fixed-N and large-N approximations.

        Args:
            target_coord (int): index of the target unit in the estimates arrays
            ind_estimates (np.ndarray): An array of individual parameter estimates
                (thetas in notation of docs and paper).
            ind_covar_ests (np.ndarray): An array of covariance matrices for
                individual parameter estimates.
            gradient_estimatetarget (np.ndarray): 1D NumPy array with the estimated
                gradient of the focus function for the target unit
            unrestr_units_bool (np.ndarray | None, optional): Boolean array indicating
                unrestricted units. True means the corresponding unit is unrestricted.
                If None, all units are unrestricted.

        Returns:
            np.ndarray: the estimated MSE matrix (psi or Q in notation of paper)
        """

        # Compute the MSE matrix of unrestricted units
        unrstrct_coefs = ind_estimates[unrestr_units_bool]
        unrstrct_covar = ind_covar_ests[unrestr_units_bool]

        # Fill MSE matrix element-by-element
        psi = np.empty((len(unrstrct_coefs), len(unrstrct_coefs)), dtype="float64")
        for i in range(len(unrstrct_coefs)):
            for j in range(len(unrstrct_coefs)):
                # Difference between estimates
                coef_dif_i = unrstrct_coefs[i] - ind_estimates[target_coord]
                coef_dif_j = unrstrct_coefs[j] - ind_estimates[target_coord]
                psi_ij = np.outer(coef_dif_i, coef_dif_j)
                # add covariance when appropriate
                if i == j:
                    psi_ij += unrstrct_covar[i]
                # Multiply by gradient
                psi_ij = gradient_estimatetarget @ psi_ij @ gradient_estimatetarget
                # Set the corresponding element
                psi[i, j] = psi_ij

        # If in large-N regime, add an outer row and column of restricted units
        if sum(unrestr_units_bool) < len(ind_estimates):
            q = np.empty(
                (len(unrstrct_coefs) + 1, len(unrstrct_coefs) + 1),
                dtype="float64",
            )
            q[:-1, :-1] = psi
            b = np.empty(len(unrstrct_coefs), dtype="float64")

            # Fill out the elements of the b vector
            mg = ind_estimates.mean(axis=0)
            for i in range(len(b)):
                b_i = np.outer(
                    unrstrct_coefs[i] - ind_estimates[target_coord],
                    ind_estimates[target_coord] - mg,
                )
                q[i, -1] = -gradient_estimatetarget @ b_i @ gradient_estimatetarget
                q[-1, i] = q[i, -1]

            # Insert the last element
            q[-1, -1] = np.power(
                gradient_estimatetarget @ (ind_estimates[target_coord] - mg),
                2,
            )
        else:
            q = psi

        return q

    def _allocate_optimal_weights(
        self,
        solution: np.ndarray,
        unrestricted_units_bool: np.ndarray,
        ind_estimates: np.ndarray,
    ) -> np.ndarray:
        """Allocate fixed-N and large-N weights across units.

        Args:
            solution (np.ndarray): The solution vector from the optimization problem.
            unrestricted_units_bool (np.ndarray): Boolean array indicating which
                units are unrestricted.
            ind_estimates (np.ndarray): Array of individual unit estimates.

        Returns:
            np.ndarray: The allocated optimal weights.
        """
        num_restr_units = len(ind_estimates) - sum(unrestricted_units_bool)
        opt_weights = np.zeros(ind_estimates.shape[0])

        if num_restr_units == 0:
            opt_weights = solution
        else:
            unrestr_weights = solution[:-1]
            opt_weights[unrestricted_units_bool] = unrestr_weights
            weight_per_restr_unit = solution[-1] / num_restr_units
            np.putmask(opt_weights, ~unrestricted_units_bool, weight_per_restr_unit)

        return opt_weights

    def _validate_all_dicts_or_none(
        self,
        ind_estimates: list | np.ndarray | dict,
        ind_covar_ests: list | np.ndarray | dict,
        unrestricted_units_bool: list | np.ndarray | dict | None = None,
    ) -> None:
        """Validate that all inputs are dictionaries or none are dictionaries.

        Args:
            ind_estimates (list | np.ndarray | dict): Individual unit estimates.
            ind_covar_ests (list | np.ndarray | dict): Individual unit covariance
                estimates.
            unrestricted_units_bool (list | np.ndarray | dict | None, optional):
                Boolean array indicating which units are unrestricted.

        Raises:
            TypeError: If some inputs are dictionaries and others are not.
        """
        is_dict_estimates = isinstance(ind_estimates, dict)
        is_dict_covar = isinstance(ind_covar_ests, dict)
        is_dict_unrestricted = (
            isinstance(unrestricted_units_bool, dict)
            if unrestricted_units_bool is not None
            else is_dict_estimates * is_dict_covar
        )

        conditions = [is_dict_estimates, is_dict_covar, is_dict_unrestricted]

        if any(conditions) and not all(conditions):
            raise TypeError(
                "If any input is a dictionary, all inputs must be dictionaries."
            )
