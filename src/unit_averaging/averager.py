from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np

from unit_averaging.focus_function import FocusFunction


class BaseUnitAverager(ABC):
    """Class to encapsulate fit and averaging behavior"""

    def __init__(
        self,
        focus_function: FocusFunction,
        ind_estimates: np.ndarray | list | dict[str | int, np.ndarray | list],
    ):
        self.focus_function = focus_function
        self.weights_ = None
        self.estimate_ = None

        self.keys, self.ind_estimates = self._convert_inputs_to_array(ind_estimates)

    def fit(self, target_id: int | str):
        """Compute the unit averaging weights and the averaging estimator

        Args:
            target_id (int | str): ID of target unit
        """

        self.target_id_ = target_id

        # Look up index of target ID in the keys array
        target_coord = np.searchsorted(self.keys, target_id)
        if target_coord == 0 and self.keys[0] != target_id:
            raise ValueError("Target unit not in the keys")
        else:
            self._target_coord_ = target_coord

        # Compute weights
        self._compute_weights()

        # Compute appropriate unit averaging estimate
        self.estimate_ = self.average(
            self.focus_function,
        )

    def average(self, focus_function: FocusFunction | None = None) -> float:
        """Perform unit averaging with the fitted weights

        Args:
            focus_function (FocusFunction | None): focus function to use in
                computing the average. mu in notation of docs and paper. If
                None, defaults to self.focus_function

        Returns:
            float: unit averaging estimate
        """
        # Check if weights have been fitted
        if self.weights_ is None:
            raise TypeError(
                "Weights have not been fitted. Call the 'fit' method first."
            )

        # If no new focus function is supplied, use the base one
        if focus_function is None:
            focus_function = self.focus_function

        # Compute unit weighed average
        weighted_ind_estimates = [
            weight * focus_function.focus_function(ind_est)
            for ind_est, weight in zip(self.ind_estimates, self.weights_, strict=True)
        ]
        return sum(weighted_ind_estimates)

    @abstractmethod
    def _compute_weights(self):
        """Compute unit averaging weights"""

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
    """Unit averaging scheme that assigns all weight to the target unit."""

    def _compute_weights(self):
        num_units = len(self.ind_estimates)
        weights = np.zeros(num_units)
        weights[self._target_coord_] = 1.0
        self.weights_ = weights


class MeanGroupUnitAverager(BaseUnitAverager):
    """Unit averaging scheme that assigns equal weights to all units"""

    def _compute_weights(self):
        num_units = len(self.ind_estimates)
        weights = np.ones(num_units) / num_units
        self.weights_ = weights


class OptimalUnitAverager(BaseUnitAverager):
    """Optimal unit averaging weight scheme that minimizes the plug-in MSE."""

    def __init__(
        self,
        focus_function: FocusFunction,
        ind_estimates: list | np.ndarray | dict,
        ind_covar_ests: list | np.ndarray | dict,
        unrestricted_units_bool: np.ndarray | list | dict | None = None,
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
        gradient_estimate_target = self._clean_gradient(
            self.focus_function.gradient(self.ind_estimates[self._target_coord_])
        )

        # Construct the objective function
        quad_term = self._build_mse_matrix(
            self._target_coord_,
            self.ind_estimates,
            self.ind_covar_ests,
            gradient_estimate_target,
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

        self.weights_ = opt_weights

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
        gradient_estimate_target: np.ndarray,
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
            gradient_estimate_target (np.ndarray): 1D NumPy array with the estimated
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
                psi_ij = gradient_estimate_target @ psi_ij @ gradient_estimate_target
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
            mg = ind_estimates.mean()
            for i in range(len(b)):
                b_i = np.outer(
                    unrstrct_coefs[i] - ind_estimates[target_coord],
                    ind_estimates[target_coord] - mg,
                )
                q[i, -1] = -gradient_estimate_target @ b_i @ gradient_estimate_target
                q[-1, i] = q[i, -1]

            # Insert the last element
            q[-1, -1] = np.power(
                gradient_estimate_target @ (ind_estimates[target_coord] - mg),
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
        """Allocate fixed-N and large-N weights across units."""
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
        """Validate that all inputs are dictionaries or none are dictionaries."""
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
