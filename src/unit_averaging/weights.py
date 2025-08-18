from typing import Protocol, runtime_checkable

import cvxpy as cp
import numpy as np

from unit_averaging.focus_function import FocusFunction


# Protocol for weight functions used in unit averaging.
# A weight function must behave like a callable that accepts:
# - a focus function (used to extract relevant features),
# - a target ID (typically an integer),
# - an array of individual estimates,
# - an optional array of covariances,
# - and any number of additional keyword arguments.
# It must return an array of weights (np.ndarray).
@runtime_checkable
class WeightFunction(Protocol):
    def __call__(
        self,
        focus_fn: FocusFunction,
        some_int: int,
        arr1: np.ndarray,
        arr2: np.ndarray | None,
        **kwargs: object,
    ) -> np.ndarray: ...


def individual_weights(
    focus_function: FocusFunction,
    target_id: int,
    ind_estimates: np.ndarray,
    ind_covar_ests: np.ndarray | None = None,
) -> np.ndarray:
    """Unit averaging weight scheme that assigns all weight to the target unit

    Args:
        focus_function (FocusFunction): instance of a focus function
        target_id (int): index of the target unit in the estimates arrays
        ind_estimates (np.ndarray): An array of individual parameter estimates
            (thetas in notation of docs and paper).
        ind_covar_ests (np.ndarray | None, optional): An array of covariance
            matrices for individual parameter estimates. This argument is
            optional for this weight scheme and not used. Defaults to None.

    Returns:
        np.ndarray: unit weights
    """
    num_units = len(ind_estimates)
    weights = np.zeros(num_units)
    weights[target_id] = 1.0
    return weights


def mean_group_weights(
    focus_function: FocusFunction,
    target_id: int,
    ind_estimates: np.ndarray,
    ind_covar_ests: np.ndarray | None = None,
) -> np.ndarray:
    """Unit averaging weight scheme that assigns equal weights to all units

    Args:
        focus_function (FocusFunction): instance of a focus function
        target_id (int): index of the target unit in the estimates arrays
        ind_estimates (np.ndarray): An array of individual parameter estimates
            (thetas in notation of docs and paper).
        ind_covar_ests (np.ndarray | None, optional): An array of covariance
            matrices for individual parameter estimates. This argument is
            optional for this weight scheme and not used. Defaults to None.

    Returns:
        np.ndarray: equal unit weights
    """
    num_units = len(ind_estimates)
    weights = np.ones(num_units) / num_units
    return weights


def _build_mse_matrix(
    target_id: int,
    ind_estimates: np.ndarray,
    ind_covar_ests: np.ndarray,
    gradient_estimate_target: np.ndarray,
    unrestr_units_bool: np.ndarray,
) -> np.ndarray:
    """Build the objective matrix for optimal-weight unit averaging.

    This function accommodates both fixed-N and large-N approximations.

    Args:
        target_id (int): index of the target unit in the estimates arrays
        ind_estimates (np.ndarray): An array of individual parameter estimates
            (thetas in notation of docs and paper).
        ind_covar_ests (np.ndarray): An array of covariance matrices for
            individual parameter estimates.
        gradient_estimate_target (np.ndarray): 1D NumPy array with the estimated
            gradient of the focus function for the target unit
        unrestr_units_bool (np.ndarray): Boolean array indicating
            unrestricted units. True means the corresponding unit is unrestricted.

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
            coef_dif_i = unrstrct_coefs[i] - ind_estimates[target_id]
            coef_dif_j = unrstrct_coefs[j] - ind_estimates[target_id]
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
                unrstrct_coefs[i] - ind_estimates[target_id],
                ind_estimates[target_id] - mg,
            )
            q[i, -1] = -gradient_estimate_target @ b_i @ gradient_estimate_target
            q[-1, i] = q[i, -1]

        # Insert the last element
        q[-1, -1] = np.power(
            gradient_estimate_target @ (ind_estimates[target_id] - mg),
            2,
        )
    else:
        q = psi

    return q


def _clean_gradient(gradient: np.ndarray) -> np.ndarray:
    """Ensure the gradient is a 1D numpy array."""
    gradient = np.array(gradient)
    if gradient.ndim == 0:
        gradient = np.expand_dims(gradient, axis=0)
    return gradient


def optimal_weights(
    focus_function: FocusFunction,
    target_id: int,
    ind_estimates: np.ndarray,
    ind_covar_ests: np.ndarray,
    unrestricted_units_bool: np.ndarray | None = None,
) -> np.ndarray:
    """Optimal unit averaging weight scheme that minimizes the plug-in MSE.

    Args:
        focus_function (FocusFunction): instance of a focus function
        target_id (int): index of the target unit in the estimates arrays
        ind_estimates (np.ndarray): An array of individual parameter estimates
            (thetas in notation of docs and paper).
        ind_covar_ests (np.ndarray | None, optional): An array of covariance
            matrices for individual parameter estimates. This argument is not
            optional for this weight scheme.
        unrestricted_units_bool: (np.ndarray | None, optional): A Boolean array
            indicated which of the units have free weights (True) and which
            belong to the restricted group (False). Specifying this array
            triggers the large-N regime.

    Returns:
        np.ndarray: optiomal unit weights

    Raises:
        TypeError: If covariances are None or if the optimizer could not find a
            feasible solution.
    """

    # Checking for fixed-N/large-N regime
    # If no value is supplied, fixed-N regime is the default
    if unrestricted_units_bool is None:
        unrestricted_units_bool = np.full(len(ind_estimates), True)

    # Optimal weights need covariances
    if ind_covar_ests is None:
        raise TypeError("Covariances cannot be None for optimal weights.")

    # Estimate gradient and ensure it is a 1D numpy array
    gradient_estimate_target = _clean_gradient(
        focus_function.gradient(ind_estimates[target_id])
    )

    # Construct the objective function
    quad_term = _build_mse_matrix(
        target_id,
        ind_estimates,
        ind_covar_ests,
        gradient_estimate_target,
        unrestricted_units_bool,
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
        cp.Minimize((1 / 2) * cp.quad_form(weights, quad_term) + lin_term.T @ weights),
        [
            ineq_lhs @ weights <= ineq_rhs,
            eq_lhs @ weights == eq_rhs,
        ],
    )
    # TODO: Increase cvxpy to at least 1.7.2 when it's released
    # https://github.com/cvxpy/cvxpy/issues/2851
    prob.solve()
    if weights.value is None:
        raise TypeError("Optimizer could not find a feasible solution, returned None.")

    return weights.value
