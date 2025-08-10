import numpy as np

from unit_averaging.focus_function import FocusFunction


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
