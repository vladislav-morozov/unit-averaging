import numpy as np

from unit_averaging.focus_function import FocusFunction


def individual_weights(
    focus_function: FocusFunction,
    target_id: int,
    ind_estimates: np.ndarray,
    ind_covar_ests: np.ndarray | None = None,
) -> np.ndarray:
    """Compute individual weights, assigning all weight to the target unit."""
    num_units = len(ind_estimates)
    weights = np.zeros(num_units)
    weights[target_id] = 1.0
    return weights
