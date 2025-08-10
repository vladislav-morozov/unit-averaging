import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, UnitAverager

# Test data for testing various inputs
test_data = [
    # Inputs: lists of scalars
    (
        InlineFocusFunction(lambda x: x, lambda x: 1),
        "individual",
        [0, 1],
        [1, 1],
        np.array([1, 0]),
        0,
    ),
    (
        InlineFocusFunction(lambda x: x, lambda x: 1),
        "mean_group",
        [0, 1],
        [1, 1],
        np.array([0.5, 0.5]),
        0.5,
    ),
    # Inputs: lists of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        "individual",
        [np.array([3, 2]), np.array([4, 5])],
        [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
        np.array([1, 0]),
        3,
    ),
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        "mean_group",
        [np.array([3, 2]), np.array([4, 5])],
        [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
        np.array([0.5, 0.5]),
        3.5,
    ),
]


@pytest.mark.parametrize(
    "focus_function, weight_scheme, ind_estimates, "
    "ind_covar_ests, expected_weights, expected_estimate",
    test_data,
)
def test_unit_averager_various_inputs(
    focus_function,
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected_weights,
    expected_estimate,
):
    """Test the UnitAverager with various inputs and weight schemes."""
    ua = UnitAverager(
        focus_function,
        weight_scheme,
        ind_estimates,
        ind_covar_ests,
    )
    ua.fit(target_id=0)

    # Compound assertion to check both weights and estimate
    assert (
        np.allclose(ua.weights_, expected_weights, rtol=1e-03)
        and np.allclose(ua.estimate_, expected_estimate, rtol=1e-03)
    ), "Weights or estimate do not match expected values."
