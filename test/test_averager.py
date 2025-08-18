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
    # Inputs: arrays of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        "individual",
        np.array([np.array([3, 2]), np.array([4, 5])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([1, 0]),
        3,
    ),
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        "mean_group",
        np.array([np.array([3, 2]), np.array([4, 5])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([0.5, 0.5]),
        3.5,
    ),
]
test_ids = [
    "scalar_list_individual",
    "scalar_list_mean_group",
    "array_list_individual",
    "array_list_mean_group",
    "array_of_arrays_individual",
    "array_of_arrays_mean_group",
]


@pytest.mark.parametrize(
    "focus_function, weight_scheme, ind_estimates, "
    "ind_covar_ests, expected_weights, expected_estimate",
    test_data,
    ids=test_ids,
)
def test_unit_averager_various_inputs(
    focus_function,
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected_weights,
    expected_estimate,
):
    """Test the ability of UnitAverager to take various inputs."""
    ua = UnitAverager(
        focus_function,
        weight_scheme,
        ind_estimates,
        ind_covar_ests,
    )
    ua.fit(target_id=0)

    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."


# Test data for testing various inputs
test_data = [
    # Same unit data (expect equal weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([0.5, 0.5]),
        1,
    ),
    # Other unit has crazy variance (expect individual weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[10e10, 0], [0, 10e10]])]),
        np.array([1, 0]),
        1,
    ),
    # Other unit has crazy bias
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([10e10, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([1, 0]),
        1,
    ),
    # Target has crazy variance, other is biased
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([2, 1])]),
        np.array([np.array([[10e10, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([0, 1]),
        2,
    ),
    # Meaningful averaging with two units
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 1]])]),
        np.array([2 / 3, 1 / 3]),
        1,
    ),
    # Meaningful averaging with three units
    # Two useful units, one crazy one
    # Ten identical units
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0, 0, 0, 0])),
        np.ones((10, 5)),
        np.array([np.eye(5)] * 10),
        np.ones((10, 1)) / 10,
        1,
    ),
    # Other nine units are crazy
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0, 0, 0, 0])),
        np.ones((10, 5)),
        np.stack([np.eye(5), *np.array([np.eye(5) * 10e10] * 9)]),
        np.stack([np.array(1), *np.zeros(9)]),
        1,
    ),
    # Nonlinear focus function
]
test_ids = [
    "2 units: identical",
    "2 units: non-target with crazy variance",
    "2 units: non-target with crazy bias",
    "2 units: target has crazy variance, other is biased",
    "2 units: reasonable",
    "10 units: identical",
    "10 units: non-targets have crazy variance",
]


@pytest.mark.parametrize(
    "focus_function, ind_estimates, "
    "ind_covar_ests, expected_weights, expected_estimate",
    test_data,
    ids=test_ids,
)
def test_unit_averager_optional_fixed_n(
    focus_function,
    ind_estimates,
    ind_covar_ests,
    expected_weights,
    expected_estimate,
):
    """Test fixed-N (agnostic) optimal weights in UnitAverager."""
    ua = UnitAverager(
        focus_function,
        "optimal",
        ind_estimates,
        ind_covar_ests,
    )
    ua.fit(target_id=0)

    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."


# Test data for testing large-n regime
test_data = [
    # Same unit data (expect equal weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([True, True]),
        np.array([0.5, 0.5]),
        1,
    ),
]
test_ids = [
    "Large-N with all units unrestricted", 
]

@pytest.mark.parametrize(
    "focus_function, ind_estimates, "
    "ind_covar_ests, unrestricted_units_bool, "
    "expected_weights, expected_estimate",
    test_data,
    ids=test_ids,
)
def test_unit_averager_large_N(
    focus_function,
    ind_estimates,
    ind_covar_ests,
    unrestricted_units_bool,
    expected_weights,
    expected_estimate,
):
    """Test fixed-N (agnostic) optimal weights in UnitAverager."""
    ua = UnitAverager(
        focus_function,
        "optimal",
        ind_estimates,
        ind_covar_ests,
    )
    ua.fit(target_id=0, unrestricted_units_bool=unrestricted_units_bool)

    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."
