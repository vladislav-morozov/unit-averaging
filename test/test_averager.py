import numpy as np
import pytest

from unit_averaging import (
    IndividualUnitAverager,
    InlineFocusFunction,
    MeanGroupUnitAverager,
    OptimalUnitAverager,
)

# Test data for IndividualUnitAverager
individual_test_data = [
    # Inputs: lists of scalars
    (
        InlineFocusFunction(lambda x: x, lambda x: 1),
        [0, 1],
        0,
        np.array([1, 0]),
        0,
    ),
    # Inputs: lists of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        [np.array([3, 2]), np.array([4, 5])],
        0,
        np.array([1, 0]),
        3,
    ),
    # Inputs: arrays of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([3, 2]), np.array([4, 5])]),
        0,
        np.array([1, 0]),
        3,
    ),
    # Inputs: dicts of numpy arrays, string keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {"a": np.array([3, 2]), "b": np.array([4, 5])},
        "a",
        np.array([1, 0]),
        3,
    ),
    # Inputs: dicts of numpy arrays, int keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {1: np.array([3, 2]), 3: np.array([4, 5])},
        3,
        np.array([0, 1]),
        4,
    ),
]

individual_test_ids = [
    "scalar_list_individual",
    "array_list_individual",
    "array_of_arrays_individual",
    "dict_of_arrays_str_keys_individual",
    "dict_of_arrays_int_keys_individual",
]


@pytest.mark.parametrize(
    "focus_function, ind_estimates, target_id, expected_weights, expected_estimate",
    individual_test_data,
    ids=individual_test_ids,
)
def test_individual_unit_averager(
    focus_function,
    ind_estimates,
    target_id,
    expected_weights,
    expected_estimate,
):
    """Test the IndividualUnitAverager with various inputs."""
    ua = IndividualUnitAverager(focus_function, ind_estimates)
    ua.fit(target_id=target_id)
    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."


# Test data for MeanGroupUnitAverager
mean_group_test_data = [
    # Inputs: lists of scalars
    (
        InlineFocusFunction(lambda x: x, lambda x: 1),
        [0, 1],
        0,
        np.array([0.5, 0.5]),
        0.5,
    ),
    # Inputs: lists of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        [np.array([3, 2]), np.array([4, 5])],
        0,
        np.array([0.5, 0.5]),
        3.5,
    ),
    # Inputs: arrays of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([3, 2]), np.array([4, 5])]),
        0,
        np.array([0.5, 0.5]),
        3.5,
    ),
    # Inputs: dicts of numpy arrays, string keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {"a": np.array([3, 2]), "b": np.array([4, 5])},
        "a",
        np.array([0.5, 0.5]),
        3.5,
    ),
    # Inputs: dicts of numpy arrays, int keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {1: np.array([3, 2]), 3: np.array([4, 5])},
        3,
        np.array([0.5, 0.5]),
        3.5,
    ),
]

mean_group_test_ids = [
    "scalar_list_mean_group",
    "array_list_mean_group",
    "array_of_arrays_mean_group",
    "dict_of_arrays_str_keys_mean_group",
    "dict_of_arrays_int_keys_mean_group",
]


@pytest.mark.parametrize(
    "focus_function, ind_estimates, target_id, expected_weights, expected_estimate",
    mean_group_test_data,
    ids=mean_group_test_ids,
)
def test_mean_group_unit_averager(
    focus_function,
    ind_estimates,
    target_id,
    expected_weights,
    expected_estimate,
):
    """Test the MeanGroupUnitAverager with various inputs."""
    ua = MeanGroupUnitAverager(focus_function, ind_estimates)
    ua.fit(target_id=target_id)
    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."


# Test data for fixed-N optimal averaging
fixed_n_test_data = [
    # Same unit data (expect equal weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        0,
        np.array([0.5, 0.5]),
        1,
    ),
    # Other unit has crazy variance (expect individual weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[10e10, 0], [0, 10e10]])]),
        0,
        np.array([1, 0]),
        1,
    ),
    # Other unit has crazy bias
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([10e10, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        0,
        np.array([1, 0]),
        1,
    ),
    # Target has crazy variance, other is biased
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([2, 1])]),
        np.array([np.array([[10e10, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        0,
        np.array([0, 1]),
        2,
    ),
    # Meaningful averaging with two units
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 1]])]),
        0,
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
        0,
        np.ones((10, 1)) / 10,
        1,
    ),
    # Other nine units are crazy
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0, 0, 0, 0])),
        np.ones((10, 5)),
        np.stack([np.eye(5), *np.array([np.eye(5) * 10e10] * 9)]),
        0,
        np.stack([np.array(1), *np.zeros(9)]),
        1,
    ),
    # Nonlinear focus function
    # Dictionary inputs with string keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {"a": np.array([1, 1]), "b": np.array([1, 1])},
        {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
        "a",
        np.array([0.5, 0.5]),
        1,
    ),
    # Dictionary inputs with int keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {1: np.array([1, 1]), 3: np.array([1, 1])},
        {1: np.array([[1, 0], [0, 1]]), 3: np.array([[1, 0], [0, 1]])},
        3,
        np.array([0.5, 0.5]),
        1,
    ), 
    # Dictionary inputs with string keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {"a": np.array([1, 1]), "b": np.array([1, 1])},
        {"b": np.array([[1, 0], [0, 1]]), "a": np.array([[1, 0], [0, 1]])},
        "a",
        np.array([0.5, 0.5]),
        1,
    ),
]
fixed_n_test_ids = [
    "2 units: identical",
    "2 units: non-target with crazy variance",
    "2 units: non-target with crazy bias",
    "2 units: target has crazy variance, other is biased",
    "2 units: reasonable",
    "10 units: identical",
    "10 units: non-targets have crazy variance",
    "2 units: identical dict inputs with str keys",
    "2 units: identical dict inputs with int keys",
    "2 units: identical dict inputs with keys in different order",
]


@pytest.mark.parametrize(
    "focus_function, ind_estimates, ind_covar_ests, "
    "target_id, expected_weights, expected_estimate",
    fixed_n_test_data,
    ids=fixed_n_test_ids,
)
def test_fixed_n_averaging(
    focus_function,
    ind_estimates,
    ind_covar_ests,
    target_id,
    expected_weights,
    expected_estimate,
):
    """Test the fixed-N regime of OptimalUnitAverager"""
    ua = OptimalUnitAverager(focus_function, ind_estimates, ind_covar_ests)
    ua.fit(target_id=target_id)
    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."


# Test data for testing large-n regime
large_n_test_data = [
    # Same unit data (expect equal weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([1.0, 1]), np.array([1, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([True, True]),
        0,
        np.array([0.5, 0.5]),
        1,
    ),
    # All units restricted (expect equal weights)
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([-1000, 1]), np.array([1000, 1])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
        np.array([False, False]),
        0,
        np.array([0.5, 0.5]),
        0,
    ),
    # Dictionary inputs with string keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {"a": np.array([1, 1]), "b": np.array([1, 1])},
        {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
        {"a": False, "b": False},
        "a",
        np.array([0.5, 0.5]),
        1,
    ),
    # Dictionary inputs with int keys
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {1: np.array([1, 1]), 3: np.array([1, 1])},
        {1: np.array([[1, 0], [0, 1]]), 3: np.array([[1, 0], [0, 1]])},
        {1: False, 3: False},
        3,
        np.array([0.5, 0.5]),
        1,
    ),
    # Dictionary inputs with with different orders
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        {"a": np.array([1, 1]), "b": np.array([1, 1])},
        {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
        {"b": False, "a": False},
        "a",
        np.array([0.5, 0.5]),
        1,
    ),
]
large_n_test_ids = [
    "Large-N with all units unrestricted",
    "Large-N with all units restricted",
    "Large-N with all units restricted and dict inputs (str keys)",
    "Large-N with all units restricted and dict inputs (int keys)",
    "Large-N with all units restricted, dict inputs with different orders)",
]


@pytest.mark.parametrize(
    "focus_function, ind_estimates, "
    "ind_covar_ests, unrestricted_units_bool, "
    "target_id, "
    "expected_weights, expected_estimate",
    large_n_test_data,
    ids=large_n_test_ids,
)
def test_large_n_averaging(
    focus_function,
    ind_estimates,
    ind_covar_ests,
    unrestricted_units_bool,
    target_id,
    expected_weights,
    expected_estimate,
):
    """Test the large-N regime of OptimalUnitAverager"""
    ua = OptimalUnitAverager(
        focus_function,
        ind_estimates,
        ind_covar_ests,
        unrestricted_units_bool,
    )
    ua.fit(target_id=target_id)
    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."
