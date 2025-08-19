import numpy as np
import pytest

from unit_averaging import (
    InlineFocusFunction,
    IndividualUnitAverager,
    MeanGroupUnitAverager,
)


# Test data for IndividualUnitAverager
individual_test_data = [
    # Inputs: lists of scalars
    (
        InlineFocusFunction(lambda x: x, lambda x: 1),
        [0, 1],
        np.array([1, 0]),
        0,
    ),
    # Inputs: lists of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        [np.array([3, 2]), np.array([4, 5])],
        np.array([1, 0]),
        3,
    ),
    # Inputs: arrays of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([3, 2]), np.array([4, 5])]),
        np.array([1, 0]),
        3,
    ),
]

individual_test_ids = [
    "scalar_list_individual",
    "array_list_individual",
    "array_of_arrays_individual",
]

# Test data for MeanGroupUnitAverager
mean_group_test_data = [
    # Inputs: lists of scalars
    (
        InlineFocusFunction(lambda x: x, lambda x: 1),
        [0, 1],
        np.array([0.5, 0.5]),
        0.5,
    ),
    # Inputs: lists of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        [np.array([3, 2]), np.array([4, 5])],
        np.array([0.5, 0.5]),
        3.5,
    ),
    # Inputs: arrays of numpy arrays
    (
        InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0])),
        np.array([np.array([3, 2]), np.array([4, 5])]),
        np.array([0.5, 0.5]),
        3.5,
    ),
]

mean_group_test_ids = [
    "scalar_list_mean_group",
    "array_list_mean_group",
    "array_of_arrays_mean_group",
]


@pytest.mark.parametrize(
    "focus_function, ind_estimates, expected_weights, expected_estimate",
    individual_test_data,
    ids=individual_test_ids,
)
def test_individual_unit_averager(
    focus_function,
    ind_estimates,
    expected_weights,
    expected_estimate,
):
    """Test the IndividualUnitAverager with various inputs."""
    ua = IndividualUnitAverager(focus_function, ind_estimates)
    ua.fit(target_id=0)
    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."


@pytest.mark.parametrize(
    "focus_function, ind_estimates, expected_weights, expected_estimate",
    mean_group_test_data,
    ids=mean_group_test_ids,
)
def test_mean_group_unit_averager(
    focus_function,
    ind_estimates,
    expected_weights,
    expected_estimate,
):
    """Test the MeanGroupUnitAverager with various inputs."""
    ua = MeanGroupUnitAverager(focus_function, ind_estimates)
    ua.fit(target_id=0)
    # Check weights and estimates
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    ), "Weights or estimate do not match expected values."
