import pytest
import numpy as np
from unit_averaging import InlineFocusFunction, IndividualUnitAverager


@pytest.fixture
def identity_focus_function():
    return InlineFocusFunction(lambda x: x, lambda x: 1)


@pytest.fixture
def first_coord_focus_function():
    return InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0]))


@pytest.mark.parametrize(
    "ind_estimates, target_id, expected_weights, expected_estimate",
    [
        ([0, 1], 0, np.array([1, 0]), 0),
        (np.array([0, 1]), 0, np.array([1, 0]), 0),
        ({0: 0, 1: 1}, 0, np.array([1, 0]), 0),
        ({"a": 0, "b": 1}, "b", np.array([0, 1]), 1),
    ],
    ids=[
        "list_of_scalars",
        "array_of_scalars",
        "dict_of_scalars_int_keys",
        "dict_of_scalars_str_keys",
    ],
)
def test_individual_averager_with_scalars(
    identity_focus_function,
    ind_estimates,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = IndividualUnitAverager(identity_focus_function, ind_estimates)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, target_id, expected_weights, expected_estimate",
    [
        ([np.array([3, 2]), np.array([4, 5])], 0, np.array([1, 0]), 3),
        ([[3, 2], [4, 5]], 0, np.array([1, 0]), 3),
    ],
    ids=[
        "list_of_arrays",
        "list_of_lists",
    ],
)
def test_individual_averager_with_lists(
    first_coord_focus_function,
    ind_estimates,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = IndividualUnitAverager(first_coord_focus_function, ind_estimates)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, target_id, expected_weights, expected_estimate",
    [
        (np.array([np.array([3, 2]), np.array([4, 5])]), 0, np.array([1, 0]), 3),
        (np.array([[3, 2], [4, 5]]), 0, np.array([1, 0]), 3),
    ],
    ids=[
        "array_of_arrays",
        "array_of_lists",
    ],
)
def test_individual_averager_with_arrays(
    first_coord_focus_function,
    ind_estimates,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = IndividualUnitAverager(first_coord_focus_function, ind_estimates)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, target_id, expected_weights, expected_estimate",
    [
        ({"a": np.array([3, 2]), "b": np.array([4, 5])}, "a", np.array([1, 0]), 3),
        ({1: np.array([3, 2]), 3: np.array([4, 5])}, 3, np.array([0, 1]), 4),
        ({"a": [4, 2], "b": [-5, 1]}, "b", np.array([0, 1]), -5),
    ],
    ids=[
        "dict_of_arrays_str_keys",
        "dict_of_arrays_int_keys",
        "dict_of_lists_str_keys",
    ],
)
def test_individual_averager_with_dicts(
    first_coord_focus_function,
    ind_estimates,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = IndividualUnitAverager(first_coord_focus_function, ind_estimates)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    )
