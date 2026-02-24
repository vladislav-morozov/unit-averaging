import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, SteinUnitAverager


@pytest.fixture
def first_coord_focus_function():
    return InlineFocusFunction(lambda x: x[0], lambda x: np.eye(len(x))[0])


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, target_id, expected_weights, expected_estimate",
    [
        (
            np.array([np.array([1.0, 1]), np.array([1, 1])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            0,
            np.array([0, 1]),
            1,
        ),
        (
            np.array([np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]),
            np.array([np.eye(2), np.eye(2), np.eye(2)]),
            0,
            np.array([0, 0.5, 0.5]),
            0,
        ),
        (
            np.array([np.array([0, 1])] * 10000),
            np.array([np.eye(2)] * 10000),
            0,
            np.concatenate(([0], np.full(9999, 1 / 9999))),
            0,
        ),
    ],
    ids=["2 units: identical", "3 units: identical", "10000 units:"],
)
def test_stein_averaging_with_arrays(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = SteinUnitAverager(first_coord_focus_function, ind_estimates, ind_covar_ests)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, target_id, expected_weights, expected_estimate",
    [
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
            "a",
            np.array([0, 1]),
            1,
        ),
        (
            {1: np.array([1, 1]), 3: np.array([1, 1])},
            {1: np.array([[1, 0], [0, 1]]), 3: np.array([[1, 0], [0, 1]])},
            3,
            np.array([1, 0]),
            1,
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"b": np.array([[1, 0], [0, 1]]), "a": np.array([[1, 0], [0, 1]])},
            "a",
            np.array([0, 1]),
            1,
        ),
    ],
    ids=[
        "String keys",
        "Int keys",
        "Dict inputs with different orders",
    ],
)
def test_stein_averaging_with_dicts(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = SteinUnitAverager(first_coord_focus_function, ind_estimates, ind_covar_ests)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate, expected_estimate, rtol=1e-03
    )
