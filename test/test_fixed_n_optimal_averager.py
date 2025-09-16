import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, OptimalUnitAverager


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
            np.array([0.5, 0.5]),
            1,
        ),
        (
            np.array([np.array([1.0, 1]), np.array([10e10, 1])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            0,
            np.array([1, 0]),
            1,
        ),
        (
            np.array([np.array([1.0, 1]), np.array([2, 1])]),
            np.array([np.array([[10e10, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            0,
            np.array([0, 1]),
            2,
        ),
        (
            np.array([np.array([1.0, 1]), np.array([1, 1])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 1]])]),
            0,
            np.array([2 / 3, 1 / 3]),
            1,
        ),
    ],
    ids=[
        "2 units: identical",
        "2 units: non-target with crazy bias",
        "2 units: target has crazy variance, other is biased",
        "2 units: reasonable",
    ],
)
def test_fixed_n_averaging_with_arrays(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = OptimalUnitAverager(first_coord_focus_function, ind_estimates, ind_covar_ests)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, target_id, expected_weights, expected_estimate",
    [
        (
            np.ones((10, 5)),
            np.array([np.eye(5)] * 10),
            0,
            np.ones((10, 1)) / 10,
            1,
        ),
        (
            np.ones((10, 5)),
            np.stack([np.eye(5), *np.array([np.eye(5) * 10e10] * 9)]),
            0,
            np.stack([np.array(1), *np.zeros(9)]),
            1,
        ),
    ],
    ids=[
        "10 units: identical",
        "10 units: non-targets have crazy variance",
    ],
)
def test_fixed_n_averaging_with_large_arrays(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = OptimalUnitAverager(first_coord_focus_function, ind_estimates, ind_covar_ests)
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
            np.array([0.5, 0.5]),
            1,
        ),
        (
            {1: np.array([1, 1]), 3: np.array([1, 1])},
            {1: np.array([[1, 0], [0, 1]]), 3: np.array([[1, 0], [0, 1]])},
            3,
            np.array([0.5, 0.5]),
            1,
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"b": np.array([[1, 0], [0, 1]]), "a": np.array([[1, 0], [0, 1]])},
            "a",
            np.array([0.5, 0.5]),
            1,
        ),
    ],
    ids=[
        "2 units: identical dict inputs with str keys",
        "2 units: identical dict inputs with int keys",
        "2 units: identical dict inputs with keys in different order",
    ],
)
def test_fixed_n_averaging_with_dicts(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = OptimalUnitAverager(first_coord_focus_function, ind_estimates, ind_covar_ests)
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate, expected_estimate, rtol=1e-03
    )
