import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, OptimalUnitAverager


@pytest.fixture
def first_coord_focus_function():
    return InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0]))


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, unrestricted_units_bool, target_id, expected_weights, expected_estimate",
    [
        (
            np.array([np.array([1.0, 1]), np.array([1, 1])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            np.array([True, True]),
            0,
            np.array([0.5, 0.5]),
            1,
        ),
        (
            np.array([np.array([-1000, 1]), np.array([1000, 1])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            np.array([False, False]),
            0,
            np.array([0.5, 0.5]),
            0,
        ),
    ],
    ids=[
        "Large-N with all units unrestricted",
        "Large-N with all units restricted",
    ],
)
def test_large_n_averaging_with_arrays(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    unrestricted_units_bool,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = OptimalUnitAverager(
        first_coord_focus_function,
        ind_estimates,
        ind_covar_ests,
        unrestricted_units_bool,
    )
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, unrestricted_units_bool, target_id, expected_weights, expected_estimate",
    [
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
            {"a": False, "b": False},
            "a",
            np.array([0.5, 0.5]),
            1,
        ),
        (
            {1: np.array([1, 1]), 3: np.array([1, 1])},
            {1: np.array([[1, 0], [0, 1]]), 3: np.array([[1, 0], [0, 1]])},
            {1: False, 3: False},
            3,
            np.array([0.5, 0.5]),
            1,
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
            {"b": False, "a": False},
            "a",
            np.array([0.5, 0.5]),
            1,
        ),
    ],
    ids=[
        "Large-N with all units restricted and dict inputs (str keys)",
        "Large-N with all units restricted and dict inputs (int keys)",
        "Large-N with all units restricted, dict inputs with different orders",
    ],
)
def test_large_n_averaging_with_dicts(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    unrestricted_units_bool,
    target_id,
    expected_weights,
    expected_estimate,
):
    ua = OptimalUnitAverager(
        first_coord_focus_function,
        ind_estimates,
        ind_covar_ests,
        unrestricted_units_bool,
    )
    ua.fit(target_id=target_id)
    assert np.allclose(ua.weights_, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate_, expected_estimate, rtol=1e-03
    )
