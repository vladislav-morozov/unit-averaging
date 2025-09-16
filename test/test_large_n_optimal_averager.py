import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, OptimalUnitAverager


@pytest.fixture
def first_coord_focus_function():
    return InlineFocusFunction(lambda x: x[0], lambda x: np.eye(len(x))[0])


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, unrestricted_units_bool, "
    "target_id, expected_weights, expected_estimate",
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
        (
            np.array([np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]),
            np.array([np.eye(2), np.eye(2), np.eye(2)]),
            np.array([True, False, False]),
            0,
            np.array([0, 0.5, 0.5]),
            0,
        ),
        (
            np.array([np.array([0, 1]), np.array([0, 4]), np.array([0, -15])]),
            np.array([np.eye(2), np.eye(2), np.eye(2)]),
            np.array([True, False, False]),
            0,
            np.array([0, 0.5, 0.5]),
            0,
        ),
        (
            np.array([np.array([0, 1])] * 10000),
            np.array([np.eye(2)] * 10000),
            np.arange(10000) == 0,
            0,
            np.concatenate(([0], np.full(9999, 1 / 9999))),
            0,
        ),
    ],
    ids=[
        "Large-N with all units unrestricted",
        "Large-N with all units restricted",
        "Large-N with some units unrestricted",
        "Large-N with irrelevant coordinate differences",
        "Stein-like with many identical restricted units",
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
    assert np.allclose(ua.weights, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, unrestricted_units_bool, "
    "target_id, expected_weights, expected_estimate",
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
    assert np.allclose(ua.weights, expected_weights, rtol=1e-03) and np.allclose(
        ua.estimate, expected_estimate, rtol=1e-03
    )


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, unrestricted_units_bool, expected_error_message",
    [
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "c": np.array([[1, 0], [0, 1]])},
            None,
            "Keys of estimates and covariances do not match.",
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
            {"a": False, "c": False},
            "Keys of estimates and unrestricted units do not match.",
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "c": np.array([[1, 0], [0, 1]])},
            {"a": False, "d": False},
            "Keys of estimates and covariances do not match.",
        ),
    ],
    ids=[
        "covariances_keys_mismatch",
        "unrestricted_units_keys_mismatch",
        "covariances_and_unrestricted_units_keys_mismatch",
    ],
)
def test_optimal_averager_key_mismatch_errors(
    first_coord_focus_function,
    ind_estimates,
    ind_covar_ests,
    unrestricted_units_bool,
    expected_error_message,
):
    """Test that OptimalUnitAverager raises ValueError for key mismatches."""
    with pytest.raises(ValueError, match=expected_error_message):
        OptimalUnitAverager(
            first_coord_focus_function,
            ind_estimates,
            ind_covar_ests,
            unrestricted_units_bool,
        )


@pytest.mark.parametrize(
    "ind_estimates, ind_covar_ests, unrestricted_units_bool",
    [
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            None,
        ),
        (
            np.array([np.array([1, 1]), np.array([1, 1])]),
            {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
            None,
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            {"a": np.array([[1, 0], [0, 1]]), "b": np.array([[1, 0], [0, 1]])},
            np.array([True, False]),
        ),
        (
            {"a": np.array([1, 1]), "b": np.array([1, 1])},
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            {"a": True, "b": False},
        ),
    ],
    ids=[
        "estimates_dict_covar_array",
        "estimates_array_covar_dict",
        "estimates_dict_covar_dict_unrestricted_array",
        "estimates_dict_covar_array_unrestricted_dict",
    ],
)
def test_optimal_averager_type_mismatch_errors(
    first_coord_focus_function, ind_estimates, ind_covar_ests, unrestricted_units_bool
):
    """Test that OptimalUnitAverager raises TypeError for mismatched input types."""
    with pytest.raises(
        TypeError,
        match="If any input is a dictionary, all inputs must be dictionaries.",
    ):
        OptimalUnitAverager(
            first_coord_focus_function,
            ind_estimates,
            ind_covar_ests,
            unrestricted_units_bool,
        )
