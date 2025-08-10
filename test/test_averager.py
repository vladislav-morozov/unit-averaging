import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, UnitAverager


@pytest.mark.parametrize(
    "weight_scheme, ind_estimates, ind_covar_ests, expected",
    [
        ("individual", [0, 0], [1, 1], np.array([1, 0])),
        ("mean_group", [0, 0], [1, 1], np.array([1 / 2, 1 / 2])),
    ],
)
def test_weight_list_scalars(
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected,
):
    """Test correct weight computation from lists of scalars"""
    # Create focus function
    identity = InlineFocusFunction(
        lambda x: x,
        lambda x: 1,
    )
    # Create and fit averager
    ua = UnitAverager(identity, weight_scheme, ind_estimates, ind_covar_ests)
    ua.fit(target_id=0)
    assert np.array_equal(ua.weights_, expected)


@pytest.mark.parametrize(
    "weight_scheme, ind_estimates, ind_covar_ests, expected",
    [
        ("individual", [0.5, 0], [1, 1], 0.5),
        ("mean_group", [0.5, 0], [1, 1], 0.25),
    ],
)
def test_average_list_scalars(
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected,
):
    """Test correct averaging from lists of scalars"""
    # Create focus function
    identity = InlineFocusFunction(
        lambda x: x,
        lambda x: 1,
    )
    # Create and fit averager
    ua = UnitAverager(identity, weight_scheme, ind_estimates, ind_covar_ests)
    ua.fit(target_id=0)
    assert ua.estimate_ == expected


@pytest.mark.parametrize(
    "weight_scheme, ind_estimates, ind_covar_ests, expected",
    [
        (
            "individual",
            [np.array([3, 2]), np.array([4, 5])],
            [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
            3,
        ),
        (
            "mean_group",
            [np.array([3, 2]), np.array([4, 5])],
            [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
            3.5,
        ),
    ],
)
def test_average_list_arrays(
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected,
):
    """Test correct averaging from lists of NumPy arrays"""
    # Create focus function
    first_coord = InlineFocusFunction(
        lambda x: x[0],
        lambda x: np.array([1, 0]),
    )
    # Create and fit averager
    ua = UnitAverager(first_coord, weight_scheme, ind_estimates, ind_covar_ests)
    ua.fit(target_id=0)
    assert ua.estimate_ == expected


@pytest.mark.parametrize(
    "weight_scheme, ind_estimates, ind_covar_ests, expected",
    [
        (
            "individual",
            np.array([np.array([3, 2]), np.array([4, 5])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            3,
        ),
        (
            "mean_group",
            np.array([np.array([3, 2]), np.array([4, 5])]),
            np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
            3.5,
        ),
    ],
)
def test_average_array_arrays(
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected,
):
    """Test averaging with array of arrays as input"""
    # Create focus function
    first_coord = InlineFocusFunction(
        lambda x: x[0],
        lambda x: np.array([1, 0]),
    )
    # Create and fit averager
    ua = UnitAverager(first_coord, weight_scheme, ind_estimates, ind_covar_ests)
    ua.fit(target_id=0)
    assert ua.estimate_ == expected
