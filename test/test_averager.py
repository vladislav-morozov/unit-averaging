import numpy as np
import pytest

from unit_averaging import InlineFocusFunction, UnitAverager


# Fixture for creating a focus function
@pytest.fixture
def identity_focus_function():
    return InlineFocusFunction(lambda x: x, lambda x: 1)


@pytest.fixture
def first_coord_focus_function():
    return InlineFocusFunction(lambda x: x[0], lambda x: np.array([1, 0]))


# Parametrize the test data
test_data = [
    # Test data for weight computation
    ("individual", [0, 0], [1, 1], np.array([1, 0])),
    ("mean_group", [0, 0], [1, 1], np.array([1 / 2, 1 / 2])),
    # Test data for averaging
    ("individual", [0.5, 0], [1, 1], 0.5),
    ("mean_group", [0.5, 0], [1, 1], 0.25),
    # Test data for array inputs
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
]


@pytest.mark.parametrize(
    "weight_scheme, ind_estimates, ind_covar_ests, expected", test_data
)
def test_unit_averager(
    weight_scheme,
    ind_estimates,
    ind_covar_ests,
    expected,
    identity_focus_function,
    first_coord_focus_function,
):
    # Choose the appropriate focus function based on the test data
    focus_function = (
        first_coord_focus_function
        if isinstance(ind_estimates[0], np.ndarray)
        else identity_focus_function
    )

    # Create and fit averager
    ua = UnitAverager(focus_function, weight_scheme, ind_estimates, ind_covar_ests)
    ua.fit(target_id=0)

    # Check if the test is for weights or averaging
    if isinstance(expected, np.ndarray):
        assert np.array_equal(ua.weights_, expected)
    else:
        assert ua.estimate_ == expected
