import numpy as np

from unit_averaging import FocusFunction, UnitAverager


def test_weight_list_scalars():
    """Test correct weight computation from lists of scalars"""
    # Create focus function
    identity = FocusFunction(
        lambda x: x,
        lambda x: 1,
    )
    # Create averager
    ua = UnitAverager(identity, "individual", [0, 0], [1, 1])
    ua.fit(target_id=0)
    assert np.array_equal(ua.weights_, np.array([1, 0]))


def test_average_list_scalars():
    """Test correct averaging from lists of scalars"""
    # Create focus function
    identity = FocusFunction(
        lambda x: x,
        lambda x: 1,
    )
    # Create averager
    ua = UnitAverager(identity, "individual", [0.5, 0], [1, 1])
    ua.fit(target_id=0)
    assert ua.estimate_ == 0.5


def test_average_list_arrays():
    """Test correct averaging from lists of NumPy arrays"""
    # Create focus function
    first_coord = FocusFunction(
        lambda x: x[0],
        lambda x: np.array([1, 0]),
    )
    # Create averager
    ua = UnitAverager(
        first_coord,
        "individual",
        [np.array([3, 2]), np.array([4, 5])],
        [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
    )
    ua.fit(target_id=1)
    assert ua.estimate_ == 4


def test_average_array_arrays():
    """Test averaging with array of arrays as input"""
    # Create focus function
    first_coord = FocusFunction(
        lambda x: x[0],
        lambda x: np.array([1, 0]),
    )
    # Create averager
    ua = UnitAverager(
        first_coord,
        "individual",
        np.array([np.array([3, 2]), np.array([4, 5])]),
        np.array([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]),
    )
    ua.fit(target_id=1)
    assert ua.estimate_ == 4
