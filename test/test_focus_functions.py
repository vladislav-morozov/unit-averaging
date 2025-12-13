import numpy as np
import pytest

from unit_averaging import IdentityFocusFunction


@pytest.mark.parametrize(
    "ind_estimate",
    [
        ([2.0]),
        (np.array(2.0)),
        (np.array([2.0])),
        (np.array([[2.0]])),
        (2),
    ],
    ids=[
        "List with scalar",
        "0D array",
        "1D array with one element",
        "2D array with one element",
        "Scalar integer",
    ],
)
def test_identity_focus_function(
    ind_estimate,
):
    identity_focus = IdentityFocusFunction()

    assert np.allclose(
        identity_focus.focus_function(ind_estimate), ind_estimate, rtol=1e-05
    ) and np.allclose(identity_focus.gradient(ind_estimate), 1.0, rtol=1e-05)


@pytest.mark.parametrize(
    "ind_estimate",
    [
        ([2.0, 1.0]),
        (np.array([3, 4.0])),
        (np.array([[3, 4.0], [1, -1]])),
    ],
    ids=[
        "List of scalar",
        "Array of scalars",
        "Matrix",
    ],
)
def test_identity_focus_function_non_scalar_function(
    ind_estimate,
):
    identity_focus = IdentityFocusFunction()

    with pytest.raises(
        ValueError, match="Inputs to IdentityFocusFunction must be scalar"
    ):
        identity_focus.focus_function(ind_estimate)


@pytest.mark.parametrize(
    "ind_estimate",
    [
        ([2.0, 1.0]),
        (np.array([3, 4.0])),
        (np.array([[3, 4.0], [1, -1]])),
    ],
    ids=[
        "List of scalar",
        "Array of scalars",
        "Matrix",
    ],
)
def test_identity_focus_function_non_scalar_gradient(
    ind_estimate,
):
    identity_focus = IdentityFocusFunction()

    with pytest.raises(
        ValueError, match="Inputs to IdentityFocusFunction must be scalar"
    ):
        identity_focus.gradient(ind_estimate)
