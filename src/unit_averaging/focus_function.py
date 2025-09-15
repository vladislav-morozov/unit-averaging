from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class BaseFocusFunction(ABC):
    """Abstract base class to encapsulate a focus function and its gradient.

    This class provides a framework for implementing focus functions and their
    gradients. A focus function represents a transformation of interest applied
    to parameter estimates. Subclasses must implement both the
    ``focus_function`` and ``gradient`` methods to define the specific
    transformation and its derivative.

    The focus function concept is central to unit averaging methods, particularly
    the optimal approach which relies on the function and its derivative.

    Attributes:
        None (abstract class)

    Examples:
        >>> from unit_averaging import BaseFocusFunction
        >>> import numpy as np
        >>> class LinearFocusFunction(BaseFocusFunction):
        ...     def focus_function(self, ind_estimate):
        ...         return ind_estimate[0] + ind_estimate[1]
        ...
        ...     def gradient(self, ind_estimate):
        ...         return np.array([1.0, 1.0])
        >>> # Create an instance of the custom focus function
        >>> focus_func = LinearFocusFunction()
        >>> # Apply the focus function to an estimate
        >>> estimate = np.array([2.0, 3.0])
        >>> result = focus_func.focus_function(estimate)
        >>> print(result)  # Output: 5.0
        >>> # Get the gradient at the estimate
        >>> grad = focus_func.gradient(estimate)
        >>> print(grad)  # Output: [1. 1.]
    """

    @abstractmethod
    def focus_function(
        self,
        ind_estimate: float | np.floating[Any] | np.ndarray,
    ) -> float | np.floating[Any]:
        """Compute the focus function for a given estimate.

        This method applies the focus function transformation to an individual
        estimate. The focus function defines the parameter of interest that the
        unit averaging process aims to estimate.

        Args:
            ind_estimate (float | np.floating[Any] | np.ndarray):
                Individual specific estimate. Can be a scalar or an array-like object.

        Returns:
            float | np.floating[Any]:
                Scalar result of applying the focus function to the input estimate.
        """
        pass

    @abstractmethod
    def gradient(
        self,
        ind_estimate: float | np.floating[Any] | np.ndarray,
    ) -> np.ndarray:
        """Compute the gradient of the focus function for a given estimate.

        This method calculates the gradient (vector of first derivatives) of the focus
        function with respect to the input estimate. The gradient is used in the
        optimization process to determine the optimal weights for unit averaging.

        Args:
            ind_estimate (float | np.floating[Any] | np.ndarray):
                Individual specific estimate. Can be a scalar or an array-like object.

        Returns:
            np.ndarray:
                Vector of first derivatives of the focus function with respect to
                the input estimate. The shape of the array should match the shape
                of the input estimate.
        """
        pass


class InlineFocusFunction(BaseFocusFunction):
    """Convenience class for creating focus functions using callable objects.

    This class allows you to create a focus function by directly passing
    callable objects for both the focus function and its gradient. It's
    useful when the focus function and its gradient are simple enough to be
    defined as lambda functions or other callable objects, eliminating the
    need to create a full subclass of ``BaseFocusFunction``.

    Args:
        focus_function (Callable): A callable that implements the focus function
            This function should take an individual estimate (scalar or array)
            and return a scalar result representing the parameter of interest.

            Signature: ``(float | np.ndarray) -> (float | np.ndarray)``.
        gradient (Callable):
            A callable that implements the gradient of the focus function with
            respect to the parameters.

            Signature: ``(float | np.ndarray) -> np.ndarray``.

    Attributes:
        _focus_function (Callable):
            The stored callable implementing the focus function.
        _gradient (Callable):
            The stored callable implementing the gradient of the focus function.

    Example:
        >>> from unit_averaging import InlineFocusFunction
        >>> import numpy as np
        >>> # Estimate
        >>> estimate = np.array([2.0, 3.0])
        >>> linear_focus = InlineFocusFunction(
        ...     lambda x: 2*x[0] + 3*x[1],  # Focus function: 2x0 + 3x1
        ...     lambda x: np.array([2.0, 3.0])  # Gradient: [2, 3]
        ... )
        >>> result = linear_focus.focus_function(np.array([1.0, 1.0]))
        >>> print(result)  # Output: 5.0
        >>> grad = linear_focus.gradient(np.array([1.0, 1.0]))
        >>> print(grad)  # Output: [2. 3.]
    """

    def __init__(
        self,
        focus_function: Callable[
            [float | np.floating[Any] | np.ndarray], float | np.floating[Any]
        ],
        gradient: Callable[[float | np.floating[Any] | np.ndarray], np.ndarray],
    ):
        self._focus_function = focus_function
        self._gradient = gradient

    def focus_function(
        self,
        ind_estimate: float | np.floating[Any] | np.ndarray,
    ) -> float | np.floating[Any]:
        return self._focus_function(ind_estimate)

    def gradient(
        self,
        ind_estimate: float | np.floating[Any] | np.ndarray,
    ) -> np.ndarray:
        return self._gradient(ind_estimate)
