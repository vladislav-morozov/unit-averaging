from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class FocusFunction(ABC):
    """Class to encapsulate the focus function and its gradient."""

    @abstractmethod
    def focus_function(
        self,
        ind_estimate: float | np.floating[Any] | np.ndarray,
    ) -> float | np.floating[Any]:
        """Compute the focus function for a given estimate

        Args:
            ind_estimate (float | np.floating[Any] | np.ndarray): individual
                specific estimates (theta in docs and paper notation)

        Returns:
            float | np.floating[Any]: scalar result of applying the focus func
        """
        pass

    @abstractmethod
    def gradient(
        self,
        ind_estimate: float | np.floating[Any] | np.ndarray,
    ) -> np.ndarray:
        """Compute the gradient of the focus function for a given estimate

        Args:
            ind_estimate (float | np.floating[Any] | np.ndarray): individual
                specific estimates (theta in docs and paper notation)

        Returns:
            np.ndarray: vector of first derivatives of focus_functions with
                respect to ind_estimates
        """
        pass


class InlineFocusFunction(FocusFunction):
    def __init__(
        self,
        focus_function: Callable[
            [float | np.floating[Any] | np.ndarray], float | np.floating[Any]
        ],
        gradient: Callable[[float | np.floating[Any] | np.ndarray], np.ndarray],
    ):
        """Create focus function by passing function and its gradient directly

        Args:
            focus_function (Callable[ [float  |  np.floating[Any]  |  np.ndarray],
                                      float  |  np.floating[Any] ]):
                focus function (mu), the target parameter
            gradient (Callable[[float  |  np.floating[Any]  |  np.ndarray],
                                np.ndarray]):
                gradient of focus function
        """
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
