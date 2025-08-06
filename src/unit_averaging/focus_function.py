from collections.abc import Callable

import numpy as np


class FocusFunction:
    """Class to encapsulate the focus function and its gradient."""

    def __init__(
        self,
        focus_function: Callable[[np.ndarray], float | int],
        gradient: Callable[[np.ndarray], np.ndarray],
    ):
        """
        Initialize the FocusFunction with a focus function and its gradient.

        Args:
            focus_function (callable): the focus function that computes the
                parameter of interest. mu in notation of docs and paper.
            gradient (callable): The gradient function of the focus function.
        """
        self.focus_function = focus_function
        self.gradient = gradient
