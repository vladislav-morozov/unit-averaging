from collections.abc import Callable


class FocusFunction:
    """Class to encapsulate the focus function and its gradient."""

    def __init__(
        self,
        focus_function: Callable,
        gradient: Callable,
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
