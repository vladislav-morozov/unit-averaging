from collections.abc import Callable
from typing import Literal

import numpy as np

from unit_averaging.focus_function import FocusFunction
from unit_averaging.weights import (
    WeightFunction,
    individual_weights,
    mean_group_weights,
    optimal_weights,
)


class UnitAverager:
    """A class for performing unit averaging"""

    def __init__(
        self,
        focus_function: FocusFunction,
        weight_scheme: Literal["individual", "mean_group", "optimal"] | WeightFunction,
        ind_estimates: list | np.ndarray,
        ind_covar_ests: list | np.ndarray | None = None,
    ):
        """
        Initialize a UnitAverager with given focus, weight scheme, and data.

        Args:
            focus_function (FocusFunction): An instance of a class encapsulating
                the focus function and its gradient.
            weight_scheme (str | WeightFunction):  A predefined weight scheme
                identifier or a custom weight function.
            ind_estimates (list | np.ndarray): sequence of individual parameter
                estimates (thetas in docs and paper). These will be used as
                arguments to focus_function
            ind_covar_ests (list | np.ndarray | None): sequence of covariances
                of individual parameter estimates (V in docs and paper). Optional
                when not using "optimal" weights.
        """
        self.focus_function = focus_function
        self.ind_estimates = np.array(ind_estimates)
        self.ind_covar_ests = np.array(ind_covar_ests)
        self.weights = None
        self.estimates_ = None

        if isinstance(weight_scheme, str):
            self.weight_function = self._init_weight_scheme(weight_scheme)
        elif isinstance(weight_scheme, Callable):
            self.weight_function = weight_scheme
        else:
            raise TypeError(
                "weight_scheme must be an allowed string or a suitable weight function"
            )

    def fit(
        self,
        target_id: int,
        **kwargs,
    ):
        """Compute the unit averaging weights and estimate

        Args:
            target_id (int): ID of target unit
        """
        # Compute and store weights
        self.weights_ = self.weight_function(
            self.focus_function,
            target_id,
            self.ind_estimates,
            self.ind_covar_ests,
            **kwargs,
        )

        # Compute appropriate unit averaging estimate
        self.estimate_ = self.average(
            self.focus_function,
        )

    def average(self, focus_function: FocusFunction | None = None) -> float:
        """Perform unit averaging with the fitted weights

        Args:
            focus_function (FocusFunction | None): focus function to use in
                computing the average. mu in notation of docs and paper. If
                None, defaults to self.focus_function

        Returns:
            float: unit averaging estimate
        """
        # Check if weights have been fitted
        if self.weights_ is None:
            raise ValueError(
                "Weights have not been fitted. Call the 'fit' method first."
            )

        # If no new focus function is supplied, use the base one
        if focus_function is None:
            focus_function = self.focus_function

        # Compute unit weighed average
        weighted_ind_estimates = [
            weight * focus_function.focus_function(ind_est)
            for ind_est, weight in zip(self.ind_estimates, self.weights_, strict=True)
        ]
        return sum(weighted_ind_estimates)

    def _init_weight_scheme(self, scheme_name: str):
        """
        Initialize an appropriate implemented weight scheme based on string input

        Args:
            scheme_name (str): string name of an implemented weigth class
        """
        match scheme_name:
            case "individual":
                return individual_weights
            case "mean_group":
                return mean_group_weights
            case "optimal":
                return optimal_weights
            case _:
                raise KeyError(f"Weight scheme '{scheme_name}' not found")
