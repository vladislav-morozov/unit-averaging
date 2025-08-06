from typing import Literal

import numpy as np

from unit_averaging.focus_function import FocusFunction
from unit_averaging.weights import IndividualWeights, UnitAveragingWeights


class UnitAverager:
    """A class for performing unit averaging"""

    def __init__(
        self,
        focus_function: FocusFunction,
        weight_scheme: Literal["individual"] | UnitAveragingWeights,
        ind_estimates: list | np.ndarray,
        ind_covar_ests: list | np.ndarray | None = None,
    ):
        """
        Initialize a UnitAverager with given focus, weight scheme, and data.

        Args:
            focus_function (FocusFunction): An instance of a class encapsulating
                the focus function and its gradient.
            weight_scheme (str): Specifies the weighting scheme to be used.
                It can be a string indicating an implemented scheme (e.g.,
                "individual") or an instance of custom UnitAveragingWeights.
            ind_estimates (list | np.ndarray): sequence of individual parameter
                estimates (thetas in docs and paper). These will be used as
                arguments to focus_function
            ind_covar_ests (list | np.ndarray | None): sequence of covariances
                of individual parameter estimates (V in docs and paper). Optional
                when not using "optimal" weights.
        """
        self.focus_function = focus_function
        self.ind_estimates = ind_estimates
        self.ind_covar_ests = ind_covar_ests

        if isinstance(weight_scheme, str):
            self.weight_scheme = self._init_weight_scheme(weight_scheme)
        elif isinstance(weight_scheme, UnitAveragingWeights):
            self.weight_scheme = weight_scheme
        else:
            raise TypeError(
                "weight_scheme must be an allowed string "
                "or instance of UnitAveragingWeights"
            )

    def fit(
        self,
        target_id: int,
    ):
        """Compute the unit averaging weights and estimate

        Args:
            target_id (int): ID of target unit
        """
        # Compute and store weights
        self.weights_ = self.weight_scheme.compute_weights(
            self.focus_function,
            self.ind_estimates,
            self.ind_covar_ests,
            target_id=target_id,
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
                return IndividualWeights()
            case _:
                raise KeyError(f"Weight scheme '{scheme_name}' not found")
