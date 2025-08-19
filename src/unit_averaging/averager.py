from abc import ABC, abstractmethod 

import numpy as np

from unit_averaging.focus_function import FocusFunction


class BaseUnitAverager(ABC):
    """Class to encapsulate fit and averaging behavior"""

    def __init__(
        self,
        focus_function: FocusFunction,
        ind_estimates: np.ndarray | list,
    ):
        self.focus_function = focus_function
        self.ind_estimates = np.array(ind_estimates)
        self.weights = None
        self.estimate_ = None

    def fit(self, target_id: int):
        """Compute the unit averaging weights and the averaging estimator

        Args:
            target_id (int): ID of target unit
        """
        self.target_id_ = target_id

        # Compute weights
        self._compute_weights()

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
            raise TypeError(
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

    @abstractmethod
    def _compute_weights(self):
        """Compute unit averaging weights"""


class IndividualUnitAverager(BaseUnitAverager):
    """Unit averaging scheme that assigns all weight to the target unit."""

    def _compute_weights(self):
        num_units = len(self.ind_estimates)
        weights = np.zeros(num_units)
        weights[self.target_id_] = 1.0
        self.weights_ = weights


class MeanGroupUnitAverager(BaseUnitAverager):
    """Unit averaging scheme that assigns equal weights to all units"""

    def _compute_weights(self):
        num_units = len(self.ind_estimates)
        weights = np.ones(num_units) / num_units
        self.weights_ = weights
