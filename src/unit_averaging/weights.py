from abc import ABC, abstractmethod

import numpy as np

from .focus_function import FocusFunction


class UnitAveragingWeights(ABC):
    """Abstract base class for unit averaging weights."""

    @abstractmethod
    def compute_weights(
        self, focus_function: FocusFunction, ind_estimates, ind_covar_ests, target_id
    ) -> np.ndarray | list | dict:
        """Compute the weights for unit averaging."""
        pass


class IndividualWeights(UnitAveragingWeights):
    """Assigns all weight to the target unit"""
    def compute_weights(
        self, focus_function: FocusFunction, ind_estimates, ind_covar_ests, target_id
    ):
        num_units = len(ind_estimates)
        weights = np.zeros(num_units)
        weights[target_id] = 1.0
        return weights
