import numpy as np

from .focus_function import FocusFunction
from .weights import IndividualWeights, UnitAveragingWeights


class UnitAverager:
    """Basic class"""

    def __init__(
        self,
        focus_function: FocusFunction,
        weight_scheme: str | UnitAveragingWeights,
        ind_estimates: list | np.ndarray,
        ind_covar_ests: list | np.ndarray,
    ):
        """
        Initialize

        Args:
            focus_function (_type_): _description_
            weight_scheme (str): _description_
            ind_estimates (list | np.ndarray): _description_
            ind_covar_ests (list | np.ndarray): _description_
            target_id (int): _description_
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
                "weight_scheme must be an allowed string or instance of UnitAveragingWeights"
            )

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

    def fit(
        self,
        target_id: int,
    ):
        """Fit

        Args:
            target_id (int): _description_
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
            target_id,
        )

    def average(self, focus_function: FocusFunction, target_id):
        """_summary_

        Args:
            focus_function (FocusFunction): _description_
            target_id (_type_): _description_
        """
        weighted_ind_estimates = [
            weight * focus_function.focus_function(ind_est)
            for ind_est, weight in zip(self.ind_estimates, self.weights_, strict=True)
        ]
        return sum(weighted_ind_estimates)
