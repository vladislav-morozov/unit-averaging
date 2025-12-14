# `unit_averaging`

**Optimal Estimation for Heterogeneous Data**
 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12+](https://img.shields.io/badge/python-3.12|3.13|3.14-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://github.com/vladislav-morozov/unit-averaging/actions/workflows/build-deploy-docs.yml/badge.svg)](https://vladislav-morozov.github.io/unit-averaging/) [![codecov](https://codecov.io/gh/vladislav-morozov/unit-averaging/branch/develop/graph/badge.svg?token=E3ONL6JLYX)](https://codecov.io/gh/vladislav-morozov/unit-averaging)

 
<table>
  <tr>
    <td style="width:70%; vertical-align:top;"> 
      <p>
        A Python package for estimating unit-specific parameters in heterogeneous data settings (panel data and meta-analysis). Implements unit averaging: an ensemble method that efficiently combines information across multiple units (e.g., countries, firms, or studies) while accounting for their individual differences.
      </p>
    </td>
    <td style="width:30%; vertical-align:top;">
      <img src="docs/source/_static/logo_transp.png" alt="Logo" width="100%" />
    </td>
  </tr>
</table>
 

## Key Features

- **Optimal weighting**: Automatically computes weights that balance bias and variance
- **Flexible workflow**: Works with panel data, meta-analysis, and other heterogeneous datasets
- **Versatility**: Can be used on top of various standard estimation packages
- **Customizable**: Implement your own weighting schemes by subclassing base classes
- **Theoretically grounded**: Based on statistical theory with proven performance

## Installation

Install the package with pip:

```bash
pip install unit_averaging
```

## Documentation

- **[Tutorials](https://vladislav-morozov.github.io/unit-averaging/tutorials/)**: Step-by-step guides to get started
- **[API Reference](https://vladislav-morozov.github.io/unit-averaging/reference/)**: Detailed documentation for all classes/functions
- **[Theory](https://vladislav-morozov.github.io/unit-averaging/theory/)**: Mathematical foundations of unit averaging
- **[Original Paper](https://arxiv.org/abs/2210.14205)**: Complete theoretical treatment


## Quick Start

The core workflow of the package in a small synthetic example: 
```python
import numpy as np
from unit_averaging import OptimalUnitAverager, InlineFocusFunction

# Example: Forecasting with a linear focus function
x_value = 1.0  # Your covariate (e.g., lagged value)
estimates = {   # Dict of unit-specific coefficient estimates
    "unit1": np.array([0.5, 0.3]),
    "unit2": np.array([0.7, 0.1])
}
covariances = { # Dict of unit-specific covariance matrices
    "unit1": np.array([[0.1, 0.0], [0.0, 0.1]]),
    "unit2": np.array([[0.1, 0.0], [0.0, 0.1]])
}

# Define focus function: e.g., μ(θ) = θ₀ + θ₁ * x
focus = InlineFocusFunction(
    focus_function=lambda coef: coef[0] + coef[1] * x_value,
    gradient=lambda coef: np.array([1, x_value])
)

# Create and fit averager
averager = OptimalUnitAverager(
    focus_function=focus,
    ind_estimates=estimates,
    ind_covar_ests=covariances
)
averager.fit(target_id="unit1")
```


## Citation

If you use `unit_averaging` in your research, please cite:

> Brownlees C. and Morozov V. (2025). *Unit Averaging for Heterogeneous Panels*. Journal of Business & Economic Statistics. doi: 10.1080/07350015.2025.2584579.

```bibtex
@article{Brownlees2024UnitAveragingHeterogeneous, 
	author = {Brownlees, Christian and Morozov, Vladislav},
	title = {{Unit Averaging for Heterogeneous Panels}},
	journal = {Journal of Business \& Economic Statistics}, 
	year = {2025}, 
	doi = {10.1080/07350015.2025.2584579},  
}
```

## Support

For questions, issues, or contributions:
- Report bugs on our [GitHub issues](https://github.com/vladislav-morozov/unit-averaging/issues)
- Contribute via [pull requests](https://github.com/vladislav-morozov/unit-averaging/pulls)
 