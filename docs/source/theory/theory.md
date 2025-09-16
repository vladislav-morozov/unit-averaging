# About Unit Averaging

This page provides a brief overview of the essentials of unit averaging.

## Overview

The key nature of optimal unit averaging can be expressed in three sentences: 

:::{admonition} What is optimal unit averaging?
:class: admonition

**Optimal unit averaging** is an *ensemble* method for *efficiently* estimating unit- or study-specific parameters in heterogeneous panel data or meta-analysis settings. It uses weighted averages of all unit-specific parameter estimates. The weights can optimally trade off the increase in bias caused by using non-target units with the decrease in variance due to using more data.
 
:::


For example, imagine that you want to forecast the gross domestic product (GDP) in France for next year. You have data both on France and on other European countries in a panel dataset. You have three main options on what to do:

- Build your forecast using only French data. This approach has the least bias and the highest variance.
- Pool all the data and fit a single predictive model. This approach has the highest bias.
- Compromise and use the panel-wide information to optimally reduce the variance of the individual-only approach.

Unit averaging is an ensemble method that embodies this compromise. It may be viewed as generalized and flexible shrinkage approach.


## Focus Parameter and Goal

To describe the approach let $i=1, \dots, N$ be the different units in your data (countries, customers, firms; or studies in a meta-analysis). The data is *heterogeneous* in the sense that each unit $i$ has their own version of some parameter $\theta_i$.

We are interested in the *unit-specific* "focus parameter" for some fixed unit $i$: 

$$
\mu(\theta_i)
$$




The transformed parameters $\mu(\theta_i)$ may be purely *predictive* (GDP forecast, probability of quitting the service) or be *causal*/structural (individual treatment effects, multipliers, etc.).

The goal is to estimate $\mu(\theta_i)$ optimally in the sense of minimizing the mean squared error (MSE).

:::{admonition} Focus functions in `unit_averaging`
:class: admonition
Use [`InlineFocusFunction`](../reference/InlineFocusFunction.rst) for simple $\mu(\cdot)$ (e.g. lambda functions) or subclass [`BaseFocusFunction`](../reference/BaseFocusFunction.rst) for complex cases. See [Getting Started](../tutorials/plot_1_basics) for an example.
:::




## Approach



### Unit Averaging Estimator

The unit averaging approach proceeds in two steps:

1. Estimate each $\theta_i$ with the individual (or unit-specific) estimator $\hat{\theta}_i$. Compute $\mu(\hat{\theta}_i)$ for all $i$. 
2. Compute a weighted average of the individual estimators:

$$
\hat{\mu}(\mathbf{w}) = \sum_{i=1}^N w_i \mu(\hat{\theta}_i),
$$

where the weights $\mathbf{w} = (w_1, \dots, w_N)$ are non-negative and sum to 1.
 
Different weights define different schemes, and this package implements several universally-applicable ones, including MSE-optimal weights, and offers an interface for implementing custom ones.

### Why Averaging Makes Sense

To understand the intuition behind $\hat{\mu}(\mathbf{w})$, suppose that we can write the individual-specific true parameters $\theta_i$ as a sum of two pieces:

$$
\theta_i = \mathbb{E}[\theta_i] + \eta_i.
$$

Here 

- $\mathbb{E}[\theta_i]$ is the average value that is *common* to all units (e.g. broad global economic laws).
- $\eta_i$ is a mean-zero value that is *specific* to unit $i$ (e.g. France-specific dynamics).

Unit averaging is motivated by the fact that each unit carries information about $\mathbb{E}[\theta_i]$. Using data on non-target units may help reduce the uncertainty about $\mathbb{E}[\theta_i]$ and reduce the overall variance of the estimator. 

By averaging $\mu(\hat{\theta}_i)$ after estimation (not pooling data beforehand), unit averaging can exploit shared information across units without assuming a specific model form.


:::{admonition} Why not pool the data?
:class: admonition 

An alternative estimation approach is to pool the data on all units and estimate a single model. However, in dynamic and/or nonlinear contexts this approach typically suffers from biases that are hard to characterize and control. Already in simple settings (e.g. linear dynamic panel models), the bias may be arbitrarily large.

:::

### Optimal Weights


At the same time, adding information of other units creates a trade-off. While it may decrease the uncertainty regarding $\mathbb{E}[\theta_i]$, it may create bias by adding potentially irrelevant information about $\eta_i$.

Optimally exploiting this bias-variance trade-off leads to the MSE-optimal weights implemented by the [`OptimalUnitAverager`](../reference/OptimalUnitAverager.rst) class. Formally, these optimal weights solve a convex quadratic problem of the form

$$
\hat{\mathbf{w}} = \mathrm{argmin} \mathbf{w}'\mathbf{\hat{\Psi}}\mathbf{w},
$$

where the matrix $\mathbf{\hat{\Psi}}$ encodes the variances and estimated biases between the target unit and the other units. The weights $\hat{\mathbf{w}}$ are non-negative and sum to 1. Higher weights are assigned to units with lower variances and to units that are more similar to the target unit.




There are two main approaches to optimal unit averaging that differ in whether they use prior information:

- No priors: ("fixed-$N$" in the original paper): in this regime, each weight $w_i$ may freely vary between 0 and 1 (up to summing to 1). The algorithm is free to choose to which units it pays more attention.
- With priods ("large-$N$" in the original paper): the user splits the units into two categories: unrestricted and restricted units.
    - The weights of unrestricted units vary independently.
    - All restricted units receive equal weights.
    - The algorithm only chooses the weight of the restricted set as a whole.

    If the number of restricted units is large, the average of a large restricted set will closely approximate $\mathbb{E}[\theta_i]$, allowing for more efficient shrinkage.
 
The prior information for selecting the unrestricted units may be obtained on the basis of domain knowledge, other studies, or in a data-driven manner.

As [Brownlees and Morozov](https://arxiv.org/abs/2210.14205) show, such optimal weights lead to fairly robust improvements in the MSE, both in simulations and with real data.
 





:::{admonition} More mathematical details 

For a full theoretical justification and explicit expressions for the matrix $\Psi$ see the [original paper](https://arxiv.org/abs/2210.14205).

:::


 

## Averaging Beyond Optimal Weights

While MSE-optimal weights are powerful for unit-specific estimation, other weighting schemes may better suit different goals. For example:

- Mean group estimation that uses *equal weights* ($w_i=1/N$ for all $i$). This scheme targets $\mathbb{E}[\mu(\theta_i)]$ and is implemented as [`MeanGroupUnitAverager`](../reference/MeanGroupUnitAverager.rst)
- Likelihood-weighted schemes that minimize the Kullback-Leibler divergence to the focus parameter instead of the MSE. 

Custom averaging schemes can be implemented by subclassing from [`BaseUnitAverager`](../reference/BaseUnitAverager.rst) and implementing the desired likelihood function or model. As an example, see the [tutorial](../tutorials/plot_2_custom_basic) on defining custom averaging schemes.

:::{admonition} Next steps

- See [Getting Started](../tutorials/plot_1_basics) for an end-to-end example of unit averaging.
- See the [API documentation](../reference/index) for the various averager classes available. 

:::
