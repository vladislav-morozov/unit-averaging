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

Unit averaging is an ensemble method in the spirit of compromise estimators.



## Target Parameter and Goal

To describe the approach let $i=1, \dots, N$ be the different units in your data (countries, customers, firms; or studies in a meta-analysis). The data is *heterogeneous* in the sense that each unit $i$ has their own version of some parameter $\theta_i$.

We are interested in the *unit-specific* "focus parameter" for some fixed unit $i$: 

$$
\mu(\theta_i)
$$




The transformed parameters $\mu(\theta_i)$ may be purely *predictive* (GDP forecast, probability of quitting the service) or be *causal*/structural (individual treatment effects, multipliers, etc.).

The goal is to estimate $\mu(\theta_i)$ optimally in the sense of minimizing the mean squared error (MSE).

:::{admonition} Focus functions in `unit_averaging`
:class: admonition
 In `unit_averaging`, these focus functions $\mu(\cdot)$ are represented by [`InlineFocusFunction`](../reference/InlineFocusFunction.rst) and concrete implementations of [`BaseFocusFunction`](../reference/BaseFocusFunction.rst). See [Getting Started](../tutorials/plot_1_basics) for an example.
:::




## Approach



### Unit Averaging Estimator

The unit averaging approach proceeds in two steps:

1. Estimate $\hat{\theta}_i$ and $\mu(\hat{\theta}_i)$ separately for each unit (producing "individual" or "unit-specific" estimators).
2. Compute a weighted average of the individual estimators:

$$
\hat{\mu}(\mathbf{w}) = \sum_{i=1}^N w_i \mu(\hat{\theta}_i),
$$

where the weights $\mathbf{w} = (w_1, \dots, w_N)$ are non-negative and sum to 1.
 
Different weights define different schemes, and this package implements several universally-applicable ones, including MSE-optimal weights, and offers an interface for implementing custom ones.

### Why Averaging Makes Sense

To understand the intuition behind $\hat{\mu}(\mathbf{w})$, suppose that we can write the individual-specific true parameters $\theta_i$ as 

$$
\theta_i = \mathbb{E}[\theta_i] + \eta_i.
$$

Here $\mathbb{E}[\theta_i]$ is the average value that is *common* to all units, while $\eta_i$ is a mean-zero value that is *specific* to unit $i$. 

Unit averaging is motivated by the fact that each unit carries information about $\mathbb{E}[\theta_i]$. Using data on non-target units may help reduce the uncertainty about $\mathbb{E}[\theta_i]$ and reduce the overall variance of the estimator. 

:::{admonition} Why a linear combination? 
:class: admonition 

Taking a linear combination of $\mu(\hat{\theta}_i)$ (post-estimation averaging) is what allows unit averaging to exploit the above intuition about $\mathbb{E}[\theta_i]$ regardless of the model context. 

The approach stands as an alternative to pre-estimation data pooling (estimating a single model on the combined data). In dynamic and/or nonlinear contexts, the latter approach typically suffers from biases that are hard to characterize and control, making it less attractive.

:::

### Optimal Weights


At the same time, adding information of other units creates a trade-off. While it may decrease the uncertainty regarding $\mathbb{E}[\theta_i]$, it may create bias by adding potentially irrelevant information about $\eta_i$.

Optimally exploiting this bias-variance trade-off leads to the MSE-optimal weights implemented by the [`OptimalUnitAverager`](../reference/OptimalUnitAverager.rst) class. These optimal weights solve a convex quadratic problem of the form

$$
\hat{\mathbf{w}} = \mathrm{argmin} \mathbf{w}'\mathbf{\hat{\Psi}}\mathbf{w},
$$

where the matrix $\mathbf{\hat{\Psi}}$ encodes the variances and estimated biases between the target unit and the other units. The weights $\hat{\mathbf{w}}$ are non-negative and sum to 1.


There are two main approaches to optimal unit averaging that differ in whether they use prior information:

- Agnostic ("fixed-N") unit averaging requires no prior information. In this regime, each weight $w_i$ may freely vary between 0 and 1 (up to summing to 1). The algorithm is free to choose which units to pay more attention to.
- Optimal averaging with prior information ("large-$N$") requires the user to split the units into two categories: unrestricted and restricted units.
    - The weights of unrestricted units vary independently.
    - All restricted units receive equal weights.
    - The algorithm only chooses the weight of the restricted set as a whole.

    This approach is particularly useful when you have a large number of restricted
    units. The average of a large restricted set will closely approximate the true
    average of the parameters. This allows for more efficient and precise shrinkage,
    as the algorithm can focus on optimizing the weights of the unrestricted units
    and the total weight of the restricted set.
 
The prior information for selecting the unrestricted units may be obtained on the basis of domain knowledge, other studies, or in a data-driven manner.

As [Brownlees and Morozov](https://arxiv.org/abs/2210.14205) show, such optimal weights lead to fairly robust improvements in the MSE, both in simulations and with real data.
 





:::{admonition} More mathematical details 

For a full theoretical justification and explicit expressions for the matrix $\Psi$ see the [original paper](https://arxiv.org/abs/2210.14205).

:::


 

## Averaging Beyond Optimal Weights

The MSE-optimal weights described above are tailored to the task of efficiently estimating unit-specific parameters.

However, there are other different schemes with different targets that one can consider. For example,

- Mean group estimation that uses *equal weights* ($w_i=1/N$ for all $i$). This scheme targets $\mathbb{E}[\mu(\theta_i)]$ implemented as [`MeanGroupUnitAverager`](../reference/MeanGroupUnitAverager.rst)
- Likelihood-weighted schemes that minimize the Kullback-Leibler divergence to the target parameter instead of the MSE. These can be implemented by subclassing from [`BaseUnitAverager`](../reference/OptimalUnitAverager.rst) and implementing the desired likelihood function or model. As an example, see the [tutorial](../tutorials/plot_2_custom_basic) on defining custom averaging schemes.

:::{admonition} Next steps

- See [Getting Started](../tutorials/plot_1_basics) for an end-to-end example of unit averaging.
- See the [API documentation](../reference/index) for the various averager classes available. 

:::
