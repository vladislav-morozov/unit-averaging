# About Unit Averaging

## Overview

The key nature of unit average can be expressed in three sentences: 

:::{admonition} What is unit averaging?
:class: admonition

In short, **unit averaging** is an *ensemble* method for *efficiently* estimating unit- or study-specific parameters in heterogeneous panel data or meta-analysis settings. It uses weighted averages of all unit-specific parameter estimates. The weights can optimally trade off the increase in bias caused by using non-target units with the decrease in variance due to using more data.


:::


For example, imagine that you want to forecast the gross domestic product (GDP) in France for next year. You have data both on France and on other European countries in a panel dataset. You have three main options on what to do:

- Build your forecast using only French data. This approach has the least bias and the highest variance.
- Pool all the data and fit a single predictive model. This approach has the highest bias.
- Compromise and use the panel-wide information to optimally reduce the variance of the individual-only approach.

Unit averaging is a compromise approach.



## Target Parameter and Goal

To describe the approach let $i=1, \dots, N$ be the different units in your data (countries, customers, firms; or studies in a meta-analysis). The data is *heterogeneous* in the sense that each unit $i$ has their own version of some parameter $\theta_i$.

We are interested in the *unit-specific* "focus parameter" for some fixed unit $i$: 

$$
\mu(\theta_i)
$$

The transformed parameters $\mu(\theta_i)$ may be purely predictive (GDP forecast, probability of quitting the service) or be causal/structural (individual treatment effects, multipliers, etc.). These focus functions $\mu(\cdot)$ are represented by the [`BaseFocusFunction`](../reference/BaseFocusFunction.rst)  and [`InlineFocusFunction`](../reference/InlineFocusFunction.rst) classes. 

We want to estimate $\mu(\theta_i)$ optimally in terms of the MSE.

## Approach



### Unit Averaging Estimator

The unit averaging approach proceeds in two steps:

1. Estimate $\hat{\theta}_i$ and $\mu(\hat{\theta}_i)$ separately for each unit (producing "individual" estimators)
2. Compute a weighted average of the individual estimators:

$$
\hat{\mu}(\mathbf{w}) = \sum_{i=1}^N w_i \mu(\hat(\theta)_i),
$$

where the weights $\mathbf{w} = (w_1, \dots, w_N)$ are non-negative and sum to 1.
 
Different weights define different schemes, and this package implements several key ones, including optimal weights.

### Why It Makes Sense

The intuition behind $\hat{\mu}(\mathbf{w})$ is quite simple.

### Optimal Weights

Implemented by the [`OptimalUnitAverager`](../reference/OptimalUnitAverager.rst) class

$$
\mathrm{argmin} \mathbf{w}'\mathbf{\hat{\Psi}}\mathbf{w}
$$

The weights ensured to be non-negative and sum to 1.


There are two main approaches to optimal unit averaging

- Agnostic (fixed-N) approach
- Approach with prior information for the unit averaging 

A fairly broad class, that also nests Stein-like shrinkage weights where the individual estimator for the target unit is shrunk towards the sample mean. 

For a full theoretical justification and explicit expressions for the matrix $\Psi$ see the paper.

Does this work in practice? 
 




 

### Other Weight Schemes

The MSE-optimal weights described above are tailored to the task of efficiently estimating unit-specific parameters.

However, there are also other different schemes with different targets that one can consider. For example,

- Mean group estimation that uses *equal weights* ($w_i=1/N$ for all $i$). This scheme targets $\mathbb{E}[\mu(\theta_i)]$ implemented as [`MeanGroupUnitAverager`](../reference/MeanGroupUnitAverager.rst)
- Likelihood-weighted schemes that minimize the Kullback-Leibler divergence to the target parameter instead of the MSE. These can be implemented by subclassing from [`BaseUnitAverager`](../reference/OptimalUnitAverager.rst) and implementing the desired likelihood function or model.