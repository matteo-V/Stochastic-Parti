## Fitting stochastic volatility models with Particle Marginal Metropolis-Hastings (PMMH) MCMC
This repository contains example code which fits a stochastic volatility model to simulated data.
The model is fit using the Bayesian paradigm and particle marginal metropolis-hastings MCMC.

### Stochastic volatility model
$$
\\nu_0 \\sim
\\y_0 \\sim \\mathcal{N}(0, \exp{\\sigma_0/2})

\\nu_{t} \\sim \\mu + \\phi(\\nu_{t-1}-\\mu) + \\mathcal{N}(0, \\sigma_v)
\\y_{t} \\sim \\mathcal{N}(0, \\exp{\\frac{\\nu_t}}{2})
$$
