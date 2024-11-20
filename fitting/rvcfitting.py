import numpy as np
from numba import njit
import emcee


@njit
def sine_fit(times, K2, gamma, t0, period):
    RVs = K2 * np.sin((2 * np.pi * (times - t0)) / period) + gamma
    return RVs


@njit
def llh(pars, times, rvs, rv_errs, period=None):
    if period is None:
        K2, gamma, t0, period = pars
    else:
        K2, gamma, t0 = pars

    rv_model = sine_fit(times, K2, gamma, t0, period)
    return -0.5 * ((rvs - rv_model)**2 / rv_errs**2).sum()


def rvmc(
    times,
    rvs,
    rv_errs,
    init_pars,
    init_scatter,
    n_walkers=32,
    n_samples=1000,
    n_burnin=100,
    fit_period=False,
    progress=True,
    return_log_prob=False
) -> np.ndarray[float]:
    """
    """
    if fit_period:
        n_dim = 4
        assert len(init_pars) == len(init_scatter) == n_dim
        init_state = init_pars + init_scatter * np.random.uniform(-1, 1, size=(n_walkers, n_dim))
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, llh, args=(times, rvs, rv_errs))
    else:
        n_dim = 3
        assert len(init_pars) == 4
        assert len(init_scatter) == 3 or len(init_scatter) == 4
        period = init_pars[3]
        init_state = init_pars[0:3] + init_scatter[0:3] * np.random.uniform(-1, 1, size=(n_walkers, n_dim))
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, llh, args=(times, rvs, rv_errs, period))

    state = sampler.run_mcmc(init_state, n_samples, progress=progress)
    r = sampler.get_chain(flat=True, thin=1, discard=n_burnin)

    if return_log_prob:
        return r, sampler.get_log_prob()
    else:
        return r
