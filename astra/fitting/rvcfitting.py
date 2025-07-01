import numpy as np
from numba import njit
import emcee


@njit
def sine_rvc(times: np.ndarray[float], K2: float, gamma: float, t0: float, period: float):
    """
    Simple sinusoidal radial velocity curve as a function of time, i.e.:
    RV(t) = K2 * sin( ( 2 * pi * (t - t0) ) / period) + gamma
    """
    RVs = K2 * np.sin((2 * np.pi * (times - t0)) / period) + gamma
    return RVs

@njit
def _rv_llh(
    pars: list[float],
    times: np.ndarray[float],
    rvs: np.ndarray[float],
    rv_errs: np.ndarray[float],
    gamma: float | None = None,
    t0: float | None = None,
    period: float | None = None
) -> float:
    """
    Log-likelihood function for radial velocity sine curve fitting.
    """
    K2 = pars[0]
    gamma = pars[1] if gamma is None else gamma
    t0 = pars[2] if t0 is None else t0
    period = pars[3] if period is None else period

    rv_model = sine_rvc(times, K2, gamma, t0, period)
    return -0.5 * ((rvs - rv_model)**2 / rv_errs**2).sum()


def rvmc(
    times: np.ndarray[float],
    rvs: np.ndarray[float],
    rv_errs: np.ndarray[float],
    init_pars: list[float],
    init_scatter: list[float],
    n_walkers: int = 32,
    n_samples: int = 1000,
    n_burnin: int = 100,
    fit_gamma: bool = True,
    fit_t0: bool = True,
    fit_period: bool = False,
    progress: bool = True,
    return_log_prob: bool = False
) -> np.ndarray[float]:
    """
    Fit radial velocities with emcee.
    """

    if fit_t0 == False and fit_period == True:
        raise ValueError("If t0 is fixed, period must also be fixed.")
    if fit_gamma == False and (fit_period == True or fit_t0 == True):
        raise ValueError("If gamma is fixed, period and t0 must also be fixed.")

    n_dim = 1 + sum([int(f) for f in [fit_gamma, fit_t0, fit_period]])
    fixed_pars = init_pars[n_dim - 4:] if n_dim != 4 else []

    init_state = init_pars[:n_dim] + init_scatter[:n_dim] * np.random.uniform(-1, 1, size=(n_walkers, n_dim))

    llh_args = [times, rvs, rv_errs] + [None] * (n_dim - 1) + list(fixed_pars)
    llh_args = tuple(llh_args)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, _rv_llh, args=llh_args)

    state = sampler.run_mcmc(init_state, n_samples, progress=progress)
    r = sampler.get_chain(flat=True, thin=1, discard=n_burnin)

    if return_log_prob:
        return r, sampler.get_log_prob()
    else:
        return r
