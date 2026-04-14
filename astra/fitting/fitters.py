from astra.fitting.core import (
    BasePrior, UniformPrior, GaussianPrior, TruncnormPrior,
    _PriorOrPriors
)
import astra.fitting.specmodels as specmodels

import os
import warnings
from typing import Any, Callable
from typing_extensions import Sentinel
from functools import partial
import pickle
import numpy as np

from tqdm.auto import tqdm

import emcee

from dynesty import NestedSampler
from dynesty.utils import print_fn, get_print_func
from dynesty import __version__ as DYNESTY_VERSION

SpectrumInterpolator = specmodels.SpectrumInterpolator

_bg_models = {model: getattr(specmodels, model) for model in specmodels.ALL_MODELS}
BG_MODELS = _bg_models.copy()

_NoKwargs = Sentinel('NoKwargs')
_NoMaskProvided = Sentinel('NoMaskProvided')

def _spec_model(
    pars: dict[str, float],
    interpolator: SpectrumInterpolator,
    bg: str | Callable | None = None,
    bg_par_map: dict[str, str] = {},
) -> np.ndarray[float]:

    wvs = interpolator._wvs

    if isinstance(bg, str):
        if bg not in _bg_models:
            raise ValueError(f"Invalid choice for background model: {bg}. Valid choices are {list(_bg_models.keys())}.")
        bg_func = _bg_models[bg]
    elif callable(bg):
        bg_func = bg
    elif bg is None:
        def bg_none(*args, **kwargs):
            return np.zeros(wvs.size)
        bg_func = bg_none
    else:
        raise TypeError(f"Invalid type for bg: {type(bg)}")

    interp_pars = {p: pars[p] for p in interpolator.param_names}
    bg_pars = {bg_par_map[p]: pars[p] for p in bg_par_map}

    radius, distance = pars['radius'], pars['distance']
    fstar = pars['fstar']

    # flux = scaled_spec(interp_pars, radius, distance, interpolator)
    flux = specmodels.interpspec_flux_scaled(
        np.array([]),
        np.array([]),
        interpolator,
        interp_pars,
        radius,
        distance
    )

    bgspec = bg_func(wvs, flux, **bg_pars)

    out = fstar * flux + (1 - fstar) * bgspec

    return out


class SpectrumFitter:
    """
    Fit a spectrum with a model consisting of an interpolated and background spectrum.
    Uses emcee.EnsembleSampler for MCMC sampling.
    Supports priors distributions using prior classes provided by astra.

    Parameters:
    param_config: dict of str: float, None or a prior/list of priors
        Configuration for fit parameters.
        Must have a key for all fit parameters.
        This includes `radius`, `distance` (for the flux-scaled interpolated model)
        and `fstar`, the fraction of stellar light.
        This also includes all parameters of `interpolator` (interpolator.param_names),
        and all parameters of `bg_model`, as given as keys of `bg_param_map`.
        Every value must be either None, float, or a prior/list of priors, where:
            - None indicates the parameter is unbound and will be fit
            - float indicates the parameter will be fixed to this value and not fit
            - A prior or list of priors will constrain the parameter appropriately
            Note that parameters of the interpolator will be automatically bounded to their
            maximum ranges using a UniformPrior if they are left unconstrained.
    interpolator: an astra.specfitting.SpectrumInterpolator instance
        The interpolator instance used for fitting the model.
        The interpolator must have its param_names specified.
    bg_model: str or callable, optional
        Background model, applied with proportion (1 - fstar).
        Can be specified with either a string (referencing a model in astra.specfitting.specmodels)
        or a callable which accepts wavelengths, fluxes as the first two arguments.
        If not given, a background of 0 is used (in which case, fstar should be fixed to 1).
    bg_param_map: dict of str: str, optional
        dict mapping param_config constraints onto the provided `bg_model`.
        This allows flexibility and prevents parameter name conflicts.
        E.g. if `bg_model` is a function blackbody(wvs, flux, teff), you could have
        `teff_bb` in your param_config, and set `bg_param_map` to be {`teff_bb`: `teff`}.
    nwalkers: int, default 16
        Number of walkers for MCMC ensemble (AIES) sampling.

    Methods:
    initialise(self, spectrum, init_config, mask=None)
        Initialise the fitter with a spectrum to fit to, and configure the starting conditions
        for the walkers with `init_config`. An optional mask may be provided.
        `init_config` must be a dict of str: tuple(float, float), with keys for each fit parameter.
        Each tuple specifies the central value (first value) and scatter size (second value)
        for initialising the walkers.

    run(self, nsteps, continue_=True, progress=True, spectrum=None, init_config=None, mask=None)
        Run the sampling process for `nsteps` number of steps.
        Will continue from an existing run if continue_ is not set to False.
        The spectrum, init_config, and mask may be set (or overridden) here.
        Set progress=False to disable the tqdm progress bar.

    """

    _non_interp_keys = {'radius', 'distance', 'fstar'}
    _scatter_methods = {'normal': np.random.normal, 'uniform': np.random.uniform}

    def __init__(
        self,
        param_config: dict[str, float | _PriorOrPriors | None],
        interpolator: SpectrumInterpolator,
        bg_model: str | Callable | None = None,
        bg_param_map: dict[str, str] = {},
        nwalkers: int = 16,
        **kwargs
    ):
        # set model
        self._model = _spec_model

        # verify and set background model
        if isinstance(bg_model, str):
            if bg_model not in _bg_models:
                raise ValueError(
                    f"Invalid choice for background model: {bg_model}. "
                    f"Valid choices are {list(_bg_models.keys())}."
                )
            bg_func = _bg_models[bg_model]
        elif callable(bg_model):
            bg_func = bg_model
        elif bg_model is None:
            def bg_none(*args, **kwargs):
                return np.zeros(interpolator._wvs.size)
            bg_func = bg_none
        else:
            raise TypeError(f"Invalid type for bg: {type(bg_model)}")
        self._bg_model = bg_func
        self._bg_param_map = bg_param_map

        #param_config_interp = {p: v for p, v in param_config.items() if p not in self._non_interp_keys}
        #bg_pars = {bg_par_map[p]: pars[p] for p in bg_par_map}
        #radius, distance = pars['radius'], pars['distance']

        if interpolator.param_names is None:
            raise ValueError("interpolator must have param_names specified for use in SpectrumFitter.")
        self._interpolator = interpolator

        # # find index of teff param in interpolator
        # self._teff_param_kw = teff_param_kw
        # self._teff_param_idx = self.interpolator.param_names.index(self._teff_param_kw)

        # verify param_config and organise priors into dicts
        self._verify_param_config(param_config)
        self._sort_priors()

        self._nwalkers = int(nwalkers)

        scatter_method = kwargs.get('scatter_method', 'normal')

        if isinstance(scatter_method, str):
            if scatter_method not in self._scatter_methods:
                raise ValueError(
                    f"Invalid choice for scatter_method: {scatter_method}. "
                    f"Valid options are {list(self._scatter_methods.keys())}."
                )
            self._scatter_method = self._scatter_methods[scatter_method]
        elif callable(scatter_method):
            self._scatter_method = scatter_method
        else:
            raise TypeError(f"Invalid type for scatter_method: {type(scatter_method)}")

        if 'pool' in kwargs:
            raise NotImplementedError("Multiprocessing not currently implemented.")
        self._pool = kwargs.get('pool')

        self.wv_tol = kwargs.get('wv_tol', 1e-6)

        #self._init_sampler(nwalkers, pool=kwargs.get('pool'))

    def _verify_param_config(self, param_config):
        """
        Verify a given paramter config and store attributes in class.
        Performs the following operations:
        - Checks match against interpolator parameters
        - Consolidates priors into lists for each parameter
        - Bounds interpolator with additional priors if needed
        - Sets attributes of class with results

        Sets/modifies the following attributes:
        self._param_config
        self._interp_bounding_priors
        self._npars
        self._fit_pars
        self._fixed_pars
        self._ndims
        """

        param_config_interp = {p: v for p, v in param_config.items() if p not in self._non_interp_keys}
        param_config_interp = {p: param_config[p] for p in self.interpolator.param_names}

        # verify param_config matches interp_params
        # if self._interpolator.param_names is not None:
        #     if not all(k in self.interpolator.param_names for k in param_config_interp.keys()):
        #         raise KeyError(f"param_config keys ({list(param_config_interp.keys())}) do not match interpolator ({self.interpolator.param_names}).")

        param_config_full = {}
        # consolidate priors - any single priors made into list of length one, floats forced, check types
        for par, res in param_config.items():
            if isinstance(res, list):
                if all(isinstance(r, BasePrior) for r in res):
                    param_config_full[par] = res
                    continue
                else:
                    raise TypeError(f"Config for parameter {par} contains mixed constraints, must contain only instances of priors.")
            elif isinstance(res, BasePrior):
                param_config_full[par] = [res]
                continue
            elif isinstance(res, (float, int)):
                param_config_full[par] = float(res)
            elif res is None:
                param_config_full[par] = None
            else:
                raise TypeError(f"Invalid type in param_config for parameter {par}: {type(res)}")

        # bound interpolator with priors if needed
        param_config_full, interp_bounding_priors = self._bound_interpolator(param_config_full)

        self._param_config = param_config_full
        self._interp_bounding_priors = interp_bounding_priors

        # total parameters (fixed/unfixed)
        self._npars = len(self.param_config)

        # ndims is number of non-fixed parameters take int just in case
        fixed_pars = [p for p, v in self.param_config.items() if isinstance(v, (float, int))]
        fit_pars = [p for p in self.param_config if p not in fixed_pars]

        self._fit_pars = fit_pars
        self._fixed_pars = fixed_pars
        self._ndims = len(fit_pars)
        self._nsteps = 0

    def _bound_interpolator(self, param_config):
        """
        Bounds self.interpolator with uniform priors if it is unbounded by
        existing priors in param_config

        Returns modified param_config and dictionary containing only the new priors.
        """

        new_priors = {par: [] for par in self.interpolator.param_names}

        # add uniform priors for bounds of interpolator
        for par, bounds in zip(self.interpolator.param_names, self.interpolator.bounds):

            res = param_config[par]
            # fixed parameter, check bounds then ignore
            if isinstance(res, float):
                if bounds[0] <= res <= bounds[1]:
                    continue
                else:
                    raise ValueError(f"Fixed value {res} for parameter {par} is out of bounds of interpolator ({bounds})")

            # free parameter, bound
            elif res is None:
                prior = UniformPrior(bounds[0], bounds[1])
                #param_config[par].append(prior)
                param_config[par] = [prior]
                new_priors[par].append(prior)
                continue

            # get any existing uniform priors
            existing_UPs = [p for p in res if isinstance(p, UniformPrior)]
            if len(existing_UPs) > 0:
                lb = min([p.lower for p in existing_UPs])
                ub = max([p.upper for p in existing_UPs])

                # if existing priors already bound interpolator, skip
                if lb > bounds[0] and ub < bounds[1]:
                    continue

            prior = UniformPrior(bounds[0], bounds[1])
            param_config[par].append(prior)
            new_priors[par].append(prior)

        return param_config, new_priors

    def _sort_priors(self):
        """
        Sorts priors from self.param_config into self._uniform_priors and
        self._other_priors, so that uniform priors can be handled first in
        the sampling function
        """

        # consolidate uniform priors so they can be applied first
        uniform_prior_dict = {}
        other_prior_dict = {}

        for par, res in self.param_config.items():

            if res is None or isinstance(res, float):
                continue

            uniform_priors = [p for p in res if isinstance(p, UniformPrior)]
            other_priors = [p for p in res if not isinstance(p, UniformPrior)]

            if len(uniform_priors) > 0:
                uniform_prior_dict[par] = uniform_priors

            if len(other_priors) > 0:
                other_prior_dict[par] = other_priors

        self._uniform_priors = uniform_prior_dict
        self._other_priors = other_prior_dict

    def _eval_priors(self, param_dict):
        """
        Evaluates priors on a dict of proposed parameters.
        """

        # first go through uniform priors
        uniform_priors = [prior.eval(param_dict[par]) for par, priors in self._uniform_priors.items() for prior in priors]

        if any(p == -np.inf for p in uniform_priors):
            return -np.inf

        probs = [prior.eval(param_dict[par]) for par, priors in self._other_priors.items() for prior in priors]

        return np.prod(probs)

    def _get_llh_func(self):
        """
        Assemble and return the log-likelihood function for emcee sampling
        """

        def llh_func(pars, pars_fixed, spec, spec_errors, cls):
            param_dict = {k: v for k, v in zip(cls.fit_pars, pars)}
            param_dict = param_dict | {k: v for k, v in zip(cls.fixed_pars, pars_fixed)}

            prior = cls._eval_priors(param_dict)

            if prior == -np.inf:
                return -np.inf

            mask = cls.mask
            # interp_pars = {k: param_dict[k] for k in cls.interpolator.param_names}

            model = self._model(
                pars=param_dict,
                interpolator=cls.interpolator,
                bg=cls.bg_model,
                bg_par_map=cls._bg_param_map
            )

            chisq = ((spec[mask] - model[mask])**2 / spec_errors[mask]**2).sum()

            # errorlnf = ?
            # errors_adj_sq = spec_errors**2 + model**2 * np.exp(2 * errorlnf)
            # chisq = (((spec - model)**2 / errors_adj_sq) + np.log(errors_adj_sq)).sum()

            llh = -0.5 * chisq + np.log(prior)

            return llh

        return llh_func

    def _init_sampler(self, flux, flux_errors):
        """
        Initialise sampling for a given spectrum
        """

        llh_func = self._get_llh_func()

        pars_fixed = [self.param_config[p] for p in self.fixed_pars]

        sampler = emcee.EnsembleSampler(
            self._nwalkers,
            self.ndims,
            llh_func,
            # pool=self._pool,
            args=(pars_fixed, flux, flux_errors, self),
            kwargs=None,
        )

        self._sampler = sampler

    def _verify_spectrum(self, spectrum):
        """
        Verify shape of spectrum and wavelength scale against interpolator
        """

        if spectrum.shape[1] != 3:
            raise IndexError("Spectrum must have three columns: wavelength, flux, and flux errors.")

        wv_tol = self.wv_tol
        wv_dev = np.abs(self.interpolator._wvs - spectrum[:, 0]).max()
        if wv_dev > wv_tol:
            raise ValueError(f"spectrum wavelengths deviate from interpolator's spectra above tolerance ({wv_dev} > {wv_tol}).")

        self._spectrum = spectrum

    def _verify_mask(self, mask):
        """
        Verify and apply mask against spectrum
        """

        wvs = self.interpolator._wvs

        if mask is None:
            mask_arr = np.ones(wvs.size, dtype=bool)
        elif isinstance(mask, list) and isinstance(mask[0], (tuple, list)):
            mask_arr = np.ones(wvs.size, dtype=bool)

            for lb, ub in mask:
                if lb > ub:
                    mask_arr = mask_arr & ~((wvs > ub) & (wvs < lb))
                else:
                    mask_arr = mask_arr & ~((wvs > lb) & (wvs < ub))

        elif isinstance(mask, list) and isinstance(mask[0], bool):
            mask_arr = np.array(mask, dtype=bool)
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            mask_arr = mask.copy()
        else:
            raise TypeError("Invalid mask format provided.")

        self._mask = mask_arr

    def _verify_init_config(self, init_config):
        """
        Verify init_config against param_config and check types
        """
        # accept list if matches length of fit pars
        if isinstance(init_config, (list, tuple)):
            if not len(init_config) == self.ndims:
                raise IndexError(f"Expected {self.ndims} items in init_config, found {len(init_config)}.")
            init_config_adj = {p: v for p, v in zip(self.fit_pars, init_config)}
        else:
            init_config_adj = init_config.copy()

        # deal with incorrect keys
        missing = [p for p in self.fit_pars if p not in init_config_adj.keys()]
        extra = [p for p in init_config_adj.keys() if p not in self.fit_pars]

        if len(missing) > 0:
            raise KeyError(f"Missing keys in init_config: {missing}")

        if len(extra) > 0:
            raise KeyError(f"Invalid keys provided: {extra}. Fit parameters are: {self.fit_pars}")

        init_config_cons = {}
        for p, conf_tuple in init_config_adj.items():
            if not isinstance(conf_tuple, (tuple, list)):
                raise TypeError(f"Invalid type in init_config: {type(conf_tuple)}")

            if len(conf_tuple) != 2:
                raise IndexError(f"Invalid item in init_config: {conf_tuple}. Must be length 2: (mean, sigma).")

            if conf_tuple[1] <= 0:
                raise ValueError(f"Invalid value for sigma for {p}. sigma must be positive and non-zero.")

            init_config_cons[p] = conf_tuple if isinstance(conf_tuple, tuple) else tuple(conf_tuple)

        self._init_config = init_config_cons

    def _create_init_samples(self, init_config):
        """
        Create samples from normal distribution using init_config
        """

        init_means = np.array([init_config[p][0] for p in self.fit_pars])
        init_sigmas = np.array([init_config[p][1] for p in self.fit_pars])

        scatter_method = self.scatter_method
        init_samples = init_means + init_sigmas * scatter_method(size=(self.nwalkers, self.ndims))

        return init_samples

    def initialise(
        self,
        spectrum: np.ndarray[float],
        init_config: dict[str, tuple[float, float]],
        mask: np.ndarray[bool] | None = None
    ):
        """
        Initialise the fitter with a spectrum to fit to, and configure the starting conditions
        for the walkers with `init_config`. An optional mask may be provided.

        Parameters:
        spectrum: np.ndarray[float]
            A spectrum with 3 columns: wavelength, flux, flux errors.
            Must have the same wavelengths as `self.interpolator`.
        init_config: dict of str: tuple[float, float]
            Configuration for scattering the initial state of the MCMC walkers.
            Must contain a key for each fit (i.e. not fixed) parameter.
            Each tuple is a pair of (central value, scatter size).
            Walkers are scattered according to self.scatter_method (default: np.random.normal).
        mask: np.ndarray[bool], optional
            Boolean mask for spectrum, ranges set to False are ignored in fitting.
        """

        # overwrite check here?

        # verify spectrum, sets self.spectrum
        self._verify_spectrum(spectrum)

        # verify mask, sets self.mask
        self._verify_mask(mask)

        self._init_sampler(spectrum[:, 1], spectrum[:, 2])

        # verify init_config, sets self.init_config
        self._verify_init_config(init_config)

        # create initial samples from init config
        init_samples = self._create_init_samples(self.init_config)
        self._init_samples = init_samples

        self._initialised = True
        self._nsteps = 0

    def _run_sampler(self, nsteps, continue_=True, progress=True):
        """
        Internal sampling method
        """

        # run from init_samples if no runs done so far or continue is False
        if self.nsteps == 0 or continue_ is False:
            self.sampler.run_mcmc(self.init_samples, nsteps, progress=progress)
        else:
            self.sampler.run_mcmc(None, nsteps, progress=progress)

        self._nsteps = self._nsteps + int(nsteps)

    def run(
        self,
        nsteps: int,
        continue_: bool = True,
        progress: bool = True,
        spectrum: np.ndarray[float] | None = None,
        init_config: dict[str, tuple[float, float]] | None = None,
        mask: np.ndarray[bool] | None = None
    ):
        """
        Run the sampling process for `nsteps` number of steps.
        Will continue from any previous runs unless continue_ = False (default True).
        Set progress=False to disable the tqdm progress bar.

        The spectrum, init_config, and mask may be set (or overridden) here.
        Both `spectrum` and `init_config` must have been set previously to start a run.
        See SpectrumFitter.initialise for more information on these parameters.

        """

        # raise error if not initialised before run
        # and parameters not given to initialise from
        if not (self.initialised or (spectrum is not None and init_config is not None)):
            if spectrum is None and init_config is None:
                raise ValueError("Sampler not initialised: please provide `spectrum` and `init_config`.")
            elif spectrum is None:
                raise ValueError("Sampler not initialised: please provide `spectrum`.")
            else:
                raise ValueError("Sampler not initialised: please provide `init_config`.")

        # initialise from provided spectrum or config
        if spectrum is not None:
            # initialise from spectrum, grab existing config if not provided
            # init_config must exist somewhere due to checks at start
            init_config = self.init_config if init_config is None else init_config
            mask = self.mask if mask is None else mask

            # this should never happen
            if init_config is None:
                raise Exception("Unexpected exception: initialised but existing init_config is None.")

            self.initialise(spectrum, init_config, mask)

        elif init_config is not None:
            # initialise from config, grab spectrum from self
            spectrum = self.spectrum
            mask = self.mask if mask is None else mask

            # this should never happen
            if spectrum is None:
                raise Exception("Unexpected exception: initialised but existing spectrum is None.")

            self.initialise(spectrum, init_config, mask)

        elif mask is not None:
            # initialise from existing
            spectrum = self.spectrum
            init_config = self.init_config

            # this should never happen
            if init_config is None:
                raise Exception("Unexpected exception: initialised but existing init_config is None.")
            # this should never happen
            if spectrum is None:
                raise Exception("Unexpected exception: initialised but existing spectrum is None.")

            self.initialise(spectrum, init_config, mask)

        self._run_sampler(nsteps, continue_=continue_, progress=progress)

    def get_chain(self, *args, **kwargs):
        return self.sampler.get_chain(*args, **kwargs)

    @property
    def param_config(self):
        return self._param_config

    @property
    def interpolator(self):
        return self._interpolator

    @property
    def fit_pars(self):
        return self._fit_pars

    @property
    def fixed_pars(self):
        return self._fixed_pars

    @property
    def ndims(self):
        return self._ndims

    @property
    def npars(self):
        return self._npars

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def bg_model(self):
        return self._bg_model

    @property
    def scatter_method(self):
        return self._scatter_method

    @property
    def sampler(self):
        return self._sampler if hasattr(self, '_sampler') else None

    @property
    def spectrum(self):
        return self._spectrum if hasattr(self, '_spectrum') else None

    @property
    def mask(self):
        return self._mask if hasattr(self, '_mask') else None

    @property
    def init_config(self):
        return self._init_config if hasattr(self, '_init_config') else None

    @property
    def initialised(self):
        return self._initialised if hasattr(self, '_initialised') else False

    @property
    def init_samples(self):
        return self._init_samples if hasattr(self, '_init_samples') else None

    @property
    def nsteps(self):
        return self._nsteps


class LinkedSpectrumFitter:
    """
    Fit a spectrum with a model consisting of an interpolated and background spectrum.
    Uses emcee.EnsembleSampler for MCMC sampling.
    Supports priors distributions using prior classes provided by astra.

    Parameters:
    param_config: dict of str: float, None or a prior/list of priors
        Configuration for fit parameters.
        Must have a key for all fit parameters.
        This includes `radius`, `distance` (for the flux-scaled interpolated model)
        and `fstar`, the fraction of stellar light.
        This also includes all parameters of `interpolator` (interpolator.param_names),
        and all parameters of `bg_model`, as given as keys of `bg_param_map`.
        Every value must be either None, float, or a prior/list of priors, where:
            - None indicates the parameter is unbound and will be fit
            - float indicates the parameter will be fixed to this value and not fit
            - A prior or list of priors will constrain the parameter appropriately
            Note that parameters of the interpolator will be automatically bounded to their
            maximum ranges using a UniformPrior if they are left unconstrained.
    interpolator: an astra.specfitting.SpectrumInterpolator instance
        The interpolator instance used for fitting the model.
        The interpolator must have its param_names specified.
    bg_model: str or callable, optional
        Background model, applied with proportion (1 - fstar).
        Can be specified with either a string (referencing a model in astra.specfitting.specmodels)
        or a callable which accepts wavelengths, fluxes as the first two arguments.
        If not given, a background of 0 is used (in which case, fstar should be fixed to 1).
    bg_param_map: dict of str: str, optional
        dict mapping param_config constraints onto the provided `bg_model`.
        This allows flexibility and prevents parameter name conflicts.
        E.g. if `bg_model` is a function blackbody(wvs, flux, teff), you could have
        `teff_bb` in your param_config, and set `bg_param_map` to be {`teff_bb`: `teff`}.

    Methods:
    initialise(self, spectrum, init_config, mask=None)
        Initialise the fitter with a spectrum to fit to, and configure the starting conditions
        for the walkers with `init_config`. An optional mask may be provided.
        `init_config` must be a dict of str: tuple(float, float), with keys for each fit parameter.
        Each tuple specifies the central value (first value) and scatter size (second value)
        for initialising the walkers.

    run(self, nsteps, continue_=True, progress=True, spectrum=None, init_config=None, mask=None)
        Run the sampling process for `nsteps` number of steps.
        Will continue from an existing run if continue_ is not set to False.
        The spectrum, init_config, and mask may be set (or overridden) here.
        Set progress=False to disable the tqdm progress bar.

    """

    _non_interp_params = ['radius', 'distance', 'fstar']
    #_model_params = {'radius', 'distance', 'fstar'}
    #_scatter_methods = {'normal': np.random.normal, 'uniform': np.random.uniform}

    # crop gauss priors to 5 sigma
    _trunc_to_sigma = 5

    # parameters for testing background model
    _bg_test_samples = 1000
    _bg_test_limits = (-1e100, 1e100)

    def __init__(
        self,
        param_config: dict[str, float | _PriorOrPriors | None],
        interpolator: SpectrumInterpolator,
        linked_pars: list[str] | None = None,
        bg_model: str | Callable | None = None,
        bg_param_map: dict[str, str] | None = None,
        **kwargs
    ):
        #param_config_interp = {p: v for p, v in param_config.items() if p not in self._non_interp_keys}
        #bg_pars = {bg_par_map[p]: pars[p] for p in bg_par_map}
        #radius, distance = pars['radius'], pars['distance']

        self._model = _spec_model

        # flag that will prevent running until a valid param config is set
        # allows interpolator, bg model, etc. to be changed without
        # requiring param config to be correct, but must be changed before running
        self._verified = {
            p: False for p in
            [
                'param_config',
                'interpolator',
                'linked_pars',
                'bg_model',
                'bg_param_map',
                'data',
                'mask',
                'linemask'
            ]
        }

        # set interpolator - skip param_config comparison
        # interpolator is read only, may not be set unless param config forced to None
        # self._param_config = None
        # self.interpolator = interpolator
        if isinstance(linked_pars, str):
            linked_pars = [linked_pars]
        elif linked_pars is None:
            linked_pars = []

        if bg_param_map is None:
            bg_param_map = dict()

        self._init_verify(
            param_config=param_config,
            interpolator=interpolator,
            linked_pars=linked_pars,
            bg_model=bg_model,
            bg_param_map=bg_param_map
        )

        # set after initialising
        self._sampler = None
        self._data = None
        self._mask = None
        self._linemask = None

        self._sampler_kwargs = dict()
        self._run_kwargs = dict()

        self._mask_arr = np.ones_like(interpolator.wvs, dtype=bool)
        self._linemask_arr = np.zeros_like(interpolator.wvs, dtype=bool)

        # self.nwalkers = nwalkers

        if 'pool' in kwargs:
            raise NotImplementedError("Multiprocessing not currently implemented.")
        self._pool = kwargs.get('pool')

        self.wv_tol = kwargs.get('wv_tol', 1e-6)

        #self._init_sampler(nwalkers, pool=kwargs.get('pool'))

    ####  Verification and testing of inputs  ####

    def _init_verify(
        self,
        param_config,
        interpolator,
        linked_pars,
        bg_model,
        bg_param_map
    ):
        """
        Verify all inputs on init
        """
        # simply verify interpolator
        self._verify_interpolator(interpolator)

        # verify background model is valid (not that it works yet)
        if isinstance(bg_model, str):
            if bg_model not in _bg_models:
                raise ValueError(
                    f"Invalid choice for background model: {bg_model}. "
                    f"Valid choices are {list(_bg_models.keys())}."
                )
            bg_func = _bg_models[bg_model]
        elif callable(bg_model):
            bg_func = bg_model
        elif bg_model is None:
            def bg_none(*args, **kwargs):
                return np.zeros(interpolator._wvs.size)
            bg_func = bg_none
        else:
            raise TypeError(f"Invalid type for bg: {type(bg_model)}")

        # verify param config but raise error on invalid linked pars
        self._verify_param_config(
            param_config,
            bg_model=bg_func,
            bg_param_map=bg_param_map,
            linked_pars=linked_pars,
            remove_linked_pars=False
        )

    def _verify_interpolator(self, interpolator):
        """
        """
        if interpolator.param_names is None:
            raise ValueError("interpolator must have param_names specified for use in SpectrumFitter.")

        # # if not first run, check against existing param config
        # if self._param_config_verified:
        #     interp_params = {p: self.param_config.get(p) for p in interpolator.param_names}

        #     # check parameters missing from interp
        #     missing_interp_params = [p for p in interp_params if interp_params[p] is None]
        #     if len(missing_interp_params) > 0:
        #         warnings.warn("")

        self._interp_params = interpolator.param_names
        self._interpolator = interpolator

        self._verified['interpolator'] = True

    def _verify_bg_param_map(self, bg_param_map):
        """
        """
        # verify vs. param_config
        bg_params = {p: self.param_config.get(p) for p in bg_param_map}
        missing_bg_params = [p for p in bg_params if bg_params[p] is None]

        if len(missing_bg_params) > 0:
            warnings.warn(f"Background params missing from par_config: {missing_bg_params}. Please set a valid param_config before running fitting.")
            self._verified['param_config'] = False
            # could not test bg_model with changes so also set to False - need to test later when valid param_config given
            self._verified['bg_model'] = False
            self._verified['bg_param_map'] = False
        else:
            # param config appears valid, check model
            # now check background model works with param_config and map
            # if it fails, assume map is wrong. correct order is to provide a new model first

            success, err = self._run_bg_model_tests(self.bg_model, bg_param_map=bg_param_map)

            if success:
                psuccess, perr = self._try_verify_param_config(self.param_config, bg_model=self.bg_model, bg_param_map=bg_param_map)
                if not psuccess:
                    print(perr)
            else:
                raise RuntimeError(f"bg_model failed to run with new bg_param_map with error: {err}.")

        self._bg_param_map = bg_param_map
        self._bg_params = list(bg_param_map)

    def _verify_bg_model(self, bg_model):
        """
        Verify background model selection is valid
        """
        # verify and set background model
        if isinstance(bg_model, str):
            if bg_model not in _bg_models:
                raise ValueError(
                    f"Invalid choice for background model: {bg_model}. "
                    f"Valid choices are {list(_bg_models.keys())}."
                )
            bg_func = _bg_models[bg_model]
        elif callable(bg_model):
            bg_func = bg_model
        elif bg_model is None:
            def bg_none(*args, **kwargs):
                return np.zeros(self.interpolator.wvs.size)
            bg_func = bg_none
        else:
            raise TypeError(f"Invalid type for bg: {type(bg_model)}.")

        # check new bg model works based on param config
        bg_params = {p: self.param_config.get(p) for p in self.bg_param_map}
        missing_bg_params = [p for p in bg_params if bg_params[p] is None]

        if len(missing_bg_params) > 0:
            warnings.warn(f"Background params missing from par_config: {missing_bg_params}. Please set a valid param_config before running fitting.")
            self._verified['param_config'] = False
            # could not test bg_model with changes so also set to False - need to test later when valid param_config given
            self._verified['bg_model'] = False
            self._verified['bg_param_map'] = False
        else:
            # param config appears valid, check model
            # if it fails, invalidate map and param_config
            success, err = self._run_bg_model_tests(bg_model=bg_func)

            if success:
                psuccess, perr = self._try_verify_param_config(self.param_config, bg_model=bg_func)
                if not psuccess:
                    print(perr)
            else:
                warnings.warn(f"New bg_model failed to run with error: {err}. Please set a valid bg_param_map before running fitting.")
                self._verified['bg_model'] = False
                self._verified['bg_param_map'] = False

        self._bg_model = bg_func

    def run_bg_model_tests(self):
        """
        Test the background model, sampling across the bounds of the priors.
        """
        test_passed, error = self._run_bg_model_tests()
        if test_passed:
            print("bg_model passed tests.")
        else:
            # warn in error if something is unbounded
            print(f"bg_model failed tests with error: {error}.")

    def _run_bg_model_tests(self, bg_model=None, bg_param_map=None, param_config=None):
        """
        Test the background model, sampling over the priors in param_config.
        """
        # overrides to check against new options
        bg_model = self.bg_model if bg_model is None else bg_model
        bg_param_map = self.bg_param_map if bg_param_map is None else bg_param_map
        param_config = self.param_config if param_config is None else param_config

        bg_model_params = {}

        for p, bgp in bg_param_map.items():
            # need to get test pars for bg model
            res = param_config[p]

            # need to handle the possible cases appropriately
            # not assuming the dynesty case of no unbounded priors
            # if it is None, we just pick a random positive number
            # if it fails, we raise the error with the chosen parameters
            # and suggest adding a prior if the bg_model should be bounded
            if isinstance(res, list):
                # must all be prior instances (enforced by sort_param_config)
                # dynesty enforces single priors for the transform func, but here is a general test
                lower, upper = self._bg_test_limits
                for prior in res:
                    # truncate at set sigma for gaussians (default 5)
                    if isinstance(prior, GaussianPrior) and prior.bounded is False:
                        lower = max(lower, prior.mean - self._trunc_to_sigma * prior.sigma)
                        upper = min(upper, prior.mean + self._trunc_to_sigma * prior.sigma)
                    else:
                        lower = max(lower, prior.lower)
                        upper = min(upper, prior.upper)

                # sample uniformly from bounds
                samples = np.random.uniform(size=self._bg_test_samples, low=lower, high=upper)

            elif isinstance(res, float):
                # single fixed value, has already been forced to float by sort_param_config
                samples = res * np.ones(self._bg_test_samples)

            elif res is None:
                # unbounded value
                samples = np.random.uniform(size=self._bg_test_samples, low=self._bg_test_limits[0], high=self._bg_test_limits[1])

            else:
                # shouldn't happen because of sort_param_config
                raise TypeError(f"Invalid type in param_config for parameter {par}: {type(res)}.")

            bg_model_params[bgp] = samples

        wvs = self.interpolator.wvs
        err = None

        try:
            for i in range(self._bg_test_samples):
                bg_model_pars = {k: v[i] for k, v in bg_model_params.items()}
                _ = bg_model(wvs, np.random.rand(*wvs.shape), **bg_model_pars)

        except Exception as e:
            err = e
            err.args = (err.args[0] + f"\nFailed with parameters: {bg_model_pars}",)

        return (err is None, err)

    def _verify_param_config(self, param_config, bg_model=None, bg_param_map=None, linked_pars=None, remove_linked_pars=True):
        """
        Verify a given paramter config against bg_param_map, bg_model, linked_pars,
        and the interpolator.
        """
        # overrides to check against new options
        bg_model = self.bg_model if bg_model is None else bg_model
        bg_param_map = self.bg_param_map if bg_param_map is None else bg_param_map
        linked_pars = self.linked_pars if linked_pars is None else linked_pars

        param_config_sorted, interp_bounding_priors = self._sort_param_config(param_config)

        # now check bg_param_map works vs param_config
        bg_params = {p: param_config_sorted.get(p) for p in bg_param_map}
        missing_bg_params = [p for p in bg_params if bg_params[p] is None]
        if len(missing_bg_params) > 0:
            raise ValueError(f"Background params missing from par_config: {missing_bg_params}.")

        # now check background model works with param_config and map
        success, err = self._run_bg_model_tests(bg_model, bg_param_map=bg_param_map, param_config=param_config_sorted)
        if not success:
            raise RuntimeError(f"bg_model failed to run with error: {err}.")

        # need check for any useless parameters - not for interpolator, model, or bg_param_map
        all_params = self._interp_params + list(bg_param_map) + self._non_interp_params
        unused_params = [p for p in param_config_sorted if p not in all_params]
        if len(unused_params) > 0:
            warnings.warn(f"Unused parameters in param_config: {unused_params}.")
            #raise ValueError(f"Unused parameters in param_config: {unused_params}.")

        priors_cons = self._consolidate_priors_dynesty(
            param_config_sorted,
            bg_param_map=bg_param_map
        )
        param_config_cons = param_config_sorted | priors_cons

        # (re)validate linked pars with remove flag
        self._verify_linked_pars(linked_pars, param_config=param_config_cons, bg_param_map=bg_param_map, remove=remove_linked_pars)

        self._update_config(param_config_cons, bg_model=bg_model, bg_param_map=bg_param_map)
        self._interp_bounding_priors = interp_bounding_priors

    def _try_verify_param_config(self, param_config, bg_model=None, bg_param_map=None):
        """
        """
        err = None

        try:
            self._verify_param_config(
                param_config,
                bg_param_map=bg_param_map,
                bg_model=bg_model
            )
        except (ValueError, RuntimeError) as e:
            err = e
            #err.args = (err.args[0] + f"\nFailed to verify against par_config",)

        return (err is None, err)

    def _verify_linked_pars(self, linked_pars=None, param_config=None, bg_param_map=None, remove=False):
        """
        Check if provided linked parameters (if given) are valid based on param config.
        """
        linked_pars = self.linked_pars if linked_pars is None else linked_pars
        param_config = self.param_config if param_config is None else param_config
        bg_param_map = self.bg_param_map if bg_param_map is None else bg_param_map

        all_params = self._interp_params + list(bg_param_map) + self._non_interp_params
        unused_params = [p for p in param_config if p not in all_params]
        fixed_pars = [p for p, v in param_config.items() if isinstance(v, (float, int))]
        fit_pars = [p for p in param_config if p not in fixed_pars + unused_params]

        if linked_pars is None:
            self._linked_pars = []
        else:
            # check invalid (missing from config)
            invalid_pars = [p for p in linked_pars if p not in param_config]
            if len(invalid_pars) > 0:
                if remove:
                    linked_pars = [p for p in linked_pars if p not in invalid_pars]
                    warnings.warn(f"linked_pars: {invalid_pars} no longer in param_config and have been removed.")
                else:
                    raise ValueError(f"Invalid linked_pars: {invalid_pars} not in param_config.")

            # check parameters that are fixed
            invalid_pars = [p for p in linked_pars if p not in fit_pars]
            if len(invalid_pars) > 0:
                if remove:
                    linked_pars = [p for p in linked_pars if p not in invalid_pars]
                    warnings.warn(f"linked_pars: {invalid_pars} no longer fit parameters and have been removed.")
                else:
                    raise ValueError(f"Invalid linked_pars: {invalid_pars} not fit parameter{'s' if len(invalid_pars) > 1 else ''}.")

            self._linked_pars = linked_pars

        self._verified['linked_pars'] = True

    def _sort_param_config(self, param_config):
        """
        Sort a given paramter config against the interpolator and store attributes in class.
        Performs the following operations:
        - Checks match against interpolator parameters
        - Consolidates priors into lists for each parameter
        - Bounds interpolator with additional priors if needed
        """
        param_config_full = {}
        # consolidate priors - any single priors made into list of length one, floats forced, check types
        for par, res in param_config.items():
            if isinstance(res, list):
                if all(isinstance(r, BasePrior) for r in res):
                    param_config_full[par] = res
                    continue
                else:
                    raise TypeError(f"Config for parameter {par} contains mixed constraints, must contain only instances of priors.")
            elif isinstance(res, BasePrior):
                param_config_full[par] = [res]
                continue
            elif isinstance(res, (float, int)):
                param_config_full[par] = float(res)
            elif res is None:
                param_config_full[par] = None
            else:
                raise TypeError(f"Invalid type in param_config for parameter {par}: {type(res)}.")

        # bound interpolator with priors if needed
        param_config_full, interp_bounding_priors = self._bound_interpolator(param_config_full)

        # this part can be removed for e.g. emcee
        # or better, add this on in a subclass and call super of the base class first
        unbounded_params = [k for k in param_config_full if param_config_full[k] is None]
        if len(unbounded_params) > 0:
            raise ValueError(f"Parameters {unbounded_params} are unbounded and require a prior for fitting with dynesty.")

        return param_config_full, interp_bounding_priors

    def _bound_interpolator(self, param_config):
        """
        Bounds self.interpolator with uniform priors if it is unbounded by
        existing priors in param_config

        Returns modified param_config and dictionary containing only the new priors.
        """

        new_priors = {par: [] for par in self.interpolator.param_names}

        # add uniform priors for bounds of interpolator
        for par, bounds in zip(self.interpolator.param_names, self.interpolator.bounds):

            res = param_config[par]
            # fixed parameter, check bounds then ignore
            if isinstance(res, float):
                if bounds[0] <= res <= bounds[1]:
                    continue
                else:
                    raise ValueError(f"Fixed value {res} for parameter {par} is out of bounds of interpolator ({bounds})")

            # free parameter, bound
            elif res is None:
                prior = UniformPrior(bounds[0], bounds[1])
                #param_config[par].append(prior)
                param_config[par] = [prior]
                new_priors[par].append(prior)
                continue

            # get any existing uniform priors
            existing_UPs = [p for p in res if isinstance(p, UniformPrior)]
            if len(existing_UPs) > 0:
                lb = min([p.lower for p in existing_UPs])
                ub = max([p.upper for p in existing_UPs])

                # if existing priors already bound interpolator, skip
                if lb > bounds[0] and ub < bounds[1]:
                    continue

            prior = UniformPrior(bounds[0], bounds[1])
            param_config[par].append(prior)
            new_priors[par].append(prior)

        return param_config, new_priors

    def _consolidate_priors_dynesty(self, param_config, bg_param_map=None):
        """
        Perform additional checks on param config as is necessary for creating a prior transform.
        - consolidate into one prior per parameter
        - ensure every free parameter is bounded by a prior
        """

        # clip gauss priors to this range (default 5)
        sigma_trunc = self._trunc_to_sigma

        bg_param_map = self.bg_param_map if bg_param_map is None else bg_param_map

        all_params = self._interp_params + list(bg_param_map) + self._non_interp_params
        unused_params = [p for p in param_config if p not in all_params]

        # ndims is number of non-fixed parameters take int just in case
        fixed_pars = [p for p, v in param_config.items() if isinstance(v, (float, int))]
        fit_pars = [p for p in param_config if p not in fixed_pars + unused_params]

        priors = {p: param_config[p] for p in fit_pars}
        priors_simplified = {}

        # need to consolidate priors first
        for par, prior_list in priors.items():

            # if we have only one prior for this parameter, this is fine
            if len(prior_list) == 1:
                priors_simplified[par] = prior_list[0]
                continue

            # if we have more than one non-uniform prior, we cannot trivially combine
            uniform_priors = [prior for prior in prior_list if isinstance(prior, UniformPrior)]
            non_uniform_priors = [prior for prior in prior_list if not isinstance(prior, UniformPrior)]

            if len(non_uniform_priors) > 1:
                raise ValueError(f"Invalid priors for '{par}'"
                                 f" - cannot combine priors: {non_uniform_priors}.")

            # find union of all uniform priors
            uniform_lower, uniform_upper = -np.inf, np.inf
            for prior in uniform_priors:
                uniform_lower = max(prior.lower, uniform_lower)
                uniform_upper = min(prior.upper, uniform_upper)

            # if we only have multiple uniform priors we are (potentially) done
            if len(non_uniform_priors) == 0:
                # if all uniform priors together are still unbounded,
                union_prior = UniformPrior(uniform_lower, uniform_upper)
                priors_simplified[par] = union_prior
                continue

            other_prior = non_uniform_priors[0]

            if not isinstance(other_prior, (GaussianPrior, TruncnormPrior)):
                # if we have a non Gaussian/Truncnorm prior, we do not know how to combine
                # must only pass that prior instead
                raise ValueError(
                    f"Invalid priors for '{par}' - cannot combine prior: {other_prior} with {uniform_priors}. "
                    "Please either remove the UniformPriors or use a Gaussian/Truncnorm prior instead."
                )

            else:
                # otherwise, we have a Gaussian (or truncated Gaussian) distribution, which we need to truncate
                # to the union of the uniform priors and the truncated edges
                other_lower = other_prior.lower if hasattr(other_prior, 'lower') else -np.inf
                other_upper = other_prior.upper if hasattr(other_prior, 'upper') else np.inf

                lower = max(other_lower, uniform_lower)
                upper = min(other_upper, uniform_upper)

                mean, sigma = other_prior.mean, other_prior.sigma
                trunc_lower, trunc_upper = (mean - lower) / sigma, (upper - mean) / sigma

                union_prior = TruncnormPrior(mean, sigma, trunc_lower=trunc_lower, trunc_upper=trunc_upper)
                priors_simplified[par] = union_prior

        priors_bounded = {}
        # now check simplified priors are bounded
        for par, prior in priors_simplified.items():

            if isinstance(prior, UniformPrior):
                if prior.bounded is False:
                    raise ValueError(f"Union of priors ({prior}) for parameter `{par}` are unbounded.")
                priors_bounded[par] = [prior]
            elif isinstance(prior, (GaussianPrior, TruncnormPrior)):
                if hasattr(prior, 'lower') and hasattr(prior, 'upper'):
                    # truncated prior, clip sigmas
                    warn = False
                    if prior.trunc_lower > sigma_trunc:
                        lower_new = sigma_trunc
                        warn = True
                    else:
                        lower_new = prior.trunc_lower
                    if prior.trunc_upper > sigma_trunc:
                        upper_new = sigma_trunc
                        warn = True
                    else:
                        upper_new = prior.trunc_upper

                    new_prior = TruncnormPrior(prior.mean, prior.sigma, trunc_lower=lower_new, trunc_upper=upper_new)

                    if warn:
                        warnings.warn(f"Prior {prior} bounds for parameter '{par}'' exceed sigma tolerance ({sigma_trunc}). Truncated to {new_prior}.")
                else:
                    # gaussian prior, clip sigmas?
                    warnings.warn(f"Prior {prior} for parameter '{par}' is not truncated. Truncating to {sigma_trunc} sigma range.")
                    new_prior = TruncnormPrior(prior.mean, prior.sigma, trunc=sigma_trunc)

                priors_bounded[par] = [new_prior]

        return priors_bounded

    def _update_config(self, param_config, bg_model=None, bg_param_map=None):
        """
        Updates the param_config, model, param_map and some auxilliary attributes.
        """
        # overrides to check against new options
        bg_model = self.bg_model if bg_model is None else bg_model
        bg_param_map = self.bg_param_map if bg_param_map is None else bg_param_map

        self._param_config = param_config
        self._bg_model = bg_model
        self._bg_param_map = bg_param_map

        self._verified['bg_param_map'] = True
        self._verified['bg_model'] = True
        self._verified['param_config'] = True
        self._verified['linked_pars'] = True

        all_params = self._interp_params + list(bg_param_map) + self._non_interp_params
        unused_params = [p for p in param_config if p not in all_params]

        # ndims is number of non-fixed parameters take int just in case
        fixed_pars = [p for p, v in param_config.items() if isinstance(v, (float, int))]
        fit_pars = [p for p in param_config if p not in fixed_pars + unused_params]

        self._fit_pars = fit_pars
        self._fixed_pars = fixed_pars

        self._sort_priors()

    def _sort_priors(self):
        """
        Sorts priors from self.param_config into self._uniform_priors and
        self._other_priors, so that uniform priors can be handled first in
        the sampling function
        """

        # consolidate uniform priors so they can be applied first
        uniform_prior_dict = {}
        other_prior_dict = {}

        for par, res in self.param_config.items():

            if res is None or isinstance(res, float):
                continue

            uniform_priors = [p for p in res if isinstance(p, UniformPrior)]
            other_priors = [p for p in res if not isinstance(p, UniformPrior)]

            if len(uniform_priors) > 0:
                uniform_prior_dict[par] = uniform_priors

            if len(other_priors) > 0:
                other_prior_dict[par] = other_priors

        self._uniform_priors = uniform_prior_dict
        self._other_priors = other_prior_dict

    ####  Verifying data and masks  ####

    def _verify_data(self, data: list[np.ndarray], _set=False):
        """
        Verify shape of data against wavelengths from interpolator
        """
        spec_shapes = [spec.shape for spec in data]

        if len(set(spec_shapes)) != 1:
            raise ValueError("Spectra in data do not have matching dimensions.")

        spec_shape = spec_shapes[0]

        if spec_shape[1] != 3:
            raise IndexError(f"Spectra in data must have 3 columns: wavelength, flux, and flux errors (found {spec_shape[1]}).")

        wv_tol = self.wv_tol
        wv_devs = np.array([np.abs(self.interpolator.wvs - spec[:, 0]).max() for spec in data])

        n_invalid = np.count_nonzero(wv_devs > wv_tol)
        if n_invalid > 0:
            invalid_idxs = np.arange(len(data))[wv_devs > wv_tol]
            if n_invalid == 1:
                raise ValueError(f"Spectrum at index {invalid_idxs[0]} deviates above tolerance: ({wv_devs.max()} > {wv_tol}).")
            else:
                raise ValueError(f"Spectra at indices {invalid_idxs} deviate above tolerance: ({wv_devs} > {wv_tol}).")

        if _set:
            self._data = data
            self._verified['data'] = True

    def _convert_mask(self, mask):
        """
        Verify and apply mask against spectrum
        """

        wvs = self.interpolator.wvs

        if mask is None:
            mask_arr = np.ones(wvs.size, dtype=bool)
        elif isinstance(mask, list) and isinstance(mask[0], (tuple, list)):
            mask_arr = np.ones(wvs.size, dtype=bool)

            for lb, ub in mask:
                if lb > ub:
                    mask_arr = mask_arr & ~((wvs > ub) & (wvs < lb))
                else:
                    mask_arr = mask_arr & ~((wvs > lb) & (wvs < ub))

        elif isinstance(mask, list) and isinstance(mask[0], bool):
            mask_arr = np.array(mask, dtype=bool)
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            mask_arr = mask.copy()
        else:
            raise TypeError("Invalid mask format provided.")

        return mask_arr

    def _verify_mask(self, mask, _set=False):
        mask_arr = self._convert_mask(mask)

        if np.count_nonzero(mask_arr) == 0:
            raise ValueError(f"Mask covers entire wavelength range ({self.interpolator.wvs.min()} to {self.interpolator.wvs.max()}).")

        if _set:
            self._mask = mask
            self._mask_arr = mask_arr
            self._verified['mask'] = True

    def _verify_linemask(self, mask, _set=False):
        mask_arr = self._convert_mask(mask)

        if mask is not None and np.count_nonzero(mask_arr) == 0:
            raise ValueError(f"Line mask does not overlap data wavelength range ({self.interpolator.wvs.min()} to {self.interpolator.wvs.max()}).")

        if _set:
            self._linemask = mask
            self._linemask_arr = ~mask_arr
            self._verified['linemask'] = True

    ####  Initialising sampler  ####

    def initialise(
        self,
        data: list[np.ndarray[float]],
        mask: list[tuple[float, float]] | None | _NoMaskProvided = _NoMaskProvided,
        linemask: list[tuple[float, float]] | None | _NoMaskProvided = _NoMaskProvided,
        sampler_kwargs: dict[str, Any] | None | _NoKwargs = None,
        run_kwargs: dict[str, Any] | None | _NoKwargs = None
    ):
        """
        Initialise the fitter with a spectrum to fit to, and configure the starting conditions
        for the walkers with `init_config`. An optional mask may be provided.

        Parameters:
        spectrum: np.ndarray[float]
            A spectrum with 3 columns: wavelength, flux, flux errors.
            Must have the same wavelengths as `self.interpolator`.
        init_config: dict of str: tuple[float, float]
            Configuration for scattering the initial state of the MCMC walkers.
            Must contain a key for each fit (i.e. not fixed) parameter.
            Each tuple is a pair of (central value, scatter size).
            Walkers are scattered according to self.scatter_method (default: np.random.normal).
        mask: np.ndarray[bool], optional
            Boolean mask for spectrum, ranges set to False are ignored in fitting.
        """
        # defaults with sentinel (allows for e.g. mask=None to remove mask)
        mask = self.mask if mask is _NoMaskProvided else mask
        linemask = self.linemask if linemask is _NoMaskProvided else linemask
        sampler_kwargs = self.sampler_kwargs if sampler_kwargs is _NoKwargs else sampler_kwargs
        run_kwargs = self.run_kwargs if run_kwargs is _NoKwargs else run_kwargs

        sampler_kwargs = dict() if sampler_kwargs is None else sampler_kwargs
        run_kwargs = dict() if run_kwargs is None else run_kwargs

        # verify all inputs without setting
        self._verify_data(data, _set=False)
        self._verify_mask(mask, _set=False)
        self._verify_linemask(linemask, _set=False)

        self._init_sampler(data, mask, linemask, sampler_kwargs, run_kwargs)

        # if they pass, set all
        self._data = data
        self._verified['data'] = True
        self._verify_mask(mask, _set=True)
        self._verify_linemask(linemask, _set=True)

    def _init_sampler(self, data, mask, linemask, sampler_kwargs=None, run_kwargs=None):
        """
        Initialise dynesty sampler for given data
        """
        # defaults - no sentinel, this method is only called via initialise()
        sampler_kwargs = dict() if sampler_kwargs is None else sampler_kwargs
        run_kwargs = dict() if run_kwargs is None else run_kwargs

        llh_func = self._get_llh_func(data, mask, linemask)
        pt_func = self._get_pt_func(data)

        #pars_fixed = [self.param_config[p] for p in self.fixed_pars]
        n_dims = len(data) * (len(self.fit_pars) - len(self.linked_pars)) + len(self.linked_pars)

        self._sampler = NestedSampler(
            llh_func,
            pt_func,
            n_dims,
            **sampler_kwargs
        )

        self._sampler_kwargs = sampler_kwargs
        self._run_kwargs = run_kwargs

    def _get_pt_func(self, data):
        """
        """
        # simplify param config to single elements
        priors = {k: self.param_config[k][0] for k in self.fit_pars}

        # create list of par names for each par in u vector
        fit_nonlinked_pars = [p for p in self.fit_pars if p not in self.linked_pars]
        pars = self.linked_pars + [p for p in fit_nonlinked_pars for i in range(len(data))]

        def prior_transform(u):

            x = [priors[p].transform(v) for p, v in zip(pars, u)]

            return np.array(x)

        return prior_transform

    def _get_llh_func(self, data, mask, linemask):

        # get some class attributes needed for llh function
        interpolator = self.interpolator
        bg_model = self.bg_model
        bg_param_map = self.bg_param_map

        # cut wavelengths off data - not needed for fitting
        data = [spec[:, 1:] for spec in data]
        mask = self._convert_mask(mask)
        do_linemask = linemask is not None
        linemask = ~self._convert_mask(linemask)

        n_masked = mask.sum()
        n_linemasked = linemask.sum()

        fit_pars = self.fit_pars
        linked_pars = self.linked_pars
        fit_nonlinked_pars = [p for p in fit_pars if p not in linked_pars]

        n_linked = len(linked_pars)
        n_spec = len(data)

        fixed_pars_dict = {k: self.param_config[k] for k in self.fixed_pars}

        #print('nmasked:', n_masked, '  nlinemasked:', n_linemasked)

        def llh_func(pars):
            # pars is the vector of parameters
            # need to separate into linked/non-linked
            # we will pass parameters in such that:
            # the first n_linked parameters will be the linked values
            # the next n_spec parameters will be the first parameter repeated n_spec times
            # ... repeat for each parameter
            # e.g. [linked_par_1, linked_par_2, fit_par_1, fit_par_1, fit_par_2, fit_par_2, ...]
            # for 2 spectra with 2 linked parameters

            # get linked pars
            linked_pars_dict = {p: v for p, v in zip(linked_pars, pars[:n_linked])}

            # list of params per spec, combined with fixed and linked pars
            param_dict_list = [
                {
                    p: v
                    for p, v in zip(fit_nonlinked_pars, pars[n_linked + i_spec::n_spec])
                } | fixed_pars_dict | linked_pars_dict for i_spec in range(n_spec)
            ]

            # # evaluate priors on every spectrum individually
            # priors = np.array([_eval_priors(param_dict) for param_dict in param_dict_list])

            # if np.any(~np.isfinite(priors)):
            #     return -np.inf

            models = [
                self._model(
                    pars=param_dict,
                    interpolator=interpolator,
                    bg=bg_model,
                    bg_par_map=bg_param_map
                )
                for param_dict in param_dict_list
            ]

            chisqs = np.array([
                np.nansum((spec[:, 0][mask] - model[mask])**2 / spec[:, 1][mask]**2)
                for model, spec in zip(models, data)
            ])

            # optional linemasking
            if do_linemask:
                chisqs_lm = np.array([
                    np.nansum((spec[:, 0][linemask] - model[linemask])**2 / spec[:, 1][linemask]**2)
                    for model, spec in zip(models, data)
                ])
                # evenly weighted chi-squared
                chisqs = 0.5 * (chisqs + (n_masked / n_linemasked) * chisqs_lm)

            llh = -0.5 * chisqs.sum()  # + np.log(priors).sum()

            return llh

        return llh_func

    ####  Running sampler  ####

    def run(
        self,
        progress: bool = True,
        overwrite: bool = False,
        run_kwargs: dict[str, Any] | None | _NoKwargs = None,
        sampler_kwargs: dict[str, Any] | None | _NoKwargs = None,
        data: list[np.ndarray[float]] | None = None,
        mask: list[tuple[float, float]] | None | _NoMaskProvided = _NoMaskProvided,
        linemask: list[tuple[float, float]] | None | _NoMaskProvided = _NoMaskProvided,
    ):
        """
        Run the sampling process for `nsteps` number of steps.
        Will continue from any previous runs unless continue_ = False (default True).
        Set progress=False to disable the tqdm progress bar.

        The spectrum, init_config, and mask may be set (or overridden) here.
        Both `spectrum` and `init_config` must have been set previously to start a run.
        See SpectrumFitter.initialise for more information on these parameters.

        """
        # defaults with sentinel for masks (allows for mask=None to remove mask)
        run_kwargs = self.run_kwargs if run_kwargs is _NoKwargs else run_kwargs
        sampler_kwargs = self.sampler_kwargs if sampler_kwargs is _NoKwargs else sampler_kwargs
        data = self.data if data is None else data
        mask = self.mask if mask is _NoMaskProvided else mask
        linemask = self.linemask if linemask is _NoMaskProvided else linemask

        run_kwargs = dict() if run_kwargs is None else run_kwargs
        sampler_kwargs = dict() if sampler_kwargs is None else sampler_kwargs

        # if something changed, reinitialise
        if (
            data is not self.data
            or mask is not self.mask
            or linemask is not self.linemask
            or sampler_kwargs is not self.sampler_kwargs
        ):
            self.initialise(data, mask, linemask, sampler_kwargs, run_kwargs)

        # check if we can run
        error_report = self.run_check()
        if error_report is not None:
            raise RuntimeError("Cannot run, encountered errors verifying configuration:\n" + error_report)

        # run
        self._run_sampler(progress=progress, overwrite=overwrite, run_kwargs=run_kwargs)

    def run_check(self):
        """
        Check fitter is correctly initialised and verified and run may proceed.
        """
        error_report = []

        for par, status in self._verified.items():
            if status is False:
                error_report.append(f"{par} is invalid.")

        error_report = None if error_report == [] else error_report
        return error_report

    ####  Backend model methods  ####

    def _eval_priors(self, param_dict):
        """
        Evaluates priors on a dict of proposed parameters.
        """

        # first go through uniform priors
        #uniform_priors = [prior.eval(param_dict[par]) for par, priors in self._uniform_priors.items() for prior in priors]
        # this version should only deal with parameters in param_dict, so e.g. bg_model can be sampled over
        uniform_priors = [prior.eval(val) for par, val in param_dict.items() for prior in self._uniform_priors.get(par, [])]

        if any(p == -np.inf for p in uniform_priors):
            return -np.inf

        #probs = [prior.eval(param_dict[par]) for par, priors in self._other_priors.items() for prior in priors]
        # this version should only deal with parameters in param_dict, so e.g. bg_model can be sampled over
        probs = [prior.eval(val) for par, val in param_dict.items() for prior in self._other_priors.get(par, [])]

        return np.prod(probs)

    def _run_sampler(self, progress=True, overwrite=False, run_kwargs=None):
        """
        Internal sampling method
        """

        format_version = 1

        # default run_kwargs if None
        run_kwargs = self.run_kwargs if run_kwargs is None else run_kwargs

        progress = run_kwargs.pop('print_progress', progress)

        resume = run_kwargs.pop('resume', False)
        checkpoint_file = run_kwargs.pop('checkpoint_file', None)
        checkpoint_every = run_kwargs.pop('checkpoint_every', 60)

        llh_func = self.sampler.loglikelihood.loglikelihood.func
        pt_func = self.sampler.prior_transform.func

        if resume:
            # load sampler
            with open(checkpoint_file, 'rb') as p:
                res = pickle.load(p)

            resume_sampler = res['sampler']
            resume_sampler_kwargs = res['sampler_kwargs']

            # ensure new sampler is valid in current class
            if resume_sampler.ndim != self.sampler.ndim:
                raise ValueError(
                    f"Sampler dimensions ({resume_sampler.ndim}) from {checkpoint_file} does not "
                    f"match dimensions of current sampler ({self.sampler.ndim})"
                )

            if resume_sampler_kwargs != self.sampler_kwargs:
                raise ValueError(
                    f"Sampler kwargs from {checkpoint_file} do not match those of current sampler."
                    f"Checkpoint file: \n{resume_sampler_kwargs}\n"
                    f"Current sampler: \n{self.sampler_kwargs}"
                )

            if DYNESTY_VERSION != res['version']:
                warnings.warn(
                    f"The dynesty version in the checkpoint file ({res['version']}) "
                    f"does not match the current dynesty version ({DYNESTY_VERSION}). "
                    "This is NOT guaranteed to work."
                )

            if format_version != res['format_version']:
                raise RuntimeError(
                    f"Incorrect format version ({checkpoint_file} has {res['format_version']}, "
                    f"expected {format_version})."
                )

            # add current llh and pt funcs
            resume_sampler.loglikelihood.loglikelihood.func = llh_func
            resume_sampler.prior_transform.func = pt_func

            sampler = resume_sampler

        else:
            if (
                checkpoint_file is not None
                and os.path.exists(checkpoint_file)
                and overwrite is False
            ):
                raise FileExistsError(
                    f"Checkpoint file {checkpoint_file} already exists. "
                    "Did you mean to resume from this checkpoint file? "
                    "If so, pass resume=True in run_kwargs. "
                    "Otherwise, pass overwrite=True to overwrite the existing checkpoint file."
                )

            sampler = self.sampler

        # any checkpoint checks?

        # set up progress bar
        if progress is True:
            custom_bar_fmt = "{n_fmt}it [{elapsed}, {rate_fmt}, {postfix}]"
            pbar, print_func = get_print_func(
                run_kwargs.pop(
                    'print_func',
                    partial(print_fn, pbar=tqdm(
                        total=run_kwargs.get('maxiter', np.inf),
                        bar_format=custom_bar_fmt,
                    ))
                ),
                progress
            )
        else:
            def print_func(*args, **kwargs):
                pass
            pbar = None

        add_live = run_kwargs.get('add_live', True)
        nlive = run_kwargs.get('nlive', 500)
        dlogz = run_kwargs.get('dlogz', 1e-3 * (nlive - 1) + 0.01 if add_live else 0.01)
        logl_max = run_kwargs.get('logl_max', np.inf)

        ncall = sampler.ncall

        # update if changed
        if run_kwargs != self.run_kwargs:
            self._run_kwargs = run_kwargs

        # update sampler before starting (if resume
        if resume:
            self._sampler = sampler

        for it, res in enumerate(sampler.sample(**run_kwargs)):

            ncall += res.nc

            print_func(res, sampler.it - 1, ncall, dlogz=dlogz, logl_max=logl_max)

            if checkpoint_file is not None and it % checkpoint_every == 0:
                try:
                    with open(checkpoint_file, 'wb') as p:
                        out = {
                            'sampler': sampler,
                            'sampler_kwargs': self.sampler_kwargs,
                            'version': DYNESTY_VERSION,
                            'format_version': format_version
                        }
                        out['sampler'].loglikelihood.loglikelihood.func = None
                        out['sampler'].prior_transform.func = None
                        pickle.dump(out, p)
                except Exception as err:
                    raise
                finally:
                    sampler.loglikelihood.loglikelihood.func = llh_func
                    sampler.prior_transform.func = pt_func

        # for i, res in enumerate(self.sampler.add_live_points()):
        #     print_func(res, it, res.ncall, add_live_it=i + 1, dlogz=dlogz, logl_max=logl_max)
        sampler.add_final_live(progress, print_func)

        # final write
        if checkpoint_file is not None:
            try:
                with open(checkpoint_file, 'wb') as p:
                    out = {
                        'sampler': sampler,
                        'sampler_kwargs': self.sampler_kwargs,
                        'version': DYNESTY_VERSION,
                        'format_version': format_version
                    }
                    out['sampler'].loglikelihood.loglikelihood.func = None
                    out['sampler'].prior_transform.func = None
                    pickle.dump(out, p)
            except Exception as err:
                raise
            finally:
                sampler.loglikelihood.loglikelihood.func = llh_func
                sampler.prior_transform.func = pt_func

        if pbar is not None:
            pbar.close()

    # def get_chain(self, *args, **kwargs):
    #     return self.sampler.get_chain(*args, **kwargs)

    def get_results_index(self, spec_index: int | list | None = None) -> dict[str, int]:
        """
        Return the a dictionary of parameter: result index for a spectrum of
        index `spec_index` in the data.
        """

        if self.data is None:
            raise RuntimeError("Cannot determine results indices - no data has been set.")

        if spec_index is None:
            spec_indices = list(range(self.nspec))
        elif isinstance(spec_index, int):
            spec_indices = [spec_index]
        else:
            spec_indices = spec_index

        fit_pars = self.fit_pars
        linked_pars = self.linked_pars
        fit_nonlinked_pars = [p for p in fit_pars if p not in linked_pars]
        n_linked = len(linked_pars)
        n_spec = self.nspec

        indices = list(range(self.ndims))

        # get linked pars
        linked_pars_dict = {p: v for p, v in zip(linked_pars, indices[:n_linked])}

        index_dict_list = [None] * len(spec_indices)

        for i_, i_spec in enumerate(spec_indices):

            fit_nonlinked_pars_dict = {
                p: i
                for p, i
                in zip(fit_nonlinked_pars, indices[n_linked + i_spec::n_spec])
            }

            # combine
            fit_pars_dict = linked_pars_dict | fit_nonlinked_pars_dict

            # reorder
            fit_pars_dict = {p: fit_pars_dict[p] for p in self.param_config if p in fit_pars_dict}

            index_dict_list[i_] = fit_pars_dict

        if isinstance(spec_index, int):
            return index_dict_list[0]
        else:
            return index_dict_list

    ####  I/O  ####

    def to_pickle(self, path):
        """
        Write fitter to pickle file.
        """
        # handle local functions
        if self.sampler is not None:
            llh_func = self.sampler.loglikelihood.loglikelihood.func
            self.sampler.loglikelihood.loglikelihood.func = None
            pt_func = self.sampler.prior_transform.func
            self.sampler.prior_transform.func = None

        try:
            with open(path, 'wb') as p:
                pickle.dump(self, p)
        except Exception as err:
            raise
        finally:
            # set local functions back to their values
            if self.sampler is not None:
                self.sampler.loglikelihood.loglikelihood.func = llh_func
                self.sampler.prior_transform.func = pt_func

    @classmethod
    def from_pickle(cls, path):
        """
        Construct fitter from pickle file.
        """
        with open(path, 'rb') as p:
            fitter = pickle.load(p)

        if fitter.sampler is not None:
            llh_func = fitter.sampler.loglikelihood.loglikelihood.func
            pt_func = fitter.sampler.prior_transform.func

            if pt_func is None:
                pt_func = fitter._get_pt_func(fitter.data)
            if llh_func is None:
                llh_func = fitter._get_llh_func(fitter.data, fitter.mask, fitter.linemask)

            fitter.sampler.loglikelihood.loglikelihood.func = llh_func
            fitter.sampler.prior_transform.func = pt_func

        return fitter

    ####  Properties and setters  ####

    @property
    def param_config(self):
        return self._param_config

    @param_config.setter
    def param_config(self, param_config):
        self._verify_param_config(param_config)

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, interpolator):
        if self.param_config is None:
            self._verify_interpolator(interpolator)
        else:
            raise NotImplementedError("Changing interpolator in an already initialised fitter instance not supported.")

    @property
    def linked_pars(self):
        return self._linked_pars

    @linked_pars.setter
    def linked_pars(self, linked_pars):
        if isinstance(linked_pars, str):
            linked_pars = [linked_pars]
        self._verify_linked_pars(linked_pars)

    @property
    def bg_model(self):
        return self._bg_model

    @bg_model.setter
    def bg_model(self, bg_model):
        self._verify_bg_model(bg_model)

    @property
    def bg_param_map(self):
        return self._bg_param_map

    @bg_param_map.setter
    def bg_param_map(self, bg_param_map):
        self._verify_bg_param_map(bg_param_map)

    @property
    def bg_params(self):
        return list(self.bg_param_map)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._verify_data(data, _set=True)
        self.initialise(data, mask=self.mask, linemask=self.linemask)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._verify_mask(mask, _set=True)
        self.initialise(data=self.data, mask=mask, linemask=self.linemask)

    @property
    def mask_arr(self):
        return self._mask_arr

    @property
    def linemask(self):
        return self._linemask

    @linemask.setter
    def linemask(self, linemask):
        self._verify_linemask(linemask, _set=True)
        self.initialise(data=self.data, mask=self.mask, linemask=linemask)

    @property
    def linemask_arr(self):
        return self._linemask_arr

    @property
    def fit_pars(self):
        return self._fit_pars

    @property
    def fixed_pars(self):
        return self._fixed_pars

    @property
    def ndims(self):
        n_spec = 1 if self.data is None else len(self.data)
        return n_spec * (len(self.fit_pars) - len(self.linked_pars)) + len(self.linked_pars)

    @property
    def npars(self):
        return len(self.param_config)

    @property
    def nspec(self):
        return len(self.data) if self.data is not None else 0

    @property
    def sampler(self):
        return self._sampler

    @property
    def sampler_kwargs(self):
        return self._sampler_kwargs

    @property
    def run_kwargs(self):
        return self._run_kwargs

    @property
    def initialised(self):
        return self.run_check() is None
