from astra.utils.utils import _mask_interp
from astra.utils._helpers import automask, xcheck_spectra, check_vbinned, deprecated_, dummy_pbar
from astra.fitting.core import BasePrior, UniformPrior, GaussianPrior, _PriorOrPriors
import astra.fitting.specmodels as specmodels

from typing import Callable
import warnings
import numpy as np
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d
import emcee

SpectrumInterpolator = specmodels.SpectrumInterpolator

#### backend ####


def optsub_standard(
    x1: np.ndarray[float],
    x2: np.ndarray[float],
    x1_err: np.ndarray[float] | None = None,
    mask: np.ndarray[bool] | None = None,
    fwhm: float = 0.
) -> (float, float, float, int):
    """
    Calculate the optimal factor for subtraction between two spectra.
    Optimal subtraction factor minimises the residuals in the equation:

    x1 - factor * x2 = residuals

    Parameters:
    x1: numpy.ndarray
        Optimal subtraction input.
    x2: numpy.ndarray
        Subtractor to remove from input, same size as `x1`.
    x1_err: numpy.ndarray or None, default None
        Errors on `x1`, optional, same size as `x1`.
    mask: numpy.ndarray or None, default None
        Boolean array to mask `x1`, optional, same size as `x1`.

    Returns:
    chisq: float
        Chi-squared value of the residuals
    factor: float
        Optimal subtraction factor
    factor_err: float
        Error on the optimal subtraction factor
    dof: int
        Degrees of freedom (number of unmasked pixels) in the subtraction

    """
    # input checks - negligible impact on runtime
    if x1.ndim != 1: raise ValueError('x1 must be one-dimensional.')
    if x2.ndim != 1: raise ValueError('x2 must be one-dimensional.')

    if not (x1.shape == x2.shape):
        raise IndexError(f"x1 and x2 must all be the same shape: ({x1.shape}, {x2.shape})")
    if mask is not None and mask.shape != x1.shape:
        raise IndexError(f"mask does not match sizes of x1 and x2: ({x1.shape})")
    if x1_err is not None and x1_err.shape != x1.shape:
        raise IndexError(f"x1_err and x1 do not have the same shape: ({x1_err.shape}, {x1.shape})")
    if 0 < fwhm < 1:
        warnings.warn("Warning: non-zero fwhm < 1 ignored", RuntimeWarning)

    mask = np.ones(x1.shape, dtype=bool) if mask is None else mask

    if fwhm >= 1:
        # molly fwhm approx. translates to this
        fwhm_ = fwhm / (2 * (2 * np.log(2))**0.5)
        # replace masked regions with linear interpolation
        x1, x2 = _mask_interp(x1, mask), _mask_interp(x2, mask)

        # smooth
        x1_smoothed = gaussian_filter1d(x1, fwhm_, mode='nearest', radius=50)
        x2_smoothed = gaussian_filter1d(x2, fwhm_, mode='nearest', radius=50)

    dof = np.count_nonzero(mask)
    weights = 1 / x1_err[mask]**2 if x1_err is not None else np.ones(dof)

    if fwhm > 1:
        d = x1[mask] - x1_smoothed[mask]
        t = x2[mask] - x2_smoothed[mask]
    else:
        d = x1[mask]
        t = x2[mask]

    sum1 = (weights * d * t).sum()
    sum2 = (weights * t * t).sum()
    sum3 = (weights * d * d).sum()

    factor = sum1 / sum2
    chisq = sum3 - sum1**2 / sum2

    factor_err = 1 / sum2**0.5

    return chisq, factor, factor_err, dof


#### frontend ####


def optsub(
    obs: np.ndarray[float] | list[np.ndarray[float]],
    templ: list[np.ndarray[float]],
    mask: np.ndarray[float] | list[tuple[float, float]] | None = None,
    progress: bool = True,
) -> np.ndarray[float]:
    """
    Perform optimal subtraction using the standard method.
    Optimal subtraction is performed for each observed spectrum,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.
    Any systemic radial velocities should be removed also (see sincshift).

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs: np.ndarray or list of np.ndarray
        Array or list of arrays of the observed spectra. If list, a list of results is returned.
        All spectra must have identical wavelength scales.
        Spectra should have two, or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templ: list of np.ndarray
        List of arrays of template spectra.
        All spectra must have identical wavelength scales, which match the observed spectra.
        Each spectrum should have two columns: wavelength and flux.
        Additional columns will be ignored. It is assumed noise is dominated by observed spectra.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array matching the
        observed and template spectra, or a list of 2-tuples defining upper and lower bounds to exclude.
    progress: bool, default True
        Controls display of tqdm progress bar, optional. False to disable

    Returns:
    optsub_results: np.ndarray or list of np.ndarray
        Output array, or list of arrays, containing optsub results, 3 columns, one row per template spectrum.
        Columns are chi-squared of subtraction, optimal subtraction factor, and error on the factor.
        If a single observed spectrum is provided, a single results array is returned.

    """
    obs_ = [obs] if isinstance(obs, np.ndarray) else obs

    # check spectra match
    try:
        obs_w_errors, templ_w_errors = xcheck_spectra(obs_, templ)
    except Exception as e:
        msg = e.args[0].replace('spectra1', 'obs').replace('spectra2', 'template')
        raise type(e)(msg, *e.args[1:])

    n_obs = len(obs_)
    n_templ = len(templ)
    obs_wvs = obs_[0][:, 0]

    if not check_vbinned(obs_wvs):
        warnings.warn("Spectra not uniform in velocity space - results may be meaningless.", RuntimeWarning)

    # check and apply mask (convert to boolean mask)
    mask_ = automask(obs_wvs, mask)

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar

    optsub_results = [None] * n_obs

    with pbar_manager(desc='optsub: ', total=n_obs * n_templ) as pbar:

        for i_obs, o in enumerate(obs_):

            obs_wvs, obs_flux = o[:, 0], o[:, 1]
            obs_flux_err = o[:, 2] if obs_w_errors else None

            # mask out any nans
            obs_mask_ = (mask_ & ~np.isnan(obs_flux))
            obs_mask_ = (obs_mask_ & (obs_flux_err >= 0)) if obs_w_errors else obs_mask_

            optsub_result = np.zeros((n_templ, 3))

            for i_templ, t in enumerate(templ):

                t_wvs, t_flux = t[:, 0], t[:, 1]

                # mask out any nans
                t_mask_ = (obs_mask_ & ~np.isnan(t_flux))

                # perform optsub
                chisq, factor, factor_err, dof = optsub_standard(obs_flux, t_flux, obs_flux_err, t_mask_)

                # save results to array
                optsub_result[i_templ] = chisq, factor, factor_err

                pbar.update(1)

            optsub_results[i_obs] = optsub_result

    return optsub_results[0] if isinstance(obs, np.ndarray) else optsub_results


#### MCMC spectrum fitting ####


# _bg_models = {
#     'flat': bg_flat, 
#     'bb': bg_blackbody, 
#     'bbscaled': bg_blackbody_scaled, 
#     'power': bg_powerlaw,
#     'powerscaled': bg_powerlaw_scaled,
#     'one': bg_ones
# }

_bg_models = {}

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
        np.array(),
        np.array(),
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

        # verify and set background model
        if isinstance(bg_model, str):
            if bg_model not in _bg_models:
                raise ValueError(f"Invalid choice for background model: {bg_model}. Valid choices are {list(_bg_models.keys())}.")
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
                    raise TypeError(f"Config for parameter {par} contains mixed constraints, must contain only instances of UniformPrior and GaussianPrior.")
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
        # this should be faster than a for loop
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

            model = _spec_model(
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

    def _create_init_samples(self, init_config, method='normal'):
        """
        Create samples from normal distribution using init_config
        """

        if method not in self._scatter_methods.keys():
            raise ValueError(f"Invalid choice for method: {method}. Valid options are {list(self._scatter_methods.keys())}.")

        scatter_method = self._scatter_methods[method]

        init_means = np.array([init_config[p][0] for p in self.fit_pars])
        init_sigmas = np.array([init_config[p][1] for p in self.fit_pars])

        init_samples = init_means + init_sigmas * scatter_method(size=(self.nwalkers, self.ndims))

        return init_samples

    def initialise(self, spectrum, init_config, mask=None):
        """
        Initialise sampler with spectrum and prepare initial samples
        """

        # overwrite check here?

        # verify spectrum, sets self.spectrum
        self._verify_spectrum(spectrum)

        self._verify_mask(mask)

        self._init_sampler(spectrum[:, 1], spectrum[:, 2])

        # verify init_config, sets self.init_config
        self._verify_init_config(init_config)

        # create intial samples from init config
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

    def run(self, nsteps, continue_=True, progress=True, spectrum=None, mask=None, init_config=None):
        """
        Run MCMC sampling
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


#### legacy ####

@deprecated_("astra.fitting.optsub_multi has been superceded by astra.fitting.optsub.")
def optsub_multi(
    obs: np.ndarray[float] | list[np.ndarray[float]],
    templ: list[np.ndarray[float]],
    mask: np.ndarray[float] | list[tuple[float, float]] | None = None,
    progress: bool = True,
) -> np.ndarray[float]:
    """
    Perform optimal subtraction using the standard method.
    Optimal subtraction is performed for each observed spectrum,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.
    Any systemic radial velocities should be removed also (see sincshift).

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs: np.ndarray or list of np.ndarray
        Array or list of arrays of the observed spectra. If list, a list of results is returned.
        All spectra must have identical wavelength scales.
        Spectra should have two, or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templ: list of np.ndarray
        List of arrays of template spectra.
        All spectra must have identical wavelength scales, which match the observed spectra.
        Each spectrum should have two columns: wavelength and flux.
        Additional columns will be ignored. It is assumed noise is dominated by observed spectra.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array matching the
        observed and template spectra, or a list of 2-tuples defining upper and lower bounds to exclude.
    progress: bool, default True
        Controls display of tqdm progress bar, optional. False to disable

    Returns:
    optsub_results: np.ndarray or list of np.ndarray
        Output array, or list of arrays, containing optsub results, 3 columns, one row per template spectrum.
        Columns are chi-squared of subtraction, optimal subtraction factor, and error on the factor.
        If a single observed spectrum is provided, a single results array is returned.

    """

    return optsub(obs, templ, mask, progress)
