from astra.utils import apply_mask
from astra.utils.utils import _mask_interp
from astra.utils.helpers import dummy_pbar, xcheck_spectra, check_vbinned

import warnings
import numpy as np
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d
#from numba import njit
#from numba.types import bool_


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
    obs: np.ndarray[float],
    templates: list[np.ndarray[float]],
    mask: np.ndarray[float] | list[tuple[float, float]] | None = None,
    progress: bool = True,
    _skip_checks: bool = False,
    ) -> np.ndarray[float]:
    """
    Perform optimal subtraction using the standard method.
    Optimal subtraction is performed for the given observed spectrum,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.
    Any systemic radial velocities should be removed also (see sincshift).

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs: np.ndarray
        Array of the observed spectrum. 
        Spectrum should have two, or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templates: list of np.ndarray
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
    optsub_results: np.ndarray
        Output array containing optsub results, 3 columns, one row per template spectrum.
        Columns are chi-squared of subtraction, optimal subtraction factor, and error on the factor.
    
    """

    if not _skip_checks:
        # check spectra match
        try:
            obs_w_errors, templ_w_errors = xcheck_spectra([obs], templates)
        except Exception as e:
            msg = e.args[0].replace('spectra1', 'obs').replace('spectra2', 'template')
            raise type(e)(msg, *args[1:])
    else:
        obs_w_errors = True if obs.shape[1] > 2 else False
        templ_w_errors = True if templates[0].shape[1] > 2 else False
    
    n_templ = len(templates)
    obs_wvs = obs[:, 0]

    #### masking checks #####
    
    if isinstance(mask, np.ndarray):
        if mask.size != obs_wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(obs_wvs, mask)
    else:
        mask_ = np.ones(obs_wvs.size, dtype=bool)
    
    ####

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar
    
    with pbar_manager(desc='optsub: ', total=n_templ) as pbar:

        obs_wvs, obs_flux = obs[:, 0], obs[:, 1]
        obs_flux_err = obs[:, 2] if obs_w_errors else None

        # mask out any nans
        obs_mask_ = (mask_ & ~np.isnan(obs_flux))
        obs_mask_ = (obs_mask_ & (obs_flux_err >= 0)) if obs_w_errors else obs_mask_
        
        chisq_array, factor_array, factor_err_array = np.zeros(n_templ), np.zeros(n_templ), np.zeros(n_templ)
        
        for i_templ, template in enumerate(templates):
        
            t_wvs, t_flux = template[:, 0], template[:, 1]
            
            # mask out any nans
            t_mask_ = (obs_mask_ & ~np.isnan(t_flux))

            # perform optsub
            chisq, factor, factor_err, dof = optsub_standard(obs_flux, t_flux, obs_flux_err, t_mask_)

            # save results to arrays
            chisq_array[i_templ] = chisq
            factor_array[i_templ], factor_err_array[i_templ] = factor, factor_err

            pbar.update(1)

        # add result data
        optsub_result = np.vstack([chisq_array, factor_array, factor_err_array]).T
                
    return optsub_result


def optsub_multi(
    obs: list[np.ndarray[float]],
    templates: list[np.ndarray[float]],
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
    obs: list of np.ndarray
        Arrays of the observed spectra. 
        All spectra must have identical wavelength scales.
        Spectra should have two, or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templates: list of np.ndarray
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
    optsub_results: np.ndarray
        Output array containing optsub results, 3 columns, one row per template spectrum.
        Columns are chi-squared of subtraction, optimal subtraction factor, and error on the factor.
    
    """

    # check spectra match
    try:
        obs_w_errors, templ_w_errors = xcheck_spectra(obs, templates)
    except Exception as e:
        msg = e.args[0].replace('spectra1', 'obs').replace('spectra2', 'template')
        raise type(e)(msg, *e.args[1:])
    
    n_obs = len(obs)
    obs_wvs = obs[0][:, 0]

    if not check_vbinned(obs_wvs):
        warnings.warn("Spectra not uniform in velocity space - results may be meaningless.", RuntimeWarning)

    #### masking checks #####
    
    if isinstance(mask, np.ndarray):
        if mask.size != obs_wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(obs_wvs, mask)
    else:
        mask_ = np.ones(obs_wvs.size, dtype=bool)
    
    ####

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar

    optsub_results = [None] * n_obs
    
    with pbar_manager(desc='optsub: ', total=n_obs) as pbar:
        
        for i_obs, obs_spec in enumerate(obs):

            optsub_result = optsub(obs=obs_spec, 
                                   templates=templates,
                                   mask=mask_,
                                   progress=False,
                                   _skip_checks=True)
            
            optsub_results[i_obs] = optsub_result

            pbar.update(1)
                
    return optsub_results