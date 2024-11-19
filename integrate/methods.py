from astra.utils import apply_mask
from astra.utils.helpers import dummy_pbar

import warnings
import numpy as np
from numba import njit
from numba.types import bool_
from tqdm.auto import tqdm

#### backend ####

@njit
def f77_ew_nj(wvs: np.ndarray, 
              flux: np.ndarray, 
              flux_err: np.ndarray | None = None, 
              mask: np.ndarray | None = None
              ):
    """
    Calculates the equivalent width over a spectrum using the same method
    as the F77 program molly. Spectrum must be normalised to 0.

    Parameters:
    wvs: numpy.ndarray
        Array of wavelengths, 1D
    flux: numpy.ndarray
        Fluxes of spectrum, same size as wvs
    flux_err: numpy.ndarray or None, default None
        Errors of fluxes, optional, same size as wvs
    mask: numpy.ndarray or None, default None
        Boolean array to mask wavelengths, optional, same size as wvs

    Returns:
    ew: float
        Measured equivalent width over the unmasked range
    ew_error: float or None
        Error on the measured equivalent width
        If flux_err not provided, ew_error is np.nan.

    """
    
    # input checks
    if not (wvs.size == flux.size):
        raise IndexError("wvs and flux must have the same size")
    if mask is not None and mask.size != wvs.size:
        raise IndexError("mask and wvs must be the same size")
    if flux_err is not None and flux_err.size != flux.size:
        raise IndexError("flux_err and flux must be the same size")

    # nb change to bool_ and mask_ for njit
    mask_ = np.ones(flux.shape, dtype=bool_) if mask is None else mask
    flux_err_ = flux_err if flux_err is not None else np.ones(flux.shape)
    
    dws = np.zeros_like(wvs)
    dws[:-1] = wvs[1:] - wvs[:-1]
    # extrapolate for final value
    dws[-1] = dws[-1] + (dws[-1] - dws[-2])

    # subtracting continuum would go here
    ew = (dws[mask_] * (flux[mask_])).sum()
    ew_err = ((dws[mask_] * flux_err[mask_])**2).sum()**0.5 if flux_err is not None else np.nan
        
    return ew, ew_err

#### frontend ####

@njit
def equivalent_width(
    spectrum: np.ndarray[float],
    mask: np.ndarray[bool] | None = None,
    ) -> (float, float):

    wvs, flux = spectrum[:, 0], spectrum[:, 1]
    has_errors = True if spectrum.shape[1] > 2 else False
    flux_err = spectrum[:, 2] if has_errors else np.ones(flux.shape)

    mask_ = np.ones(flux.shape, dtype=bool_) if mask is None else mask

    dws = np.zeros_like(wvs)
    dws[:-1] = wvs[1:] - wvs[:-1]
    # extrapolate for final value
    dws[-1] = dws[-1] + (dws[-1] - dws[-2])

    # subtracting continuum would go here
    ew = (dws[mask_] * (flux[mask_])).sum()
    ew_err = ((dws[mask_] * flux_err[mask_])**2).sum()**0.5 if has_errors is not None else np.nan
        
    return ew, ew_err

_method_dict = {'ew': equivalent_width, 'flux': None}
_integrate_methods = list(_method_dict.keys())