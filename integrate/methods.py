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

def do_ews(spectra: list,
           mask: np.ndarray | list | None = None,
           initial_shifts: list | None = None
           ):
    """
    Measure equivalent widths using the same F77 method as molly.
    Equivalent widths are measured across a series of spectra.
    Spectra must be normalised to a continuum of 0.

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    spectra: list of path-like str or list of np.ndarray
        List of paths to spectra, or list of arrays of spectra. 
        Each spectrum should have two, or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array the same size as the 
        wavelengths of the spectra, or a list of 2-tuples defining upper and lower bounds to exclude.
    initial_shifts: list of ints or None, default None
        Initial pixel shifts to apply. Must have the same length as spectra.

    Returns:
    ews: np.ndarray
        Measured equivalent widths, one element per spectrum.
    ew_errors: np.ndarray
        Errors on equivalent widths, one element per spectrum.
        Set to np.nan if no errors present in a given spectrum.
    
    """
    # need first wvs to compare against template wvs, and check other inputs
    wvs = spectra[0][:, 0] if isinstance(spectra[0], np.ndarray) else np.loadtxt(spectra[0], usecols=0)
    
    n_spec = len(spectra)
    
    # check initial shifts if provided, including bounds
    if initial_shifts is not None:
        if len(initial_shifts) != len(spectra):
            raise IndexError("initial_shifts must have the same length as obs_spectra.")
        for initial_shift in initial_shifts:
            if not isinstance(initial_shift, (np.integer, int)):
                raise TypeError("initial_shifts must be given as ints")
            if abs(initial_shift) > wvs.size // 2:
                raise IndexError(f"shift ({initial_shift}) out of bounds for wavelengths with size {wvs.size}") 
                
    #### spectra checks ####
    # check data, open if not already arrays
    if not isinstance(spectra[0], np.ndarray):
        spec_data = [np.loadtxt(f) for f in spectra]
    else:
        spec_data = spectra

    # check shapes are all identical, flag errors if all have at least 3 columns
    shapes = np.array([a.shape for a in spec_data])
    cols_unique = np.unique(shapes[:, 1])
    if cols_unique.min() < 2:
        raise IndexError("spectra do not all have at least two columns")
    spec_have_errors = True if cols_unique.min() > 2 else False

    #### masking checks #####
    if isinstance(mask, np.ndarray):
        # check shapes are all identical, flag errors if all have at least 3 columns
        wvs_unique = np.unique(shapes[:, 0])
        if wvs_unique.size != 1:
            raise IndexError("mask is an array, and spectra do not all have the same number of points")
    
        # check wavelength scales
        wv_dev = np.abs(np.array([a[:, 0] for a in spec_data]) - wvs).max()
        if wv_dev > wv_tol:
            raise ValueError(f"mask is an array, and spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")
        
        if mask.size != wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    else:
        mask_ = None
    
    ####
    
    initial_shifts = [0] * n_spec if initial_shifts is None else initial_shifts

    ews, ew_errors = np.zeros(n_spec), np.zeros(n_spec)
        
    for i_spec, (spec, initial_shift) in enumerate(zip(spec_data, initial_shifts)):

        wvs, flux = spec[:, 0], spec[:, 1]
        flux_err = spec[:, 2] if spec_have_errors else None

        # apply shift
        # effectively, shift the mask to where the shift specifies
        # functionally, it is better to shift the flux in the opposite direction,
        # and then pad with nans which will get masked anyways
        if initial_shift > 0:
            flux = np.r_[flux[initial_shift:], np.zeros(initial_shift) + np.nan]
            if flux_err is not None:
                flux_err = np.r_[flux_err[initial_shift:], np.zeros(initial_shift) + np.nan]
        elif initial_shift < 0:
            flux = np.r_[np.zeros(-initial_shift) + np.nan, flux[:initial_shift]]
            if flux_err is not None:
                flux_err = np.r_[np.zeros(-initial_shift) + np.nan, flux_err[:initial_shift]]
        
        # apply mask if needed
        if mask_ is None:
            if isinstance(mask, list):
                spec_mask_ = apply_mask(wvs, mask)
            else:
                spec_mask_ = np.ones(wvs.size, dtype=bool)
        else:
            spec_mask_ = mask_.copy()

        # mask out any nans
        spec_mask_ = (spec_mask_ & ~np.isnan(flux))
        spec_mask_ = (spec_mask_ & (flux_err >= 0)) if spec_have_errors else spec_mask_
        
        # measure ews
        ew, ew_error = f77_ew(wvs, flux, flux_err, spec_mask_)

        # save to arrays
        ews[i_spec], ew_errors[i_spec] = ew, ew_error
                
    return ews, ew_errors


_method_dict = {'ew': equivalent_width, 'flux': None}
_integrate_methods = list(_method_dict.keys())