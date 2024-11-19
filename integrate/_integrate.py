import astra.integrate.methods as methods
from astra.utils import apply_mask
from astra.utils.helpers import check_spectra

import warnings
import numpy as np

_methods = methods._integrate_methods
_default_method = _methods[0]


def integrate(
    spectrum: np.ndarray[float],
    mask: np.ndarray[bool] | list[tuple[float, float]] | None = None,
    method: {methods._integrate_methods} | callable = _default_method,
    _skip_checks: bool = False
    ) -> (float, float):

    if isinstance(method, str):
        method_ = methods._method_dict.get(method, None)
        if method_ is None:
            raise ValueError(
                f"Invalid choice of integration method: {method}. Possible choices are {_methods}.")
    elif isinstance(method, callable):
        method_ = method
    else:
        raise TypeError(
            f"Invalid type for `method`: {type(method)}. `method` should be of type string or callable.")

    has_errors = check_spectra([spectrum])

    wvs, flux = spectrum[:, 0], spectrum[:, 1]
    flux_err = spectrum[:, 2] if has_errors else None

    if isinstance(mask, np.ndarray):
        if mask.size != wvs.size:
            raise IndexError(
                "If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(wvs, mask)
    else:
        mask_ = np.ones(wvs.size, dtype=bool)

    integral, integral_err = method_(spectrum, mask_)

    return integral, integral_err


def integrate_multi(
    spectra: list[np.ndarray[float]],
    mask: np.ndarray[bool] | list[tuple[float, float]] | None = None,
    method: {methods._integrate_methods} | callable = _default_method
    ) -> np.ndarray[float]:
    pass


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