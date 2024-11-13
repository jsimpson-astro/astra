__all__ = [
    'apply_mask',
    'taper_spectrum',
    'calculate_vpx'
    ]

import numpy as np

c_ = 299792.458

def apply_mask(wvs: np.ndarray, mask_bounds: list) -> np.ndarray:
    """
    Apply a list of mask bounds, containing tuples of upper and lower bounds,
    to an array of wavelengths, returning a boolean mask of the same size.

    Parameters:
    wvs: np.ndarray
        Wavelength array to mask
    mask_bounds: list of tuples of 2 floats
        List of tuples where each tuple is an upper and lower bound to 
        exclude. The order does not matter.

    Returns:
    mask: np.ndarray
        Boolean array, same size as wvs. False where excluded by mask
    
    """

    mask = np.ones(wvs.size, dtype=bool)

    for lb, ub in mask_bounds:
        if lb > ub:
            mask = mask & ~((wvs > ub) & (wvs < lb))
        else:
            mask = mask & ~((wvs > lb) & (wvs < ub))

    return mask


def taper_spectrum(
    spectrum: np.ndarray, 
    taper: float, 
    taper_errors: bool = False
    ) -> np.ndarray:
    """
    Taper the ends of a spectrum using a cosine envelope.
    `taper` is the fraction of the spectrum to taper, from 0 to 1.
    Taper fractions greater than 0.5 will overlap in the centre of the spectrum.

    """
    if taper > 1:
        raise ValueError("taper fraction cannot exceed 1.")
    if taper < 0.:
        raise ValueError("taper fraction cannot be negative.")

    spectrum_ = spectrum.copy()

    if taper == 0:
        return spectrum_

    flux = spectrum_[:, 1]
    flux_err = spectrum_[:, 2] if spectrum_.shape[1] > 1 else None

    # identify ends to slice
    slice0 = slice(idx0, idx0 + int(taper*flux.size) + 1)
    slice1 = slice(idx1 - int(taper*flux.size), idx1 + 1)

    # compute factors to multiply by based on (index - end pixel)
    idxs = np.arange(flux.size)
    factor0 = ((1 - np.cos(np.pi * (idxs[slice0] - idx0) / flux.size / taper)) / 2)
    factor1 = ((1 - np.cos(np.pi * (idxs[slice1] - idx1) / flux.size / taper)) / 2)

    # apply taper
    flux[slice0] = factor0 * flux[slice0]
    flux[slice1] = factor1 * flux[slice1]

    spectrum_[:, 1] = flux
    if flux_err is not None:
        if taper_errors:
            flux_err[slice0] = factor0 * flux_err[slice0]
            flux_err[slice1] = factor1 * flux_err[slice1]
        spectrum_[:, 2] = flux_err

    return spectrum_


def calculate_vpx(wvs: np.ndarray) -> float:
    """
    Calculate the average velocity per pixel from an array of wavelengths.

    """
    return c_ * (np.exp(np.log(wvs.max()/wvs.min()) / (wvs.size-1) ) - 1)


