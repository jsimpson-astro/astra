__all__ = [
    'apply_mask',
    'taper_spectrum',
    'calculate_vpx',
    'mask_interp'
]

import numpy as np

c_ = 299792.458


def apply_mask(wvs: np.ndarray, mask_bounds: list[tuple[float, float]]) -> np.ndarray[bool]:
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
    taper_errors: bool = False,
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

    # taper from edges of mask if provided
    idx0 = 0
    idx1 = len(flux) - 1

    # identify ends to slice
    slice0 = slice(idx0, idx0 + int(taper * flux.size) + 1)
    slice1 = slice(idx1 - int(taper * flux.size), idx1 + 1)

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
    return c_ * (np.exp(np.log(wvs.max() / wvs.min()) / (wvs.size - 1)) - 1)


def _mask_interp(
    flux: np.ndarray[float],
    mask: np.ndarray[bool]
) -> np.ndarray[float]:
    """
    Linearly interpolate flux from edges of given mask.
    Masked edges are set to the nearest unmasked value.

    Parameters:
    flux: np.ndarray
        Fluxes of spectrum used for interpolation.
    mask: np.ndarray of bool
        Boolean array to interpolate over where False, same size as flux.

    Returns:
    flux_i: np.ndarray
        Fluxes of spectrum with interpolation over masked regions, same size as flux.

    """
    flux_i = flux.copy()

    # indices included in mask - so slice i:f+1
    mask_diff = np.diff(mask.astype(np.int8))
    mask_is = (np.r_[0, mask_diff] == -1).nonzero()[0]
    mask_fs = (np.r_[mask_diff, 0] == 1).nonzero()[0]
    # print(mask_is, mask_fs)

    # must catch unbounded (or all True) cases
    if mask_is.size != 0 and mask_fs.size != 0:
        if mask_is[0] < mask_fs[0]:
            # left edge is not masked, go i[0]:f[0], i[1]:f[1]
            slices = [slice(i, f + 1) for i, f in zip(mask_is[:-1], mask_fs)]
        else:
            # left edge is masked, go :f[0], i[0]:f[1] etc.
            slices = [slice(None, mask_fs[0] + 1)]
            slices += [slice(i, f + 1) for i, f in zip(mask_is[:-1], mask_fs[1:])]

        if mask_is[-1] < mask_fs[-1]:
            # right edge is not masked, i[-1]:f[-1] should be fine
            slices += [slice(mask_is[-1], mask_fs[-1] + 1)]
        else:
            # right edge is masked, will have to use i[-1]:
            slices += [slice(mask_is[-1], None)]
    elif mask_is.size != mask_fs.size:
        slices = [slice(mask_is[0], None) if mask_fs.size == 0 else slice(None, mask_fs[0] + 1)]
    else:
        slices = []

    for sl in slices:
        if sl.start and sl.stop:
            # linear interpolation
            flux_i[sl] = np.linspace(flux[sl.start - 1], flux[sl.stop], sl.stop - sl.start + 2)[1:-1]
        else:
            flux_i[sl] = flux[sl.start - 1] if sl.start else flux[sl.stop]

    return flux_i


def mask_interp(
    spectrum: np.ndarray[float],
    mask: np.ndarray[bool] | list[tuple[float, float]]
) -> np.ndarray[float]:
    """
    Linearly interpolate flux from edges of given mask.
    Masked edges are set to the nearest unmasked value.

    Parameters:
    spectrum: np.ndarray
        Spectrum to interpolate over, first column wavelengths, second flux.
    mask: np.ndarray of bool
        Boolean array to interpolate over where False, same length as spectrum.

    Returns:
    spectrum_i: np.ndarray
        Spectrum with interpolation over masked regions.

    """

    if isinstance(mask, np.ndarray):
        if mask.size != spectrum[:, 0].size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(spectrum[:, 0], mask)
    else:
        raise TypeError(f"Invalid type for mask: {type(mask)}")

    flux_i = _mask_interp(spectrum[:, 1], mask_)

    spectrum_i = spectrum.copy()

    spectrum_i[:, 1] = flux_i

    # mask errors that have been interpolated
    if spectrum.shape[1] > 2:
        spectrum_i[:, 2][spectrum_i[:, 1] != spectrum[:, 1]] = np.nan

    return spectrum_i
