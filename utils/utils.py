__all__ = [
    'taper_spectrum',
    'calculate_vpx',
    'mask_interp',
    'phase_average'
]

from astra.utils.helpers import check_spectra

import numpy as np
import warnings

c_ = 299792.458


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


def phase_average(
    spec: list[np.ndarray[float]],
    phases: np.ndarray[float],
    phase_bins: list[tuple[float, float]] | None = None,
    width: float | None = None,
    n_bins: int | None = None,
) -> list[np.ndarray[float]]:
    """
    Phase-bin a list of spectra, i.e. average according to orbital phase.

    Phase bins must be either provided explicitly, with argument `phase_bins`,
    or specified by `width` and `n_bins`.

    Parameters:
    spec: list of np.ndarray
        List of arrays of spectra to phase-average.
        All spectra must have identical wavelength scales. Each spectrum should have two,
        or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    phases: np.ndarray
        Orbital phases of each spectrum, same 1D array of same length as `spec`.
        Will be clipped to between 0 and 1 if not done already.
    phase_bins: list of 2-tuples of floats, optional
        Phase bins to average in, as a list of tuples of (lower bound, upper bound).
        This will override `width` and `n_bins` if they are set.
    width: float, optional
        Width of phase bins, such that a phase bin around a phase 'p' will be defined:

            p - (width / 2) < phases < p + (width / 2)

        Ignored if `phase_bins` set explicitly.
    n_bins: int, optional
        Number of phase bins. Ignored if `phase_bins` set explicitly.

    Returns:
    phase_averaged_spec: list of np.ndarray (or None)
        List of phase-averaged spectra, one per phase bin.
        If no spectra were found in a given phase bin, a warning will be given,
        and'None' will be returned instead.

    """
    errors = check_spectra(spec)

    # handle optional phase bins/widths + n_bins
    if phase_bins is None:
        if width is None or n_bins is None:
            raise ValueError("`width` and `n_bins` must be provided if `phase_bins` not set.")
        if width <= 0:
            raise ValueError("`width` must be greater than zero.")
        if n_bins < 1:
            raise ValueError("Number of bins must be 1 or higher.")

        phase_bins = np.linspace(0 - width / 2, 1 - width / 2, n_bins + 1)
        phase_bins = [(p_l, p_u) for p_l, p_u in zip(phase_bins[:-1], phase_bins[1:])]

    else:
        if width is not None:
            warnings.warn("`width` provided but phase bins already set - ignoring.")
        if n_bins is not None:
            warnings.warn("`n_bins` provided but phase bins already set - ignoring.")

    out = [None] * len(phase_bins)
    phases_mod = phases % 1  # ensure phases clipped to 0 - 1 range

    for i, (lower, upper) in enumerate(phase_bins):

        # clip phase bin edges to 0 - 1
        lower_mod, upper_mod = lower % 1, upper % 1

        # separate into two cases:
        #  - bin enclosed by lower < p < upper
        #  - bin wraps i.e. upper > 1 or lower < 0, so p < (upper % 1) and p > (lower % 1)
        if lower_mod < upper_mod:
            sel = (phases_mod > lower_mod) & (phases_mod < upper_mod)
        elif lower_mod > upper_mod:
            sel = (phases_mod > lower_mod) | (phases_mod < upper_mod)
        else:
            warnings.warn(f"Lower and upper phase bounds are the same: {lower_mod}, {upper_mod}.")

        to_bin = [t for t, s in zip(spec, sel) if s]

        # pass if no spectra in bin
        if len(to_bin) == 0:
            warnings.warn(f"No spectra between phases {lower_mod} and {upper_mod} - skipping.")
            continue

        # wavelength scales are the same from check_spectra
        wvs = to_bin[0][:, 0]

        fluxes = np.array([t[:, 1] for t in to_bin])
        fluxes_ma = np.ma.MaskedArray(fluxes, mask=np.isnan(fluxes))

        # if errors present, weight averages
        if errors:
            flux_errors = np.array([t[:, 2] for t in to_bin])
            flux_errors_ma = np.ma.MaskedArray(flux_errors, mask=np.isnan(flux_errors))

            weights = 1 / flux_errors_ma**2
            flux_avg = np.ma.average(fluxes_ma, weights=weights, axis=0).data
            flux_errors_avg = np.ma.sqrt(1 / weights.sum(axis=0)).data

            out[i] = np.c_[wvs, flux_avg, flux_errors_avg]
        else:
            flux_avg = np.ma.average(fluxes_ma, axis=0).data
            out[i] = np.c_[wvs, flux_avg]

    return out
