__all__ = [
    'taper_spec',
    'calculate_vpx',
    'mask_interp',
    'average_spec',
    'phase_average',
    'sincshift'
]

from astra.utils._helpers import check_spectra, check_vbinned

import numpy as np
import warnings

c_ = 299792.458


def taper_spec(
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


def average_spec(spec: list[np.ndarray[float]]) -> np.ndarray[float]:
    """
    Average a list of spectra together, weighted by errors

    """
    errors = check_spectra(spec)

    # wavelength scales are the same from check_spectra
    wvs = spec[0][:, 0]

    fluxes = np.array([s[:, 1] for s in spec])
    fluxes_ma = np.ma.MaskedArray(fluxes, mask=np.isnan(fluxes))

    # if errors present, weight averages
    if errors:
        flux_errors = np.array([s[:, 2] for s in spec])
        flux_errors_ma = np.ma.MaskedArray(flux_errors, mask=np.isnan(flux_errors))

        weights = 1 / flux_errors_ma**2
        flux_avg = np.ma.average(fluxes_ma, weights=weights, axis=0).data
        flux_errors_avg = np.ma.sqrt(1 / weights.sum(axis=0)).data

        return np.c_[wvs, flux_avg, flux_errors_avg]
    else:
        flux_avg = np.ma.average(fluxes_ma, axis=0).data
        return np.c_[wvs, flux_avg]


def phase_average(
    spec: list[np.ndarray[float]],
    phases: np.ndarray[float],
    phase_bins: list[tuple[float, float]] | None = None,
    n_bins: int | None = None,
    width: float | None = None,
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
    n_bins: int, optional
        Number of phase bins.
        Ignored if `phase_bins` set explicitly, required if `phase_bins` is not set.
    width: float, optional
        Width of phase bins, such that a phase bin around a phase 'p' will be defined:

            p - (width / 2) < phases < p + (width / 2)

        Defaults to 1 / n_bins if `phase_bins` is not set.
        Ignored if `phase_bins` set explicitly.

    Returns:
    phase_averaged_spec: list of np.ndarray (or None)
        List of phase-averaged spectra, one per phase bin.
        If no spectra were found in a given phase bin, a warning will be given,
        and'None' will be returned instead.

    """
    errors = check_spectra(spec)

    # handle optional phase bins/widths + n_bins
    if phase_bins is None:
        if n_bins is None:
            raise ValueError("`n_bins` must be provided if `phase_bins` not set.")
        elif n_bins < 1:
            raise ValueError("Number of bins must be 1 or higher.")
        elif n_bins == 1 and (width is None or width >= 1):
            return [average_spec(spec)]

        if width is None:
            width = 1 / n_bins
        elif width <= 0:
            raise ValueError("`width` must be greater than zero.")

        # create bins
        bins = np.linspace(0, 1 - (1 / n_bins), n_bins)
        lbs, ubs = bins - width / 2, bins + width / 2
        phase_bins = [(lb, ub) for lb, ub in zip(lbs, ubs)]

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
            sel = phases_mod == lower_mod

        to_bin = [t for t, s in zip(spec, sel) if s]

        # pass if no spectra in bin
        if len(to_bin) == 0:
            warnings.warn(f"No spectra between phases {lower_mod} and {upper_mod} - skipping.")
            continue

        out[i] = average_spec(to_bin)

    return out


def sincshift(
    spec: np.ndarray[float],
    vshift: float = 0.,
    pad: float = np.nan
) -> np.ndarray[float]:
    """
    Shift the flux (and error) of a spectrum by a given velocity, in km/s.
    Rebinning is done with a sinc rebinning method.
    This convolves a windowed sinc function with the spectrum.
    The window is defined as:

    f(x) = 4/3 - 8 (x-1) x^2    0.0 < x <= 0.5
         = 8/3 (1-x)^3          0.5 < x < 1.0

    Parameters:
    spec: numpy.ndarray
        Spectrum to shift
    vshift: float, default 0.
        Velocity shift to apply to spectrum, in km/s
    pad: float, default np.nan.
        Value to fill padded fluxes with, optional

    Returns:
    spec_shifted: np.ndarray
        Spectrum, rebinned with velocity shift applied to flux and any errors.

    """

    errors = check_spectra([spec])

    wvs, flux = spec[:, 0], spec[:, 1]
    flux_err = spec[:, 2] if errors else None

    if not check_vbinned(wvs):
        warnings.warn("Spectra not uniform in velocity space.", RuntimeWarning)

    maxsinc = 15
    v_avg = c_ * (np.exp(np.log(wvs.max() / wvs.min()) / (wvs.size - 1)) - 1)

    # compute pixel shift, separate into integer (rounded) and decimal parts
    pxshift = -np.log(1 + vshift / c_) / np.log(1 + v_avg / c_)  # px shift
    nshift, subpxshift = int(pxshift), pxshift - int(pxshift)

    # compute sinc function, shifted by the decimal part of the shift (xshift)~
    x1 = np.pi * (subpxshift - np.arange(-maxsinc, maxsinc + 1))
    x2 = np.abs((subpxshift - np.arange(-maxsinc, maxsinc + 1)) / (maxsinc + 0.5))
    sinc = np.zeros(2 * maxsinc + 1)

    # use taylor series approx at small x1 (avoid div/0)
    x1_mask = (np.abs(x1) < 1e-4)
    sinc[x1_mask] = 1 - (x1[x1_mask]**2) / 6
    sinc[~x1_mask] = np.sin(x1[~x1_mask]) / x1[~x1_mask]

    # apply window to sinc function
    x2_mask = (x2 <= 0.5)
    sinc[x2_mask] = sinc[x2_mask] * (4 / 3 + 8 * (x2[x2_mask] - 1) * x2[x2_mask]**2)
    sinc[~x2_mask] = sinc[~x2_mask] * 8 / 3 * (1 - x2[~x2_mask])**3

    sinc = sinc / sinc.sum()

    # now do convolution, depending on nshift
    if nshift < 0:
        # pad fluxes with end values for convolution, which will be removed with mode='valid'
        flux_padded = np.r_[flux[0] * np.ones(maxsinc),
                            flux[:nshift + maxsinc if nshift + maxsinc < 0 else None],
                            flux[-1] * np.ones(max(nshift + maxsinc, 0))]

        # convolve, reversing sinc function as numpy reverses the shorter array
        flux_shifted = np.convolve(flux_padded, sinc[::-1], mode='valid')
        # pad zeros where fluxes have been shifted away
        flux_shifted = np.r_[np.zeros(-nshift) + pad, flux_shifted]

        # repeat for errors
        if flux_err is not None:
            flux_err_padded = np.r_[flux_err[0] * np.ones(maxsinc),
                                    flux_err[:nshift + maxsinc if nshift + maxsinc < 0 else None],
                                    flux_err[-1] * np.ones(max(nshift + maxsinc, 0))]

            flux_err_shifted = np.convolve(flux_err_padded, sinc[::-1], mode='valid')
            flux_err_shifted = np.r_[np.zeros(-nshift) + pad, flux_err_shifted]
        else:
            flux_err_shifted = np.zeros_like(flux_shifted) + pad

    elif nshift > 0:
        # pad fluxes with end values for convolution, which will be removed with mode='valid'
        flux_padded = np.r_[flux[0] * np.ones(max(maxsinc - nshift, 0)),
                            flux[max(nshift - maxsinc, 0):],
                            flux[-1] * np.ones(maxsinc)]

        # convolve, reversing sinc function as numpy reverses the shorter array
        flux_shifted = np.convolve(flux_padded, sinc[::-1], mode='valid')
        # pad zeros where fluxes have been shifted away
        flux_shifted = np.r_[flux_shifted, np.zeros(nshift) + pad]

        # repeat for errors
        if flux_err is not None:
            flux_err_padded = np.r_[flux_err[0] * np.ones(max(maxsinc - nshift, 0)),
                                    flux_err[max(nshift - maxsinc, 0):],
                                    flux_err[-1] * np.ones(maxsinc)]

            flux_err_shifted = np.convolve(flux_err_padded, sinc[::-1], mode='valid')
            flux_err_shifted = np.r_[flux_err_shifted, np.zeros(nshift) + pad]
        else:
            flux_err_shifted = np.zeros_like(flux_shifted) + pad

    else:
        # pad fluxes with end values for convolution, which will be removed with mode='valid'
        flux_padded = np.r_[flux[0] * np.ones(maxsinc), flux, flux[-1] * np.ones(maxsinc)]
        # convolve, reversing sinc function as numpy reverses the shorter array
        flux_shifted = np.convolve(flux_padded, sinc[::-1], mode='valid')

        # repeat for errors
        if flux_err is not None:
            flux_err_padded = np.r_[flux_err[0] * np.ones(maxsinc),
                                    flux_err, flux_err[-1] * np.ones(maxsinc)]

            flux_err_shifted = np.convolve(flux_err_padded, sinc[::-1], mode='valid')
        else:
            flux_err_shifted = np.zeros_like(flux_shifted) + pad

    spec_shifted = spec.copy()
    spec_shifted[:, 1] = flux_shifted
    if errors:
        spec_shifted[:, 2] = flux_err_shifted

    return spec_shifted
