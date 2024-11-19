from astra.utils.helpers import check_spectra, check_vbinned, automask

import warnings
import numpy as np
from numba import njit
from numba.types import bool_
from tqdm.auto import tqdm


@njit
def ew_compute(
    spectrum: np.ndarray[float],
    mask: np.ndarray[bool] | None = None,
) -> (float, float):
    """
    Calculates the equivalent width over a spectrum within a mask.

    Parameters:
    spectrum: numpy.ndarray
    mask: numpy.ndarray or None, default None
        Boolean array to mask wavelengths, optional, same size as wvs

    Returns:
    ew: float
        Measured equivalent width over the unmasked range
    ew_error: float or None
        Error on the measured equivalent width
        If flux_err not provided, ew_error is np.nan.
    """

    wvs, flux = spectrum[:, 0], spectrum[:, 1]
    has_errors = True if spectrum.shape[1] > 2 else False
    flux_err = spectrum[:, 2] if has_errors else np.ones(flux.shape)

    mask_ = np.ones(flux.shape, dtype=bool_) if mask is None else mask

    dws = np.zeros_like(wvs)
    dws[:-1] = wvs[1:] - wvs[:-1]
    # extrapolate for final value
    dws[-1] = dws[-1] + (dws[-1] - dws[-2])

    # subtracting continuum would go here
    ew = (dws[mask_] * flux[mask_]).sum()
    ew_err = ((dws[mask_] * flux_err[mask_])**2).sum()**0.5 if has_errors is not None else np.nan

    return ew, ew_err


def ew(
    spec: np.ndarray[float] | list[np.ndarray[float]],
    mask: np.ndarray[bool] | list[float] | None = None,
    continuum: np.ndarray[float] | float = 1.,
) -> np.ndarray[float] | list[np.ndarray[float]]:
    """
    """

    spec_ = [spec] if isinstance(spec, np.ndarray) else spec
    n_spec = len(spec_)

    for i, s in enumerate(spec_):
        try:
            # check spectrum has correct shape
            _ = check_spectra([s])
            # check monotonic
            _ = check_vbinned(s[:, 0])
        except Exception as e:
            # raise error with specific spectrum if more than 1 given
            msg = f"Error with spectrum at index {i}: {e.args[0]}" if n_spec > 1 else e.args[0]
            raise type(e)(msg, *e.args[1:])

    shapes_match = len(set(s.shape for s in spec_)) == 1

    # if mask/continuum are boolean arrays, need to check they match spectra
    # if spectra have varying shapes, raise error
    if shapes_match:
        if isinstance(mask, np.ndarray) and mask.shape != spec_[0].shape:
            raise IndexError("If `mask` is an array, it must match the number of points of all spectra.")
        if isinstance(continuum, np.ndarray) and continuum.shape != spec_[0].shape:
            raise IndexError("If `continuum` is an array, it must match the number of points of all spectra.")
    else:
        if isinstance(mask, np.ndarray):
            raise IndexError("If `mask` is an array, spectra must have identical shapes.")
        if isinstance(continuum, np.ndarray):
            raise IndexError("If `continuum` is an array, spectra must have identical shapes.")
        warnings.warn("Spectra do not have consistent shapes.", RuntimeWarning)

    ####

    ews, ew_errors = np.zeros(n_spec), np.zeros(n_spec)

    for i_s, s in enumerate(spec_):

        spec_have_errors = True if s.shape[1] > 2 else False
        wvs = s[:, 0]

        # apply mask (convert to boolean mask)
        s_mask_ = automask(wvs, mask)

        # mask out any nans
        spec_mask_ = (s_mask_ & ~np.isnan(s[:, 1]))
        spec_mask_ = (s_mask_ & (s[:, 2] >= 0)) if spec_have_errors else spec_mask_

        # apply continuum
        s_norm = s.copy()
        if isinstance(continuum, np.ndarray):
            # normalise, then zero spectra
            s_norm[:, 1] = s_norm[:, 1] / continuum - 1
            s_norm[:, 2] = s_norm[:, 2] / continuum - 1
        else:
            s_norm[:, 1] = s_norm[:, 1] - continuum
            s_norm[:, 2] = s_norm[:, 2] - continuum

        # measure ews
        ew, ew_error = ew_compute(s_norm, s_mask_)

        # save to arrays
        ews[i_s], ew_errors[i_s] = ew, ew_error

    # return array if list of spectra provided, else single value + error
    return (ew, ew_error) if isinstance(spec, np.ndarray) else np.vstack([ews, ew_errors]).T
