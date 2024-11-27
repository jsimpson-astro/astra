from astra.utils import calculate_vpx, taper_spec
from astra.utils._helpers import automask, xcheck_spectra, check_vbinned, deprecated_, dummy_pbar

import warnings
import numpy as np
from tqdm.auto import tqdm
from numba import njit
from numba.types import bool_


c_ = 299792.458
# wv_tol = 1e-6


#### backend functions ####


@njit
def xcor_standard_compute(
    x1: np.ndarray[float],
    x2: np.ndarray[float],
    x1_err: np.ndarray[float] | None = None,
    mask: np.ndarray[bool] | None = None,
    shifts: tuple[int, int] = (-10, 10)
) -> (np.ndarray[float], np.ndarray[float]):
    """
    Perform cross-correlation using the standard computation at each integer shift.
    Returns the value of the cross-correlation at every shift.

    Parameters:
    x1: numpy.ndarray
        Correlation input 1
    x2: numpy.ndarray
        Correlate input 2, same size as `x1`
    x1_err: numpy.ndarray or None, default None
        Errors on `x1`, optional, same size as `x1`
    mask: numpy.ndarray or None, default None
        Boolean array to mask `x1` and `x2`, optional, same size as `x1` and `x2`
    shifts: tuple of ints, length 2
        Tuple of pixel shifts, (negative, positive), shifts[1] > shifts[0]

    Returns:
    xcor: numpy.ndarray
        Array of correlation results at every shift
        Size of (shifts[1] - shifts[0] + 1)
    xcor_error: numpy.ndarray
        Error on correlation results, same size as `xcor`

    """
    mask = np.ones(x1.shape, dtype=bool_) if mask is None else mask

    # first and last valid pixels
    idx0, idx1 = mask.argmax(), len(mask) - mask[::-1].argmax() - 1
    total_unmasked = np.count_nonzero(mask)  # number of unmasked pixels

    x4, x5 = np.zeros((shifts[1] - shifts[0] + 1)), np.zeros((shifts[1] - shifts[0] + 1))

    for out_idx, i in enumerate(range(shifts[0], shifts[1] + 1)):

        # slice according to shifts
        xcor_slice1 = slice(max(idx0, idx0 + i), min(idx1, idx1 + i) + 1)
        xcor_slice2 = slice(max(idx0, idx0 + i) - i, min(idx1, idx1 + i) + 1 - i)

        # slice mask to match
        xcor_mask1 = mask[xcor_slice1]
        xcor_mask2 = mask[xcor_slice1]

        # slice data, applying slice first followed by slice mask
        xcor_x1 = x1[xcor_slice1][xcor_mask1]
        xcor_x2 = x2[xcor_slice2][xcor_mask2]

        if x1_err is not None:
            xcor_x1_err = x1_err[xcor_slice1][xcor_mask1]
            weights = (1 / xcor_x1_err**2)
            sum1 = (weights * xcor_x2**2).sum()
            sum2 = (weights * xcor_x1 * xcor_x2).sum()
        else:
            sum1 = (xcor_x2**2).sum()
            sum2 = (xcor_x1 * xcor_x2).sum()

        current_unmasked = np.count_nonzero(xcor_mask1)

        if sum1 > 0:
            fac = total_unmasked / current_unmasked
            x4[out_idx] = fac * sum2  # / weights.sum()**0.5 # normalisation, not applied in molly
            # / weights.sum()**0.5 # normalisation, not applied in molly
            x5[out_idx] = fac * np.sqrt(sum1)
        else:
            x4[out_idx] = 0
            x5[out_idx] = -1

    return x4, x5


@njit
def xcor_quad_max(xcor: np.ndarray) -> (float, float):
    """
    Find the maximum of a cross-correlation output with quadratic
    fitting of the maximum and two nearest points.

    Parameters:
    xcor: numpy.ndarray
        Cross-correlation output data

    Returns:
    max_loc: float
        Calculated location of the maximum
    error: float
        Error on calculated location of maximum

    """

    max_idx = xcor.argmax()
    xcor_max = xcor[max_idx]

    # clip edges
    max_idx = 1 if max_idx == 0 else max_idx
    max_idx = max_idx - 1 if max_idx == xcor.size - 1 else max_idx

    xcor_max_a, xcor_max_b = xcor[max_idx - 1], xcor[max_idx + 1]
    z1, z2 = xcor_max_b - xcor_max_a, xcor_max_a + xcor_max_b - 2 * xcor_max
    max_loc = max_idx - 0.5 * (z1 / z2)
    error = 1 / (-0.5 * z2)**0.5

    return max_loc, error


def xcor_standard(
    x1: np.ndarray[float],
    x2: np.ndarray[float],
    x1_err: np.ndarray[float] | None = None,
    mask: np.ndarray[bool] | None = None,
    shifts: tuple[int, int] = (-10, 10)
) -> (float, float):
    """
    Perform cross-correlation using the standard computation at each integer shift.
    Returns location of maximum cross-correlation, and its error,
    using a quadratic fit to the peak.

    Parameters:
    x1: numpy.ndarray
        Correlation input 1
    x2: numpy.ndarray
        Correlate input 2, same size as `x1`
    x1_err: numpy.ndarray or None, default None
        Errors on `x1`, optional, same size as `x1`
    mask: numpy.ndarray or None, default None
        Boolean array to mask `x1` and `x2`, optional, same size as `x1` and `x2`
    shifts: tuple of ints, length 2
        Tuple of pixel shifts, (negative, positive), shifts[1] > shifts[0]

    Returns:
    loc: float
        Shift with maximum cross-correlation
    loc_err: float
        Error on `loc`

    """
    # input checks - negligible impact on runtime
    if x1.ndim != 1:
        raise ValueError('x1 must be one-dimensional.')
    if x2.ndim != 1:
        raise ValueError('x2 must be one-dimensional.')

    if not (x1.shape == x2.shape):
        raise IndexError(f"x1 and x2 must all be the same shape: ({x1.shape}, {x2.shape})")
    if mask is not None and mask.shape != x1.shape:
        raise IndexError(f"mask does not match sizes of x1 and x2: ({x1.shape})")
    if x1_err is not None and x1_err.shape != x1.shape:
        raise IndexError(f"x1_err and x1 do not have the same shape: ({x1_err.shape}, {x1.shape})")
    if not all(isinstance(i, (np.integer, int)) for i in shifts):
        raise TypeError(f"shifts ({shifts}) must be given as ints")
    if not shifts[0] < shifts[1]:
        raise ValueError(f"shifts[0] ({shifts[0]}) must be less than shifts[1] ({shifts[1]})")
    if any(abs(s) > x1.size // 2 for s in shifts):
        raise IndexError(f"shifts ({shifts}) out of bounds for x1 with size {x1.size}")

    # mask = np.ones_like(x1, dtype=bool) if mask is None else mask

    xcor, xcor_err = xcor_standard_compute(x1, x2, x1_err, mask, shifts)
    max_loc, max_loc_err = xcor_quad_max(xcor)  # xcor_err is unused

    # max_loc is simply the index - offset to shifts
    max_loc = max_loc + shifts[0]

    return max_loc, max_loc_err


#### frontend ####


def xcorrelate(
    obs: list[np.ndarray[float]],
    templ: np.ndarray[float] | list[np.ndarray[float]],
    mask: np.ndarray[bool] | list[float] | None = None,
    shifts: tuple[int, int] = (-10, 10),
    initial_shifts: list[int] | None = None,
    taper: float = 0.,
    xcor_func: callable = xcor_standard,
    progress: bool = True
) -> list[np.ndarray]:
    """
    Perform cross-correlation over a series of observed spectra,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs: list of numpy.ndarray
        List of arrays of observed spectra.
        All spectra must have identical wavelength scales. Each spectrum should have two,
        or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templ: numpy.ndarray or list of numpy.ndarray
        Array or list of arrays of template spectra. If list, a list of results is returned.
        All spectra must have identical wavelength scales, which match the observed spectra.
        Each spectrum should have two columns: wavelength and flux.
        Additional columns will be ignored. It is assumed noise is dominated by observed spectra.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array the same size as
        the wavelengths of the observed and template spectra, or a list of 2-tuples defining
        upper and lower bounds to exclude.
    shifts: tuple of ints, length 2
        Tuple of pixel shifts, (negative, positive), where shifts[1] > shifts[0]
    initial_shifts: list of ints or None, default None
        Initial pixel shifts for correlation. Must have the same length as obs.
    taper: float, default 0.
        From 0 to 1, fraction to taper from ends of the flux of both observed and template spectra.
    xcor_func: callable, default xcor_standard
        Function for computing the cross-correlation between two spectra, with the form:

        xcor_func(flux1: numpy.ndarray[float],
                  flux2: numpy.ndarray[float],
                  flux1_err: numpy.ndarray[float] | None = None,
                  mask: numpy.ndarray[bool] | None = None,
                  shifts: tuple[int, int] = None
                  ) -> float, float

        which returns the maximum cross-correlation shift (in pixels) and its error (in pixels).
    progress: bool, default True
        Controls display of tqdm progress bar, optional. False to disable

    Returns:
    xcor_results: numpy.ndarray or list of numpy.ndarray
        2D arrays, one per template, each containing a radial velocity curve.
        Two columns (radial velocity + errors), one row per observed spectrum.
        If a single template is provided, only a single 2D array is returned.

    """

    templates = [templ] if isinstance(templ, np.ndarray) else templ

    # check spectra match
    try:
        obs_w_errors, templ_w_errors = xcheck_spectra(obs, templates)
    except Exception as e:
        msg = e.args[0].replace('spectra1', 'obs').replace('spectra2', 'templates')
        raise type(e)(msg, *e.args[1:])

    obs_wvs = obs[0][:, 0]

    if not check_vbinned(obs_wvs):
        warnings.warn("Spectra not uniform in velocity space - results may be meaningless.", RuntimeWarning)

    n_obs = len(obs)
    n_templ = len(templates)

    # check shift tuple
    if not (isinstance(shifts[0], (np.integer, int)) and isinstance(shifts[1], (np.integer, int))):
        raise TypeError("shifts must be given as ints")
    if not shifts[0] < shifts[1]:
        raise ValueError(f"shifts[0] ({shifts[0]}) must be less than shifts[1] ({shifts[1]})")

    # check initial shifts if provided, including bounds
    if initial_shifts is not None:
        if len(initial_shifts) != len(obs):
            raise IndexError("initial_shifts must have the same length as obs.")
        for initial_shift in initial_shifts:
            if not isinstance(initial_shift, (np.integer, int)):
                raise TypeError("initial_shifts must be given as ints")
            if any(abs(s + initial_shift) > obs_wvs.size // 2 for s in shifts):
                raise IndexError(f"shifts ({shifts}, {initial_shift}) out of bounds"
                                 f"for wavelengths with size {obs_wvs.size}")

    # check taper
    if taper > 1:
        raise ValueError(f"taper fraction cannot exceed 1 (taper={taper})")
    if taper > 0.5:
        warnings.warn(f"Warning: taper > 0.5 ({taper}), tapering will overlap in centre of spectra", RuntimeWarning)

    # check func
    if not callable(xcor_func):
        raise TypeError(f"xcor_func {xcor_func} is not callable.")

    # check and apply mask (convert to boolean mask)
    mask_ = automask(obs_wvs, mask)

    initial_shifts = [0] * n_obs if initial_shifts is None else initial_shifts

    # calculate average velocity using first spectrum
    v_avg = calculate_vpx(obs_wvs)

    if taper > 0.:
        # clip to edges of mask, and scale taper so same % of spectrum cut
        idx0, idx1 = mask_.argmax(), len(mask_) - mask_[::-1].argmax()
        taper_scaled = taper * obs_wvs.size / (idx1 - idx0)

        obs_data = [None] * n_obs
        for i, s in enumerate(obs):
            obs_data[i] = s.copy()
            obs_data[i][idx0:idx1] = taper_spec(s[idx0:idx1], taper_scaled)

        templ_data = [None] * n_templ
        for i, s in enumerate(templates):
            templ_data[i] = s.copy()
            templ_data[i][idx0:idx1] = taper_spec(s[idx0:idx1], taper_scaled)
    else:
        obs_data = obs
        templ_data = templates

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar

    xcor_results = [None] * n_templ

    with pbar_manager(desc='xcor: ', total=n_obs * n_templ) as pbar:

        for i_templ, t in enumerate(templ_data):

            t_wvs, t_flux = t[:, 0], t[:, 1]

            # mask out any nans
            t_mask_ = (mask_ & ~np.isnan(t_flux))

            xcor_result = np.zeros((n_obs, 2))

            for i_obs, o in enumerate(obs_data):

                initial_shift = initial_shifts[i_obs]

                obs_wvs, obs_flux = o[:, 0], o[:, 1]
                obs_flux_err = o[:, 2] if obs_w_errors else None

                # mask out any nans
                obs_mask_ = (t_mask_ & ~np.isnan(obs_flux))
                obs_mask_ = (obs_mask_ & (obs_flux_err >= 0)) if obs_w_errors else obs_mask_

                # perform xcor

                shifts_ = (shifts[0] + initial_shift, shifts[1] + initial_shift)

                max_loc, max_loc_error = xcor_func(obs_flux, t_flux, obs_flux_err, obs_mask_, shifts_)

                rv, rv_err = v_avg * max_loc, v_avg * max_loc_error

                # save results to arrays
                xcor_result[i_obs] = rv, rv_err

                pbar.update(1)

            # add result data
            xcor_results[i_templ] = xcor_result

    # return list of arrays if list of templates provided, else single array
    return xcor_results[0] if isinstance(templ, np.ndarray) else xcor_results


#### legacy ####

@deprecated_("astra.correlate.xcorrelate_multi has been superceded by astra.correlate.xcorrelate.")
def xcorrelate_multi(
    obs: list[np.ndarray[float]],
    templ: np.ndarray[float] | list[np.ndarray[float]],
    mask: np.ndarray[bool] | list[float] | None = None,
    shifts: tuple[int, int] = (-10, 10),
    initial_shifts: list[int] | None = None,
    taper: float = 0.,
    xcor_func: callable = xcor_standard,
    progress: bool = True
) -> list[np.ndarray]:
    """
    Perform cross-correlation over a series of observed spectra,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs: list of numpy.ndarray
        List of arrays of observed spectra.
        All spectra must have identical wavelength scales. Each spectrum should have two,
        or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templ: numpy.ndarray or list of numpy.ndarray
        Array or list of arrays of template spectra. If list, a list of results is returned.
        All spectra must have identical wavelength scales, which match the observed spectra.
        Each spectrum should have two columns: wavelength and flux.
        Additional columns will be ignored. It is assumed noise is dominated by observed spectra.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array the same size as
        the wavelengths of the observed and template spectra, or a list of 2-tuples defining
        upper and lower bounds to exclude.
    shifts: tuple of ints, length 2
        Tuple of pixel shifts, (negative, positive), where shifts[1] > shifts[0]
    initial_shifts: list of ints or None, default None
        Initial pixel shifts for correlation. Must have the same length as obs.
    taper: float, default 0.
        From 0 to 1, fraction to taper from ends of the flux of both observed and template spectra.
    xcor_func: callable, default xcor_standard
        Function for computing the cross-correlation between two spectra, with the form:

        xcor_func(flux1: numpy.ndarray[float],
                  flux2: numpy.ndarray[float],
                  flux1_err: numpy.ndarray[float] | None = None,
                  mask: numpy.ndarray[bool] | None = None,
                  shifts: tuple[int, int] = None
                  ) -> float, float

        which returns the maximum cross-correlation shift (in pixels) and its error (in pixels).
    progress: bool, default True
        Controls display of tqdm progress bar, optional. False to disable

    Returns:
    xcor_results: numpy.ndarray or list of numpy.ndarray
        2D arrays, one per template, each containing a radial velocity curve.
        Two columns (radial velocity + errors), one row per observed spectrum.
        If a single template is provided, only a single 2D array is returned.

    """
    return xcorrelate(obs, templ, mask, shifts, initial_shifts, taper, xcor_func, progress)
