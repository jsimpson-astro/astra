__all__ = [
    'deprecated_',
    'dummy_pbar',
    'check_spectra',
    'xcheck_spectra',
    'check_vbinned',
    'apply_mask',
    'automask'
]

from typing_extensions import deprecated
# from inspect import signature
import functools
import numpy as np

def deprecated_import(msg):
    import warnings
    warnings.warn(msg, DeprecationWarning, stacklevel=3)

def deprecated_(*dep_args, **dep_kwargs):

    def decorator(func):

        stacklevel = dep_kwargs.get('stacklevel', 1)
        func = deprecated(*dep_args, **dep_kwargs | dict(stacklevel=stacklevel + 1))(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # get deprecation message, add to docs
        msg = dep_args[0] if dep_args else dep_kwargs['msg']
        newdoc = f"- {msg}" + '\n' + ("" if func.__doc__ is None else func.__doc__)
        wrapper.__doc__ = newdoc

        # # set signature to match
        # wrapper.__signature__ = signature(func)

        return wrapper

    return decorator


class dummy_pbar():

    """
    Placeholder for tqdm pbar that does nothing
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, num, *args, **kwargs):
        pass


def check_spectra(
    spectra: list[np.ndarray],
    wv_tol: float = 1e-6
) -> bool:
    """
    Check a list of spectra all match i.e.:
    - All have the same shapes
    - All have the same wavelength scales
    - All have at least two columns
    Returns True if 3 or more columns are present in the spectra.

    """

    # check shapes
    spectra_shapes = [a.shape for a in spectra]
    if len(set(spectra_shapes)) > 1:
        raise IndexError("Spectra do not have the same shapes.")

    n_cols = spectra_shapes[0][1]
    if n_cols < 2:
        raise IndexError("Spectra do not have at least two columns.")

    # check wavelength scales
    if len(spectra) > 1:
        wvs_0 = spectra[0][:, 0]
        wv_dev = np.abs(np.array([a[:, 0] for a in spectra[1:]]) - wvs_0).max()
        if wv_dev > wv_tol:
            raise ValueError(f"Spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")

    # return True if errors present
    if n_cols == 2:
        return False
    else:
        return True


def xcheck_spectra(
    spectra1: list[np.ndarray],
    spectra2: list[np.ndarray],
    wv_tol: float = 1e-6
) -> (bool, bool):
    """
    Check two lists of spectra match i.e.:
    - All spectra within each list have the same shapes
    - All spectra within each list have the same wavelength scales
    - All spectra have at least two columns
    - Spectra in `spectra1` have the same lengths as those in `spectra2`
    - Spectra in `spectra1` have the same wavelength scales as those in `spectra2`

    Returns a 2-tuple of bools, one for each list, True if 3 or more columns are present
    in the respective list.

    """
    # check spectra1 are consistent
    try:
        spectra1_have_errs = check_spectra(spectra1, wv_tol=wv_tol)
    except Exception as e:
        msg = 'Error with spectra1: ' + e.args[0]
        raise type(e)(msg, *e.args[1:])

    # check spectra2 are consistent
    try:
        spectra2_have_errs = check_spectra(spectra2, wv_tol=wv_tol)
    except Exception as e:
        msg = 'Error with spectra2: ' + e.args[0]
        raise type(e)(msg, *e.args[1:])

    # check lengths are consistent between both
    len1 = spectra1[0].shape[0]
    len2 = spectra2[0].shape[0]
    if len1 != len2:
        raise IndexError(f"Spectra have unequal lengths ({len1}, {len2}).")

    # check wavelength deviation between the two
    wvs_1 = spectra1[0][:, 0]
    wv_dev = np.abs(np.array([a[:, 0] for a in spectra2]) - wvs_1).max()
    if wv_dev > wv_tol:
        raise ValueError(f"Spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")

    return spectra1_have_errs, spectra2_have_errs


def check_vbinned(
    wvs: np.ndarray[float],
    wv_tol: float = 1e-6
) -> bool:
    """
    Check a spectrum's wavelength scale is logarithmic,
    i.e. pixels are uniform in velocity space.

    """

    # check monotonic
    if not np.all(wvs[1:] > wvs[:-1]):
        raise ValueError("Wavelength scale is not monotonically increasing.")

    # find log coefficient + const. for scale
    a = (np.log(wvs[-1]) - np.log(wvs[0])) / (wvs.size - 1)
    b = np.log(wvs[0])

    # subtract actual scale from this
    wvs_log = np.exp(a * np.arange(wvs.size) + b)

    # if deviations above wv_tol, return False
    wv_dev = np.abs(wvs - wvs_log).max()

    if wv_dev > wv_tol:
        return False
    else:
        return True


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


def automask(
    wvs: np.ndarray[float],
    mask: np.ndarray[bool] | list[float] | None = None,
) -> np.ndarray[bool]:
    """
    Automatic masking helper.
    Returns a boolean mask matching the provided wavelengths.
    Automatically applies any bounds if given, or generates a blank mask if None.

    """

    if mask is None:
        return np.ones(wvs.shape, dtype=bool)
    elif isinstance(mask, np.ndarray):
        if mask.shape != wvs.shape:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        return mask.copy()
    elif isinstance(mask, list):
        return apply_mask(wvs, mask)
    else:
        return np.ones(wvs.size, dtype=bool)
