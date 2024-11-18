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
	) -> (float, float):
	
	if isinstance(method, str):
		method_ = methods._method_dict.get(method, None)
		if method_ is None:
			raise ValueError(f"Invalid choice of integration method: {method}. Possible choices are {_methods}.")
	elif isinstance(method, callable):
		method_ = method
	else:
		raise TypeError(f"Invalid type for `method`: {type(method)}. `method` should be of type string or callable.")

	has_errors = check_spectra(spectrum)

	wvs, flux = spectrum[:, 0], spectrum[:, 1]
	flux_err = spectrum[:, 2] if has_errors else None

	if isinstance(mask, np.ndarray):
        if mask.size != wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(wvs, mask)
    else:
        mask_ = np.ones(wvs.size, dtype=bool)

    integral, integral_err = method_(spectrum, mask_)

    return integral, integral_err