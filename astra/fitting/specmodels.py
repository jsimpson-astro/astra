__all__ = [
    'SpectrumInterpolator',
    'blackbody',
    'blackbody_arb_scaled',
    'blackbody_cen_scaled',
    'blackbody_flux_scaled',
    'blackbody_quantile_scaled',
    'flat',
    'flat_arb_scaled',
    'flat_cen_scaled',
    'flat_quantile_scaled',
    'interpspec',
    'interpspec_arb_scaled',
    'interpspec_cen_scaled',
    'interpspec_flux_scaled',
    'interpspec_quantile_scaled',
    'powerlaw',
    'powerlaw_arb_scaled',
    'powerlaw_cen_scaled',
    'powerlaw_quantile_scaled'
]

from astra.utils.constants import c, rsun_pc_scale, bb_prefac, bb_expfac
from astra.utils._helpers import deprecated_

import numpy as np
from scipy import spatial

class SpectrumInterpolator:
    """
    N-D spectrum interpolator.
    For a provided grid of spectra and associated parameters, this class can be called to produce
    a new spectrum at arbitrary parameter within the grid bounds using interpolation.
    Needed for spectrum fitting.

    Parameters:
    wvs: np.ndarray
        Wavelengths to be used for interpolation
    spec: list of np.ndarray
        List of 1D flux arrays of model spectra.
        Must match length of wvs
    params: list of tuples of dicts, or numpy.ndarray
        List of parameters associated to each spectrum.
        Must have the same length as `spec`.
        For 1D, a list of floats is accepted.
        For higher dimensions, a list of tuples (or lists) or np.ndarray is acceptable,
        but `param_names` must be given if the interpolator is to be used for fitting.
        Alternatively, a list of dicts with keys for parameter names can be given.
    param_names: list of str, optional
        List of parameter names, must match number of paramters in `params`.
        If `params` is a list of dict, `param_names` is not needed and will be assembled
        from dict keys in `params`.
        Must be set if interpolator is to be provided to SpectrumFitter and `params` is
        not a list of dict.

    Methods:
    __call__(*args, **kwargs)
        Interpolate to the provided parameters.
        Parameters can either be provided as positional arguments, with the order matching the
        order of `params` when the class was initialised, or as keyword arguments matching
        self.param_names, or a mix.
        Returns interpolated flux at the wavelengths of the input spectra.

    """

    def __init__(
        self,
        wvs: np.ndarray,
        spec: list[np.ndarray],
        params: list[float | list | tuple | dict] | np.ndarray,
        param_names: list[str] | None = None
    ):

        # check number of spectra = number of parameters
        if len(spec) != len(params):
            raise IndexError(f"Length of spectra list ({len(spec)}) does not match length of parameter list ({len(params)}).")

        # # check wavelengths all match
        # if not all(np.all(spec[0][:, 0] == s[:, 0]) for s in spec[1:]):
        #     raise ValueError("Wavelength scales must be identical for all spectra.")
        # check all sizes match
        if not all(spec[0].shape == s.shape for s in spec[1:]):
            raise ValueError("All spectra must have identical shapes.")
        if not all(s.shape == wvs.shape for s in spec):
            raise IndexError("Shape of wavelength array does not match shape of flux arrays.")

        self._wvs = wvs
        self._spec = np.array(spec)

        if isinstance(params[0], dict):
            # check all are dicts and have the same keys
            for pars in params:
                assert isinstance(pars, dict)
                assert pars.keys() == params[0].keys()

            if param_names is not None:
                raise ValueError("param_names given but parameter names already specified by dicts.")

            self._spec_param_names = tuple(params[0].keys())
            self._spec_points = np.array([list(pars.values()) for pars in params])
            self._spec_params = params

        else:
            if isinstance(params, np.ndarray) and len(params.shape) == 1:
                params_ = params[:, None]
            elif isinstance(params, (list, tuple)) and not hasattr(params[0], '__len__'):
                params_ = np.array(params)[:, None]
            else:
                params_ = params

            for pars in params_:
                first_dims = len(params_[0])
                assert isinstance(pars, (list, tuple, np.ndarray))
                assert len(pars) == first_dims

            if param_names is not None:
                self.param_names = param_names
            else:
                self._spec_param_names = None

            self._spec_points = np.array(params_)
            self._spec_params = params_

        self._quiet = True

        self._npars = self._spec_points.shape[1]
        self._dims = [i for i in range(self._npars) if len(set(self._spec_points[:, i])) > 1]
        self._ndims = len(self._dims)

        if self._ndims == 0:
            raise ValueError("Parameters consistent with a single point.")

        self._interp = self._simple_interp if self._ndims == 1 else self._delaunay_interp
        self._init_interp()

    def _simple_interp(self, pars):
        # interpolator for a single parameter dimension

        points1d = self._spec_points[:, self._dims[0]]
        point1d = pars[self._dims[0]]

        if point1d in points1d:
            return self._spec[points1d == point1d][0]

        idxs = np.arange(len(points1d))
        mask_lower = (points1d <= point1d)
        mask_higher = (points1d >= point1d)

        nn_lower = idxs[mask_lower][np.abs(points1d[mask_lower] - point1d).argmin()]
        nn_higher = idxs[mask_higher][np.abs(points1d[mask_higher] - point1d).argmin()]

        nns = np.array([nn_lower, nn_higher])
        #nns = np.abs(points1d - point1d).argsort()[:2]
        nn_points = self._spec_points[nns]
        nn_points1d = nn_points[:, self._dims[0]]

        frac_dist = (nn_points1d - point1d) / (nn_points1d.max() - nn_points1d.min())

        if np.any(frac_dist == 0):
            weights = np.zeros(self._ndims + 1, dtype=float)
            weights[frac_dist == 0] = 1
        else:
            weights = 1 - np.abs(frac_dist)

        #spec_to_average = np.vstack(self._spec[nns])
        #out = np.average(spec_to_average, axis=0, weights=weights)

        out = (self._spec[nns] * weights[:, None]).sum(axis=0)

        if not self._quiet:
            print(f"nearest: {[p[0] for p in nn_points]}")
            #print(f"fractional distance: {frac_dist}")
            print(f"weights: {weights}")

        return out

    def _delaunay_interp(self, pars):
        # delaunay interpolation for 2D+ interpolation

        point = np.array(pars)

        # find nearest neighbours
        simplex_idx = self._tri.find_simplex(point)
        simplex = self._tri.simplices[simplex_idx]

        # weights of N + 1 nearest neighbours for N dims
        weights = np.zeros(self._ndims + 1, dtype=float)

        # barycentric transform
        transform = self._tri.transform[simplex_idx, :self._ndims]
        barydist = point - self._tri.transform[simplex_idx, self._ndims]

        # calculate weights for average (normalised)
        weights[:self._ndims] = transform.dot(barydist)
        weights[self._ndims] = 1 - weights[:self._ndims].sum()

        out = (self._spec[simplex] * weights[:, None]).sum(axis=0)

        if not self._quiet:
            nn_points = self._spec_points[simplex]
            print(f"nearest: {nn_points}")
            #print(f"fractional distance: {frac_dist}")
            print(f"weights: {weights}")

        return out

    def _init_interp(self):
        if self._interp == self._simple_interp:
            self._bounds = [(self._spec_points.min(), self._spec_points.max())]
        elif self._interp == self._delaunay_interp:
            try:
                self._tri = spatial.Delaunay(self._spec_points)
            except spatial.QhullError as qhe:
                raise ValueError("Parameter space is flat:", qhe)

            self._bounds = [(min_, max_) for min_, max_ in zip(self._spec_points.min(axis=0), self._spec_points.max(axis=0))]
        else:
            raise Exception(f"Could not initialise interpolator: {self._interp}")

    def _arg_extract(self, *args, **kwargs):
        # allows calling interpolator with positional and kwargs simultaneously

        if self._spec_param_names is None:
            # no dictionary params (no param names)
            if len(args) == len(self._spec_params[0]):
                params = args
            else:
                raise ValueError(f"Expected {len(self._spec_params[0])} positional arguments but {len(args)} were given.")

        else:
            # dictionary params possible
            if len(args) == len(self._spec_param_names):
                # already given in args, use args, check for duplicates
                dup = [kw for kw in kwargs if kw in self._spec_param_names]
                if len(dup) > 0:
                    raise KeyError(f"{dup} given as keyword argument but already specified by positional argument.")
                else:
                    params = args
            elif len(args) == 0:
                # no args, just check kwargs
                params = [kwargs.get(p) for p in self._spec_param_names]
                missing = [p for p in self._spec_param_names if kwargs.get(p) is None]
                #missing = [p for p in self._spec_param_names if p is None]
                if len(missing) > 0:
                    raise KeyError(f"Missing parameters: {missing}")
            elif len(args) > len(self._spec_param_names):
                # too many args
                raise ValueError(f"Expected {len(self._spec_param_names)} positional arguments but {len(args)} were given.")
            else:
                # mix of args and kwargs - retrieve args first, check kwargs for rest
                params = [a for a, p in zip(args, self._spec_param_names)]
                found = [p for a, p in zip(args, self._spec_param_names)]
                dup = [kw for kw in kwargs if kw in found]
                if len(dup) > 0:
                    raise KeyError(f"{dup} given as keyword argument but already specified by positional argument.")

                params_ = [kwargs.get(p) for p in self._spec_param_names if p not in found]
                missing = [p for p in params_ if p is None]
                if len(missing) > 0:
                    raise KeyError(f"Missing parameters: {missing}")
                else:
                    params = params + params_

        return params

    def check_bounds(self, params):
        # check interpolator bounds satisfied for a call

        check = [b[0] <= p <= b[1] for p, b in zip(params, self._bounds)]

        return all(check)

    def __call__(self, *args, **kwargs):
        # evaluate interpolator

        params = self._arg_extract(*args, **kwargs)

        if not self.check_bounds(params):
            oob = [(p, b) for p, b in zip(params, self._bounds) if not b[0] <= p <= b[1]]
            msg = '\n'.join([f"Value: {p}, bounds: {b}" for p, b in oob])
            raise ValueError(f"Parameters out of bounds: \n{msg}")
        else:
            out = self._interp(params)
            return out

    @property
    def param_names(self):
        return self._spec_param_names

    @param_names.setter
    def param_names(self, param_names):
        if hasattr(self, '_spec_params') and isinstance(self._spec_params[0], dict):
            raise ValueError("Cannot set param_names: parameter names already specified by dicts.")

        if hasattr(self, '_npars') and len(param_names) != self.npars:
            raise IndexError(f"Length of param_names {len(param_names)} does not match number of parameters {self.npars}.")

        self._spec_param_names = param_names

    @property
    def wvs(self):
        return self._wvs

    @property
    def bounds(self):
        return self._bounds

    @property
    def ndims(self):
        return self._ndims

    @property
    def npars(self):
        return self._npars


####  Basic spectrum models  ####

# base models
# all functions start with wavelengths, fluxes as first two args

def flat(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
) -> np.ndarray[float]:
    """
    Flat spectrum like `wvs`, with all fluxes = 1
    """
    return np.ones_like(wvs)


def powerlaw(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    expo: float
) -> np.ndarray[float]:
    """
    Simple power law, frequency^(-expo)
    """
    freq = (c * 1e10) / wvs
    power = freq ** (-expo)
    return power


def blackbody(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    teff: float
) -> np.ndarray[float]:
    """
    Blackbody radiation from Planck's law, in erg/s/cm^2/AA
    """
    # wavelengths to m
    wav = wvs * 1e-10
    # plancks law
    model = (bb_prefac / wav**5) / (np.exp(bb_expfac / (wav * teff)) - 1)
    # rescale to erg/s/cm^2/AA
    model = model / 1e7
    return model


def interpspec(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    interp: SpectrumInterpolator,
    pars: dict[str, float]
) -> np.ndarray[float]:
    """
    Wrapper for an instance of SpectrumInterpolator, to provide a similar interface
    to the other spectral models here.
    """
    model = interp(**pars)
    # model = interp(**pars) if isinstance(pars, dict) else interp(pars)
    return model


_model_funcs = [flat, powerlaw, blackbody, interpspec]

# scalings

def _arb_scale(model_flux, scale):
    return scale * model_flux


def _quantile_scale(model_flux, flux, quantile):
    return model_flux * (np.quantile(flux, quantile) / np.quantile(model_flux, quantile))


def _cen_scale(model_flux, wvs, flux, quantile, window):
    cen_wv = (wvs[0] + wvs[-1]) / 2
    cen_wv_idx = np.abs(wvs - cen_wv).argmin()

    if window is None:
        averaging_mask = np.ones_like(flux, dtype=bool)
    else:
        averaging_mask = (wvs > cen_wv - window / 2) & (wvs < cen_wv + window / 2)

    return model_flux * (np.quantile(flux[averaging_mask], quantile) / model_flux[cen_wv_idx])


def _flux_scale(model_flux, radius, distance):
    return (rsun_pc_scale * (radius / distance))**2 * model_flux


_scale_funcs = [_arb_scale, _quantile_scale, _cen_scale, _flux_scale]

####

# models with scaling
_scaling_options = {
    flat: [_arb_scale, _quantile_scale, _cen_scale],
    powerlaw: [_arb_scale, _quantile_scale, _cen_scale],
    blackbody: [_arb_scale, _quantile_scale, _cen_scale, _flux_scale],
    interpspec: [_arb_scale, _quantile_scale, _cen_scale, _flux_scale]
}

# there are better ways of doing this, but this is the simplest
# for the user to follow (function signatures, docs, etc.)
# could move to model class in the future

# flat

def flat_arb_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    scale: float
) -> np.ndarray[float]:
    """
    Flat spectrum like `wvs`,
    with flux set to `scale`.
    """
    model = flat(wvs, flux)
    model = _arb_scale(model, scale)
    return model


def flat_quantile_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    quantile: float = 0.995
) -> np.ndarray[float]:
    """
    Flat spectrum like `wvs`,
    scaled to the `quantile` quantile of `flux`.
    """
    model = flat(wvs, flux)
    model = _quantile_scale(model, flux, quantile)
    return model


def flat_cen_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    quantile: float = 0.995,
    window: float | None = None
) -> np.ndarray[float]:
    """
    Flat spectrum like `wvs`,
    scaled to the `quantile` quantile of `flux` in a central window of width `window` (in AA).
    """
    model = flat(wvs, flux)
    model = _cen_scale(model, wvs, flux, quantile, window)
    return model

# powerlaw

def powerlaw_arb_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    expo: float,
    scale: float
) -> np.ndarray[float]:
    """
    Simple power law, frequency^(-expo),
    multiplied by arbitrary scaling `scale`.
    """
    model = powerlaw(wvs, flux, expo)
    model = _arb_scale(model, scale)
    return model


def powerlaw_quantile_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    expo: float,
    quantile: float = 0.995
) -> np.ndarray[float]:
    """
    Simple power law, frequency^(-expo),
    scaled to the `quantile` quantile of `flux`.
    """
    model = flat(wvs, flux)
    model = _quantile_scale(model, flux, quantile)
    return model


def powerlaw_cen_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    expo: float,
    quantile: float = 0.995,
    window: float | None = None
) -> np.ndarray[float]:
    """
    Simple power law, frequency^(-expo),
    scaled to the `quantile` quantile of `flux` in a central window of width `window` (in AA).
    """
    model = powerlaw(wvs, flux, expo)
    model = _cen_scale(model, wvs, flux, quantile, window)
    return model

# blackbody

def blackbody_arb_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    teff: float,
    scale: float
) -> np.ndarray[float]:
    """
    Blackbody radiation from Planck's law, in erg/s/cm^2/AA,
    multiplied by arbitrary scaling `scale`.
    """
    model = blackbody(wvs, flux, teff)
    model = _arb_scale(model, scale)
    return model


def blackbody_quantile_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    teff: float,
    quantile: float = 0.995
) -> np.ndarray[float]:
    """
    Blackbody radiation from Planck's law, in erg/s/cm^2/AA,
    scaled to the `quantile` quantile of `flux`.
    """
    model = blackbody(wvs, flux, teff)
    model = _quantile_scale(model, flux, quantile)
    return model


def blackbody_cen_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    teff: float,
    quantile: float = 0.995,
    window: float | None = None
) -> np.ndarray[float]:
    """
    Blackbody radiation from Planck's law, in erg/s/cm^2/AA,
    scaled to the `quantile` quantile of `flux` in a central window of width `window` (in AA).
    """
    model = blackbody(wvs, flux, teff)
    model = _cen_scale(model, wvs, flux, quantile, window)
    return model


def blackbody_flux_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    teff: float,
    radius: float,
    distance: float
) -> np.ndarray[float]:
    """
    Blackbody radiation from Planck's law, in erg/s/cm^2/AA,
    scaled to a star with radius `radius` (in Rsun) and distance `distance` (in pc).
    """
    model = blackbody(wvs, flux, teff)
    model = _flux_scale(model, radius, distance)
    return model

# interpolator-based

def interpspec_arb_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    interp: SpectrumInterpolator,
    pars: dict[str, float],
    scale: float
) -> np.ndarray[float]:
    """
    Wrapper for an instance of SpectrumInterpolator,
    multiplied by arbitrary scaling `scale`.
    """
    model = interpspec(wvs, flux, interp, pars)
    model = _arb_scale(model, scale)
    return model


def interpspec_quantile_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    interp: SpectrumInterpolator,
    pars: dict[str, float],
    quantile: float = 0.995
) -> np.ndarray[float]:
    """
    Wrapper for an instance of SpectrumInterpolator,
    scaled to the `quantile` quantile of `flux`.
    """
    model = interpspec(wvs, flux, interp, pars)
    model = _quantile_scale(model, flux, quantile)
    return model


def interpspec_cen_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    interp: SpectrumInterpolator,
    pars: dict[str, float],
    quantile: float = 0.995,
    window: float | None = None
) -> np.ndarray[float]:
    """
    Wrapper for an instance of SpectrumInterpolator,
    scaled to the `quantile` quantile of `flux` in a central window of width `window` (in AA).
    """
    model = interpspec(wvs, flux, interp, pars)
    model = _cen_scale(model, wvs, flux, quantile, window)
    return model


def interpspec_flux_scaled(
    wvs: np.ndarray[float],
    flux: np.ndarray[float],
    interp: SpectrumInterpolator,
    pars: dict[str, float],
    radius: float,
    distance: float
) -> np.ndarray[float]:
    """
    Wrapper for an instance of SpectrumInterpolator,
    scaled to a star with radius `radius` (in Rsun) and distance `distance` (in pc).
    """
    model = interpspec(wvs, flux, interp, pars)
    model = _cen_scale(model, radius, distance)
    return model


# dict of each model and their scaled versions
_models_and_scales = {
    model_func.__name__: [
        model_func.__name__ + scale_func.__name__ + 'd' for scale_func in scale_funcs
    ]
    for model_func, scale_funcs in _scaling_options.items()
}

# exposed lists for reference
BASE_MODELS = [func.__name__ for func in _model_funcs]
SCALED_MODELS = [model for models in _models_and_scales.values() for model in models]
ALL_MODELS = sorted(BASE_MODELS + SCALED_MODELS)

@deprecated_("scaled_spec is deprecated, use interpspec (or interpspec_flux_scaled) instead.")
def scaled_spec(pars, radius, distance, interp):
    #spec = interp(pars)
    spec = interp(**pars) if isinstance(pars, dict) else interp(pars)
    spec = (2.254610138e-8 * (radius / distance))**2 * spec
    return spec


if __name__ == '__main__':
    generated_all = (
        "__all__ = [\n    "
        + ',\n    '.join(['\'SpectrumInterpolator\''] + [f"\'{m}\'" for m in ALL_MODELS])
        + "\n]"
    )
    print(generated_all)
