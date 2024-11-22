from astra.utils.helpers import dummy_pbar, check_spectra, check_vbinned

import numpy as np
from numba import njit
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import emcee

c_ = 299792.458


@njit
def scaled_gauss(
    xs: np.ndarray,
    x0: float = 0.,
    sigma: float = 1.,
    A: float = 1.,
) -> np.ndarray:
    """
    Simple function to generate a scaled Gaussian, multiplied by scaling factor A.

    """
    return A * np.exp(-(xs - x0)**2 / (2 * sigma**2))


@njit
def generate_multilines(
    wvs: np.ndarray,
    line_wvs: np.ndarray,
    line_heights: np.ndarray,
    rv: float,
    fwhm: float
) -> np.ndarray:
    """
    Creates multiple synthetic lines using `scaled_gauss`.

    Parameters:
    wvs: np.ndarray
        Wavelengths to compute lines at.
    line_wvs: np.ndarray
        Central wavelengths of lines.
    line_heights: np.ndarray
        Scale height of each line, same length as `line_wvs`.
    rv: float
        Radial velocity shift of lines, in km/s.
    fwhm: float
        Full-width half-maximum of lines, in km/s.

    """

    line_array = np.zeros_like(wvs)

    for line_wv, line_height in zip(line_wvs, line_heights):
        line_array += scaled_gauss(wvs,
                                   line_wv * (1 + (rv / c_)),
                                   line_wv * (fwhm / c_),
                                   line_height
                                   )

    return line_array


@njit
def generate_conti(
    wvs: np.ndarray,
    block_start_wvs: np.ndarray,
    block_end_wvs: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """

    """

    conti_flux = np.zeros_like(wvs)

    for start, end, offset, slope in zip(block_start_wvs, block_end_wvs, offsets, slopes):
        block_mask = (wvs >= start) & (wvs <= end)
        conti_flux[block_mask] += offset + slope * (wvs[block_mask] - start)

    return conti_flux


def linefitmc(
    spectra: list[np.ndarray],
    line_wvs: list[float],
    n_walkers: int = 16,
    n_samples: int = 1000,
    n_burnin: int = 100,
    init_fwhm: float = 100.,
    init_heights: list[float] | float = -1.,
    init_offset: float = 0.,
    fit_fwhm: bool = True,
    fit_offsets: bool = True,
    fit_slopes: bool = False,
    fwhm_bounds: tuple | None = None,
    height_bounds: tuple | None = None,
    offset_bounds: tuple | None = None,
    multiple_offsets: bool = False,
    search_width: float = 2000.,
    mask_width: float | None = None,
    mask_width_scale: float = 2.,
    v_shifts: list[float] | None = None,
    return_fwhm: bool = False,
    return_all: bool = False,
    **kws
) -> np.ndarray[float]:
    """
    MCMC Gaussian line-fitting function.

    Parameters
    ----------
    spectra: list of np.ndarray
        List of input spectra to measure radial velocities from.
        Each spectrum should have two or optionally three columns: wavelength, flux, and flux error.
    line_wvs: list of floats
        List of rest wavelengths of lines of interest, in angstroms.
    init_fwhm: float, default 100 km/s
        Starting value of FWHM of lines, in km/s.
    init_heights: float or list of floats, default -1.

    init_offsets: float or list, default 1.

    init_slopes: float or list of floats, default 0.

    fit_offsets: bool, default True

    fit_slopes: bool, default False

    search_width: float, default 1000 km/s

    v_shifts: list of floats, optional

    Returns
    -------
    rv_array: np.ndarray
        2D array of results, one row per spectrum.
        Column 1: radial velocities, column 2: radial velocity errors.

    """
    c_ = 299792.458

    spec_w_errors = check_spectra(spectra)

    if return_fwhm and not fit_fwhm:
        raise ValueError(f"Cannot return fwhm if not fitting fwhm.")

    plot_line_fits = kws.get('plot_line_fits', False)
    print_info = kws.get('print_info', False)
    progress = kws.get('progress', True)

    # check params
    if not hasattr(init_heights, '__len__'):
        init_heights_ = [init_heights] * len(line_wvs)
    else:
        init_heights_ = init_heights
    n_lines = len(init_heights_)

    if v_shifts is None:
        v_shifts = [0.] * len(line_wvs)

    if fwhm_bounds is None:
        fwhm_bounds = (0., search_width)
    else:
        if fwhm_bounds[0] > fwhm_bounds[1]:
            raise ValueError("Invalid fwhm_bounds - fwhm_bounds[1] must be greater than fwhm_bounds[0].")
        if fwhm_bounds[0] < 0:
            raise ValueError("Invalid fwhm_bounds - cannot contain negative values.")

    if height_bounds is not None and height_bounds[0] > height_bounds[1]:
        raise ValueError("Invalid height_bounds - height_bounds[1] must be greater than height_bounds[0].")

    if offset_bounds is not None and offset_bounds[0] > offset_bounds[1]:
        raise ValueError("Invalid offset_bounds - offset_bounds[1] must be greater than offset_bounds[0].")

    # set mask width from scale if not explicitly given
    mask_width = mask_width_scale * search_width if mask_width is None else mask_width

    ####

    # setup masks and blocking, determine number of free params
    wvs = spectra[0][:, 0]  # all spectra have same wv scales
    mask = np.ones(wvs.shape, dtype=bool)

    # construct mask around lines
    for line_wv in line_wvs:
        wv_width = line_wv * (mask_width / c_)
        lb, ub = line_wv - wv_width / 2, line_wv + wv_width / 2
        mask = mask & ~((wvs > lb) & (wvs < ub))
    mask = ~mask

    # find edges of mask
    block_search = np.diff(np.r_[False, mask, False].astype(int))
    starts0, ends0 = wvs[block_search[:-1] == 1], wvs[block_search[1:] == -1]
    n_blocks = len(starts0)

    # prepare starting samples
    offsets0 = init_offset * np.ones(n_blocks)
    slopes0 = np.zeros(n_blocks)

    n_dim = (2 if fit_fwhm else 1) + n_lines  # 1 for rv, 1 for fwhm, each line's height
    init_pars = [None] + ([init_fwhm] if fit_fwhm else [])  # placeholder rv
    init_pars += init_heights_
    init_scatter = [search_width / 2] + ([init_fwhm / 10] if fit_fwhm else [])
    init_scatter += [h / 10 for h in init_heights_]

    if fit_offsets:
        n_dim = n_dim + (n_blocks if multiple_offsets else 1)
        init_pars += list(offsets0) if multiple_offsets else [init_offset]
        init_scatter += [0.1 for o in offsets0] if multiple_offsets else [0.1]

    if fit_slopes:
        n_dim = n_dim + n_blocks
        init_pars += list(slopes0)
        init_scatter += [1e-6 for s in slopes0]

    # assemble cost_function
    def cost_func(pars, rv0):
        rv = pars[0]

        if fit_fwhm:
            fwhm = pars[1]
            par_idx = 2
        else:
            fwhm = init_fwhm
            par_idx = 1

        # check uniform 'priors'
        if not rv0 - search_width < rv < rv0 + search_width:
            return -np.inf

        if not fwhm_bounds[0] < fwhm < fwhm_bounds[1]:
            return -np.inf

        heights = pars[par_idx:n_lines + par_idx]
        par_idx += n_lines

        if (height_bounds is not None
                and np.any((heights < height_bounds[0]) | (heights > height_bounds[1]))):
            return -np.inf

        # deal with variable number of parameters
        if fit_offsets:
            offsets = pars[par_idx:par_idx + n_blocks] if multiple_offsets else np.array([pars[par_idx]] * n_blocks)
            par_idx += n_blocks if multiple_offsets else 1

            if (offset_bounds is not None
                    and np.any((offsets < offset_bounds[0]) | (offsets > offset_bounds[1]))):
                return -np.inf
        else:
            offsets = offsets0
        if fit_slopes:
            slopes = pars[par_idx:]
        else:
            slopes = slopes0

        # compute fluxes
        line_fluxes = generate_multilines(wvs_masked, np.array(line_wvs), heights, rv, fwhm)
        conti_flux = generate_conti(wvs_masked, starts, ends, offsets, slopes)
        total_flux = line_fluxes + conti_flux

        # compute log likelihood
        chisq = ((flux_masked - total_flux)**2 / flux_errs_masked**2).sum()
        # lnprior = np.log(1.0 /(np.sqrt(2 * np.pi) * rv0_err))-0.5*(rv - rv0)**2 / rv0_err**2
        lnprior = 0

        return -0.5 * chisq + lnprior

    ##############

    # setup figure if requested
    if plot_line_fits:

        fig, axes = plt.subplots(figsize=(10, 1.5 * len(spectra)), ncols=n_blocks, sharey=True)
        fig.subplots_adjust(wspace=0)
        axes = [axes] if n_blocks == 1 else axes

    progress = True

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar

    # rvs, rv_errs = np.zeros(len(spectra)), np.zeros(len(spectra))

    # double for param + errs
    output_array = np.zeros((len(spectra), 2 * n_dim))

    with pbar_manager(desc='linefitmc: ', total=len(spectra)) as pbar:
        for ispec, spec in enumerate(spectra):

            wvs, flux = spec[:, 0], spec[:, 1]
            flux_errs = spec[:, 2] if spec.shape[1] > 2 else np.ones_like(flux)

            # create mask over all wvs
            mask = np.ones(wvs.size, dtype=bool)

            rv0 = v_shifts[ispec]
            rv0_err = 10
            init_pars[0] = rv0

            # construct mask around lines
            for line_wv in line_wvs:
                # shifted wv
                wv_width = line_wv * (mask_width / c_)
                line_wv_shifted = line_wv * (1 + (rv0 / c_))
                lb, ub = line_wv_shifted - wv_width / 2, line_wv_shifted + wv_width / 2
                mask = mask & ~((wvs > lb) & (wvs < ub))
            mask = ~mask

            wvs_masked, flux_masked, flux_errs_masked = wvs[mask], flux[mask], flux_errs[mask]

            # find edges of mask
            starts, ends = starts0 * (1 + rv0 / c_), ends0 * (1 + rv0 / c_)

            #####################

            # sample
            init_state = init_pars + init_scatter * np.random.uniform(-1, 1, size=(n_walkers, n_dim))
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, cost_func, args=(rv0,))
            _ = sampler.run_mcmc(init_state, nsteps=n_samples, progress=False)
            r = sampler.get_chain(flat=True, thin=1, discard=n_burnin)

            # get outputs and errors, save
            pars_out = np.median(r, axis=0)
            par_errs = 0.5 * (np.quantile(r, 0.84, axis=0) - np.quantile(r, 0.16, axis=0))
            output_array[ispec, 0::2] = pars_out
            output_array[ispec, 1::2] = par_errs

            pbar.update(1)

            par_lerrs = pars_out - np.quantile(r, 0.16, axis=0)
            par_uerrs = np.quantile(r, 0.84, axis=0) - pars_out

            if plot_line_fits or print_info:

                rv_out, fwhm_out = pars_out[0], pars_out[1]
                rv_out = pars_out[0]

                # deal with variable number of parameters
                if fit_fwhm:
                    fwhm_out = pars_out[1]
                    par_idx = 2
                else:
                    fwhm_out = init_fwhm
                    par_idx = 1
                heights_out = pars_out[par_idx:par_idx + n_lines]
                par_idx += n_lines

                if fit_offsets:
                    offsets_out = pars_out[par_idx:par_idx + n_blocks] if multiple_offsets else np.array([pars_out[par_idx]] * n_blocks)
                    par_idx += n_blocks if multiple_offsets else 1
                else:
                    offsets_out = offsets0
                if fit_slopes:
                    slopes_out = pars_out[par_idx:]
                else:
                    slopes_out = slopes0

                # # save results
                # rvs[ispec] = rv_out
                # rv_errs[ispec] = (par_uerrs[0] + par_lerrs[0]) / 2

                #### extra diagnostics ####

                if plot_line_fits:
                    line_fluxes = generate_multilines(wvs_masked, np.array(line_wvs), heights_out, rv_out, fwhm_out)
                    conti_flux = generate_conti(wvs_masked, starts, ends, offsets_out, slopes_out)
                    total_flux = line_fluxes + conti_flux

                    for ax in axes:

                        ax.plot(wvs, flux - ispec, color='k', lw=1)

                        for start, end in zip(starts, ends):
                            block_mask = (wvs_masked > start) & (wvs_masked < end)
                            ax.plot(wvs_masked[block_mask], total_flux[block_mask] - ispec, color='r', lw=1)

                        ax.vlines([line_wv * (1 + (rv_out / c_)) for line_wv in line_wvs], -ispec - 0.2, -ispec, color='r', lw=1)

                        ax.vlines(line_wvs, -ispec - 0.4, -ispec - 0.2, color='b', lw=1)
                        ax.vlines([line_wv * (1 + (rv0 / c_)) for line_wv in line_wvs], -ispec - 0.4, -ispec - 0.2, color='g', lw=1)

                if print_info:
                    info = f"init rv: {rv0:.0f} km/s  "
                    info += f"rv: {rv_out:.0f}+{par_uerrs[0]:.0f}-{par_lerrs[0]:.0f} km/s"
                    info += f"fwhm: {fwhm_out:.0f}+{par_uerrs[1]:.0f}-{par_lerrs[1]:.0f} km/s"
                    info += '  heights: ' + ', '.join([f'{h:.2f}' for h in heights_out])
                    info += '  offsets: ' + ', '.join([f'{o:.2f}' for o in offsets_out])
                    info += '  slopes: ' + ', '.join([f'{s:.4f}' for s in slopes_out])

                    print(info)

    if plot_line_fits:
        edge_spacing = 0.8
        for ax, start, end in zip(axes, starts0, ends0):
            ax.set_xlim(start - edge_spacing * (end - start), end + edge_spacing * (end - start))

    if return_all:
        return output_array
    elif return_fwhm:
        return output_array[:, 0:4]
    else:
        return output_array[:, 0:2]
