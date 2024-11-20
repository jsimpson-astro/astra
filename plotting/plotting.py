from astra.fitting.rvcfitting import sine_fit

import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import corner


_corner_kws = dict(show_titles=True,
                   title_fmt='.1f',
                   plot_density=False,
                   plot_datapoints=False,
                   fill_contours=True,
                   )


def multicorner(
    chains,
    fig=None,
    ax=None,
    labels=None,
    axis_names=None,
    param_names=None,
    param_units=None,
    param_formats=None,
    colors=None,
    param_labels=None,
    corner_kws=None
) -> (plt.Figure, plt.Axes):
    """
    Plot multiple MCMC (flat) chains as stacked corner plots.
    Also serves as a convenience function for precise formatting of parameter labels.

    Parameters
    ----------
    chains: list of np.ndarray or np.ndarray
        List of flattened chains, with each chain being a np.ndarray of shape (n_samples, n_dim).
        Also supports plotting of a single chain if a single array is provided.

    fig: matplotlib Figure object, optional
        Existing figure object to plot to - if given, must also provide `ax`.
        Created if not provided.
    ax: matplotlib Axes object, optional
        Existing axes object to plot to - if given, must also provide `fig`.
        Created if not provided.

    labels: list of str, optional
        Labels of each corner plot, for figure legend.
        Must have length equal to number of chains.
    axis_names: list of str, optional
        Names of each parameter, for axis labels.
        Must have length equal to number of dimensions.
    param_names: list of str, optional
        Names of each parameter, added over histograms.
        Must have length equal to number of dimensions.
        Taken from `axis_names` if it is provided and `param_names` is not.
    param_units: list of str, optional
        Units for each parameter, added over histograms.
        Must have length equal to number of dimensions.
    param_formats: list of str, optional
        Format string for each parameter (value and errors), e.g. `.2f`.
        Must have length equal to number of dimensions.

    color: list pf matplotlib colors, optional
        Color of each corner plot.
        Must have length equal to number of chains.
    param_labels: list of str, optional
        Additional subscripts to add to each name in `param_names`.
        Must have length equal to number of *chains* (not dimensions).
        Note that these are already rendered in mathTeX, so `$` signs should not be used,
        and `\\mathrm{}` should be used to get Roman text.

    corner_kws: dict, optional
        Additional keyword arguments to pass to corner.corner

    Returns
    -------
    fig, ax


    """
    # if single chain given, convert to list
    if hasattr(chains, 'shape'):
        sample_list = [chains]
    else:
        sample_list = chains

    corner_kws = _corner_kws if corner_kws is None else _corner_kws | corner_kws

    # check dimensions match
    n_dims = set([s.shape[1] for s in sample_list])
    if len(n_dims) != 1:
        raise ValueError(f"chains do not have matching dimensions: ({n_dims})")
    n_dim = sample_list[0].shape[1]

    # create figure if needed
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=n_dim, ncols=n_dim)
    elif fig is not None and ax is not None:
        pass
    else:
        raise ValueError("Must define both fig and ax or neither.")

    if axis_names is not None and param_names is None:
        param_names = axis_names

    param_units = [None] * n_dim if param_units is None else param_units

    if param_labels is None:
        param_labels = [None] * len(sample_list)
    else:
        if param_names is None:
            raise ValueError("Must define `param_names` if `param_labels` is defined.")

    param_formats = ['.4g'] * n_dim if param_formats is None else param_formats
    param_formats = [param_formats] * n_dim if not isinstance(param_formats, list) else param_formats

    # create a color list of sufficient length if not provided
    if colors is None:
        colors = plt.color_sequences['tab10']
        colors = (len(sample_list) // len(colors) + 1) * colors
        colors = colors[:len(sample_list)]

    ####

    # check lengths of units, formats, names match number of dimensions from chains
    assert_len = [param_units, param_formats]
    assert_len += [axis_names] if axis_names is not None else []
    assert_len += [param_names] if param_names is not None else []
    assert_len = [len(i) == n_dim for i in assert_len]

    if not all(assert_len):
        assert_len_names = ['param_units', 'param_formats']
        assert_len_names += ['axis_names'] if axis_names is not None else []
        assert_len_names += ['param_names'] if param_names is not None else []
        assert_len_names = [n for i, n in enumerate(assert_len_names) if assert_len[i] is False]
        raise IndexError(f"{assert_len_names} do not match number of dimensions in chains ({n_dim}).")

    # check lengths of labels, colors, param_labels match number of chains provided
    assert_len = [param_labels, colors]
    assert_len += [labels] if labels is not None else []
    assert_len = [len(i) == len(sample_list) for i in assert_len]

    if not all(assert_len):
        assert_len_names = ['param_labels', 'colors']
        assert_len_names += ['labels'] if labels is not None else []
        assert_len_names = [n for i, n in enumerate(assert_len_names) if assert_len[i] is False]
        raise IndexError(f"{assert_len_names} do not match number of chains ({len(sample_list)}).")

    # plotting loop
    for i, samples in enumerate(sample_list):
        fig = corner.corner(samples, fig=fig, labels=axis_names, color=colors[i], **_corner_kws)

    # assemble titles (i.e. param_names + param_labels + values + param_units)
    for dim in range(n_dim):

        name = param_names[dim] if param_names is not None else None
        fmt = param_formats[dim]
        unit = param_units[dim]
        title_axes = [a for a in fig.axes if len(a.patches) != 0]
        titles = []

        for i, samples in enumerate(sample_list):

            chain = samples[:, dim]
            # get median value and errors
            val = np.quantile(chain, 0.5)
            lerr, uerr = val - np.quantile(chain, 0.16), np.quantile(chain, 0.84) - val

            title = ""
            lbl = param_labels[i]
            # add name with optional subscript
            title += name if name is not None else ''
            title += f'$_{{{lbl}}}$' if lbl is not None else ''
            # add value with errors
            title += f"$ = {val:{fmt}}_{{-{lerr:{fmt}}}}^{{+{uerr:{fmt}}}}$"
            # add unit if provided
            title += f"$\\,${unit}" if unit is not None else ''

            titles.append(title)

        title_axes[dim].set_title('\n'.join(titles))

    # add legend if more than one corner plotted
    if i > 0 and labels is not None:
        a = ax[0][0] if isinstance(ax[0], np.ndarray) else ax[0]
        handles = a.patches
        legend = fig.legend(handles=handles, labels=labels)

    return fig, ax


_rv_kws = dict(marker='.', capsize=3, ls='None', alpha=0.8)
_fit_kws = dict(ls='--', alpha=0.8)
_res_kws = dict(marker='o', alpha=0.8, s=12)
_fill_kws = dict(ls='None', alpha=0.2)


def rvplot(
    times,
    rvs,
    rv_errs=None,
    fig=None,
    ax=None,
    pars=None,
    par_errs=None,
    color=None,
    label=None,
    label_K2=True,
    label_chisq=True,
    n_free_pars=4,
    n_rvpoints=1000,
    n_rvsamples=100,
    zero=True,
    err_by_chisq=False,
    rv_kws=None,
    fit_kws=None,
    res_kws=None,
    fill_kws=None,
    fill_color=None,
) -> (plt.Figure, plt.Axes):
    """
    Parameter order: K2, gamma, T0, period,
    Errors must be provided to sample RV curves.

    Parameters
    ----------
    times: np.ndarray of floats
        (MJD) times of radial velocity measurements
    rvs: np.ndarray of floats
        Radial velocities to plot, units km/s. Same dimensions as `times`.
    rv_errs: np.ndarray of floats, optional
        Errors on radial velocities, units km/s. Same dimensions as `times` and `rvs`.

    fig: matplotlib Figure object, optional
        Existing figure object to plot to - if given, must also provide `ax`.
        Created if not provided.
    ax: matplotlib Axes object, optional
        Existing axes object to plot to - if given, must also provide `fig`.
        Created if not provided.

    pars: iterable, length 4, optional
        Paramters for RV fits, in order: K2, gamma, T0, period.
    par_errs: iterable, length 4, optional
        Parameter errors for RV sampling, same order as `par_errs`.

    color: matplotlib color, optional
        Color of artists
    label: str, optional
        Label of plot (for legend), applied to errorbar plot.
    label_K2: bool, default True
        Add K2 (+ error) to label.
    label_chisq: bool, default True
        Add reduced chi-squared to label

    n_free_pars: int, default 4
        Number of free parameters, for calculating reduced chi-squared.
    n_rvpoints: int, default 1000
        Number of points for RV curves, interpolated from phase 0 to 1.
    n_rvsamples: int, default 100
        Number of samples to pull from par_errs, to build error highlights.

    zero: bool, default True
        Remove gamma (if provided) from RVs.
    err_by_chisq: bool, default False
        Scale errors by reduced chi-squared .
        (i.e. multiply by the square root of the reduced chi-squared)

    rv_kws: dict, optional
        Axes.errorbar keywords for plotting RV measurements.
    fit_kws: dict, optional
        Axes.plot keywords for plotting RV fits.
    res_kws: dict, optional
        Axes.scatter keywords for plotting residuals.
    fill_kws: dict, optional
        Axes.fill_between keywords for plotting error highlights.

    fill_color: matplotlib color, optional
        Alternative color for fill_between error highlights

    Returns
    -------
    fig: matplotlib Figure object
    ax: matplotlib Axes object

    """

    # create figure if not given
    if fig is None and ax is None:
        # fig, ax = plt.subplots(figsize=(8, 6), nrows=2, sharex=True, gridspec_kw={'height_ratios': (4, 1)})
        fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': (4, 1)})
        fig.subplots_adjust(hspace=0)

        ax[0].set_ylabel('Radial velocity ($\\mathrm{km\\,s^{-1}}$)')
        ax[1].set_ylabel('Residuals ($\\mathrm{km\\,s^{-1}}$)')
        ax[1].hlines(0, 0, 1, color='k', ls='--', alpha=0.8)

        if pars is None:
            ax[1].set_xlabel('MJD (d)')
            ax[0].set_xlim(times.min(), times.max())
            ax[1].set_xlim(times.min(), times.max())
        else:
            ax[1].set_xlabel('Orbital phase')
            ax[0].set_xlim(0, 1)
            ax[1].set_xlim(0, 1)
    elif fig is not None and ax is not None:
        pass
    else:
        raise ValueError("Must define both fig and ax or neither.")

    # default plot kwargs
    rv_kws = _rv_kws if rv_kws is None else _rv_kws | rv_kws
    fit_kws = _fit_kws if fit_kws is None else _fit_kws | fit_kws
    res_kws = _res_kws if res_kws is None else _res_kws | res_kws
    fill_kws = _fill_kws if fill_kws is None else _fill_kws | fill_kws

    # simply plot times + rvs and exit
    if pars is None:
        ax[0].errorbar(times, rvs, rv_errs, color=color, label=label, **rv_kws)
        return fig, ax

    # proceeed with plotting a fit
    K2, gamma, t0, period = pars
    K2_err, gamma_err, t0_err, period_err = par_errs if par_errs is not None else ([None] * 4)
    par_errs_ = np.array([K2_err, gamma_err, t0_err, period_err])

    phases = ((times - t0) / period) % 1
    phases_interp = np.linspace(0, 1, 1000)

    rv_fit = sine_fit((phases * period) + t0, *pars) - (gamma if zero else 0)
    rv_fit_interp = sine_fit((phases_interp * period) + t0, *pars) - (gamma if zero else 0)

    residuals = rvs - rv_fit - (gamma if zero else 0)

    chisq = (residuals**2 / rv_errs**2).sum() if rv_errs is not None else (residuals**2).sum()
    dof = len(rvs) - n_free_pars

    if err_by_chisq and par_errs is not None:
        K2_err, gamma_err, t0_err, period_err = par_errs_ * (chisq / dof)**0.5
        par_errs_ = par_errs_ * (chisq / dof)**0.5

    label_ = '' if label is None else label

    if label_K2:
        if K2_err is not None:
            label_ += f"\n$K_2 = {K2:.1f} \\pm {K2_err:.1f} \\, \\mathrm{{km\\,s^{{-1}}}}$"
        else:
            label_ += f"\n$K_2 = {K2:.1f} \\, \\mathrm{{km\\,s^{{-1}}}}$"
    if label_chisq:
        label_ += f"\n$\\chi^2_\\nu = {chisq/dof:.2f}$"

    eb = ax[0].errorbar(phases, rvs - gamma if zero else rvs, rv_errs,
                        color=color, label=label_, **rv_kws)

    if color is None:
        color = eb[0].get_color()

    ax[0].plot(phases_interp, rv_fit_interp, color=color, **fit_kws)
    ax[1].scatter(phases, residuals, color=color, **res_kws)

    # stop here if no errors to scatter
    if par_errs is None:
        return fig, ax

    # scatter errors
    par_samples = pars + par_errs * np.random.normal(size=(n_rvsamples, 4))
    rv_fit_samples = (np.array([sine_fit((phases_interp * period) + t0, *p) for p in par_samples])
                      - (gamma if zero else 0))

    ax[0].fill_between(phases_interp,
                       y1=rv_fit_samples.min(axis=0),
                       y2=rv_fit_samples.max(axis=0),
                       color=fill_color if fill_color is not None else color, **fill_kws)

    return fig, ax
