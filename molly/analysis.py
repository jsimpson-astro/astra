from .utils import apply_mask, dummy_pbar
from .core_nj import f77_xcor_nj
from .core import f77_ew, f77_optsub, xcor_quad_max

import numpy as np
from tqdm.auto import tqdm

c = 299792.458
wv_tol = 1e-6

def do_xcor(obs_spectra: list, 
            templ_spectra: list, 
            mask: np.ndarray | list | None = None,
            shifts: tuple[int, int] = (-10, 10), 
            initial_shifts: list | None = None, 
            taper: float = 0.,
            maxima_func: callable = xcor_quad_max,
            progress: bool = True,
            ):

    """
    Perform cross-correlation using the same F77 method as molly.
    Cross-correlation is performed over a series of observed spectra,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs_spectra: list of path-like str or list of np.ndarray
        List of paths to observed spectra, or list of arrays of observed spectra. 
        All spectra must have identical wavelength scales. Each spectrum should have two, 
        or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templ_spectra: list of path-like str or list of np.ndarray
        List of paths to template spectra, or list of arrays of template spectra. 
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
        Initial pixel shifts for correlation. Must have the same length as obs_spectra.
    taper: float, default 0.
        From 0 to 1, fraction to taper from ends of the flux of both observed and template spectra.
    maxima_func: callable, default quad_xcor_max
        Function for calculating the best fit pixel of a cross-correlation output, with the form:
        
        func(xcor: np.ndarray[shape (shifts[1] - shifts[0] + 1,]) -> max_loc: float, error: float
        
        where 'xcor' is the individual cross-correlation result i.e. the value of the correlation at 
        each shift, and 'max_loc' and 'error' are the calculated location of the maximum and its error.
    progress: bool, default True
        Controls display of tqdm progress bar, optional. False to disable

    Returns:
    xcor_results: list of dict
        A list of dictionaries, one element per template spectrum, with the correlation result 
        over all observed spectra given.
    
    """
    # need first wvs to compare against template wvs, and check other inputs
    obs_wvs = obs_spectra[0][:, 0] if isinstance(obs_spectra[0], np.ndarray) else np.loadtxt(obs_spectra[0], usecols=0)
    
    n_obs = len(obs_spectra)
    n_templ = len(templ_spectra)
    n_shifts = shifts[1] - shifts[0] + 1
    
    # check shift tuple#
    if not (isinstance(shifts[0], (np.integer, int)) and isinstance(shifts[1], (np.integer, int))):
        raise TypeError("shifts must be given as ints")
    if not shifts[0] < shifts[1]:
        raise ValueError(f"shifts[0] ({shifts[0]}) must be less than shifts[1] ({shifts[1]})")

    # check initial shifts if provided, including bounds
    if initial_shifts is not None:
        if len(initial_shifts) != len(obs_spectra):
            raise IndexError("initial_shifts must have the same length as obs_spectra.")
        for initial_shift in initial_shifts:
            if not isinstance(initial_shift, (np.integer, int)):
                raise TypeError("initial_shifts must be given as ints")
            if any(abs(s + initial_shift) > obs_wvs.size // 2 for s in shifts):
                raise IndexError(f"shifts ({shifts}, {initial_shift}) out of bounds for wavelengths with size {obs_wvs.size}")

    # check taper
    if taper > 1:
        raise ValueError(f"taper fraction cannot exceed 1 (taper={taper})")
    if taper > 0.5:
        print(f"Warning: taper > 0.5 ({taper}), tapering will overlap in centre of spectra")

    # check func
    if not callable(maxima_func):
        raise TypeError(f"maxima_func {maxima_func} is not callable.")
    # test scaled parabola at x = 0 to 1
    k = 10
    xs =  np.linspace(0, 1, n_shifts)
    test_xcor = (k - k*xs) * xs
    s, s_err = maxima_func(test_xcor)
    if not (isinstance(s, float) and isinstance(s_err, float)):
        raise TypeError('Incorrect return types from maxima func - should return two floats.')    
    
    #### template checks ####
    
    # test opening all templates - templates first as they can be cleared to save memory
    if not isinstance(templ_spectra[0], np.ndarray):
        templ_data = [np.loadtxt(f) for f in templ_spectra]
        templ_are_arrays = False
    else:
        templ_data = templ_spectra
        templ_are_arrays = True

    # check shapes
    templ_shapes = np.array([a.shape for a in templ_data])
    wvs_unique, cols_unique = np.unique(templ_shapes[:, 0]), np.unique(templ_shapes[:, 1])
    if wvs_unique.size != 1:
        raise IndexError("templ_spectra do not all have the same number of points.")
    if cols_unique.min() < 2:
        raise IndexError("templ_spectra do not all have at least two columns.")

    # check wavelength scales
    wv_dev = np.abs(np.array([a[:, 0] for a in templ_data]) - obs_wvs).max()
    if wv_dev > wv_tol:
        raise ValueError(f"templ_spectra and obs_spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")
    
    del templ_data

    #### obs spectra checks ####
    
    # check data, open if not already arrays
    if not isinstance(obs_spectra[0], np.ndarray):
        obs_data = [np.loadtxt(f) for f in obs_spectra]
        obs_are_arrays = False
    else:
        obs_data = obs_spectra
        obs_are_arrays = True

    # check shapes are all identical, flag errors if all have at least 3 columns
    obs_shapes = np.array([a.shape for a in obs_data])
    wvs_unique, cols_unique = np.unique(obs_shapes[:, 0]), np.unique(obs_shapes[:, 1])
    if wvs_unique.size != 1:
        raise IndexError("obs_spectra do not all have the same number of points")
    if cols_unique.min() < 2:
        raise IndexError("obs_spectra do not all have at least two columns")
    obs_have_errors = True if cols_unique.min() > 2 else False

    # check wavelength scales
    wv_dev = np.abs(np.array([a[:, 0] for a in obs_data]) - obs_wvs).max()
    if wv_dev > wv_tol:
        raise ValueError(f"obs_spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")

    #### masking checks #####
    
    if isinstance(mask, np.ndarray):
        if mask.size != obs_wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(obs_wvs, mask)
    else:
        mask_ = np.ones(obs_wvs.size, dtype=bool)
    
    ####
    
    initial_shifts = [0] * n_obs if initial_shifts is None else initial_shifts
    
    # calculate average velocity using first spectrum
    v_avg = c * (np.exp(np.log(obs_wvs.max()/obs_wvs.min()) / (obs_wvs.size-1) ) - 1)

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar

    xcor_results = []
    
    with pbar_manager(desc='xcor: ', total=n_templ*n_obs) as pbar:
        
        for i_templ, template in enumerate(templ_spectra):
            
            # if it is a path, load it
            t_spec = template if templ_are_arrays else np.loadtxt(template)
            t_wvs, t_flux = t_spec[:, 0], t_spec[:, 1]
            
            # mask out any nans
            t_mask_ = (mask_ & ~np.isnan(t_flux))

            xcor_result = {'template': i_templ,
                           'mask': True if mask is not None else False,
                           'errors': True if obs_have_errors else False,
                           'shifts': shifts,
                           'initial_shifts': np.array(initial_shifts),
                           'v_average': v_avg,
                           }
            
            xcor_array = np.zeros((n_obs, shifts[1] - shifts[0] + 1))
            pixel_shift_array, pixel_shift_err_array = np.zeros(n_obs), np.zeros(n_obs)
            vshift_array, vshift_err_array = np.zeros(n_obs), np.zeros(n_obs)
            
            for i_obs, (obs, initial_shift) in enumerate(zip(obs_data, initial_shifts)):

                obs_wvs, obs_flux = obs[:, 0], obs[:, 1]
                obs_flux_err = obs[:, 2] if obs_have_errors else None

                # mask out any nans
                obs_mask_ = (t_mask_ & ~np.isnan(obs_flux))
                obs_mask_ = (obs_mask_ & (obs_flux_err >= 0)) if obs_have_errors else obs_mask_

                # # shift mask?
                # if initial_shift > 0:
                #     obs_mask_s = np.zeros(obs_wvs.shape, dtype=bool)
                #     #obs_mask_s[initial_shift:] = obs_mask_[:-initial_shift]
                #     obs_mask_s[:-initial_shift] = obs_mask_[initial_shift:]
                #     obs_mask_ = obs_mask_s
                # elif initial_shift < 0:
                #     obs_mask_s = np.zeros(obs_wvs.shape, dtype=bool)
                #     #obs_mask_s[:initial_shift] = obs_mask_[-initial_shift:]
                #     obs_mask_s[-initial_shift:] = obs_mask_[:initial_shift]
                #     obs_mask_ = obs_mask_s
    
                # perform xcor
                xcor, xcor_error = f77_xcor_nj(obs_wvs, obs_flux, t_flux, obs_flux_err, obs_mask_, 
                                               shifts, initial_shift, taper)
                
                # find maximum, apply shift
                max_loc, max_loc_error = maxima_func(xcor)
                max_loc = max_loc + shifts[0] + initial_shift
                vshift, vshift_error = v_avg * max_loc, v_avg * max_loc_error

                # save results to arrays
                xcor_array[i_obs] = xcor
                pixel_shift_array[i_obs], pixel_shift_err_array[i_obs] = max_loc, max_loc_error
                vshift_array[i_obs], vshift_err_array[i_obs] = vshift, vshift_error
    
                pbar.update(1)

            # add result data
            xcor_result['xcor'] = xcor_array
            xcor_result['pixel_shift'] = pixel_shift_array
            xcor_result['pixel_shift_error'] = pixel_shift_err_array
            xcor_result['v_shift'] = vshift_array
            xcor_result['v_shift_error'] = vshift_err_array 
            
            xcor_results.append(xcor_result)
                
    return xcor_results

def do_ews(spectra: list,
           mask: np.ndarray | list | None = None,
           initial_shifts: list | None = None
           ):
    """
    Measure equivalent widths using the same F77 method as molly.
    Equivalent widths are measured across a series of spectra.
    Spectra must be normalised to a continuum of 0.

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    spectra: list of path-like str or list of np.ndarray
        List of paths to spectra, or list of arrays of spectra. 
        Each spectrum should have two, or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array the same size as the 
        wavelengths of the spectra, or a list of 2-tuples defining upper and lower bounds to exclude.
    initial_shifts: list of ints or None, default None
        Initial pixel shifts to apply. Must have the same length as spectra.

    Returns:
    ews: np.ndarray
        Measured equivalent widths, one element per spectrum.
    ew_errors: np.ndarray
        Errors on equivalent widths, one element per spectrum.
        Set to np.nan if no errors present in a given spectrum.
    
    """
    # need first wvs to compare against template wvs, and check other inputs
    wvs = spectra[0][:, 0] if isinstance(spectra[0], np.ndarray) else np.loadtxt(spectra[0], usecols=0)
    
    n_spec = len(spectra)
    
    # check initial shifts if provided, including bounds
    if initial_shifts is not None:
        if len(initial_shifts) != len(spectra):
            raise IndexError("initial_shifts must have the same length as obs_spectra.")
        for initial_shift in initial_shifts:
            if not isinstance(initial_shift, (np.integer, int)):
                raise TypeError("initial_shifts must be given as ints")
            if abs(initial_shift) > wvs.size // 2:
                raise IndexError(f"shift ({initial_shift}) out of bounds for wavelengths with size {wvs.size}") 
                
    #### spectra checks ####
    # check data, open if not already arrays
    if not isinstance(spectra[0], np.ndarray):
        spec_data = [np.loadtxt(f) for f in spectra]
    else:
        spec_data = spectra

    # check shapes are all identical, flag errors if all have at least 3 columns
    shapes = np.array([a.shape for a in spec_data])
    cols_unique = np.unique(shapes[:, 1])
    if cols_unique.min() < 2:
        raise IndexError("spectra do not all have at least two columns")
    spec_have_errors = True if cols_unique.min() > 2 else False

    #### masking checks #####
    if isinstance(mask, np.ndarray):
        # check shapes are all identical, flag errors if all have at least 3 columns
        wvs_unique = np.unique(shapes[:, 0])
        if wvs_unique.size != 1:
            raise IndexError("mask is an array, and spectra do not all have the same number of points")
    
        # check wavelength scales
        wv_dev = np.abs(np.array([a[:, 0] for a in spec_data]) - wvs).max()
        if wv_dev > wv_tol:
            raise ValueError(f"mask is an array, and spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")
        
        if mask.size != wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    else:
        mask_ = None
    
    ####
    
    initial_shifts = [0] * n_spec if initial_shifts is None else initial_shifts

    ews, ew_errors = np.zeros(n_spec), np.zeros(n_spec)
        
    for i_spec, (spec, initial_shift) in enumerate(zip(spec_data, initial_shifts)):

        wvs, flux = spec[:, 0], spec[:, 1]
        flux_err = spec[:, 2] if spec_have_errors else None

        # apply shift
        # effectively, shift the mask to where the shift specifies
        # functionally, it is better to shift the flux in the opposite direction,
        # and then pad with nans which will get masked anyways
        if initial_shift > 0:
            flux = np.r_[flux[initial_shift:], np.zeros(initial_shift) + np.nan]
            if flux_err is not None:
                flux_err = np.r_[flux_err[initial_shift:], np.zeros(initial_shift) + np.nan]
        elif initial_shift < 0:
            flux = np.r_[np.zeros(-initial_shift) + np.nan, flux[:initial_shift]]
            if flux_err is not None:
                flux_err = np.r_[np.zeros(-initial_shift) + np.nan, flux_err[:initial_shift]]
        
        # apply mask if needed
        if mask_ is None:
            if isinstance(mask, list):
                spec_mask_ = apply_mask(wvs, mask)
            else:
                spec_mask_ = np.ones(wvs.size, dtype=bool)
        else:
            spec_mask_ = mask_.copy()

        # mask out any nans
        spec_mask_ = (spec_mask_ & ~np.isnan(flux))
        spec_mask_ = (spec_mask_ & (flux_err >= 0)) if spec_have_errors else spec_mask_
        
        # measure ews
        ew, ew_error = f77_ew(wvs, flux, flux_err, spec_mask_)

        # save to arrays
        ews[i_spec], ew_errors[i_spec] = ew, ew_error
                
    return ews, ew_errors
    

def do_optsub(obs_spectra: list, 
              templ_spectra: list, 
              mask: np.ndarray | list | None = None, 
              progress: bool = True
              ):

    """
    Perform optimal subtraction using the same F77 method as molly.
    Optimal subtraction is performed for each observed spectra,
    iterating over all given template spectra.
    All spectra should be in the same wavelength scale, binned into
    uniform velocity bins - otherwise results may be meaningless.
    Any systemic radial velocities should be removed also (see f77_sincshift).

    Masking is performed according to the provided mask (optional), which
    may be a list of bounds to exclude, or a boolean array.
    Note that masking is also applied where fluxes == np.nan and where errors < 0.

    Parameters:
    obs_spectra: list of path-like str or list of np.ndarray
        List of paths to observed spectra, or list of arrays of observed spectra. 
        All spectra must have identical wavelength scales. Each spectrum should have two, 
        or optionally three columns: wavelength, flux, and flux error.
        Additional columns will be ignored.
    templ_spectra: list of path-like str or list of np.ndarray
        List of paths to template spectra, or list of arrays of template spectra. 
        All spectra must have identical wavelength scales, which match the observed spectra. 
        Each spectrum should have two columns: wavelength and flux.
        Additional columns will be ignored. It is assumed noise is dominated by observed spectra.
    mask: numpy.ndarray, list of tuples of floats, or None, default None
        Wavelength mask, optional. If provided, can either be a boolean array the same size as 
        the wavelengths of the observed and template spectra, or a list of 2-tuples defining 
        upper and lower bounds to exclude.
    progress: bool, default True
        Controls display of tqdm progress bar, optional. False to disable

    Returns:
    optsub_results: list of dict
        A list of dictionaries, one element per observed spectrum, with the optimal subtraction 
        result over all template spectra given.
    
    """
    
    n_obs = len(obs_spectra)
    n_templ = len(templ_spectra)

    #### obs spectra checks ####
    
    # check data, open if not already arrays
    if not isinstance(obs_spectra[0], np.ndarray):
        obs_data = [np.loadtxt(f) for f in obs_spectra]
        obs_are_arrays = False
    else:
        obs_data = obs_spectra
        obs_are_arrays = True

    # use first wvs to compare against other wvs
    obs_wvs = obs_data[0][:, 0]

    # check shapes are all identical, flag errors if all have at least 3 columns
    obs_shapes = np.array([a.shape for a in obs_data])
    wvs_unique, cols_unique = np.unique(obs_shapes[:, 0]), np.unique(obs_shapes[:, 1])
    if wvs_unique.size != 1:
        raise IndexError("obs_spectra do not all have the same number of points")
    if cols_unique.min() < 2:
        raise IndexError("obs_spectra do not all have at least two columns")
    obs_have_errors = True if cols_unique.min() > 2 else False

    # check wavelength scales
    wv_dev = np.abs(np.array([a[:, 0] for a in obs_data]) - obs_wvs).max()
    if wv_dev > wv_tol:
        raise ValueError(f"obs_spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")

    del obs_data
    
    #### template checks ####
    
    # test opening all templates - templates first as they can be cleared to save memory
    if not isinstance(templ_spectra[0], np.ndarray):
        templ_data = [np.loadtxt(f) for f in templ_spectra]
        templ_are_arrays = False
    else:
        templ_data = templ_spectra
        templ_are_arrays = True

    # check shapes
    templ_shapes = np.array([a.shape for a in templ_data])
    wvs_unique, cols_unique = np.unique(templ_shapes[:, 0]), np.unique(templ_shapes[:, 1])
    if wvs_unique.size != 1:
        raise IndexError("templ_spectra do not all have the same number of points.")
    if cols_unique.min() < 2:
        raise IndexError("templ_spectra do not all have at least two columns.")

    # check wavelength scales
    wv_dev = np.abs(np.array([a[:, 0] for a in templ_data]) - obs_wvs).max()
    if wv_dev > wv_tol:
        raise ValueError(f"templ_spectra and obs_spectra wavelength scales deviate above tolerance ({wv_dev}>{wv_tol}).")

    #### masking checks #####
    
    if isinstance(mask, np.ndarray):
        if mask.size != obs_wvs.size:
            raise IndexError("If mask is an array, it must match the number of points of the spectra.")
        mask_ = mask
    elif isinstance(mask, list):
        mask_ = apply_mask(obs_wvs, mask)
    else:
        mask_ = np.ones(obs_wvs.size, dtype=bool)
    
    ####

    # disable progress bar if requested by replacing with dummy class that does nothing
    pbar_manager = tqdm if progress else dummy_pbar

    optsub_results = []
    
    with pbar_manager(desc='optsub: ', total=n_obs*n_templ) as pbar:
        
        for i_obs, obs in enumerate(obs_spectra):

            # if it is a path, load it
            obs_spec = obs if obs_are_arrays else np.loadtxt(obs)
            obs_wvs, obs_flux = obs_spec[:, 0], obs_spec[:, 1]
            obs_flux_err = obs_spec[:, 2] if obs_have_errors else None

            # mask out any nans
            obs_mask_ = (mask_ & ~np.isnan(obs_flux))
            obs_mask_ = (obs_mask_ & (obs_flux_err >= 0)) if obs_have_errors else obs_mask_

            optsub_result = {'obs': i_obs,
                             'mask': True if mask is not None else False,
                             'errors': True if obs_have_errors else False
                             }
            
            factor_array, factor_err_array = np.zeros(n_templ), np.zeros(n_templ)
            chisq_array, dof_array = np.zeros(n_templ), np.zeros(n_templ)
            
            for i_templ, template in enumerate(templ_data):
            
                t_wvs, t_flux = template[:, 0], template[:, 1]
                
                # mask out any nans
                t_mask_ = (obs_mask_ & ~np.isnan(t_flux))
    
                # perform optsub
                factor, factor_err, chisq, dof = f77_optsub(obs_wvs, obs_flux, t_flux, obs_flux_err, t_mask_)

                # save results to arrays
                factor_array[i_templ], factor_err_array[i_templ] = factor, factor_err
                chisq_array[i_templ], dof_array[i_templ] = chisq, dof
    
                pbar.update(1)

            # add result data
            optsub_result['chisq'] = chisq_array
            optsub_result['factor'] = factor_array
            optsub_result['factor_errors'] = factor_err_array
            optsub_result['dof'] = dof_array
            
            optsub_results.append(optsub_result)
                
    return optsub_results