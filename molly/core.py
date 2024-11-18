import numpy as np
from scipy.ndimage import gaussian_filter1d

c = 299792.458

def f77_xcor(wvs: np.ndarray,
             flux1: np.ndarray, 
             flux2: np.ndarray, 
             flux1_err: np.ndarray | None = None,
             mask: np.ndarray | None = None,
             shifts: tuple[int, int] = (-10, 10), 
             initial_shift: int = 0,
             taper: float = 0., 
             ):
    """
    Perform cross-correlation using the same F77 method as molly

    Parameters:
    wvs: numpy.ndarray
        Array of wavelengths, 1D
    flux1: numpy.ndarray
        Fluxes of observational spectrum, same size as wvs
    flux2: numpy.ndarray
        Fluxes of template spectrum, same size as wvs
    flux1_err: numpy.ndarray or None, default None
        Errors of observational fluxes, optional, same size as wvs
    mask: numpy.ndarray or None, default None
        Boolean array to mask wavelengths, optional, same size as wvs
    shifts: tuple of ints, length 2
        Tuple of pixel shifts, (negative, positive), shifts[1] > shifts[0]
    initial_shift: int, default = 0
        Initial pixel shift for correlation
    taper: float
        From 0 to 1, fraction to taper from ends of x1 and x2

    Returns:
    xcor: numpy.ndarray
        Array of correlation results at every shift
        Size of (shifts[1] - shifts[0] + 1) 
    xcor_error: numpy.ndarray
        Error on correlation results, same size as xcor
    
    """
    # input checks - negligible impact on runtime
    if not (wvs.size == flux1.size == flux2.size):
        raise IndexError(f"wvs, flux1, and flux2 must all be the same size: ({wvs.size}, {flux1.size}, {flux2.size})")
    if mask is not None and mask.size != wvs.size:
        raise IndexError(f"mask and wvs do not have the same size: ({mask.size}, {wvs.size})")
    if flux1_err is not None and flux1_err.size != flux1.size:
        raise IndexError(f"flux1_err and flux1 do not have the same size: ({flux1_err.size}, {flux1.size})")
    if not all(isinstance(i, (np.integer, int)) for i in [*shifts, initial_shift]):
        raise TypeError(f"shifts ({shifts}) must be given as ints")
    if not shifts[0] < shifts[1]:
        raise ValueError(f"shifts[0] ({shifts[0]}) must be less than shifts[1] ({shifts[1]})")
    if any(abs(s + initial_shift) > wvs.size // 2 for s in shifts):
        raise IndexError(f"shifts ({shifts}, {initial_shift}) out of bounds for wvs with size {wvs.size}")
    if taper > 1:
        raise ValueError(f"taper fraction cannot exceed 1 (taper={taper})")
    if taper > 0.5:
        print(f"Warning: taper > 0.5 ({taper}), tapering will overlap in centre of spectra")
    
    mask = np.ones_like(flux1, dtype=bool) if mask is None else mask
    
    # first and last valid pixels
    idx0, idx1 = mask.argmax(), len(mask) - mask[::-1].argmax() - 1
    total_unmasked = np.count_nonzero(mask) # number of unmasked pixels

    # taper is applied to ends using a cos function
    # based on the distance from the ends
    if taper > 0:
        # make copies
        x1, x2, x1_err = flux1.copy(), flux2.copy(), flux1_err
        # identify ends to slice
        slice0 = slice(idx0, idx0 + int(taper*x1.size) + 1)
        slice1 = slice(idx1 - int(taper*x1.size), idx1 + 1)

        # compute factors to multiply by based on (index - end pixel)
        idxs = np.arange(x1.size)
        factor0 = ((1 - np.cos(np.pi * (idxs[slice0] - idx0) / x1.size / taper)) / 2)
        factor1 = ((1 - np.cos(np.pi * (idxs[slice1] - idx1) / x1.size / taper)) / 2)

        # apply taper
        x1[slice0], x2[slice0] = factor0 * x1[slice0], factor0 * x2[slice0]
        x1[slice1], x2[slice1] = factor1 * x1[slice1], factor1 * x2[slice1]
    else:
        x1, x2, x1_err = flux1, flux2, flux1_err

    x4, x5 = np.zeros((shifts[1] - shifts[0] + 1)), np.zeros((shifts[1] - shifts[0] + 1))
    # x6 = np.zeros((shifts[1] - shifts[0] + 1)) unused
    

    for out_idx, i in enumerate(range(shifts[0] + initial_shift, shifts[1] + initial_shift + 1)):
        
        # slice according to shifts
        xcor_slice1 = slice(max(idx0, idx0+i), min(idx1, idx1+i) + 1)
        xcor_slice2 = slice(max(idx0, idx0+i) - i, min(idx1, idx1+i) + 1 - i)

        # slice mask to match
        # new mask shifting code - should add a flag for optional?
        # if initial_shift > 0:
        #     mask_s = np.zeros(flux1.shape, dtype=bool)
        #     mask_s[initial_shift:] = mask[:-initial_shift]
        #     xcor_mask1 = mask_s[xcor_slice1]
        #     xcor_mask2 = mask_s[xcor_slice1]
        # elif initial_shift < 0:
        #     mask_s = np.zeros(flux1.shape, dtype=bool)
        #     mask_s[:initial_shift] = mask[-initial_shift:]
        #     xcor_mask1 = mask_s[xcor_slice1]
        #     xcor_mask2 = mask_s[xcor_slice1]
        # else:
        #     xcor_mask1 = mask[xcor_slice1]
        #     xcor_mask2 = mask[xcor_slice1]
        # new code ends  
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
            x4[out_idx] = fac * sum2 #/ weights.sum()**0.5 # normalisation, not applied in molly
            x5[out_idx] = fac * np.sqrt(sum1) #/ weights.sum()**0.5 # normalisation, not applied in molly
            # x6[out_idx] = fac * sum1
        else:
            x4[out_idx] = 0
            x5[out_idx] = -1
            # x6[out_idx] = 0
    
    return x4, x5

def xcor_quad_max(xcor):
    """
    Find the maximum of a cross-correlation output with quadratic 
    fitting of the maximum and two nearest points.

    Parameters:
    xcor: np.ndarray
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
        
    xcor_max_a, xcor_max_b = xcor[max_idx-1], xcor[max_idx+1]
    z1, z2 = xcor_max_b - xcor_max_a, xcor_max_a + xcor_max_b - 2 * xcor_max
    max_loc = max_idx  - 0.5 * (z1 / z2)
    error = 1 / (-0.5 * z2)**0.5
    
    return max_loc, error

def f77_ew(wvs: np.ndarray, 
           flux: np.ndarray, 
           flux_err: np.ndarray | None = None, 
           mask: np.ndarray | None = None
           ):
    """
    Calculates the equivalent width over a spectrum using the same method
    as the F77 program molly. Spectrum must be normalised to 0.

    Parameters:
    wvs: numpy.ndarray
        Array of wavelengths, 1D
    flux: numpy.ndarray
        Fluxes of spectrum, same size as wvs
    flux_err: numpy.ndarray or None, default None
        Errors of fluxes, optional, same size as wvs
    mask: numpy.ndarray or None, default None
        Boolean array to mask wavelengths, optional, same size as wvs

    Returns:
    ew: float
        Measured equivalent width over the unmasked range
    ew_error: float or None
        Error on the measured equivalent width
        If flux_err not provided, ew_error is np.nan.

    """
    # PMAP = phase fold data
    # NB mask moved to heliocentric frame
    # choices determined by YAX
    # FLUX: 1, EW: 2, MJY: 3, COUNTS: 4
    
    # input checks
    if not (wvs.size == flux.size):
        raise IndexError("wvs and flux must have the same size")
    if mask is not None and mask.size != wvs.size:
        raise IndexError("mask and wvs must be the same size")
    if flux_err is not None and flux_err.size != flux.size:
        raise IndexError("flux_err and flux must be the same size")

    # nb change to bool_ and mask_ for njit
    mask = np.ones(flux.shape, dtype=bool) if mask is None else mask
    
    dws = np.zeros_like(wvs)
    dws[:-1] = wvs[1:] - wvs[:-1]
    # extrapolate for final value
    dws[-1] = dws[-1] + (dws[-1] - dws[-2])

    # subtracting continuum would go here
    ew = (dws[mask] * (flux[mask])).sum()
    ew_err = ((dws[mask] * flux_err[mask])**2).sum()**0.5 if flux_err is not None else np.nan
    
    return ew, ew_err

def f77_optsub(wvs: np.ndarray,
               flux1: np.ndarray, 
               flux2: np.ndarray, 
               flux1_err: np.ndarray | None = None, 
               mask: np.ndarray | None = None, 
               fwhm: float = 0.):
    """
    Calculate the optimal factor for subtraction between two spectra using 
    the same method as the F77 program molly.
    Optimal subtraction factor minimises the residuals in the equation:
    
    flux1 - factor * flux2 = residuals

    Parameters:
    wvs: numpy.ndarray
        Array of wavelengths, 1D
    flux1: numpy.ndarray
        Fluxes of observational spectrum, same size as wvs
    flux2: numpy.ndarray
        Fluxes of template spectrum, same size as wvs
    flux1_err: numpy.ndarray or None, default None
        Errors of observational fluxes, optional, same size as wvs
    mask: numpy.ndarray or None, default None
        Boolean array to mask wavelengths, optional, same size as wvs

    Returns:
    factor: float
        Optimal subtraction factor
    factor_err: float
        Error on the optimal subtraction factor
    chisq: float
        Chi-squared value of the residuals
    dof: int
        Degrees of freedom (number of unmasked pixels) in the subtraction
    
    """
    # input checks - negligible impact on runtime
    if not (wvs.size == flux1.size == flux2.size):
        raise IndexError("wvs, flux1, and flux2 must all be the same size")
    if mask is not None and mask.size != wvs.size:
        raise IndexError("mask and wvs must be the same size")
    if flux1_err is not None and flux1_err.size != flux1.size:
        raise IndexError("flux1_err and flux1 must be the same size")
    if 0 < fwhm < 1:
        print("Warning: non-zero fwhm < 1 ignored")
    
    mask = np.ones(flux.shape, dtype=bool) if mask is None else mask
    
    if fwhm > 1:
        # molly fwhm approx. translates to this 
        fwhm_ = fwhm / (2 * (2 * np.log(2))**0.5)
        # replace masked regions with linear interpolation
        x1, x2 = f77_replace(flux1, mask), f77_replace(flux2, mask)

        # smooth
        x1_smoothed = gaussian_filter1d(x1, fwhm_, mode='nearest', radius=50)
        x2_smoothed = gaussian_filter1d(x2, fwhm_, mode='nearest', radius=50)
    else:
        x1, x2 = flux1, flux2
    x1_err = flux1_err 

    dof = np.count_nonzero(mask)
    weights = 1 / x1_err[mask]**2 if x1_err is not None else np.ones(dof)
    
    if fwhm > 1:
        d = x1[mask] - x1_smoothed[mask]
        t = x2[mask] - x2_smoothed[mask]
    else:
        d = x1[mask]
        t = x2[mask]
        
    sum1 = (weights * d * t).sum()
    sum2 = (weights * t * t).sum()
    sum3 = (weights * d * d).sum()
    
    factor = sum1 / sum2
    chisq = sum3 - sum1**2 / sum2

    factor_err = 1 / sum2**0.5

    return factor, factor_err, chisq, dof

def f77_replace(flux, mask):
    """
    Linearly interpolate flux from edges of given mask.
    Masked edges are set to the nearest unmasked value.

    Parameters:
    flux: np.ndarray
        Fluxes of spectrum used for interpolation
    mask: np.ndarray of bool
        Boolean array to interpolate over where False, same size as flux

    Returns:
    flux_i: np.ndarray
        Fluxes of spectrum with interpolation over masked regions, same size as flux
    
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
            slices = [slice(i, f+1) for i, f in zip(mask_is[:-1], mask_fs)]
        else:
            # left edge is masked, go :f[0], i[0]:f[1] etc.
            slices = [slice(None, mask_fs[0]+1)]
            slices += [slice(i, f+1) for i, f in zip(mask_is[:-1], mask_fs[1:])]
        
        if mask_is[-1] < mask_fs[-1]:
            # right edge is not masked, i[-1]:f[-1] should be fine
            slices += [slice(mask_is[-1], mask_fs[-1]+1)]
        else:
            # right edge is masked, will have to use i[-1]:
            slices += [slice(mask_is[-1], None)]
    elif mask_is.size != mask_fs.size:
        slices = [slice(mask_is[0], None) if mask_fs.size == 0 else slice(None, mask_fs[0]+1)]
    else:
        slices = []
    
    for sl in slices:
        if sl.start and sl.stop:
            # linear interpolation 
            flux_i[sl] = np.linspace(flux[sl.start-1], flux[sl.stop], sl.stop - sl.start + 2)[1:-1]
        else:
            flux_i[sl] = flux[sl.start-1] if sl.start else flux[sl.stop]

    return flux_i

def f77_sincshift(wvs: np.ndarray, 
                  flux: np.ndarray, 
                  flux_err: np.ndarray | None = None, 
                  vshift: float = 0.,
                  pad: float = np.nan
                  ):
    """
    Shift the flux (and error) of a spectrum by a given velocity, in km/s.
    Rebinning is done with the sinc rebinning method from the F77 program molly.
    This convolves a windowed sinc function with the spectrum. 
    The window is defined as:

    f(x) = 4/3 - 8 (x-1) x^2    0.0 < x <= 0.5
         = 8/3 (1-x)^3          0.5 < x < 1.0

    Parameters:
    wvs: numpy.ndarray
        Array of wavelengths, 1D
    flux: numpy.ndarray
        Fluxes of spectrum, same size as wvs
    flux_err: numpy.ndarray or None, default None
        Errors of fluxes, optional, same size as wvs
    vshift: float, default 0.
        Velocity shift to apply to spectrum, in km/s
    pad: float, default np.nan.
        Value to fill padded fluxes with, optional
    
    Returns:
    flux_shifted:
        Fluxes of spectrum, rebinned with velocity shift applied
    flux_err_shifted:
        Errors of fluxes, rebinned with velocity shift applied
        If flux_err is None, errors are set to 'pad'.
    
    """
    maxsinc = 15
    v_avg = c * (np.exp(np.log(wvs.max()/wvs.min()) / (wvs.size-1) ) - 1)

    # compute pixel shift, separate into integer (rounded) and decimal parts
    pxshift = -np.log(1 + vshift/c)/np.log(1 + v_avg/c) # px shift
    nshift, subpxshift = int(pxshift), pxshift - int(pxshift)
    
    # compute sinc function, shifted by the decimal part of the shift (xshift)~
    x1 = np.pi * (subpxshift - np.arange(-maxsinc, maxsinc+1))
    x2 = np.abs((subpxshift - np.arange(-maxsinc, maxsinc+1)) / (maxsinc+0.5))
    sinc = np.zeros(2 * maxsinc + 1)

    # use taylor series approx at small x1 (avoid div/0)
    x1_mask = (np.abs(x1) < 1e-4)
    sinc[x1_mask] = 1 - (x1[x1_mask]**2) / 6
    sinc[~x1_mask] = np.sin(x1[~x1_mask])/x1[~x1_mask]

    # apply window to sinc function
    x2_mask = (x2 <= 0.5)
    sinc[x2_mask] = sinc[x2_mask] * (4/3 + 8*(x2[x2_mask]-1)*x2[x2_mask]**2)
    sinc[~x2_mask] = sinc[~x2_mask] * 8/3 * (1-x2[~x2_mask])**3

    sinc = sinc / sinc.sum()

    # now do convolution, depending on nshift
    if nshift < 0:
        # pad fluxes with end values for convolution, which will be removed with mode='valid'
        flux_padded = np.r_[flux[0]*np.ones(maxsinc), 
                            flux[:nshift+maxsinc if nshift+maxsinc < 0 else None], 
                            flux[-1]*np.ones(max(nshift+maxsinc, 0))]

        # convolve, reversing sinc function as numpy reverses the shorter array
        flux_shifted = np.convolve(flux_padded, sinc[::-1], mode='valid')
        # pad zeros where fluxes have been shifted away
        flux_shifted = np.r_[np.zeros(-nshift) + pad, flux_shifted]

        # repeat for errors
        if flux_err is not None:
            flux_err_padded = np.r_[flux_err[0]*np.ones(maxsinc), 
                                    flux_err[:nshift+maxsinc if nshift+maxsinc < 0 else None], 
                                    flux_err[-1]*np.ones(max(nshift+maxsinc, 0))]
            
            flux_err_shifted = np.convolve(flux_err_padded, sinc[::-1], mode='valid')
            flux_err_shifted = np.r_[np.zeros(-nshift) + pad, flux_err_shifted]
        else:
            flux_err_shifted = np.zeros_like(flux_shifted) + pad
            
    elif nshift > 0:
        # pad fluxes with end values for convolution, which will be removed with mode='valid'
        flux_padded = np.r_[flux[0]*np.ones(max(maxsinc-nshift, 0)), 
                            flux[max(nshift-maxsinc, 0):], 
                            flux[-1]*np.ones(maxsinc)]

        # convolve, reversing sinc function as numpy reverses the shorter array
        flux_shifted = np.convolve(flux_padded, sinc[::-1], mode='valid')
        # pad zeros where fluxes have been shifted away
        flux_shifted = np.r_[flux_shifted, np.zeros(nshift) + pad]

        # repeat for errors
        if flux_err is not None:
            flux_err_padded = np.r_[flux_err[0]*np.ones(max(maxsinc-nshift, 0)), 
                                    flux_err[max(nshift-maxsinc, 0):], 
                                    flux_err[-1]*np.ones(maxsinc)]
            
            flux_err_shifted = np.convolve(flux_err_padded, sinc[::-1], mode='valid')
            flux_err_shifted = np.r_[flux_err_shifted, np.zeros(nshift) + pad]
        else:
            flux_err_shifted = np.zeros_like(flux_shifted) + pad
            
    else:
        # pad fluxes with end values for convolution, which will be removed with mode='valid'
        flux_padded = np.r_[flux[0]*np.ones(maxsinc), flux, flux[-1]*np.ones(maxsinc)]
        # convolve, reversing sinc function as numpy reverses the shorter array
        flux_shifted = np.convolve(flux_padded, sinc[::-1], mode='valid')

        # repeat for errors
        if flux_err is not None:
            flux_err_padded = np.r_[flux_err[0]*np.ones(maxsinc), flux_err, flux_err[-1]*np.ones(maxsinc)]
            flux_err_shifted = np.convolve(flux_err_padded, sinc[::-1], mode='valid')
        else:
            flux_err_shifted = np.zeros_like(flux_shifted) + pad

    return flux_shifted, flux_err_shifted
