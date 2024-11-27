import numpy as np
from numba import njit
from numba.types import bool_

c = 299792.458

@njit
def f77_xcor_nj(wvs: np.ndarray,
               flux1: np.ndarray, 
               flux2: np.ndarray, 
               flux1_err: np.ndarray | None = None, 
               mask: np.ndarray | None = None,
               shifts: tuple[int, int] = (-10, 10), 
               initial_shift: int = 0,
               taper: float = 0.
               ):
    """
    Perform cross-correlation using the same method as the F77 program molly.
    Spectra should be normalised to a continuum of 0 first.

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
        raise IndexError("wvs, flux1, and flux2 must all be the same size")
    if mask is not None and mask.size != wvs.size:
        raise IndexError("mask and wvs must be the same size")
    if flux1_err is not None and flux1_err.size != flux1.size:
        raise IndexError("flux1_err and flux1 must be the same size")
    if not (isinstance(shifts[0], (np.integer, int)) and isinstance(shifts[1], (np.integer, int))):
        raise TypeError("shifts must be given as ints")
    if not isinstance(initial_shift, (np.integer, int)):
        raise TypeError("initial_shift must be given as an int")
    if not shifts[0] < shifts[1]:
        raise ValueError("shifts[0] must be less than shifts[1]")
    if abs(shifts[0] + initial_shift) > wvs.size // 2 or abs(shifts[1] + initial_shift) > wvs.size // 2:
        raise IndexError("shifts + initial shift out of bounds of wvs")
    if taper > 1:
        raise ValueError("taper fraction cannot exceed 1")
    if taper > 0.5:
        print("Warning: taper > 0.5, tapering will overlap in centre of spectra")
    
    mask_ = np.ones(flux1.shape, dtype=bool_) if mask is None else mask
    
    # first and last valid pixels
    idx0, idx1 = mask_.argmax(), len(mask_) - mask_[::-1].argmax() - 1
    total_unmasked = np.count_nonzero(mask_) # number of unmasked pixels

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
    # x6 = np.zeros((shifts[1] - shifts[0] + 1))

    for out_idx, i in enumerate(range(shifts[0] + initial_shift, shifts[1] + initial_shift + 1)):
        
        # slice according to shifts
        xcor_slice1 = slice(max(idx0, idx0+i), min(idx1, idx1+i) + 1)
        xcor_slice2 = slice(max(idx0, idx0+i) - i, min(idx1, idx1+i) + 1 - i)

        # slice mask to match
        # new mask shifting code - should add a flag for optional?
        # if initial_shift > 0:
        #     mask_s = np.zeros(flux1.shape, dtype=bool_)
        #     mask_s[initial_shift:] = mask_[:-initial_shift]
        #     xcor_mask1 = mask_s[xcor_slice1]
        #     xcor_mask2 = mask_[xcor_slice1]
        # elif initial_shift < 0:
        #     mask_s = np.zeros(flux1.shape, dtype=bool_)
        #     mask_s[:initial_shift] = mask_[-initial_shift:]
        #     xcor_mask1 = mask_s[xcor_slice1]
        #     xcor_mask2 = mask_[xcor_slice1]
        # else:
        #     xcor_mask1 = mask_[xcor_slice1]
        #     xcor_mask2 = mask_[xcor_slice1]
        # new code ends  
        xcor_mask1 = mask_[xcor_slice1]
        xcor_mask2 = mask_[xcor_slice1]

        # slice data, applying slice first followed by slice mask
        xcor_x1 = x1[xcor_slice1][xcor_mask1]
        xcor_x2 = x2[xcor_slice2][xcor_mask2]

        if flux1_err is not None:
            xcor_x1_err = x1_err[xcor_slice1][xcor_mask1]
            weights = (1 / xcor_x1_err**2)
            sum1 = (weights * xcor_x2**2).sum() #/ weights.sum()**0.5 # normalisation, not applied in molly
            sum2 = (weights * xcor_x1 * xcor_x2).sum() #/ weights.sum()**0.5
        else:
            sum1 = (xcor_x2**2).sum()
            sum2 = (xcor_x1 * xcor_x2).sum()
        
        current_unmasked = np.count_nonzero(xcor_mask1)
        
        if sum1 > 0:
            fac = total_unmasked / current_unmasked
            x4[out_idx] = fac * sum2
            x5[out_idx] = fac * np.sqrt(sum1)
            #x6[out_idx] = fac * sum1
        else:
            x4[out_idx] = 0
            x5[out_idx] = -1
            #x6[out_idx] = 0
    
    return x4, x5

@njit
def f77_ew_nj(wvs: np.ndarray, 
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
    
    # input checks
    if not (wvs.size == flux.size):
        raise IndexError("wvs and flux must have the same size")
    if mask is not None and mask.size != wvs.size:
        raise IndexError("mask and wvs must be the same size")
    if flux_err is not None and flux_err.size != flux.size:
        raise IndexError("flux_err and flux must be the same size")

    # nb change to bool_ and mask_ for njit
    mask_ = np.ones(flux.shape, dtype=bool_) if mask is None else mask
    flux_err_ = flux_err if flux_err is not None else np.ones(flux.shape)
    
    dws = np.zeros_like(wvs)
    dws[:-1] = wvs[1:] - wvs[:-1]
    # extrapolate for final value
    dws[-1] = dws[-1] + (dws[-1] - dws[-2])

    # subtracting continuum would go here
    ew = (dws[mask_] * (flux[mask_])).sum()
    ew_err = ((dws[mask_] * flux_err[mask_])**2).sum()**0.5 if flux_err is not None else np.nan
        
    return ew, ew_err
    
