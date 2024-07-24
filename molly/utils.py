import numpy as np

class dummy_pbar():
    """
    Placeholder for tqdm pbar that does nothing
    """
    def __init__(self, desc, total):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, num):
        pass

def apply_mask(wvs: np.ndarray, mask_bounds: list):
    """
    Apply a list of mask bounds, contained tuples of upper and lower bounds,
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

def read_mask(file):
    """
    Opens a molly mask, returns a list of tuples of mask bounds
    Only works with wavelength masks, not pixels or velocities.
    
    """

    with open(file, 'r') as f:
        line_list = f.readlines()[1:]
        mask_str_line_list = [l.strip() for l in line_list if len(l.strip()) != 0]
        # deal with multiple lines
        mask_list = []
        for l in mask_str_line_list:
            mask_list = mask_list + [float(i) for i in l.split()]

        # separate into pairs
        lower_bounds = mask_list[0::2]
        upper_bounds = mask_list[1::2]

        mask = []
        
        for lb, ub in zip(lower_bounds, upper_bounds):
            mask.append((lb, ub))

    return mask