"""
Open a Siemens TWIX .dat file, extract the k-space array and human-readable dimension info.
"""
import twixtools
from collections import OrderedDict
from read_siemens_scans.config import DIM_DESCRIPTION
import json


class Dimensions(OrderedDict):
    def __init__(self, d):
        super().__init__(d)

    def print(self):
        print(json.dumps(self, indent=4))



def read_raw_twix_data(twix_path: str, fix_oversampling: bool = False):
    """
    # Read the TWIX file using twixtools
    # If fix_oversampling is True, it will adjust the k-space data to fit a 1:1 aspect ratio
    
    Input:
    twix_path:           TWIX (.dat) file path.
    fix_oversampling:    Enable handling of oversampling. Optional flag to correct oversampling: if True, the code will remove oversampling along the readout axis before returning the array.
    
    Output:
    kspace_squeezed:     NumPy array of raw complex k-space data.
    active_image_dims:   OrderedDict with labeled dimensions (dimension name: length).
    """
    assert twix_path.endswith('.dat') # Sanity check for TWIX file format (.dat extension))

    twix = twixtools.read_twix(twix_path, keep_syncdata=True, keep_acqend=True) # File reading step. 
    mapped = twixtools.map_twix(twix) # Converts the low-level twix object into a mapped dictionary-like structure.
    kspace = mapped[-1]['image'] # From the mapped list of measurements, pick the last one and then access its 'image' entry.
    kspace.flags['remove_os'] = fix_oversampling # Here kspace is a twixtools MmapArray object with an attribute .flags (a dict) that controls how data is read from disk. Apple or don't apply oversampling removal.
    dim_keys = list(kspace._dim_order) # Convert k_space._dim_order to a list. kspace._dim_order is a tuple of dimension keys from the TWIX header (e.g 'Lin': phase encode lines, 'Col':readout columns, 'Cha': coil channels, 'Par': partition (for 3D phase encoding)). 
    
    """
    active_image_dims: 
    Builds an OrderedDict (wrapped in the custom Dimensions class) where:
    Key = a human-readable dimension name (from DIM_DESCRIPTION in config.py if known; otherwise 'Unnamed (KEY)').
    Value = the length of that dimension in the array.
    The if kspace.shape[dim] > 1 filter removes singleton dimension
    """
    active_image_dims = Dimensions( 
        (
            DIM_DESCRIPTION[dim_keys[dim]] if dim_keys[dim] in DIM_DESCRIPTION.keys() else f'Unnamed ({dim_keys[dim]})',
            kspace.shape[dim]
        ) for dim in range(kspace.ndim) if kspace.shape[dim] > 1
    )
    kspace_squeezed = kspace[:].squeeze() # Removes any remaining singleton axes 
    return kspace_squeezed, active_image_dims








