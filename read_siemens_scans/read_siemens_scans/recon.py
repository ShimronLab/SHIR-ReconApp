"""
Take k-space, apply an inverse FFT along the encoding axes to get back into image space.
"""

import numpy as np
from collections import OrderedDict


def ifftnd(kspace, axes=[-1]):
    """
    Perform an n-dimensional inverse Fast Fourier Transform (FFT) on the given k-space data.

    Parameters:
        kspace (ndarray): The input k-space data.
        axes (list of int): The axes over which to compute the inverse FFT.

    Returns:
        img (ndarray): The resulting image data in the spatial domain.
    """
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = ifftshift(ifftn(fftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def _find_dim_index(dims_data: OrderedDict, dim_name: str) -> int:
    """
    Given the image dimensions metadata, find which dimension is labeled by a given name.

    :param dims_data:   Image dimensions metadata. An OrderedDict of (dimension name: length).
    :param dim_name:    Dimension label we search for.
    :return:            Index of the found dimension.
    """
    list_found = [ind for ind, dim in enumerate(dims_data.keys()) if dim_name in dim]
    assert len(list_found) > 0, f'Did not find a dimension with the phrase \"{dim_name}\"'
    assert len(list_found) == 1, f'Found more than one dimension with the phrase \"{dim_name}\"'
    return list_found[0]


def adjoint_recon(kspace: np.ndarray, dims_dict: OrderedDict):
    """
    Pseudo-inverse-Fourier image reconstruction.

    :param kspace:      k-space data, with squeezed dimensions.
    :param dims_dict:   Image dimensions metadata. An OrderedDict of (dimension name: length).
    :return:
    """
    image_shape = [_find_dim_index(dims_dict, name) for name in ['Col', 'Lin']] # _find_dim_index searches dims_dict keys for a substring (e.g., 'Col' or 'Lin') and returns the axis index where it appears. This ensures we find the correct axes regardless of order in the array. 'Col' = readout direction, 'Lin'=phase encoding direction
    recons = ifftnd(kspace, image_shape) # Reconstructed complex images.
    return recons 


def kspace_fix_aspect_ratio(kspace: np.ndarray, dims_dict: OrderedDict):
    """
    Fourier interpolation of the image to fit a 1:1 aspect ratio.

    :param kspace:      k-space data, with squeezed dimensions.
    :param dims_dict:   Image dimensions metadata. An OrderedDict of (dimension name: length).
    :return:
    """
    # Find longest dimension out of {width, height}
    col_ind = _find_dim_index(dims_dict, 'Col')
    lin_ind = _find_dim_index(dims_dict, 'Lin')
    max_size = max(list(dims_dict.values())[ind] for ind in [col_ind, lin_ind])

    # Define eventual shape
    old_shape = np.array(kspace.shape)
    new_shape = np.copy(old_shape)
    new_shape[col_ind] = max_size
    new_shape[lin_ind] = max_size

    # Padding the k-space onto that shape
    lower_margin = (new_shape - old_shape) // 2
    upper_margin = new_shape - lower_margin - old_shape
    kspace_extended = np.pad(kspace, pad_width=list(zip(lower_margin, upper_margin)))

    return kspace_extended