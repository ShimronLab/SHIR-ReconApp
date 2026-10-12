"""
Open a Siemens TWIX .dat file, extract the k-space array and human-readable dimension info.
"""
import twixtools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from read_siemens_scans.config import DIM_DESCRIPTION
import json

import numpy as np


class Dimensions(OrderedDict):
    def __init__(self, d):
        super().__init__(d)

    def print(self):
        print(json.dumps(self, indent=4))


@dataclass
class TwixMeta:
    """
    What the file says about its own geometry, alongside the k-space array.

    header        : protocol header of the image measurement (twixtools 'hdr').
    axes          : twixtools dimension names of the squeezed image array, in
                    order (e.g. ['Sli', 'Lin', 'Cha', 'Col']).
    lines         : one entry per image readout (MDH), as equal-length arrays:
                    Lin, Sli, Par, Eco, Rep, Set, Ave, CenterLin, CenterCol,
                    ncol (samples in scan) and ref (also a calibration line).
    refscan       : squeezed 'refscan' array (readout oversampling kept), or None.
    refscan_axes  : dimension names of `refscan`.
    refscan_lines : Lin / Sli of every calibration readout.
    has_noise     : the file contains noise-adjust readouts.
    noise         : noise-adjust samples as (n_samples, n_coils), or None.
    """
    header: dict
    axes: List[str]
    lines: Dict[str, np.ndarray]
    refscan: Optional[np.ndarray] = None
    refscan_axes: List[str] = field(default_factory=list)
    refscan_lines: Dict[str, np.ndarray] = field(default_factory=dict)
    has_noise: bool = False
    noise: Optional[np.ndarray] = None


def read_raw_twix_data(twix_path: str, fix_oversampling: bool = False, *, with_meta: bool = False):
    """
    # Read the TWIX file using twixtools
    # If fix_oversampling is True, it will adjust the k-space data to fit a 1:1 aspect ratio

    Input:
    twix_path:           TWIX (.dat) file path.
    fix_oversampling:    Enable handling of oversampling. Optional flag to correct oversampling: if True, the code will remove oversampling along the readout axis before returning the array.
    with_meta:           Also return a TwixMeta (header + per-readout table) as a third value.

    Output:
    kspace_squeezed:     NumPy array of raw complex k-space data.
    active_image_dims:   OrderedDict with labeled dimensions (dimension name: length).
    meta:                TwixMeta — only when with_meta=True.
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
    if not with_meta:
        return kspace_squeezed, active_image_dims
    return kspace_squeezed, active_image_dims, _build_meta(twix, mapped, kspace)


_LINE_COUNTERS = ("Lin", "Sli", "Par", "Eco", "Rep", "Set", "Ave")


def _squeezed_axes(arr) -> List[str]:
    """Dimension names left after squeeze(), in order."""
    return [name for name, n in zip(arr._dim_order, arr.shape) if n > 1]


def _noise_samples(mapped) -> Optional[np.ndarray]:
    """Noise-adjust readouts (usually in the first, adjustment measurement)
    as (n_samples, n_coils), or None."""
    for meas in mapped:
        if "noise" not in meas:
            continue
        arr = meas["noise"]
        arr.flags["remove_os"] = False
        data = np.asarray(arr[:]).squeeze()
        axes = _squeezed_axes(arr)
        if "Cha" not in axes:
            continue
        data = np.moveaxis(data, axes.index("Cha"), -1)
        return data.reshape(-1, data.shape[-1])
    return None


def _build_meta(twix, mapped, image_arr) -> TwixMeta:
    """Collect the header and per-readout MDH fields of the image measurement."""
    meas = twix[-1]
    mapped_meas = mapped[-1]
    rows = {k: [] for k in (*_LINE_COUNTERS, "CenterLin", "CenterCol", "ncol", "ref")}
    ref_rows = {"Lin": [], "Sli": []}
    for mdb in meas["mdb"]:
        is_ref = mdb.is_flag_set("PATREFSCAN") or mdb.is_flag_set("PATREFANDIMASCAN")
        if is_ref:
            ref_rows["Lin"].append(mdb.cLin)
            ref_rows["Sli"].append(mdb.cSlc)
        if not mdb.is_image_scan():
            continue
        counter = mdb.mdh.Counter
        for k in _LINE_COUNTERS:
            rows[k].append(getattr(counter, k))
        rows["CenterLin"].append(mdb.mdh.CenterLin)
        rows["CenterCol"].append(mdb.mdh.CenterCol)
        rows["ncol"].append(mdb.mdh.SamplesInScan)
        rows["ref"].append(is_ref)

    refscan, refscan_axes = None, []
    if "refscan" in mapped_meas:
        ref_arr = mapped_meas["refscan"]
        ref_arr.flags["remove_os"] = False
        refscan = np.asarray(ref_arr[:]).squeeze()
        refscan_axes = _squeezed_axes(ref_arr)

    has_noise = any(mdb.is_flag_set("NOISEADJSCAN")
                    for m in twix for mdb in m["mdb"])
    return TwixMeta(
        header=meas["hdr"],
        axes=_squeezed_axes(image_arr),
        lines={k: np.asarray(v) for k, v in rows.items()},
        refscan=refscan,
        refscan_axes=refscan_axes,
        refscan_lines={k: np.asarray(v) for k, v in ref_rows.items()},
        has_noise=has_noise,
        noise=_noise_samples(mapped) if has_noise else None,
    )
