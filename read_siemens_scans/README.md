# Siemens Scanner TWIX Reader

This repository contains tools and scripts necessary for reading and visualizing TWIX files from Siemens MRI scanners.
The project supports the processing and display of both 2D and 3D scans, providing a streamlined workflow for handling this proprietary data format.

## Features
- **Unambiguous TWIX file parsing:** Reading a file, it returns both the k-space array and info on its dimensions (names, lengths).

## Requirements

Only requirements:
`twixtools == 1.0, numpy, matplotlib`

## Usage

**Demo** Go to `demos/demo.ipynb` for a full example usage.

**Reading raw data**
```
from read_siemens_scans.twix_reading import read_raw_twix_data

twix_path: str = ...
kspace, dim_data = read_raw_twix_data(twix_path, fix_oversampling=False)
```

**iFFT recon**
```
from read_siemens_scans.recon import adjoint_recon, kspace_fix_aspect_ratio

# If FOV is square-shaped
images = adjoint_recon(kspace, dim_data)

# Otherwise, using simple Fourier interpolation (note that it would introduce ringing)
kspace_fixed = kspace_fix_aspect_ratio(kspace, dim_data)
images_interp = adjoint_recon(kspace_fixed, dim_data)
```

## Acknowledgements

This repo is based on Orel Tsioni's code
https://github.com/orelts/raw_kspace_recon/tree/master

