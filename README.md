# Wave-induced-fluid-flow
This repository contains Python implementations of several **wave-induced fluid flow (WIFF)** models for studying seismic wave dispersion and attenuation in porous rocks.
## Main modules
- `biot_gassmann.py`
Biot global flow and Gassmann  models
- `EMT.py`
Effective medium theory (EMT) related functions.
- `squirt_flow.py`
Squirt-flow models and related calculations.
- `wiff_layer.py`
Wave-induced fluid flow in layered porous media.
- `wiff_layer_aniso.py`
WIFF model for anisotropic layered media.
- `wiff_sphere.py`
Spherical inclusion / patchy-saturation WIFF models.
- `utils.py`
Common utility functions used by different models.
## Examples and tests
The following Jupyter notebooks are used to test the models and reproduce numerical examples:
- `test_global_flow.ipynb`
- `test_layer.ipynb`
- `test_sphere.ipynb`
- `test_squirt_flow.ipynb`
## Requirements
Main dependencies include: Python, NumPy, SciPy, Matplotlib