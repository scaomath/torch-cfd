# Torch CFD

This is a native PyTorch port of [Google's Computational Fluid Dynamics package in Jax](https://github.com/google/jax-cfd). The main changes are documented in the `README.md` under the [`torch_cfd` directory](torch_cfd/README.md). The biggest change is many routines that rely on the functional programming of Jax have been rewritten to be a more PyTorch-friendly tensor-in to tensor-out style.

## Installation

```bash
pip install torch-cfd
```

## Contributions
PR welcome. Current the port only includes:
- Pseudospectral methods for vorticity which use anti-aliasing filtering techniques for non-linear terms to maintain stability.
- Temporal discretization: Currently only RK4 temporal discretization, using explicit time-stepping for advection and either implicit or explicit time-stepping for diffusion.
- Boundary conditions: only periodic boundary conditions.

## Examples
- Demos of different simulation setups:
  - [2D simulation with a psuedo-spectral solver](example_Kolmogrov2d_rk4_cn_forced_turbulence.ipynb)
