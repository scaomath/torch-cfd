# Computational Fluid Dynamics in PyTorch

A native PyTorch port of [Google's Computational Fluid Dynamics package in Jax](https://github.com/google/jax-cfd). The main changes are documented in the `README.md` under the [`torch_cfd` directory](torch_cfd). Biggest changes in all routines:
- Routines that rely on the functional programming of Jax have been rewritten to be a more debugger-friendly PyTorch tensor-in-tensor-out style.
- Functions and operators are in general implemented as `nn.Module`.

## Installation

```bash
pip install torch-cfd
```

## Contributions
PR welcome. Currently, the port only includes:
- Pseudospectral methods for vorticity which use anti-aliasing filtering techniques for non-linear terms to maintain stability.
- Temporal discretization: Currently only RK4 temporal discretization using explicit time-stepping for advection and either implicit or explicit time-stepping for diffusion.
- Boundary conditions: only periodic boundary conditions.

## Examples
- Demos of different simulation setups:
  - [2D simulation with a pseudo-spectral solver](example_Kolmogrov2d_rk4_cn_forced_turbulence.ipynb)

## Acknowledgments
SC appreciates the support from the National Science Foundation DMS-2309778.