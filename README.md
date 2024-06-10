# Neural Operator-Assisted Computational Fluid Dynamics in PyTorch

This repository featuers two parts:
- The first part is a native PyTorch port of [Google's Computational Fluid Dynamics package in Jax](https://github.com/google/jax-cfd). The main changes are documented in the `README.md` under the [`torch_cfd` directory](torch_cfd). Most significant changes in all routines include:
  - Routines that rely on the functional programming of Jax have been rewritten to be a more debugger-friendly PyTorch tensor-in-tensor-out style.
  - Functions and operators are in general implemented as `nn.Module` like a factory template.
  - Jax-cfd's `funcutils.trajectory` function supports to track only one field variable (vorticity or velocity), Extra fields computation and tracking are made easier, such as time derivatives and PDE residual $R(\boldsymbol{v}):=\boldsymbol{f}-\partial_t \boldsymbol{v}-(\boldsymbol{v}\cdot\nabla)\boldsymbol{v} + \nu \Delta \boldsymbol{v}$.
  - All ops takes batch dimension of tensors into consideration, not a single trajectory.
- Neural Operator-Assisted Navier-Stokes Equations solver.
  - The **Spatiotempoeral Fourier Neural Operator** (SFNO) that is a spacetime tensor-to-tensor learner (or trajectory-to-trajectory), available in the [`sfno` directory](sfno/). Inspirations are drawn from the [3D FNO in Nvidia's Neural Operator repo](https://github.com/neuraloperator/neuraloperator).
  - Data generation for the meta-example of the isotropic turbulence with energy spectra matching the inverse cascade of Kolmogorov flow in a periodic box. Ref: McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *Journal of Fluid Mechanics*, 146, 21-43.
  - Pipelines for the *a posteriori* error estimation to fine-tune the SFNO to reach the scientific computing level of accuracy ($\le 10^{-6}$) in Bochner norm using FLOPs on par with a single evaluation, and only a fraction of FLOPs of a single `.backward()`.
  - Example files will be added later after cleanup.

## Installation
To install `torch-cfd`'s current release, simply do:
```bash
pip install torch-cfd
```
If one wants to play with the neural operator part, it is recommended to clone this repo and play it locally by creating a venv using `requirements.txt`. Note: using PyTorch version >=2.0.0 for the broadcasting semantics.

## Data
The data are available at https://huggingface.co/datasets/scaomath/navier-stokes-dataset 
Data generation instructions are available in the [SFNO folder](/sfno/)


## Examples
- Demos of different simulation setups:
  - [2D simulation with a pseudo-spectral solver](/examples/Kolmogrov2d_rk4_cn_forced_turbulence.ipynb)
- Demos of Spatiotemporal FNO's training and evaluation
  - [Training of SFNO for only 15 epochs](/examples/ex2_SFNO_train.ipynb)
  - [Training of SFNO for only 5 epoch to match the inverse cascade of Kolmogorov flow](/examples/ex2_SFNO_5ep_spectra.ipynb)
  - [Baseline of FNO3d for fixed step size](/examples/ex2_FNO3d_train_normalized.ipynb)

## Licenses
The Apache 2.0 License in the root folder applies to the `torch-cfd` folder of the repo that is inherited from Google's original license file for `Jax-cfd`. The `fno` folder has the MIT license inherited from [NVIDIA's Neural Operator repo](https://github.com/neuraloperator/neuraloperator). Note: the license(s) in the subfolder takes precedence.

## Contributions
PR welcome. Currently, the port of `torch-cfd` currently includes:
- Pseudospectral method for vorticity which uses anti-aliasing filtering techniques for nonlinear terms to maintain stability.
- Temporal discretization: Currently only RK4 temporal discretization using explicit time-stepping for advection and either implicit or explicit time-stepping for diffusion.
- Boundary conditions: only periodic boundary conditions.

## Reference
```bibtex
@article{2024SpectralRefiner,
  title={Spectral-Refiner: Fine-Tuning of Accurate Spatiotemporal Neural Operator for Turbulent Flows},
  author={Shuhao Cao and Francesco Brarda and Ruipeng Li and Yuanzhe Xi},
  journal={arXiv preprint arXiv:2405.17211},
  year={2024},
  primaryClass={cs.LG}
}
```

## Acknowledgments
The research of Brarda and Xi is supported by the National Science Foundation award DMS-2208412. 
The work of Li was performed under the auspices of
the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DEAC52-07NA27344 and was supported by the LLNL-LDRD program under Project No. 24ERD033. Cao is greatful for the support from [Long Chen (UC Irvine)](https://github.com/lyc102/ifem) and 
[Ludmil Zikatanov (Penn State)](https://github.com/HAZmathTeam/hazmath) over the years, and their efforts in open-sourcing scientific computing codes. Cao also appreciates the support from the National Science Foundation DMS-2309778, and the free A6000 credits at the SSE ML cluster from the University of Missouri.