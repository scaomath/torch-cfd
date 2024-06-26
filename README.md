# Neural Operator-Assisted Computational Fluid Dynamics in PyTorch

## Summary 

This repository features two parts:

### Part I: a native PyTorch port of [Google's Computational Fluid Dynamics package in Jax](https://github.com/google/jax-cfd)
The main changes are documented in the `README.md` under the [`torch_cfd` directory](./torch_cfd/). The most significant changes in all routines include:
  - Routines that rely on the functional programming of Jax have been rewritten to be a more debugger-friendly PyTorch tensor-in-tensor-out style.
  - Functions and operators are in general implemented as `nn.Module` like a factory template.
  - Jax-cfd's `funcutils.trajectory` function supports tracking only one field variable (vorticity or velocity). For this port, extra fields computation and tracking are made more accessible, such as time derivatives $\partial_t\boldsymbol{v}$ and PDE residual $R(\boldsymbol{v}):=\boldsymbol{f}-\partial_t \boldsymbol{v}-(\boldsymbol{v}\cdot\nabla)\boldsymbol{v} + \nu \Delta \boldsymbol{v}$.
  - All ops take into consideration the batch dimension of tensors `(b, *, n, n)` regardless of `*` dimension, which is similar to PyTorch behavior, not a single trajectory like Google's original Jax-CFD package.

### Part II: Spectral-Refiner: Neural Operator-Assisted Navier-Stokes Equations solver.
  - The **Spatiotempoeral Fourier Neural Operator** (SFNO) is a spacetime tensor-to-tensor learner (or trajectory-to-trajectory), available in the [`sfno` directory](./sfno/). We draw inspiration from the [3D FNO in Nvidia's Neural Operator repo](https://github.com/neuraloperator/neuraloperator), as well as Temam's book on functional analysis for NSE. Major architectural changes can be found in [the documentation of the `SFNO` class](https://github.com/scaomath/torch-cfd/blob/c88607dd9c87e4232c01102233c4ddc74398c9d8/sfno/sfno.py#L465). 
  - Data generation for the meta-example of the isotropic turbulence in [McWilliams1984]. After the warmup phase, the energy spectra match the inverse cascade of Kolmogorov flow in a periodic box.
  - Pipelines for the *a posteriori* error estimation to fine-tune the SFNO to reach the scientific computing level of accuracy ($\le 10^{-6}$) in Bochner norm using FLOPs on par with a single evaluation, and only a fraction of FLOPs of a single `.backward()`.
  - [Examples](#examples) can be found below.

[McWilliams1984]: McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *Journal of Fluid Mechanics*, 146, 21-43.

## Installation
To install `torch-cfd`'s current release, simply do:
```bash
pip install torch-cfd
```
If one wants to play with the neural operator part, it is recommended to clone this repo and play it locally by creating a venv using `requirements.txt`. Note: using PyTorch version >=2.0.0 for the broadcasting semantics.

## Data
The data are available at [https://huggingface.co/datasets/scaomath/navier-stokes-dataset](https://huggingface.co/datasets/scaomath/navier-stokes-dataset).
Data generation instructions are available in the [SFNO folder](./sfno/).


## Examples
- Demos of different simulation setups:
  - [2D simulation with a pseudo-spectral solver](./examples/Kolmogrov2d_rk4_cn_forced_turbulence.ipynb)
- Demos of Spatiotemporal FNO's training and evaluation
  - [Training of SFNO for only 15 epochs for the isotropic turbulence example](./examples/ex2_SFNO_train.ipynb)
  - [Training of SFNO for only ***10*** epochs with 1k samples and reach `1e-2` level of relative error](./examples/ex2_SFNO_train_fnodata.ipynb) using the data in the FNO paper, which to our best knowledge no operator learner can do this in <100 epochs in the small data regime.
  - [Fine-tuning of SFNO on a `256x256` grid for only 50 ADAM iterations to reach `1e-6` residual in the functional norm using FNO data](./examples/ex2_SFNO_finetune_fnodata.ipynb)
  - [Fine-tuning of SFNO on the `256x256` grid for the McWilliams 2d isotropic turbulence](./examples/ex2_SFNO_finetune_McWilliams2d.ipynb)
  - [Training of SFNO for only 5 epoch to match the inverse cascade of Kolmogorov flow](./examples/ex2_SFNO_5ep_spectra.ipynb)
  - [Baseline of FNO3d for fixed step size that requires preloading a normalizer](./examples/ex2_FNO3d_train_normalized.ipynb)

## Licenses
The Apache 2.0 License in the root folder applies to the `torch-cfd` folder of the repo that is inherited from Google's original license file for `Jax-cfd`. The `fno` folder has the MIT license inherited from [NVIDIA's Neural Operator repo](https://github.com/neuraloperator/neuraloperator). Note: the license(s) in the subfolder takes precedence.

## Contributions
PR welcome. Currently, the port of `torch-cfd` currently includes:
- The pseudospectral method for vorticity uses anti-aliasing filtering techniques for nonlinear terms to maintain stability.
- Temporal discretization: Currently only RK4 temporal discretization uses explicit time-stepping for advection and either implicit or explicit time-stepping for diffusion.
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
the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DEAC52-07NA27344 and was supported by the LLNL-LDRD program under Project No. 24ERD033. Cao is grateful for the support from [Long Chen (UC Irvine)](https://github.com/lyc102/ifem) and 
[Ludmil Zikatanov (Penn State)](https://github.com/HAZmathTeam/hazmath) over the years, and their efforts in open-sourcing scientific computing codes. Cao also appreciates the support from the National Science Foundation DMS-2309778, and the free A6000 credits at the SSE ML cluster from the University of Missouri.