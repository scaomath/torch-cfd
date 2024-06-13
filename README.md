# Neural Operator-Assisted Computational Fluid Dynamics in PyTorch

## Summary 

This repository features two parts:

### Part I: a native PyTorch port of [Google's Computational Fluid Dynamics package in Jax](https://github.com/google/jax-cfd)
The main changes are documented in the `README.md` under the [`torch_cfd` directory](./torch_cfd/). The most significant changes in all routines include:
  - Routines that rely on the functional programming of Jax have been rewritten to be a more debugger-friendly PyTorch tensor-in-tensor-out style.
  - Functions and operators are in general implemented as `nn.Module` like a factory template.
  - Jax-cfd's `funcutils.trajectory` function supports to track only one field variable (vorticity or velocity), Extra fields computation and tracking are made easier, such as time derivatives and PDE residual $R(\boldsymbol{v}):=\boldsymbol{f}-\partial_t \boldsymbol{v}-(\boldsymbol{v}\cdot\nabla)\boldsymbol{v} + \nu \Delta \boldsymbol{v}$.
  - All ops take batch dimension of tensors `(b, *, n, n)` regardless of `*` dimension, which is similar to PyTorch behavior, not a single trajectory like Google's original Jax-CFD package.

### Part II: Spectral-Refiner: Neural Operator-Assisted Navier-Stokes Equations solver.
  - The **Spatiotempoeral Fourier Neural Operator** (SFNO) is a spacetime tensor-to-tensor learner (or trajectory-to-trajectory), available in the [`sfno` directory](./sfno/). We draw inspiration from the [3D FNO in Nvidia's Neural Operator repo](https://github.com/neuraloperator/neuraloperator), as well as Temam's book on functional analysis for NSE.
  - Data generation for the meta-example of the isotropic turbulence in [McWilliams1984]. After the warmup phase, the energy spectra match the inverse cascade of Kolmogorov flow in a periodic box.
  - Pipelines for the *a posteriori* error estimation to fine-tune the SFNO to reach the scientific computing level of accuracy ($\le 10^{-6}$) in Bochner norm using FLOPs on par with a single evaluation, and only a fraction of FLOPs of a single `.backward()`.
  - [Examples](#examples) can be found below.

[McWilliams1984]: McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *Journal of Fluid Mechanics*, 146, 21-43.

## Installation
```bash
pip install -r ./requirements.txt
```
Please install the required packages above or create a venv. Note: using PyTorch version >=2.0.0 for the broadcasting semantics.

## Data
The data are available at https://www.kaggle.com/datasets/anonymousauthor25/sfno-dataset  
Data generation instructions are available in the [SFNO folder](./sfno/)


## Examples
Please check the example folder on the left panel. Somehow the example notebook links are broken after anonymization.
- Demos of different simulation setups:
  - [2D simulation with a pseudo-spectral solver](./examples/Kolmogrov2d_rk4_cn_forced_turbulence.ipynb)
- Demos of Spatiotemporal FNO's training and evaluation
  - [Training of SFNO for only 15 epochs for the isotropic turbulence example](./examples/ex2_SFNO_train.ipynb)
  - [Training of SFNO for only ***10*** epochs with 1k samples and reach `1e-2` level of relative error](./examples/ex2_SFNO_train_fnodata.ipynb) using the data in the FNO paper, which to our best knowledge no operator learner can do this in 500 epochs in the small data regime.
  - [Fine-tuning of SFNO on a `256x256` grid for only 50 ADAM iterations to reach `1e-6` residual in the functional norm using FNO data](./examples/ex2_SFNO_finetune_fnodata.ipynb)
  - [Fine-tuning of SFNO on the `256x256` grid for the McWilliams 2d isotropic turbulence](./examples/ex2_SFNO_finetune_McWilliams2d.ipynb)
  - [Training of SFNO for only 5 epoch to match the inverse cascade of Kolmogorov flow](./examples/ex2_SFNO_5ep_spectra.ipynb)
  - [Baseline of FNO3d for fixed step size that requires preloading a normalizer](./examples/ex2_FNO3d_train_normalized.ipynb)

## Licenses
The Apache 2.0 License in the root folder applies to the `torch-cfd` folder of the repo that is inherited from Google's original license file for `Jax-cfd`. The `fno` folder has the MIT license inherited from [NVIDIA's Neural Operator repo](https://github.com/neuraloperator/neuraloperator). Note: the license(s) in the subfolder takes precedence.
