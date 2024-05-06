# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2024 S.Cao
# ported Google's Jax-CFD functional template to PyTorch's tensor ops

"""Prepare initial conditions for simulations."""
import math
from typing import Callable, Optional, Sequence

import torch
import torch.fft as fft

from . import grids, pressure

Array = torch.Tensor
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def wrap_velocities(
    v: Sequence[Array],
    grid: grids.Grid,
    bcs: Sequence[BoundaryConditions],
    device: Optional[torch.device] = None,
) -> GridVariableVector:
    """Wrap velocity arrays for input into simulations."""
    device = grid.device if device is None else device
    return tuple(
        GridVariable(GridArray(u, offset, grid).to(device), bc)
        for u, offset, bc in zip(v, grid.cell_faces, bcs)
    )


def wrap_vorticity(
    w: Array,
    grid: grids.Grid,
    bc: BoundaryConditions,
    device: Optional[torch.device] = None,
) -> GridVariable:
    """Wrap vorticity arrays for input into simulations."""
    device = grid.device if device is None else device
    return GridVariable(GridArray(w, grid.cell_faces, grid).to(device), bc)


def _log_normal_density(k, mode: float, variance=0.25):
    """
    Unscaled PDF for a log normal given `mode` and log variance 1.


    """
    mean = math.log(mode) + variance
    logk = torch.log(k)
    return torch.exp(-((mean - logk) ** 2) / 2 / variance - logk)


def McWilliams_density(k, mode: float, tau: float = 1.0):
    """Implements the McWilliams spectral density function.
    |\psi|^2 \sim k^{-1}(tau^2 + (k/k_0)^4)^{-1}
    k_0 is a prescribed wavenumber that the energy peaks.
    tau flattens the spectrum density at low wavenumbers to be bigger.

    Refs:
      McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow.
    """
    return (k * (tau**2 + (k / mode) ** 4)) ** (-1)


def _angular_frequency_magnitude(grid: grids.Grid) -> Array:
    frequencies = [
        2 * torch.pi * fft.fftfreq(size, step)
        for size, step in zip(grid.shape, grid.step)
    ]
    freq_vector = torch.stack(torch.meshgrid(*frequencies, indexing="ij"), axis=0)
    return torch.linalg.norm(freq_vector, axis=0)


def spectral_filter(
    spectral_density: Callable[[Array], Array],
    v: Array,
    grid: grids.Grid,
) -> Array:
    """Filter an Array with white noise to match a prescribed spectral density."""
    k = _angular_frequency_magnitude(grid)
    filters = torch.where(k > 0, spectral_density(k), 0.0)
    # The output signal can safely be assumed to be real if our input signal was
    # real, because our spectral density only depends on norm(k).
    return fft.ifftn(fft.fftn(v) * filters).real


def streamfunc_normalize(k, psi):
    nx, ny = psi.shape
    psih = fft.fft2(psi)
    uh_mag = k * psih
    kinetic_energy = (2 * uh_mag.abs() ** 2 / (nx * ny) ** 2).sum()
    return psi / kinetic_energy.sqrt()


def project_and_normalize(
    v: GridVariableVector, maximum_velocity: float = 1
) -> GridVariableVector:
    v = pressure.projection(v)
    vmax = torch.linalg.norm(torch.stack([u.data for u in v]), dim=0).max()
    v = tuple(GridVariable(maximum_velocity * u.array / vmax, u.bc) for u in v)
    return v


def filtered_velocity_field(
    grid: grids.Grid,
    maximum_velocity: float = 1,
    peak_wavenumber: float = 3,
    iterations: int = 3,
    random_state: int = 0,
) -> GridArray:
    """Create divergence-free velocity fields with appropriate spectral filtering.

    Args:
      rng_key: key for seeding the random initial velocity field.
      grid: the grid on which the velocity field is defined.
      maximum_velocity: the maximum speed in the velocity field.
      peak_wavenumber: the velocity field will be filtered so that the largest
        magnitudes are associated with this wavenumber.
      iterations: the number of repeated pressure projection and normalization
        iterations to apply.
    Returns:
      A divergence free velocity field with the given maximum velocity. Associates
      periodic boundary conditions with the velocity field components.
    """

    # Log normal distribution peaked at `peak_wavenumber`. Note that we have to
    # divide by `k ** (ndim - 1)` to account for the volume of the
    # `ndim - 1`-sphere of values with wavenumber `k`.
    def spectral_density(k):
        return _log_normal_density(k, peak_wavenumber) / k ** (grid.ndim - 1)

    random_states = [random_state + i for i in range(grid.ndim)]
    rng = torch.Generator()
    velocity_components = []
    boundary_conditions = []
    for k in random_states:
        rng.manual_seed(k)
        noise = torch.randn(grid.shape, generator=rng)
        velocity_components.append(spectral_filter(spectral_density, noise, grid))
        boundary_conditions.append(grids.periodic_boundary_conditions(grid.ndim))
    velocity = wrap_velocities(velocity_components, grid, boundary_conditions)

    # Due to numerical precision issues, we repeatedly normalize and project the
    # velocity field. This ensures that it is divergence-free and achieves the
    # specified maximum velocity.
    # velocity is ((n, n), (n, n)) GridVariableVector
    for _ in range(iterations):
        velocity = project_and_normalize(velocity, maximum_velocity)
    return velocity


def vorticity_field(
    grid: grids.Grid,
    peak_wavenumber: float = 3,
    random_state: int = 0,
) -> GridArray:
    """Create vorticity field with a spectral filtering
    using the McWilliams power spectrum density function.

    Args:
      rng_key: key for seeding the random initial vorticity field.
      grid: the grid on which the vorticity field is defined.
      peak_wavenumber: the velocity field will be filtered so that the largest
        magnitudes are associated with this wavenumber.

    Returns:
      A vorticity field with periodic boundary condition.
    """

    def spectral_density(k):
        return McWilliams_density(k, peak_wavenumber)

    rng = torch.Generator()
    rng.manual_seed(random_state)
    noise = torch.randn(grid.shape, generator=rng)
    k = _angular_frequency_magnitude(grid)
    psi = spectral_filter(spectral_density, noise, grid)
    psi = streamfunc_normalize(k, psi)
    vorticity = fft.ifftn(fft.fftn(psi) * k**2).real
    boundary_condition = grids.periodic_boundary_conditions(grid.ndim)
    vorticity = wrap_vorticity(vorticity, grid, boundary_condition)

    return vorticity
