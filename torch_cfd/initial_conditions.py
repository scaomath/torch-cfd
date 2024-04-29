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
from typing import Callable, Optional, Sequence
import math
import torch
import torch.fft as fft
from . import grids
from . import finite_differences as fd
from . import fast_diagonalization as solver

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


def _log_normal_pdf(x, mode: float, variance=0.25):
    """Unscaled PDF for a log normal given `mode` and log variance 1."""
    mean = math.log(mode) + variance
    logx = torch.log(x)
    return torch.exp(-((mean - logx) ** 2) / 2 / variance - logx)


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


def _rhs_transform(
    u: GridArray,
    bc: BoundaryConditions,
) -> Array:
    """Transform the RHS of pressure projection equation for stability.

    In case of poisson equation, the kernel is subtracted from RHS for stability.

    Args:
      u: a GridArray that solves ∇²x = u.
      bc: specifies boundary of x.

    Returns:
      u' s.t. u = u' + kernel of the laplacian.
    """
    u_data = u.data
    for axis in range(u.grid.ndim):
        if (
            bc.types[axis][0] == grids.BCType.NEUMANN
            and bc.types[axis][1] == grids.BCType.NEUMANN
        ):
            # if all sides are neumann, poisson solution has a kernel of constant
            # functions. We substact the mean to ensure consistency.
            u_data = u_data - torch.mean(u_data)
    return u_data


def solve_fast_diag(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    pressure_bc: Optional[grids.ConstantBoundaryConditions] = None,
    implementation: Optional[str] = None,
) -> GridArray:
    """Solve for pressure using the fast diagonalization approach."""
    del q0  # unused
    if pressure_bc is None:
        pressure_bc = grids.get_pressure_bc_from_velocity(v)
    if grids.has_all_periodic_boundary_conditions(*v):
        circulant = True
    else:
        circulant = False
        # only matmul implementation supports non-circulant matrices
        implementation = "matmul"
    grid = grids.consistent_grid(*v)
    rhs = fd.divergence(v)
    laplacians = list(map(fd.laplacian_matrix, grid.shape, grid.step))
    laplacians = [lap.to(grid.device) for lap in laplacians]
    rhs_transformed = _rhs_transform(rhs, pressure_bc)
    pinv = solver.pseudoinverse(
        rhs_transformed,
        laplacians,
        rhs_transformed.dtype,
        hermitian=True,
        circulant=circulant,
        implementation=implementation,
    )
    # return applied(pinv)(rhs_transformed)
    return GridArray(pinv, rhs.offset, rhs.grid)


def projection(
    v: GridVariableVector,
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
    """
    Apply pressure projection (a discrete Helmholtz decomposition)
    to make a velocity field divergence free.
    
    Note: this will have a non-negligible error in fp32.
    """
    grid = grids.consistent_grid(*v)
    pressure_bc = grids.get_pressure_bc_from_velocity(v)

    q0 = GridArray(torch.zeros(grid.shape), grid.cell_center, grid)
    q0 = pressure_bc.impose_bc(q0)

    q = solve(v, q0, pressure_bc)
    q = pressure_bc.impose_bc(q)
    q_grad = fd.forward_difference(q)
    v_projected = tuple(u.bc.impose_bc(u.array - q_g) for u, q_g in zip(v, q_grad))
    return v_projected


def project_and_normalize(
    v: GridVariableVector, maximum_velocity: float = 1
) -> GridVariableVector:
    v = projection(v)
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
        return _log_normal_pdf(k, peak_wavenumber) / k ** (grid.ndim - 1)

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
