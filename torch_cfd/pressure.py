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

"""Functions for computing and applying pressure."""

from typing import Callable, Optional

import torch

from . import grids, fast_diagonalization as solver, finite_differences as fd


Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


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
    """Solve for pressure using the fast diagonalization approach.

      To support backward compatibility, if the pressure_bc are not provided and
      velocity has all periodic boundaries, pressure_bc are assigned to be periodic.

      Args:
          v: a tuple of velocity values for each direction.
          q0: the starting guess for the pressure.
          pressure_bc: the boundary condition to assign to pressure. If None,
          boundary condition is infered from velocity.
          implementation: how to implement fast diagonalization.
          For non-periodic BCs will automatically be matmul.

    Returns:
      A solution to the PPE equation.
    """
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
    return GridArray(pinv, rhs.offset, rhs.grid)


def projection(
    v: GridVariableVector,
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
    """
    Apply pressure projection (a discrete Helmholtz decomposition)
    to make a velocity field divergence free.

    Note by S.Cao: this was originally implemented by the jax-cfd team
    but using FDM results having a non-negligible error in fp32.
    One resolution is to use fp64 then cast back to fp32.
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
