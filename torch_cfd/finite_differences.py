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

"""Functions for approximating derivatives.

Finite difference methods operate on GridVariable and return GridVariable without bc.
Evaluating finite differences requires boundary conditions, which are defined
for a GridVariable. But the operation of taking a derivative makes the boundary
condition undefined, which is why the bc is removed.

For example, if the variable c has the boundary condition c_b = 0, and we take
the derivate dc/dx, then it is unclear what the boundary condition on dc/dx
should be. So programmatically, after taking finite differences and doing
operations, the user has to explicitly assign boundary conditions to the result.

Example:
  c = GridVariable(c_array, offset, grid, c_boundary_condition)
  dcdx = finite_differences.forward_difference(c)  # returns GridArray
  c_new = c + dt * (-velocity * dcdx)  # operations on GridArrays
  c_new = GridVariable(c_new, offset, grid, c_boundary_condition)  # assocaite BCs
"""

import math
import typing
from typing import Any, List, Optional, Sequence, Tuple, Union
from functools import reduce
import operator
import torch
from torch_cfd import boundaries, grids

ArrayVector = Sequence[torch.Tensor]
GridVariable = grids.GridVariable
GridTensor = grids.GridTensor
GridVariableVector = Union[grids.GridVariableVector, Sequence[grids.GridVariable]]


def stencil_sum(*arrays: GridVariable, return_tensor=False) -> Union[GridVariable, torch.Tensor]:
    """
    Sum arrays across a stencil, with an averaged offset.
    After summing, the offset is averaged across the arrays and bc is set to None
    """
    result = torch.stack([array.data for array in arrays]).sum(dim=0)

    if return_tensor:
        return result
    
    offset = grids.averaged_offset_arrays(*arrays)
    grid = grids.consistent_grid_arrays(*arrays)
    
    return GridVariable(result, offset, grid)

@typing.overload
def forward_difference(u: GridVariable, dim: int) -> Union[GridVariable, torch.Tensor]: ...


@typing.overload
def forward_difference(
    u: GridVariable, dim: Optional[Tuple[int, ...]] = ...
) -> Tuple[GridVariable, ...]: ...


def forward_difference(u, dim=None):
    """Approximates grads with finite differences in the forward direction."""
    if dim is None:
        dim = range(-u.grid.ndim, 0)
    if not isinstance(dim, int):
        return tuple(
            forward_difference(u, a) for a in dim
        )  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    diff = stencil_sum(u.shift(+1, dim), -u)
    return diff / u.grid.step[dim]


@typing.overload
def central_difference(u: GridVariable, dim: int) -> GridVariable: ...


@typing.overload
def central_difference(
    u: GridVariable, dim: Optional[Tuple[int, ...]]
) -> Tuple[GridVariable, ...]: ...


def central_difference(u, dim=None):
    """Approximates grads with central differences."""
    if dim is None:
        dim = range(-u.grid.ndim, 0)
    if not isinstance(dim, int):
        return tuple(central_difference(u, a) for a in dim)
    diff = stencil_sum(u.shift(+1, dim), -u.shift(-1, dim))
    return diff / (2 * u.grid.step[dim])


@typing.overload
def backward_difference(u: GridVariable, dim: int) -> GridVariable: ...


@typing.overload
def backward_difference(
    u: GridVariable, dim: Optional[Tuple[int, ...]]
) -> Tuple[GridVariable, ...]: ...


def backward_difference(u, dim=None):
    """Approximates grads with finite differences in the backward direction."""
    if dim is None:
        dim = range(-u.grid.ndim, 0)
    if not isinstance(dim, int):
        return tuple(backward_difference(u, a) for a in dim)
    diff = stencil_sum(u, -u.shift(-1, dim))
    return diff / u.grid.step[dim]


def divergence(v: GridVariableVector) -> GridVariable:
    """Approximates the divergence of `v` using backward differences."""
    grid = grids.consistent_grid_arrays(*v)
    if len(v) != grid.ndim:
        raise ValueError(
            "The length of `v` must be equal to `grid.ndim`."
            f"Expected length {grid.ndim}; got {len(v)}."
        )
    return reduce(operator.add, [backward_difference(u, dim) for dim, u in enumerate(v)])


def centered_divergence(v: GridVariableVector) -> GridVariable:
    """Approximates the divergence of `v` using centered differences."""
    grid = grids.consistent_grid_arrays(*v)
    if len(v) != grid.ndim:
        raise ValueError(
            "The length of `v` must be equal to `grid.ndim`."
            f"Expected length {grid.ndim}; got {len(v)}."
        )
    return reduce(operator.add, [central_difference(u, dim) for dim, u in enumerate(v)])

def laplacian(u: GridVariable) -> GridVariable:
    """Approximates the Laplacian of `u`."""
    scales = tuple(1 / s**2 for s in u.grid.step)
    result = -2 * u.data * sum(scales)
    for d in range(-u.grid.ndim, 0):
        result += stencil_sum(u.shift(-1, d), u.shift(+1, d)) * scales[d]
    return result


def set_laplacian_matrix(
    grid: grids.Grid,
    bc: boundaries.BoundaryConditions,
    device: Optional[torch.device] = None,
) -> ArrayVector:
    """Initialize the Laplacian operators."""

    offset = grid.cell_center
    return laplacian_matrix_w_boundaries(grid, offset=offset, bc=bc, device=device)


def laplacian_matrix(n: int, step: float, sparse: bool = False) -> torch.Tensor:
    """
    Create 1D Laplacian operator matrix, with periodic BC.
    modified the scipy.linalg.circulant implementation to native torch
    """
    if sparse:
        values = torch.tensor([1.0, -2.0, 1.0]) / step**2
        idx_row = torch.arange(n).repeat(3)
        idx_col = torch.cat(
            [
                (torch.arange(n) - 1) % n,  # left neighbor (wrap around)
                torch.arange(n),  # center
                (torch.arange(n) + 1) % n,  # right neighbor (wrap around)
            ]
        )

        indices = torch.stack([idx_row, idx_col])
        data = torch.cat(
            [values[0].repeat(n), values[1].repeat(n), values[2].repeat(n)]
        )
        return torch.sparse_coo_tensor(indices, data, size=(n, n))
    else:
        column = torch.zeros(n)
        column[0] = -2 / step**2
        column[1] = column[-1] = 1 / step**2
        idx = (n - torch.arange(n)[None].T + torch.arange(n)[None]) % n
        return torch.gather(column[None, ...].expand(n, -1), 1, idx)


def _laplacian_boundary_dirichlet_cell_centered(
    laplacians: ArrayVector, grid: grids.Grid, axis: int, side: str
) -> None:
    """Converts 1d laplacian matrix to satisfy dirichlet homogeneous bc.

    laplacians[i] contains a 3 point stencil matrix L that approximates
    d^2/dx_i^2.
    For detailed documentation on laplacians input type see
    array_utils.laplacian_matrix.
    The default return of array_utils.laplacian_matrix makes a matrix for
    periodic boundary. For dirichlet boundary, the correct equation is
    L(u_interior) = rhs_interior and BL_boundary = u_fixed_boundary. So
    laplacian_boundary_dirichlet restricts the matrix L to
    interior points only.

    This function assumes RHS has cell-centered offset.
    Args:
      laplacians: list of 1d laplacians
      grid: grid object
      axis: axis along which to impose dirichlet bc.
      side: lower or upper side to assign boundary to.

    Returns:
      updated list of 1d laplacians.

    TODO:
    [ ]: this function is not implemented in the original Jax-CFD code.
    """
    # This function assumes homogeneous boundary, in which case if the offset
    # is 0.5 away from the wall, the ghost cell value u[0] = -u[1]. So the
    # 3 point stencil [1 -2 1] * [u[0] u[1] u[2]] = -3 u[1] + u[2].
    if side == "lower":
        laplacians[axis][0, 0] = laplacians[axis][0, 0] - 1 / grid.step[axis] ** 2
    else:
        laplacians[axis][-1, -1] = laplacians[axis][-1, -1] - 1 / grid.step[axis] ** 2
    # deletes corner dependencies on the "looped-around" part.
    # this should be done irrespective of which side, since one boundary cannot
    # be periodic while the other is.
    laplacians[axis][0, -1] = 0.0
    laplacians[axis][-1, 0] = 0.0
    return laplacians


def _laplacian_boundary_neumann_cell_centered(
    laplacians: List[Any], grid: grids.Grid, axis: int, side: str
) -> None:
    """Converts 1d laplacian matrix to satisfy neumann homogeneous bc.

    This function assumes the RHS will have a cell-centered offset.
    Neumann boundaries are not defined for edge-aligned offsets elsewhere in the
    code.

    Args:
      laplacians: list of 1d laplacians
      grid: grid object
      axis: axis along which to impose dirichlet bc.
      side: which boundary side to convert to neumann homogeneous bc.

    Returns:
      updated list of 1d laplacians.

    TODO
    [ ]: this function is not implemented in the original Jax-CFD code.
    """
    if side == "lower":
        laplacians[axis][0, 0] = laplacians[axis][0, 0] + 1 / grid.step[axis] ** 2
    else:
        laplacians[axis][-1, -1] = laplacians[axis][-1, -1] + 1 / grid.step[axis] ** 2
    # deletes corner dependencies on the "looped-around" part.
    # this should be done irrespective of which side, since one boundary cannot
    # be periodic while the other is.
    laplacians[axis][0, -1] = 0.0
    laplacians[axis][-1, 0] = 0.0
    return laplacians


def laplacian_matrix_w_boundaries(
    grid: grids.Grid,
    offset: Tuple[float, ...],
    bc: grids.BoundaryConditions,
    laplacians: Optional[ArrayVector] = None,
    device: Optional[torch.device] = None,
    sparse: bool = False,
) -> ArrayVector:
    """Returns 1d laplacians that satisfy boundary conditions bc on grid.

    Given grid, offset and boundary conditions, returns a list of 1 laplacians
    (one along each axis).

    Currently, only homogeneous or periodic boundary conditions are supported.

    Args:
      grid: The grid used to construct the laplacian.
      offset: The offset of the variable on which laplacian acts.
      bc: the boundary condition of the variable on which the laplacian acts.

    Returns:
      A list of 1d laplacians.
    """
    if not isinstance(bc, boundaries.ConstantBoundaryConditions):
        raise NotImplementedError(f"Explicit laplacians are not implemented for {bc}.")
    if laplacians is None:
        laplacians = list(map(laplacian_matrix, grid.shape, grid.step))
    for dim in range(-grid.ndim, 0):
        if math.isclose(offset[dim], 0.5):
            for i, side in enumerate(["lower", "upper"]):  # lower and upper boundary
                if bc.types[dim][i] == boundaries.BCType.NEUMANN:
                    _laplacian_boundary_neumann_cell_centered(
                        laplacians, grid, dim, side
                    )
                elif bc.types[dim][i] == boundaries.BCType.DIRICHLET:
                    _laplacian_boundary_dirichlet_cell_centered(
                        laplacians, grid, dim, side
                    )
        if math.isclose(offset[dim] % 1, 0.0):
            if (
                bc.types[dim][0] == boundaries.BCType.DIRICHLET
                and bc.types[dim][1] == boundaries.BCType.DIRICHLET
            ):
                # This function assumes homogeneous boundary and acts on the interior.
                # Thus, the laplacian can be cut off past the edge.
                # The interior grid has one fewer grid cell than the actual grid, so
                # the size of the laplacian should be reduced.
                laplacians[dim] = laplacians[dim][:-1, :-1]
            elif boundaries.BCType.NEUMANN in bc.types[dim]:
                raise NotImplementedError(
                    "edge-aligned Neumann boundaries are not implemented."
                )
    return list(lap.to(device) for lap in laplacians) if device else laplacians


def _linear_along_axis(c: GridVariable, offset: float, dim: int) -> GridVariable:
    """Linear interpolation of `c` to `offset` along a single specified `axis`."""
    offset_delta = offset - c.offset[dim]

    # If offsets are the same, `c` is unchanged.
    if offset_delta == 0:
        return c

    new_offset = tuple(offset if j == dim else o for j, o in enumerate(c.offset))

    # If offsets differ by an integer, we can just shift `c`.
    if int(offset_delta) == offset_delta:
        data = grids.shift(c, int(offset_delta), dim).data
        return GridVariable(
            data=data,
            offset=new_offset,
            grid=c.grid,
            bc=c.bc,
        )

    floor = int(math.floor(offset_delta))
    ceil = int(math.ceil(offset_delta))
    floor_weight = ceil - offset_delta
    ceil_weight = 1.0 - floor_weight
    data = (
        floor_weight * c.shift(floor, dim).data
        + ceil_weight * c.shift(ceil, dim).data
    )
    return GridVariable(data, new_offset, c.grid, c.bc)


def linear(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None,
) -> GridVariable:
    """Multi-linear interpolation of `c` to `offset`.

    Args:
      c: quantitity to be interpolated.
      offset: offset to which we will interpolate `c`. Must have the same length
        as `c.offset`.
      v: velocity field. Not used.
      dt: size of the time step. Not used.

    Returns:
      An `GridArray` containing the values of `c` after linear interpolation
      to `offset`. The returned value will have offset equal to `offset`.
    """
    del v, dt  # unused
    if len(offset) != len(c.offset):
        raise ValueError(
            "`c.offset` and `offset` must have the same length;"
            f"got {c.offset} and {offset}."
        )
    interpolated = c
    for a, o in enumerate(offset):
        interpolated = _linear_along_axis(interpolated, offset=o, dim=a)
    return interpolated


@typing.overload
def gradient_tensor(v: GridVariable) -> GridTensor: ...


@typing.overload
def gradient_tensor(v: Sequence[GridVariable]) -> GridTensor: ...


@typing.overload
def gradient_tensor(v: GridVariableVector) -> GridTensor: ...


def gradient_tensor(v):
    """Approximates the cell-centered gradient of `v`."""
    if not isinstance(v, GridVariable):
        return GridTensor(torch.stack([gradient_tensor(u) for u in v], dim=-1))
    grad = []
    for axis in range(v.grid.ndim):
        offset = v.offset[axis]
        if offset == 0:
            derivative = forward_difference(v, axis)
        elif offset == 1:
            derivative = backward_difference(v, axis)
        elif offset == 0.5:
            v_centered = linear(v, v.grid.cell_center)
            derivative = central_difference(v_centered, axis)
        else:
            raise ValueError(f"expected offset values in {{0, 0.5, 1}}, got {offset}")
        grad.append(derivative)
    return GridTensor(torch.stack(grad, dim=-1))


def curl_2d(v: GridVariableVector) -> GridVariable:
    """Approximates the curl of `v` in 2D using forward differences."""
    if len(v) != 2:
        raise ValueError(f"Length of `v` is not 2: {len(v)}")
    grid = grids.consistent_grid_arrays(*v)
    if grid.ndim != 2:
        raise ValueError(f"Grid dimensionality is not 2: {grid.ndim}")
    return forward_difference(v[1], dim=0) - forward_difference(v[0], dim=1)
