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

Finite difference methods operate on GridVariable and return GridArray.
Evaluating finite differences requires boundary conditions, which are defined
for a GridVariable. But the operation of taking a derivative makes the boundary
condition undefined, which is why the return type is GridArray.

For example, if the variable c has the boundary condition c_b = 0, and we take
the derivate dc/dx, then it is unclear what the boundary condition on dc/dx
should be. So programmatically, after taking finite differences and doing
operations, the user has to explicitly assign boundary conditions to the result.

Example:
  c = GridVariable(c_array, c_boundary_condition)
  dcdx = finite_differences.forward_difference(c)  # returns GridArray
  c_new = c + dt * (-velocity * dcdx)  # operations on GridArrays
  c_new = GridVariable(c_new, c_boundary_condition)  # assocaite BCs
"""

import typing
from typing import Optional, Sequence, Tuple
from . import grids
import numpy as np
import torch

Array = torch.Tensor
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridArrayTensor = grids.GridArrayTensor
GridVariableVector = grids.GridVariableVector


def stencil_sum(*arrays: GridArray) -> GridArray:
    """Sum arrays across a stencil, with an averaged offset."""
    offset = grids.averaged_offset(*arrays)
    # pytype appears to have a bad type signature for sum():
    # Built-in function sum was called with the wrong arguments [wrong-arg-types]
    #          Expected: (iterable: Iterable[complex])
    #   Actually passed: (iterable: Generator[Union[jax.interpreters.xla.DeviceArray, numpy.ndarray], Any, None])
    result = sum(array.data for array in arrays)  # type: ignore
    grid = grids.consistent_grid(*arrays)
    return grids.GridArray(result, offset, grid)


@typing.overload
def forward_difference(u: GridVariable, axis: int) -> GridArray: ...


@typing.overload
def forward_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]] = ...
) -> Tuple[GridArray, ...]: ...


def forward_difference(u, axis=None):
    """Approximates grads with finite differences in the forward direction."""
    if axis is None:
        axis = range(u.grid.ndim)
    if not isinstance(axis, int):
        return tuple(
            forward_difference(u, a) for a in axis
        )  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    diff = stencil_sum(u.shift(+1, axis), -u.array)
    return diff / u.grid.step[axis]


@typing.overload
def central_difference(u: GridVariable, axis: int) -> GridArray: ...


@typing.overload
def central_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]]
) -> Tuple[GridArray, ...]: ...


def central_difference(u, axis=None):
    """Approximates grads with central differences."""
    if axis is None:
        axis = range(u.grid.ndim)
    if not isinstance(axis, int):
        return tuple(central_difference(u, a) for a in axis)
    diff = stencil_sum(u.shift(+1, axis), -u.shift(-1, axis))
    return diff / (2 * u.grid.step[axis])


@typing.overload
def backward_difference(u: GridVariable, axis: int) -> GridArray: ...


@typing.overload
def backward_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]]
) -> Tuple[GridArray, ...]: ...


def backward_difference(u, axis=None):
    """Approximates grads with finite differences in the backward direction."""
    if axis is None:
        axis = range(u.grid.ndim)
    if not isinstance(axis, int):
        return tuple(backward_difference(u, a) for a in axis)
    diff = stencil_sum(u, -u.shift(-1, axis))
    return diff / u.grid.step[axis]


def divergence(v: Sequence[GridVariable]) -> GridArray:
    """Approximates the divergence of `v` using backward differences."""
    grid = grids.consistent_grid(*v)
    if len(v) != grid.ndim:
        raise ValueError(
            "The length of `v` must be equal to `grid.ndim`."
            f"Expected length {grid.ndim}; got {len(v)}."
        )
    differences = [backward_difference(u, axis) for axis, u in enumerate(v)]
    return sum(differences)


def centered_divergence(v: Sequence[GridVariable]) -> GridArray:
    """Approximates the divergence of `v` using centered differences."""
    grid = grids.consistent_grid(*v)
    if len(v) != grid.ndim:
        raise ValueError(
            "The length of `v` must be equal to `grid.ndim`."
            f"Expected length {grid.ndim}; got {len(v)}."
        )
    differences = [central_difference(u, axis) for axis, u in enumerate(v)]
    return sum(differences)


def laplacian_matrix(n: int, step: float) -> Array:
    """
    Create 1D Laplacian operator matrix, with periodic BC.
    modified the scipy.linalg.circulant implementation to native torch
    """
    column = torch.zeros(n)
    column[0] = -2 / step**2
    column[1] = column[-1] = 1 / step**2
    idx = (n - torch.arange(n)[None].T + torch.arange(n)[None]) % n
    return torch.gather(column[None, ...].expand(n, -1), 1, idx)


def _linear_along_axis(c: GridVariable, offset: float, axis: int) -> GridVariable:
    """Linear interpolation of `c` to `offset` along a single specified `axis`."""
    offset_delta = offset - c.offset[axis]

    # If offsets are the same, `c` is unchanged.
    if offset_delta == 0:
        return c

    new_offset = tuple(offset if j == axis else o for j, o in enumerate(c.offset))

    # If offsets differ by an integer, we can just shift `c`.
    if int(offset_delta) == offset_delta:
        return grids.GridVariable(
            array=grids.GridArray(
                data=c.shift(int(offset_delta), axis).data,
                offset=new_offset,
                grid=c.grid,
            ),
            bc=c.bc,
        )

    floor = int(np.floor(offset_delta))
    ceil = int(np.ceil(offset_delta))
    floor_weight = ceil - offset_delta
    ceil_weight = 1.0 - floor_weight
    data = (
        floor_weight * c.shift(floor, axis).data
        + ceil_weight * c.shift(ceil, axis).data
    )
    return grids.GridVariable(array=grids.GridArray(data, new_offset, c.grid), bc=c.bc)


def linear(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None,
) -> grids.GridVariable:
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
        interpolated = _linear_along_axis(interpolated, offset=o, axis=a)
    return interpolated


@typing.overload
def gradient_tensor(v: GridVariable) -> GridArrayTensor: ...


@typing.overload
def gradient_tensor(v: Sequence[GridVariable]) -> GridArrayTensor: ...


def gradient_tensor(v):
    """Approximates the cell-centered gradient of `v`."""
    if not isinstance(v, GridVariable):
        return GridArrayTensor(np.stack([gradient_tensor(u) for u in v], axis=-1))
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
    return GridArrayTensor(grad)


def curl_2d(v: Sequence[GridVariable]) -> GridArray:
    """Approximates the curl of `v` in 2D using forward differences."""
    if len(v) != 2:
        raise ValueError(f"Length of `v` is not 2: {len(v)}")
    grid = grids.consistent_grid(*v)
    if grid.ndim != 2:
        raise ValueError(f"Grid dimensionality is not 2: {grid.ndim}")
    return forward_difference(v[1], axis=0) - forward_difference(v[0], axis=1)
