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

from __future__ import annotations

import dataclasses
import math
import numbers
import operator
from functools import partial, reduce

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils._pytree import register_pytree_node
except:
    from torch.utils._pytree import _register_pytree_node as register_pytree_node

from .tensor_utils import split_along_axis

Array = torch.Tensor


class BCType:
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
    """
    port from jax_cfd.base.grids.Grid to pytorch
    Describes the size and shape for an Arakawa C-Grid.

    See https://en.wikipedia.org/wiki/Arakawa_grids.

    This class describes domains that can be written as an outer-product of 1D
    grids. Along each dimension `i`:
    - `shape[i]` gives the whole number of grid cells on a single device.
    - `step[i]` is the width of each grid cell.
    - `(lower, upper) = domain[i]` gives the locations of lower and upper
      boundaries. The identity `upper - lower = step[i] * shape[i]` is enforced.
    """

    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    domain: Tuple[Tuple[float, float], ...]

    def __init__(
        self,
        shape: Sequence[int],
        step: Optional[Union[float, Sequence[float]]] = None,
        domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
        device: Optional[torch.device] = "cpu",
    ):
        super().__init__()
        """Construct a grid object."""
        shape = tuple(operator.index(s) for s in shape)
        object.__setattr__(self, "shape", shape)

        if step is not None and domain is not None:
            raise TypeError("cannot provide both step and domain")
        elif domain is not None:
            if isinstance(domain, (int, float)):
                domain = ((0, domain),) * len(shape)
            else:
                if len(domain) != self.ndim:
                    raise ValueError(
                        "length of domain does not match ndim: "
                        f"{len(domain)} != {self.ndim}"
                    )
                for bounds in domain:
                    if len(bounds) != 2:
                        raise ValueError(
                            f"domain is not sequence of pairs of numbers: {domain}"
                        )
            domain = tuple((float(lower), float(upper)) for lower, upper in domain)
        else:
            if step is None:
                step = 1
            if isinstance(step, numbers.Number):
                step = (step,) * self.ndim
            elif len(step) != self.ndim:
                raise ValueError(
                    "length of step does not match ndim: " f"{len(step)} != {self.ndim}"
                )
            domain = tuple(
                (0.0, float(step_ * size)) for step_, size in zip(step, shape)
            )

        object.__setattr__(self, "domain", domain)

        step = tuple(
            (upper - lower) / size for (lower, upper), size in zip(domain, shape)
        )
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "device", device)

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of this grid."""
        return len(self.shape)

    @property
    def cell_center(self) -> Tuple[float, ...]:
        """Offset at the center of each grid cell."""
        return self.ndim * (0.5,)

    @property
    def cell_faces(self) -> Tuple[Tuple[float, ...]]:
        """Returns the offsets at each of the 'forward' cell faces."""
        d = self.ndim
        offsets = (torch.eye(d) + torch.ones([d, d])) / 2.0
        return tuple(tuple(float(o) for o in offset) for offset in offsets)

    def stagger(self, v: Tuple[Array, ...]) -> Tuple[Array, ...]:
        """Places the velocity components of `v` on the `Grid`'s cell faces."""
        offsets = self.cell_faces
        return tuple(GridArray(u, o, self) for u, o in zip(v, offsets))

    def center(self, v: Tuple[Array, ...]) -> Tuple[Array, ...]:
        """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
        offset = self.cell_center
        return lambda u: GridArray(u, offset, self), v

    def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
        """Returns a tuple of arrays containing the grid points along each axis.

        Args:
          offset: an optional sequence of length `ndim`. The grid will be shifted by
            `offset * self.step`.

        Returns:
          An tuple of `self.ndim` arrays. The jth return value has shape
          `[self.shape[j]]`.
        """
        if offset is None:
            offset = self.cell_center
        if len(offset) != self.ndim:
            raise ValueError(
                f"unexpected offset length: {len(offset)} vs " f"{self.ndim}"
            )
        return tuple(
            lower + (torch.arange(length) + offset_i) * step
            for (lower, _), offset_i, length, step in zip(
                self.domain, offset, self.shape, self.step
            )
        )

    def fft_axes(self) -> Tuple[Array, ...]:
        """Returns the ordinal frequencies corresponding to the axes.

        Transforms each axis into the *ordinal* frequencies for the Fast Fourier
        Transform (FFT). Multiply by `2 * jnp.pi` to get angular frequencies.

        Returns:
          A tuple of `self.ndim` arrays. The jth return value has shape
          `[self.shape[j]]`.
        """
        freq_axes = tuple(fft.fftfreq(n, d=s) for (n, s) in zip(self.shape, self.step))
        return freq_axes

    def mesh(
        self,
        offset: Optional[Sequence[float]] = None,
    ) -> Tuple[Array, ...]:
        """Returns an tuple of arrays containing positions in each grid cell.

        Args:
          offset: an optional sequence of length `ndim`. The grid will be shifted by
            `offset * self.step`.

        Returns:
          An tuple of `self.ndim` arrays, each of shape `self.shape`. In 3
          dimensions, entry `self.mesh[n][i, j, k]` is the location of point
          `i, j, k` in dimension `n`.
        """
        axes = self.axes(offset)
        x, y = torch.meshgrid(*axes, indexing="ij")
        return x.to(self.device), y.to(self.device)

    def fft_mesh(self) -> Tuple[Array, ...]:
        """Returns a tuple of arrays containing positions in Fourier space."""
        fft_axes = self.fft_axes()
        kx, ky = torch.meshgrid(*fft_axes, indexing="ij")
        return kx.to(self.device), ky.to(self.device)

    def rfft_mesh(self) -> Tuple[Array, ...]:
        """Returns a tuple of arrays containing positions in rfft space."""
        fft_mesh = self.fft_mesh()
        k_max = math.floor(self.shape[-1] / 2.0)
        return tuple(fmesh[..., : k_max + 1] for fmesh in fft_mesh)

    def eval_on_mesh(
        self, fn: Callable[..., Array], offset: Optional[Sequence[float]] = None
    ) -> Array:
        """Evaluates the function on the grid mesh with the specified offset.

        Args:
          fn: A function that accepts the mesh arrays and returns an array.
          offset: an optional sequence of length `ndim`.  If not specified, uses the
            offset for the cell center.

        Returns:
          fn(x, y, ...) evaluated on the mesh, as a GridArray with specified offset.
        """
        if offset is None:
            offset = self.cell_center
        return GridArray(fn(*self.mesh(offset)), offset, self)


@dataclasses.dataclass
class GridArray(torch.Tensor):
    """
    the original jax implentation uses np.lib.mixins.NDArrayOperatorsMixin
    and the __array_ufunc__ method to implement arithmetic operations
    here it is modified to use torch.Tensor as the base class
    and __torch_function__ to do various things like clone() and to()
    reference: https://pytorch.org/docs/stable/notes/extending.html

    Data with an alignment offset and an associated grid.

    Offset values in the range [0, 1] fall within a single grid cell.

    Examples:
      offset=(0, 0) means that each point is at the bottom-left corner.
      offset=(0.5, 0.5) is at the grid center.
      offset=(1, 0.5) is centered on the right-side edge.

    Attributes:
      data: array values.
      offset: alignment location of the data with respect to the grid.
      grid: the Grid associated with the array data.
      dtype: type of the array data.
      shape: lengths of the array dimensions.
    """

    # Don't (yet) enforce any explicit consistency requirements between data.ndim
    # and len(offset), e.g., so we can feel to add extra time/batch/channel
    # dimensions. But in most cases they should probably match.
    # Also don't enforce explicit consistency between data.shape and grid.shape,
    # but similarly they should probably match.

    data: Array = None
    offset: Tuple[float, ...] = None
    grid: Grid = None

    def tree_flatten(self):
        """Returns flattening recipe for GridArray JAX pytree."""
        children = (self.data,)
        aux_data = (self.offset, self.grid)
        return children, aux_data

    @staticmethod
    def __new__(cls, x, offset, grid, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def clone(self, *args, **kwargs):
        return super().clone(*args, **kwargs)

    def to(self, *args, **kwargs):
        return GridArray(self.data.to(*args, **kwargs), self.offset, self.grid)

    _HANDLED_TYPES = (numbers.Number, Array)

    @classmethod
    def __torch_function__(self, ufunc, types, args=(), kwargs=None):
        """Define arithmetic on GridArrays using NumPy's mixin."""
        if kwargs is None:
            kwargs = {}
        if not all(issubclass(t, self._HANDLED_TYPES + (GridArray,)) for t in types):
            return NotImplemented
        try:
            # get the corresponding torch function to the NumPy ufunc
            func = getattr(torch, ufunc.__name__)
        except AttributeError:
            return NotImplemented
        arrays = [x.data if isinstance(x, GridArray) else x for x in args]
        result = func(*arrays)
        offset = consistent_offset(*[x for x in args if isinstance(x, GridArray)])
        grid = consistent_grid(*[x for x in args if isinstance(x, GridArray)])
        if isinstance(result, tuple):
            return tuple(GridArray(r, offset, grid) for r in result)
        else:
            return GridArray(result, offset, grid)


GridArrayVector = Tuple[GridArray, ...]


@dataclasses.dataclass
class GridVariable:
    """Associates a GridArray with BoundaryConditions.

    Performing pad and shift operations, e.g. for finite difference calculations,
    requires boundary condition (BC) information. Since different variables in a
    PDE system can have different BCs, this class associates a specific variable's
    data with its BCs.

    Array operations on GridVariables act like array operations on the
    encapsulated GridArray.

    Attributes:
      array: GridArray with the array data, offset, and associated grid.
      bc: boundary conditions for this variable.
      grid: the Grid associated with the array data.
      dtype: type of the array data.
      shape: lengths of the array dimensions.
      data: array values.
      offset: alignment location of the data with respect to the grid.
      grid: the Grid associated with the array data.
    """

    array: GridArray
    bc: BoundaryConditions

    def __post_init__(self):
        if not isinstance(self.array, GridArray):  # frequently missed by pytype
            raise ValueError(
                f"Expected array type to be GridArray, got {type(self.array)}"
            )
        if len(self.bc.types) != self.grid.ndim:
            raise ValueError(
                "Incompatible dimension between grid and bc, grid dimension = "
                f"{self.grid.ndim}, bc dimension = {len(self.bc.types)}"
            )

    def tree_flatten(self):
        """Returns flattening recipe for GridVariable JAX pytree."""
        children = (self.array,)
        aux_data = (self.bc,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Returns unflattening recipe for GridVariable JAX pytree."""
        return cls(*children, *aux_data)

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def data(self) -> Array:
        return self.array.data

    @property
    def offset(self) -> Tuple[float, ...]:
        return self.array.offset

    @property
    def grid(self) -> Grid:
        return self.array.grid

    def shift(
        self,
        offset: int,
        dim: int,
    ) -> GridArray:
        """Shift this GridVariable by `offset`.

        Args:
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of the encapsulated GridArray, shifted by `offset`. The returned
          GridArray has offset `u.offset + offset`.
        """
        return self.bc.shift(self.array, offset, dim)

    def _interior_grid(self) -> Grid:
        """Returns only the interior grid points."""
        grid = self.array.grid
        domain = list(grid.domain)
        shape = list(grid.shape)
        for axis in range(self.grid.ndim):
            # nothing happens in periodic case
            if self.bc.types[axis][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            # this will automatically set the grid to interior.
            if torch.isclose(self.array.offset[axis], 1.0):
                shape[axis] -= 1
                domain[axis] = (domain[axis][0], domain[axis][1] - grid.step[axis])
            elif torch.isclose(self.array.offset[axis], 0.0):
                shape[axis] -= 1
                domain[axis] = (domain[axis][0] + grid.step[axis], domain[axis][1])
        return Grid(shape, domain=tuple(domain))

    def _interior_array(self) -> Array:
        """Returns only the interior points of self.array."""
        data = self.array.data
        for axis in range(self.grid.ndim):
            # nothing happens in periodic case
            if self.bc.types[axis][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            if torch.isclose(self.offset[axis], 1.0):
                data, _ = split_along_axis(data, -1, axis)
            elif torch.isclose(self.offset[axis], 0.0):
                _, data = split_along_axis(data, 1, axis)

        return data

    def interior(self) -> GridArray:
        """Returns a GridArray associated only with interior points.

         Interior is defined as the following:register
           for d in range(u.grid.ndim):
            points = u.grid.axes(offset=u.offset[d])
            interior_points =
              all points where grid.domain[d][0] < points < grid.domain[d][1]

        The exception is when the boundary conditions are periodic,
        in which case all points are included in the interior.

        In case of dirichlet with edge offset, the grid and array size is reduced,
        since one scalar lies exactly on the boundary. In all other cases,
        self.grid and self.array are returned.
        """
        interior_array = self._interior_array()
        interior_grid = self._interior_grid()
        return GridArray(interior_array, self.array.offset, interior_grid)

    def enforce_edge_bc(self, *args) -> GridVariable:
        """Returns the GridVariable with edge BC enforced, if applicable.

        For GridVariables having nonperiodic BC and offset 0 or 1, there are values
        in the array data that are dependent on the boundary condition.
        enforce_edge_bc() changes these boundary values to match the prescribed BC.

        Args:
          *args: any optional values passed into BoundaryConditions values method.
        """
        if self.grid.shape != self.array.data.shape:
            raise ValueError("Stored array and grid have mismatched sizes.")
        data = torch.tensor(self.array.data)
        for axis in range(self.grid.ndim):
            if "periodic" not in self.bc.types[axis]:
                values = self.bc.values(axis, self.grid, *args)
                for boundary_side in range(2):
                    if torch.isclose(self.array.offset[axis], boundary_side):
                        # boundary data is set to match self.bc:
                        all_slice = [
                            slice(None, None, None),
                        ] * self.grid.ndim
                        all_slice[axis] = -boundary_side
                        data = data.at[tuple(all_slice)].set(values[boundary_side])
        return GridVariable(
            array=GridArray(data, self.array.offset, self.grid), bc=self.bc
        )


GridVariableVector = Tuple[GridVariable, ...]


class GridArrayTensor(Array):
    """A numpy array of GridArrays, representing a physical tensor field.

    Packing tensor coordinates into a numpy array of dtype object is useful
    because pointwise matrix operations like trace, transpose, and matrix
    multiplications of physical tensor quantities is meaningful.

    Example usage:
      grad = fd.gradient_tensor(uv)                    # a rank 2 Tensor
      strain_rate = (grad + grad.T) / 2.
      nu_smag = np.sqrt(np.trace(strain_rate.dot(strain_rate)))
      nu_smag = Tensor(nu_smag)                        # a rank 0 Tensor
      subgrid_stress = -2 * nu_smag * strain_rate      # a rank 2 Tensor
    """

    def __new__(cls, arrays):
        return torch.asarray(arrays).view(cls)


register_pytree_node(
    GridArrayTensor,
    lambda tensor: (tensor.ravel().tolist(), tensor.shape),
    lambda shape, arrays: GridArrayTensor(torch.asarray(arrays).reshape(shape)),
)


def applied(func):
    """Convert an array function into one defined on GridArrays.

    Since `func` can only act on `data` attribute of GridArray, it implicitly
    enforces that `func` cannot modify the other attributes such as offset.

    Args:
      func: function being wrapped.

    Returns:
      A wrapped version of `func` that takes GridArray instead of Array args.
    """

    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, GridVariable):
                raise ValueError("grids.applied() cannot be used with GridVariable")

        offset = consistent_offset(
            *[
                arg
                for arg in args + tuple(kwargs.values())
                if isinstance(arg, GridArray)
            ]
        )
        grid = consistent_grid(
            *[
                arg
                for arg in args + tuple(kwargs.values())
                if isinstance(arg, GridArray)
            ]
        )
        raw_args = [arg.data if isinstance(arg, GridArray) else arg for arg in args]
        raw_kwargs = {
            k: v.data if isinstance(v, GridArray) else v for k, v in kwargs.items()
        }
        data = func(*raw_args, **raw_kwargs)
        return GridArray(data, offset, grid)

    return wrapper


def averaged_offset(*arrays: Union[GridArray, GridVariable]) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    offsets = torch.as_tensor([array.offset for array in arrays])
    offset = torch.mean(offsets, dim=0)
    return tuple(offset.tolist())


def consistent_offset(*arrays: Array) -> Tuple[float, ...]:
    """Returns the unique offset, or raises InconsistentOffsetError."""
    offsets = {array.offset for array in arrays}
    if len(offsets) != 1:
        raise Exception(f"arrays do not have a unique offset: {offsets}")
    (offset,) = offsets
    return offset


def consistent_grid(*arrays: GridArray):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids) != 1:
        raise Exception(f"arrays do not have a unique grid: {grids}")
    (grid,) = grids
    return grid


@dataclasses.dataclass(init=False, frozen=True)
class BoundaryConditions:
    """Base class for boundary conditions on a PDE variable.

    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    types: Tuple[Tuple[str, str], ...]

    def shift(
        self,
        u: GridArray,
        offset: int,
        dim: int,
    ) -> GridArray:
        """Shift an GridArray by `offset`.

        Args:
          u: an `GridArray` object.
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of `u`, shifted by `offset`. The returned `GridArray` has offset
          `u.offset + offset`.
        """
        raise NotImplementedError(
            "shift() not implemented in BoundaryConditions base class."
        )

    def values(
        self,
        dim: int,
        grid: Grid,
        offset: Optional[Tuple[float, ...]],
        time: Optional[float],
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """Returns Arrays specifying boundary values on the grid along axis.

        Args:
          dim: axis along which to return boundary values.
          grid: a `Grid` object on which to evaluate boundary conditions.
          offset: a Tuple of offsets that specifies (along with grid) where to
            evaluate boundary conditions in space.
          time: a float used as an input to boundary function.

        Returns:
          A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
          boundary. In case of periodic boundaries, returns a tuple(None,None).
        """
        raise NotImplementedError(
            "values() not implemented in BoundaryConditions base class."
        )


@dataclasses.dataclass(init=False, frozen=True)
class ConstantBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a PDE variable that are constant in space and time.

    Example usage:
      grid = Grid((10, 10))
      array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)),
                                          ((0.0, 10.0),(1.0, 0.0)))
      u = GridVariable(array, bc)

    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    types: Tuple[Tuple[str, str], ...]
    _values: Tuple[Tuple[Optional[float], Optional[float]], ...]

    def __init__(
        self,
        types: Sequence[Tuple[str, str]],
        values: Sequence[Tuple[Optional[float], Optional[float]]],
    ):
        types = tuple(types)
        values = tuple(values)
        object.__setattr__(self, "types", types)
        object.__setattr__(self, "_values", values)

    def shift(
        self,
        u: GridArray,
        offset: int,
        dim: int,
    ) -> GridArray:
        """Shift an GridArray by `offset`.

        Args:
          u: an `GridArray` object.
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of `u`, shifted by `offset`. The returned `GridArray` has offset
          `u.offset + offset`.
        """
        padded = self._pad(u, offset, dim)
        trimmed = self._trim(padded, -offset, dim)
        # print(u.shape, offset)
        # print(padded.shape, trimmed.shape)
        return trimmed

    def _is_aligned(self, u: GridArray, dim: int) -> bool:
        """Checks if array u contains all interior domain information.

        For dirichlet edge aligned boundary, the value that lies exactly on the
        boundary does not have to be specified by u.
        Neumann edge aligned boundary is not defined.

        Args:
        u: array that should contain interior data
        dim: axis along which to check

        Returns:
        True if u is aligned, and raises error otherwise.
        """
        size_diff = u.shape[dim] - u.grid.shape[dim]
        if self.types[dim][0] == BCType.DIRICHLET and np.isclose(u.offset[dim], 1):
            size_diff += 1
        if self.types[dim][1] == BCType.DIRICHLET and np.isclose(u.offset[dim], 1):
            size_diff += 1
        if self.types[dim][0] == BCType.NEUMANN and np.isclose(u.offset[dim] % 1, 0):
            raise NotImplementedError("Edge-aligned neumann BC are not implemented.")
        if size_diff < 0:
            raise ValueError("the GridArray does not contain all interior grid values.")
        return True

    def _pad(
        self,
        u: GridArray,
        width: int,
        dim: int,
    ) -> GridArray:
        """Pad a GridArray by `padding`.

        Important: _pad makes no sense past 1 ghost cell for nonperiodic
        boundaries. This is sufficient for jax_cfd finite difference code.

        Args:
          u: a `GridArray` object.
          width: number of elements to pad along axis. Use negative value for lower
            boundary or positive value for upper boundary.
          dim: axis to pad along.

        Returns:
          Padded array, elongated along the indicated axis.
        """
        if width < 0:  # pad lower boundary
            bc_type = self.types[dim][0]
            padding = (-width, 0)
        else:  # pad upper boundary
            bc_type = self.types[dim][1]
            padding = (0, width)

        full_padding = [(0, 0)] * u.grid.ndim
        full_padding[dim] = padding

        offset = list(u.offset)
        offset[dim] -= padding[0]

        if bc_type != BCType.PERIODIC and abs(width) > 1:
            raise ValueError(
                "Padding past 1 ghost cell is not defined in nonperiodic case."
            )

        if bc_type == BCType.PERIODIC:
            # self.values are ignored here
            pad_kwargs = dict(mode="circular")
        elif bc_type == BCType.DIRICHLET:
            if np.isclose(u.offset[dim] % 1, 0.5):  # cell center
                # make the linearly interpolated value equal to the boundary by setting
                # the padded values to the negative symmetric values
                data = 2 * expand_dims_pad(
                    u.data, full_padding, mode="constant", constant_values=self._values
                ) - expand_dims_pad(u.data, full_padding, mode="reflect")
                return GridArray(data, tuple(offset), u.grid)
            elif np.isclose(u.offset[dim] % 1, 0):  # cell edge
                pad_kwargs = dict(mode="constant", constant_values=self._values)
            else:
                raise ValueError(
                    "expected offset to be an edge or cell center, got "
                    f"offset[axis]={u.offset[dim]}"
                )
        elif bc_type == BCType.NEUMANN:
            if not (
                np.isclose(u.offset[dim] % 1, 0) or np.isclose(u.offset[dim] % 1, 0.5)
            ):
                raise ValueError(
                    "expected offset to be an edge or cell center, got "
                    f"offset[axis]={u.offset[dim]}"
                )
            else:
                # When the data is cell-centered, computes the backward difference.
                # When the data is on cell edges, boundary is set such that
                # (u_last_interior - u_boundary)/grid_step = neumann_bc_value.
                data = expand_dims_pad(
                    u.data, full_padding, mode="replicate"
                ) + u.grid.step[dim] * (
                    expand_dims_pad(u.data, full_padding, mode="constant")
                    - expand_dims_pad(
                        u.data,
                        full_padding,
                        mode="constant",
                        constant_values=self._values,
                    )
                )
                return GridArray(data, tuple(offset), u.grid)

        else:
            raise ValueError("invalid boundary type")
        data = expand_dims_pad(u.data, full_padding, **pad_kwargs)
        return GridArray(data, tuple(offset), u.grid)

    def _trim(
        self,
        u: GridArray,
        width: int,
        dim: int,
    ) -> GridArray:
        """Trim padding from a GridArray.

        Args:
          u: a `GridArray` object.
          width: number of elements to trim along axis. Use negative value for lower
            boundary or positive value for upper boundary.
          dim: axis to trim along.

        Returns:
          Trimmed array, shrunk along the indicated axis.
        """
        if width < 0:  # trim lower boundary
            padding = (-width, 0)
        else:  # trim upper boundary
            padding = (0, width)

        limit_index = u.data.shape[dim] - padding[1]
        data = u.data.index_select(
            dim=dim, index=torch.arange(padding[0], limit_index, device=u.data.device)
        )
        offset = list(u.offset)
        offset[dim] += padding[0]
        return GridArray(data, tuple(offset), u.grid)

    def values(self, dim: int, grid: Grid) -> Tuple[Optional[Array], Optional[Array]]:
        """Returns boundary values on the grid along axis.

        Args:
          dim: axis along which to return boundary values.
          grid: a `Grid` object on which to evaluate boundary conditions.

        Returns:
          A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
          boundary. In case of periodic boundaries, returns a tuple(None,None).
        """
        if None in self._values[dim]:
            return (None, None)
        bc = tuple(
            torch.full(grid.shape[:dim] + grid.shape[dim + 1 :], self._values[dim][-i])
            for i in [0, 1]
        )
        return bc

    def _trim_padding(self, u: GridArray, dim: int = 0, trim_side: str = "both"):
        """Trims padding from a GridArray along axis and returns the array interior.

        Args:
        u: a `GridArray` object.
        dim: axis to trim along.
        trim_side: if 'both', trims both sides. If 'right', trims the right side.
            If 'left', the left side.

        Returns:
        Trimmed array, shrunk along the indicated axis side.
        """
        padding = (0, 0)
        if u.shape[dim] >= u.grid.shape[dim]:
            # number of cells that were padded on the left
            negative_trim = 0
            if u.offset[dim] <= 0 and (trim_side == "both" or trim_side == "left"):
                negative_trim = -math.ceil(-u.offset[dim])
                # periodic is a special case. Shifted data might still contain all the
                # information.
                if self.types[dim][0] == BCType.PERIODIC:
                    negative_trim = max(negative_trim, u.grid.shape[dim] - u.shape[dim])
                # for both DIRICHLET and NEUMANN cases the value on grid.domain[0] is
                # a dependent value.
                elif np.isclose(u.offset[dim] % 1, 0):
                    negative_trim -= 1
                u = self._trim(u, negative_trim, dim)
            # number of cells that were padded on the right
            positive_trim = 0
            if trim_side == "right" or trim_side == "both":
                # periodic is a special case. Boundary on one side depends on the other
                # side.
                if self.types[dim][1] == BCType.PERIODIC:
                    positive_trim = max(u.shape[dim] - u.grid.shape[dim], 0)
                else:
                    # for other cases, where to trim depends only on the boundary type
                    # and data offset.
                    last_u_offset = u.shape[dim] + u.offset[dim] - 1
                    boundary_offset = u.grid.shape[dim]
                    if last_u_offset >= boundary_offset:
                        positive_trim = math.ceil(last_u_offset - boundary_offset)
                        if self.types[dim][1] == BCType.DIRICHLET and np.isclose(
                            u.offset[dim] % 1, 0
                        ):
                            positive_trim += 1
        if positive_trim > 0:
            u = self._trim(u, positive_trim, dim)
        # combining existing padding with new padding
        padding = (-negative_trim, positive_trim)
        return u, padding

    def trim_boundary(self, u: GridArray) -> GridArray:
        """Returns GridArray without the grid points on the boundary.

        Some grid points of GridArray might coincide with boundary. This trims those
        values. If the array was padded beforehand, removes the padding.

        Args:
        u: a `GridArray` object.

        Returns:
        A GridArray shrunk along certain dimensions.
        """
        for axis in range(u.grid.ndim):
            _ = self._is_aligned(u, axis)
            u, _ = self._trim_padding(u, axis)
        return u

    def pad_and_impose_bc(
        self,
        u: GridArray,
        offset_to_pad_to: Optional[Tuple[float, ...]] = None,
        mode: Optional[str] = "extend",
    ) -> GridVariable:
        """Returns GridVariable with correct boundary values.

        Some grid points of GridArray might coincide with boundary. This ensures
        that the GridVariable.array agrees with GridVariable.bc.
        Args:
        u: a `GridArray` object that specifies only scalar values on the internal
            nodes.
        offset_to_pad_to: a Tuple of desired offset to pad to. Note that if the
            function is given just an interior array in dirichlet case, it can pad
            to both 0 offset and 1 offset.
        mode: type of padding to use in non-periodic case.
            Mirror mirrors the flow across the boundary.
            Extend extends the last well-defined value past the boundary.

        Returns:
        A GridVariable that has correct boundary values.
        """
        if offset_to_pad_to is None:
            offset_to_pad_to = u.offset
            for axis in range(u.grid.ndim):
                _ = self._is_aligned(u, axis)
                if self.types[axis][0] == BCType.DIRICHLET and np.isclose(
                    u.offset[axis], 1.0
                ):
                    if np.isclose(offset_to_pad_to[axis], 1.0):
                        u = self._pad(u, 1, axis, mode=mode)
                    elif np.isclose(offset_to_pad_to[axis], 0.0):
                        u = self._pad(u, -1, axis, mode=mode)
        return GridVariable(u, self)

    def impose_bc(self, u: GridArray) -> GridVariable:
        """Returns GridVariable with correct boundary condition.

        Some grid points of GridArray might coincide with boundary. This ensures
        that the GridVariable.array agrees with GridVariable.bc.
        Args:
        u: a `GridArray` object.

        Returns:
        A GridVariable that has correct boundary values and is restricted to the
        domain.
        """
        offset = u.offset
        u = self.trim_boundary(u)
        return self.pad_and_impose_bc(u, offset)

    trim = _trim
    pad = _pad


class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
    """Boundary conditions for a PDE variable.

    Example usage:
      grid = Grid((10, 10))
      array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)))
      u = GridVariable(array, bc)

    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    def __init__(self, types: Sequence[Tuple[str, str]]):

        ndim = len(types)
        values = ((0.0, 0.0),) * ndim
        super(HomogeneousBoundaryConditions, self).__init__(types, values)


def is_periodic_boundary_conditions(c: GridVariable, dim: int) -> bool:
    """Returns true if scalar has periodic bc along axis."""
    if c.bc.types[dim][0] != BCType.PERIODIC:
        return False
    return True


# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> BoundaryConditions:
    """Returns periodic BCs for a variable with `ndim` spatial dimension."""
    return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def consistent_boundary_conditions(*arrays: GridVariable) -> Tuple[str, ...]:
    """Returns whether BCs are periodic.

    Mixed periodic/nonperiodic boundaries along the same boundary do not make
    sense. The function checks that the boundary is either periodic or not and
    throws an error if its mixed.

    Args:
      *arrays: a list of gridvariables.

    Returns:
      a list of types of boundaries corresponding to each axis if
      they are consistent.
    """
    bc_types = []
    for axis in range(arrays[0].grid.ndim):
        bcs = {is_periodic_boundary_conditions(array, axis) for array in arrays}
        if len(bcs) != 1:
            raise Exception(f"arrays do not have consistent bc: {arrays}")
        elif bcs.pop():
            bc_types.append("periodic")
        else:
            bc_types.append("nonperiodic")
    return tuple(bc_types)


def get_pressure_bc_from_velocity(
    v: GridVariableVector,
) -> HomogeneousBoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity."""
    # assumes that if the boundary is not periodic, pressure BC is zero flux.
    velocity_bc_types = consistent_boundary_conditions(*v)
    pressure_bc_types = []
    for velocity_bc_type in velocity_bc_types:
        if velocity_bc_type == "periodic":
            pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
        else:
            pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
    return HomogeneousBoundaryConditions(pressure_bc_types)


def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
    """Returns True if arrays have periodic BC in every dimension, else False."""
    for array in arrays:
        for axis in range(array.grid.ndim):
            if not is_periodic_boundary_conditions(array, axis):
                return False
    return True


def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable, flux_direction: int
) -> ConstantBoundaryConditions:
    """Returns advection flux boundary conditions for the specified velocity.

    Infers advection flux boundary condition in flux direction
    from scalar c and velocity u in direction flux_direction.
    The flux boundary condition should be used only to compute divergence.
    If the boundaries are periodic, flux is periodic.
    In nonperiodic case, flux boundary parallel to flux direction is
    homogeneous dirichlet.
    In nonperiodic case if flux direction is normal to the wall, the
    function supports 2 cases:
      1) Nonporous boundary, corresponding to homogeneous flux bc.
      2) Pourous boundary with constant flux, corresponding to
        both the velocity and scalar with Homogeneous Neumann bc.

    This function supports only these cases because all other cases result in
    time dependent flux boundary condition.

    Args:
      u: velocity component in flux_direction.
      c: scalar to advect.
      flux_direction: direction of velocity.

    Returns:
      BoundaryCondition instance for advection flux of c in flux_direction.
    """
    # only no penetration and periodic boundaries are supported.
    flux_bc_types = []
    flux_bc_values = []
    if not isinstance(u.bc, HomogeneousBoundaryConditions):
        raise NotImplementedError(
            f"Flux boundary condition is not implemented for velocity with {u.bc}"
        )
    for axis in range(c.grid.ndim):
        if u.bc.types[axis][0] == "periodic":
            flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
            flux_bc_values.append((None, None))
        elif flux_direction != axis:
            # This is not technically correct. Flux boundary condition in most cases
            # is a time dependent function of the current values of the scalar
            # and velocity. However, because flux is used only to take divergence, the
            # boundary condition on the flux along the boundary parallel to the flux
            # direction has no influence on the computed divergence because the
            # boundary condition only alters ghost cells, while divergence is computed
            # on the interior.
            # To simplify the code and allow for flux to be wrapped in gridVariable,
            # we are setting the boundary to homogeneous dirichlet.
            # Note that this will not work if flux is used in any other capacity but
            # to take divergence.
            flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
            flux_bc_values.append((0.0, 0.0))
        else:
            flux_bc_types_ax = []
            flux_bc_values_ax = []
            for i in range(2):  # lower and upper boundary.

                # case 1: nonpourous boundary
                if (
                    u.bc.types[axis][i] == BCType.DIRICHLET
                    and u.bc.bc_values[axis][i] == 0.0
                ):
                    flux_bc_types_ax.append(BCType.DIRICHLET)
                    flux_bc_values_ax.append(0.0)

                # case 2: zero flux boundary
                elif (
                    u.bc.types[axis][i] == BCType.NEUMANN
                    and c.bc.types[axis][i] == BCType.NEUMANN
                ):
                    if not isinstance(c.bc, ConstantBoundaryConditions):
                        raise NotImplementedError(
                            "Flux boundary condition is not implemented for scalar"
                            + f" with {c.bc}"
                        )
                    if not np.isclose(c.bc.bc_values[axis][i], 0.0):
                        raise NotImplementedError(
                            "Flux boundary condition is not implemented for scalar"
                            + f" with {c.bc}"
                        )
                    flux_bc_types_ax.append(BCType.NEUMANN)
                    flux_bc_values_ax.append(0.0)

                # no other case is supported
                else:
                    raise NotImplementedError(
                        f"Flux boundary condition is not implemented for {u.bc, c.bc}"
                    )
            flux_bc_types.append(flux_bc_types_ax)
            flux_bc_values.append(flux_bc_values_ax)
    return ConstantBoundaryConditions(flux_bc_types, flux_bc_values)


def expand_dims_pad(
    inputs: Any,
    pad: Tuple[Tuple[int, int], ...],
    dim: int = 2,
    mode: str = "constant",
    constant_values: float = 0,
) -> Any:
    """
    wrapper for F.pad with a dimension checker
    note: jnp's pad pad_width starts from the first dimension to the last dimension
    while torch's pad pad_width starts from the last dimension to the previous dimension
    example: for torch (1, 1, 2, 2) means padding last dim by (1, 1) and 2nd to last by (2, 2)

    Args:
      inputs: array or a tuple of arrays to pad.
      pad_width: padding width for each dimension.
      mode: padding mode, one of 'constant', 'reflect', 'symmetric'.
      values: constant value to pad with.

    Returns:
      Padded `inputs`.
    """
    assert len(pad) == inputs.ndim, "pad must have the same length as inputs.ndim"
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("inputs must be a torch.Tensor")
    if dim == inputs.ndim:
        inputs = inputs.unsqueeze(0)
    pad = reduce(lambda a, x: x + a, pad, ())  # flatten the pad and reverse the order
    if mode == "constant":
        array = F.pad(inputs, pad, mode=mode, value=constant_values)
    elif mode == "reflect" or mode == "circular":
        array = F.pad(inputs, pad, mode=mode)
    else:
        raise NotImplementedError(f"invalid mode {mode} for torch.nn.functional.pad")

    return array.squeeze(0) if dim != array.ndim else array
