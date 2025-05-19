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
# ported Google's Jax-CFD functional template to torch.Tensor operations

from __future__ import annotations

import dataclasses
import math
import numbers
import operator

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.fft as fft

from torch_cfd import tensor_utils


_HANDLED_TYPES = (numbers.Number, torch.Tensor)


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

    def stagger(self, v: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Places the velocity components of `v` on the `Grid`'s cell faces."""
        offsets = self.cell_faces
        return tuple(GridArray(u, o, self) for u, o in zip(v, offsets))

    def center(self, v: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
        offset = self.cell_center
        return lambda u: GridArray(u, offset, self), v

    def axes(
        self, offset: Optional[Sequence[float]] = None
    ) -> Tuple[torch.Tensor, ...]:
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

    def fft_axes(self) -> Tuple[torch.Tensor, ...]:
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
    ) -> Tuple[torch.Tensor, ...]:
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

    def fft_mesh(self) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing positions in Fourier space."""
        fft_axes = self.fft_axes()
        kx, ky = torch.meshgrid(*fft_axes, indexing="ij")
        return kx.to(self.device), ky.to(self.device)

    def rfft_mesh(self) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing positions in rfft space."""
        fft_mesh = self.fft_mesh()
        k_max = math.floor(self.shape[-1] / 2.0)
        return tuple(fmesh[..., : k_max + 1] for fmesh in fft_mesh)

    def eval_on_mesh(
        self, fn: Callable[..., torch.Tensor], offset: Optional[Sequence[float]] = None
    ) -> torch.Tensor:
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


def _binary_method(name, op):
    """
    Implement a forward binary method with an operator.
    see np.lib.mixins.NDArrayOperatorsMixin

    Notes: because GridArray is a subclass of torch.Tensor, we need to check
    if the other operand is a GridArray first, otherwise, isinstance(other, _HANDLED_TYPES) will return True as well, which is not what we want as
    there will be no offset in the other operand.
    """

    def method(self, other):
        if isinstance(other, GridArray):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            return GridArray(op(self.data, other.data), self.offset, self.grid)
        elif isinstance(other, _HANDLED_TYPES):
            return GridArray(op(self.data, other), self.offset, self.grid)

        return NotImplemented

    method.__name__ = f"__{name}__"
    return method


def _reflected_binary_method(name, op):
    """Implement a reflected binary method with an operator."""

    def method(self, other):
        if isinstance(other, GridArray):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            return GridArray(op(other.data, self.data), self.offset, self.grid)
        elif isinstance(other, _HANDLED_TYPES):
            return GridArray(op(other, self.data), self.offset, self.grid)

        return NotImplemented

    method.__name__ = f"__r{name}__"
    return method


def _inplace_binary_method(name, op):
    """Implement an in-place binary method with an operator."""

    def method(self, other):
        if isinstance(other, GridArray):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            self.data = op(self.data, other.data)
            return self
        elif isinstance(other, _HANDLED_TYPES):
            self.data = op(self.data, other)
            return self

        return NotImplemented

    method.__name__ = f"__i{name}__"
    return method


def _numeric_methods(name, op):
    """Implement forward, reflected and inplace binary methods with an operator."""
    return (
        _binary_method(name, op),
        _reflected_binary_method(name, op),
        _inplace_binary_method(name, op),
    )


def _unary_method(name, op):
    def method(self):
        return GridArray(op(self.data), self.offset, self.grid)

    method.__name__ = f"__i{name}__"
    return method


class GridArrayOperatorsMixin:

    __slots__ = ()

    __lt__ = _binary_method("lt", operator.lt)
    __le__ = _binary_method("le", operator.le)
    __eq__ = _binary_method("eq", operator.eq)
    __ne__ = _binary_method("ne", operator.ne)
    __gt__ = _binary_method("gt", operator.gt)
    __ge__ = _binary_method("ge", operator.ge)

    __add__, __radd__, __iadd__ = _numeric_methods("add", lambda x, y: x + y)
    __sub__, __rsub__, __isub__ = _numeric_methods("sub", lambda x, y: x - y)
    __mul__, __rmul__, __imul__ = _numeric_methods("mul", lambda x, y: x * y)
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
        "div", lambda x, y: x / y
    )

    # # Unary methods, ~ operator is not implemented
    __neg__ = _unary_method("neg", operator.neg)
    __pos__ = _unary_method("pos", operator.pos)
    __abs__ = _unary_method("abs", operator.abs)


@dataclasses.dataclass
class GridArray(GridArrayOperatorsMixin):
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
      data: torch.Tensor values.
      offset: alignment location of the data with respect to the grid.
      grid: the Grid associated with the array data.
      dtype: type of the array data.
      shape: lengths of the array dimensions.

    Porting note:
     - defining __init__() or using super().__init__() will cause a recursive loop not sure why.
     - Mixin defining all operator special methods using __torch_function__. Some integer-based operations are not implemented.
    The implementation refers to that of np.lib.mixins.NDArrayOperatorsMixin
    """

    # Don't (yet) enforce any explicit consistency requirements between data.ndim
    # and len(offset), e.g., so we can feel to add extra time/batch/channel
    # dimensions. But in most cases they should probably match.
    # Also don't enforce explicit consistency between data.shape and grid.shape,
    # but similarly they should probably match.

    data: torch.Tensor = None
    offset: Tuple[float, ...] = None
    grid: Grid = None

    # def __init__(
    #     self, data: torch.Tensor = None,
    #     offset: Tuple[float, ...] = None,
    #     grid: Grid = None
    # ):
    #     super().__init__()
    #     self.data = data
    #     self.offset = offset
    #     self.grid = grid

    # def tree_flatten(self):
    #     """Returns flattening recipe for GridArray JAX pytree."""
    #     children = (self.data,)
    #     aux_data = (self.offset, self.grid)
    #     return children, aux_data
    # @staticmethod
    # def __new__(cls, data, offset, grid, *args, **kwargs):
    #     return super().__new__(cls, data, *args, **kwargs)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def device(self) -> torch.device:
        return self.data.device

    def clone(self, *args, **kwargs):
        return GridArray(self.data.clone(*args, **kwargs), self.offset, self.grid)

    def to(self, *args, **kwargs):
        return GridArray(self.data.to(*args, **kwargs), self.offset, self.grid)

    @staticmethod
    def is_torch_fft_func(func):
        return getattr(func, "__module__", "").startswith("torch._C._fft")
    
    @staticmethod
    def is_torch_linalg_func(func):
        return getattr(func, "__module__", "").startswith("torch._C._linalg")

    @classmethod
    def __torch_function__(self, ufunc, types, args=(), kwargs=None):
        """Define arithmetic on GridArrays using an implementation similar to NumPy's NDArrayOperationsMixin."""
        if kwargs is None:
            kwargs = {}
        if not all(issubclass(t, _HANDLED_TYPES + (GridArray,)) for t in types):
            return NotImplemented
        try:
            # get the corresponding torch function similar to numpy ufunc
            if self.is_torch_fft_func(ufunc):
                # For FFT functions, we can use the original function
                processed_args = [
                    x.data if isinstance(x, GridArray) else x for x in args
                ]
                result = ufunc(*processed_args, **kwargs)
                offset = consistent_offset_arrays(*[x for x in args if (type(x) is GridArray)])
                grid = consistent_grid_arrays(*[x for x in args if isinstance(x, GridArray)])
                return GridArray(result, offset, grid)
            elif self.is_torch_linalg_func(ufunc):
                # For linalg functions, we can use the original function
                processed_args = [
                    x.data if isinstance(x, GridArray) else x for x in args
                ]
                return ufunc(*processed_args, **kwargs)
            else:
                ufunc = getattr(torch, ufunc.__name__)
        except AttributeError as e:
            return NotImplemented

        arrays = [x.data if isinstance(x, GridArray) else x for x in args]
        result = ufunc(*arrays, **kwargs)
        offset = consistent_offset_arrays(*[x for x in args if isinstance(x, GridArray)])
        grid = consistent_grid_arrays(*[x for x in args if isinstance(x, GridArray)])
        if isinstance(result, tuple):
            return tuple(GridArray(r, offset, grid) for r in result)
        else:
            return GridArray(result, offset, grid)


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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns torch.Tensors specifying boundary values on the grid along axis.

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


def _gridvar_binary_method(name, op):
    """Implement a forward binary method for GridVariable with an operator."""

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.bc != other.bc:
                raise ValueError(
                    f"Cannot operate on grid variables with different boundary conditions"
                )
            return GridVariable(op(self.array, other.array), self.bc)
        elif isinstance(other, _HANDLED_TYPES + (GridArray,)):
            return GridVariable(op(self.array, other), self.bc)

        return NotImplemented

    method.__name__ = f"__{name}__"
    return method


def _gridvar_reflected_binary_method(name, op):
    """Implement a reflected binary method for GridVariable with an operator."""

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.bc != other.bc:
                raise ValueError(
                    f"Cannot operate on grid variables with different boundary conditions"
                )
            return GridVariable(op(other.array, self.array), self.bc)
        elif isinstance(other, _HANDLED_TYPES + (GridArray,)):
            return GridVariable(op(other, self.array), self.bc)

        return NotImplemented

    method.__name__ = f"__r{name}__"
    return method


def _gridvar_inplace_binary_method(name, op):
    """Implement an in-place binary method for GridVariable with an operator."""

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.bc != other.bc:
                raise ValueError(
                    f"Cannot operate on grid variables with different boundary conditions"
                )
            self.array = op(self.array, other.array)
            return self
        elif isinstance(other, _HANDLED_TYPES + (GridArray,)):
            self.array = op(self.array, other)
            return self

        return NotImplemented

    method.__name__ = f"__i{name}__"
    return method


def _gridvar_numeric_methods(name, op):
    """Implement forward, reflected and inplace binary methods for GridVariable with an operator."""
    return (
        _gridvar_binary_method(name, op),
        _gridvar_reflected_binary_method(name, op),
        _gridvar_inplace_binary_method(name, op),
    )


def _gridvar_unary_method(name, op):
    def method(self):
        return GridVariable(op(self.array), self.bc)

    method.__name__ = f"__i{name}__"
    return method


class GridVariableOperatorsMixing:
    """Mixin class for GridVariable"""

    __slots__ = ()

    __lt__ = _gridvar_binary_method("lt", operator.lt)
    __le__ = _gridvar_binary_method("le", operator.le)
    __eq__ = _gridvar_binary_method("eq", operator.eq)
    __ne__ = _gridvar_binary_method("ne", operator.ne)
    __gt__ = _gridvar_binary_method("gt", operator.gt)
    __ge__ = _gridvar_binary_method("ge", operator.ge)

    __add__, __radd__, __iadd__ = _gridvar_numeric_methods("add", lambda x, y: x + y)
    __sub__, __rsub__, __isub__ = _gridvar_numeric_methods("sub", lambda x, y: x - y)
    __mul__, __rmul__, __imul__ = _gridvar_numeric_methods("mul", lambda x, y: x * y)
    __truediv__, __rtruediv__, __itruediv__ = _gridvar_numeric_methods(
        "div", lambda x, y: x / y
    )

    # Unary methods, ~ operator is not implemented
    __neg__ = _gridvar_unary_method("neg", operator.neg)
    __pos__ = _gridvar_unary_method("pos", operator.pos)
    __abs__ = _gridvar_unary_method("abs", operator.abs)


@dataclasses.dataclass
class GridVariable(GridVariableOperatorsMixing):
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
      data: torch.Tensor values.
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

    # def tree_flatten(self):
    #     """Returns flattening recipe for GridVariable JAX pytree."""
    #     children = (self.array,)
    #     aux_data = (self.bc,)
    #     return children, aux_data

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     """Returns unflattening recipe for GridVariable JAX pytree."""
    #     return cls(*children, *aux_data)

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def data(self) -> torch.Tensor:
        return self.array.data

    @data.setter
    def data(self, value: torch.Tensor):
        self.array.data = value

    @property
    def device(self) -> torch.device:
        return self.array.device

    @property
    def offset(self) -> Tuple[float, ...]:
        return self.array.offset

    @property
    def grid(self) -> Grid:
        return self.array.grid

    def clone(self, *args, **kwargs):
        """Returns a copy of the GridVariable with cloned array data."""
        return GridVariable(self.array.clone(*args, **kwargs), self.bc)

    def to(self, *args, **kwargs):
        return GridVariable(self.array.to(*args, **kwargs), self.bc)

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
            if torch.isclose(self.offset[axis], 1.0):
                shape[axis] -= 1
                domain[axis] = (domain[axis][0], domain[axis][1] - grid.step[axis])
            elif torch.isclose(self.offset[axis], 0.0):
                shape[axis] -= 1
                domain[axis] = (domain[axis][0] + grid.step[axis], domain[axis][1])
        return Grid(shape, domain=tuple(domain))

    def _interior_array(self) -> torch.Tensor:
        """Returns only the interior points of self.array."""
        data = self.array.data
        for axis in range(self.grid.ndim):
            # nothing happens in periodic case
            if self.bc.types[axis][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            if torch.isclose(self.offset[axis], 1.0):
                data, _ = tensor_utils.split_along_axis(data, -1, axis)
            elif torch.isclose(self.offset[axis], 0.0):
                _, data = tensor_utils.split_along_axis(data, 1, axis)

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
        return GridArray(interior_array, self.offset, interior_grid)

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
        data = torch.as_tensor(self.data)
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
        return GridVariable(array=GridArray(data, self.offset, self.grid), bc=self.bc)


# GridArrayVector = Tuple[GridArray, ...]
class GridArrayVector(tuple):
    """
    A tuple-like container for GridArray objects, representing a vector field.
    Supports elementwise addition and scalar multiplication.
    """

    def __new__(cls, arrays):
        if not all(isinstance(a, GridArray) for a in arrays):
            raise TypeError("All elements must be GridArray instances.")
        return super().__new__(cls, arrays)

    def __add__(self, other):
        if not isinstance(other, GridArrayVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("GridArrayVectors must have the same length.")
        return GridArrayVector([a + b for a, b in zip(self, other)])

    def __iadd__(self, other):
        # Tuples are immutable, so __iadd__ should return a new object using __add__
        return self.__add__(other)

    __iadd__ = __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, GridArrayVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("GridArrayVectors must have the same length.")
        return GridArrayVector([a - b for a, b in zip(self, other)])

    def __rsub__(self, other):
        return GridArrayVector([b - a for a, b in zip(self, other)])

    def __mul__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (GridArray,)):
            return NotImplemented
        return GridArrayVector([v * x for v in self])

    __imul__ = __rmul__ = __mul__

    def __truediv__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (GridArray,)):
            return NotImplemented
        return GridArrayVector([v / x for v in self])

    def __rtruediv__(self, x):
        """
        __rdiv__ does not really make sense for GridArrayVector, but is
        implemented for consistency.
        """
        if not isinstance(x, _HANDLED_TYPES + (GridArray,)):
            return NotImplemented
        return GridArrayVector([x / v for v in self])
    
    @property
    def device(self) -> torch.device:
        return self[0].data.device
    
    def to(self, *args, **kwargs):
        return GridArrayVector([v.to(*args, **kwargs) for v in self])
    
    def clone(self, *args, **kwargs):
        return GridArrayVector([v.clone(*args, **kwargs) for v in self])


# GridVariableVector = Tuple[GridVariable, ...]
class GridVariableVector(tuple):
    """
    A tuple-like container for GridVariable objects, representing a vector field.
    Supports elementwise addition and scalar multiplication.
    """

    def __new__(cls, variables):
        if not all(isinstance(v, GridVariable) for v in variables):
            raise TypeError("All elements must be GridVariable instances.")
        return super().__new__(cls, variables)

    def __add__(self, other):
        if not isinstance(other, GridVariableVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("GridVariableVectors must have the same length.")
        return GridVariableVector([a + b for a, b in zip(self, other)])

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, GridVariableVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("GridVariableVectors must have the same length.")
        return GridVariableVector([a - b for a, b in zip(self, other)])

    __rsub__ = __sub__

    def __mul__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (GridVariable,)):
            return NotImplemented
        return GridVariableVector([v * x for v in self])

    __rmul__ = __mul__

    def __truediv__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (GridVariable,)):
            return NotImplemented
        return GridVariableVector([v / x for v in self])

    def __rtruediv__(self, x):
        """
        __rdiv__ does not really make sense for GridVariableVector, but is
        implemented for consistency.
        """
        if not isinstance(x, _HANDLED_TYPES + (GridVariable,)):
            return NotImplemented
        return GridVariableVector([x / v for v in self])
    
    @property
    def device(self) -> torch.device:
        return self[0].array.device
    
    def to(self, *args, **kwargs):
        return GridVariableVector([v.to(*args, **kwargs) for v in self])
    
    def clone(self, *args, **kwargs):
        return GridVariableVector([v.clone(*args, **kwargs) for v in self])


def applied(func):
    """Convert an array function into one defined on GridArrays.

    Since `func` can only act on `data` attribute of GridArray, it implicitly
    enforces that `func` cannot modify the other attributes such as offset.

    Args:
      func: function being wrapped.

    Returns:
      A wrapped version of `func` that takes GridArray instead of torch.Tensor args.
    """

    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, GridVariable):
                raise ValueError("grids.applied() cannot be used with GridVariable")

        offset = consistent_offset_arrays(
            *[
                arg
                for arg in args + tuple(kwargs.values())
                if isinstance(arg, GridArray)
            ]
        )
        grid = consistent_grid_arrays(
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


# Aliases for often used `grids.applied` functions.
where = applied(torch.where)


class GridArrayTensor(torch.Tensor):
    """An array of GridArrays, representing a physical tensor field.

    Packing tensor coordinates into a torch tensor of dtype object is useful
    because pointwise matrix operations like trace, transpose, and matrix
    multiplications of physical tensor quantities is meaningful.

    TODO:
    Add supports to operations like trace, transpose, and matrix multiplication on physical tensor fields, without register_pytree_node.

    Example usage:
      grad = fd.gradient_tensor(uv)                    # a rank 2 Tensor
      strain_rate = (grad + grad.T) / 2.
      nu_smag = np.sqrt(np.trace(strain_rate.dot(strain_rate)))
      nu_smag = Tensor(nu_smag)                        # a rank 0 Tensor
      subgrid_stress = -2 * nu_smag * strain_rate      # a rank 2 Tensor
    """

    # def __new__(cls, arrays):
    #     return torch.asarray(arrays).view(cls)
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        return super().__new__(cls, data, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return super().clone(*args, **kwargs)


def applied(func):
    """Convert an array function into one defined on GridArrays.

    Since `func` can only act on `data` attribute of GridArray, it implicitly
    enforces that `func` cannot modify the other attributes such as offset.

    Args:
      func: function being wrapped.

    Returns:
      A wrapped version of `func` that takes GridArray instead of torch.Tensor args.
    """

    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, GridVariable):
                raise ValueError("grids.applied() cannot be used with GridVariable")

        offset = consistent_offset_arrays(
            *[
                arg
                for arg in args + tuple(kwargs.values())
                if isinstance(arg, GridArray)
            ]
        )
        grid = consistent_grid_arrays(
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


def averaged_offset(*offsets: Tuple[Tuple]) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    n = len(offsets)
    assert n > 0, "No offsets provided"
    m = len(offsets[0])
    return tuple(sum(o[i] for o in offsets) / n for i in range(m))


def averaged_offset_arrays(
    *arrays: Union[GridArray, GridVariable]
) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    offsets = tuple([array.offset for array in arrays])
    return averaged_offset(*offsets)


def control_volume_offsets(
    c: Union[GridArray, GridVariable],
) -> Tuple[Tuple[float, ...], ...]:
    """Returns offsets for the faces of the control volume centered at `c`."""
    return tuple(
        tuple(o + 0.5 if i == j else o for i, o in enumerate(c.offset))
        for j in range(len(c.offset))
    )


def consistent_offset_arrays(*arrays: GridArray) -> Tuple[float, ...]:
    """Returns the unique offset, or raises InconsistentOffsetError."""
    offsets = {array.offset for array in arrays}
    if len(offsets) != 1:
        raise Exception(f"arrays do not have a unique offset: {offsets}")
    (offset,) = offsets
    return offset


def consistent_grid_arrays(*arrays: GridArray):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids) != 1:
        raise Exception(f"arrays do not have a unique grid: {grids}")
    (grid,) = grids
    return grid

def consistent_grid(grid: Grid, *arrays: GridArray):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids.union({grid})) != 1:
        raise Exception(f"arrays' grids {grids} are not consistent with the grid {grid}")
    (grid,) = grids
    return grid

