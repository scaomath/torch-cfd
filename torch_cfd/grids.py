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

# Modifications copyright (C) 2025 S.Cao
# ported Google's Jax-CFD functional template to torch.Tensor operations

from __future__ import annotations

import dataclasses
import math
import numbers
import operator
from functools import reduce

from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.fft as fft
import torch.nn.functional as F
from einops import rearrange as _rearrange, repeat as _repeat

from torch_cfd import tensor_utils

_HANDLED_TYPES = (numbers.Number, torch.Tensor)
T = TypeVar("T")  # for GridVariable vector


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
    """
    Describes the size and shape for an Arakawa C-Grid.
    Ported from jax_cfd.base.grids.Grid to pytorch

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
    device: torch.device

    def __init__(
        self,
        shape: Sequence[int],
        step: Optional[Union[float, Sequence[float]]] = None,
        domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
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
        if device is None:
            device = torch.device("cpu")
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
    def cell_faces(self) -> Tuple[Tuple[float, ...], ...]:
        """Returns the offsets at each of the 'forward' cell faces."""
        d = self.ndim
        offsets = (torch.eye(d) + torch.ones([d, d])) / 2.0
        return tuple(tuple(float(o) for o in offset) for offset in offsets)

    def stagger(self, v: Tuple[torch.Tensor, ...]) -> Tuple[GridVariable, ...]:
        """Places the velocity components of `v` on the `Grid`'s cell faces."""
        offsets = self.cell_faces
        return tuple(GridVariable(u, o, self) for u, o in zip(v, offsets))

    def center(self, v: Tuple[torch.Tensor, ...]) -> Tuple[GridVariable, ...]:
        """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
        offset = self.cell_center
        return tuple(GridVariable(tensor, offset, self) for tensor in v)

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
        mesh = torch.meshgrid(*axes, indexing="ij")
        return tuple(xi.to(self.device) for xi in mesh)

    def fft_mesh(self) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing positions in Fourier space."""
        fft_axes = self.fft_axes()
        fmesh = torch.meshgrid(*fft_axes, indexing="ij")
        return tuple(xi.to(self.device) for xi in fmesh)

    def rfft_mesh(self) -> Tuple[torch.Tensor, ...]:
        """Returns a tuple of arrays containing positions in rfft space."""
        fmesh = self.fft_mesh()
        k_max = math.floor(self.shape[-1] / 2.0)
        return tuple(xi[..., : k_max + 1] for xi in fmesh)

    def eval_on_mesh(
        self,
        fn: Callable[..., torch.Tensor],
        offset: Optional[Tuple[float, ...]] = None,
    ) -> GridVariable:
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
        return GridVariable(fn(*self.mesh(offset)), offset, self)


class BCType:
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    NONE = None


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
        u: GridVariable,
        offset: int,
        dim: int,
    ) -> GridVariable:
        """Shift an GridVariable by `offset`.

        Args:
          u: an `GridVariable` object.
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of `u`, shifted by `offset`. The returned `GridVariable` has offset
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

    def trim_boundary(
        self,
        u: GridVariable,
    ) -> GridVariable:
        raise NotImplementedError(
            "trim_boundary() not implemented in BoundaryConditions base class."
        )

    def impose_bc(
        self,
        u: GridVariable,
    ) -> GridVariable:
        """Impose boundary conditions on the grid variable."""
        raise NotImplementedError(
            "impose_bc() not implemented in BoundaryConditions base class."
        )


def _binary_method(name, op):
    """
    Implement a forward binary method with an operator.
    see np.lib.mixins.NDArrayOperatorsMixin

    Notes: because GridArray is a subclass of torch.Tensor, we need to check
    if the other operand is a GridArray first, otherwise, isinstance(other, _HANDLED_TYPES) will return True as well, which is not what we want as
    there will be no offset in the other operand.
    """

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            elif self.grid != other.grid:
                raise ValueError(
                    f"Cannot operate on arrays with different grids: {self.grid} vs {other.grid}"
                )
            if self.bc is None or other.bc is None or self.bc != other.bc:
                return GridVariable(op(self.data, other.data), self.offset, self.grid)
            return GridVariable(
                op(self.data, other.data), self.offset, self.grid, self.bc
            )
        elif isinstance(other, _HANDLED_TYPES):
            return GridVariable(op(self.data, other), self.offset, self.grid, self.bc)

        return NotImplemented

    method.__name__ = f"__{name}__"
    return method


def _reflected_binary_method(name, op):
    """Implement a reflected binary method with an operator."""

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            elif self.grid != other.grid:
                raise ValueError(
                    f"Cannot operate on arrays with different grids: {self.grid} vs {other.grid}"
                )
            if self.bc is None or other.bc is None or self.bc != other.bc:
                return GridVariable(op(other.data, self.data), self.offset, self.grid)
            return GridVariable(
                op(other.data, self.data), self.offset, self.grid, self.bc
            )
        elif isinstance(other, _HANDLED_TYPES):
            return GridVariable(op(other, self.data), self.offset, self.grid, self.bc)

        return NotImplemented

    method.__name__ = f"__r{name}__"
    return method


def _inplace_binary_method(name, op):
    """
    Implement an in-place binary method with an operator.
    Note: inplace operations do not change the boundary conditions or the offset.
    """

    def method(self, other):
        if isinstance(other, GridVariable):
            if self.offset != other.offset:
                raise ValueError(
                    f"Cannot operate on arrays with different offsets: {self.offset} vs {other.offset}"
                )
            elif self.grid != other.grid:
                raise ValueError(
                    f"Cannot operate on arrays with different grids: {self.grid} vs {other.grid}"
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
        return GridVariable(op(self.data), self.offset, self.grid, self.bc)

    method.__name__ = f"__i{name}__"
    return method


class GridTensorOpsMixin:
    """
    The implementation refers to that of np.lib.mixins.NDArrayOperatorsMixin
    """

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
class GridVariable(GridTensorOpsMixin):
    """
    GridVariable class has data with
        - an alignment offset
        - an associated grid
        - a boundary condition

    Offset values in the range [0, 1] fall within a single grid cell.
    Offset enables performing pad and shift operations, e.g. for finite difference calculations, requires boundary condition (BC) information. Since different variables in a PDE system can have different BCs, this class associates a specific variable's data with its BCs.

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
      bc: boundary conditions for this variable.

    Porting note by S. Cao:
     - [x] defining __init__() or using super().__init__() will cause a recursive loop not sure why (fixed 0.0.6).
     - [x] (added 0.0.1) the original jax implentation uses np.lib.mixins.NDArrayOperatorsMixin
    and the __array_ufunc__ method to implement arithmetic operations
    here it is modified to use torch.Tensor as the base class
    and __torch_function__ to do various things like clone() and to()
    reference: https://pytorch.org/docs/stable/notes/extending.html
     - [x] (added 0.0.8) Mixin defining all operator special methods using __torch_function__. Some integer-based operations are not implemented.
     - [x] (0.1.1) In original Google Research's Jax-CFD code, the devs opted to separate GridArray (no bc) and GridVariable (bc). After carefully studied the FVM implementation, I decided to combine GridArray with GridVariable to reduce code duplication.
     - One can definitely try to use TensorClass from tensordict to implement a more generic GridVariable class, however I found using Tensorclass or @tensorclass actually slows down the code quite a bit.
     - [x] (0.2.0) Finished refactoring the whole GridVariable class for various routines, getting rid of the GridArray class, adding several helper functions for GridVariableVector, and adding batch dimension checks.
    """

    data: torch.Tensor
    offset: Tuple[float, ...]
    grid: Grid
    bc: Optional[BoundaryConditions] = None

    def __post_init__(self):
        if not isinstance(self.data, torch.Tensor):  # frequently missed by pytype
            raise ValueError(
                f"Expected data type to be torch.Tensor, got {type(self.data)}"
            )
        if self.bc is not None:
            """bc = None follows the original GridArray behavior in Jax-CFD"""
            if len(self.bc.types) != self.grid.ndim:
                raise ValueError(
                    "Incompatible dimension between grid and bc, grid dimension = "
                    f"{self.grid.ndim}, bc dimension = {len(self.bc.types)}"
                )

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
    def array(self) -> torch.Tensor:
        """added for back-compatibility"""
        return self.data

    @array.setter
    def array(self, value: torch.Tensor):
        self.data = value

    @property
    def device(self) -> torch.device:
        return self.data.device

    def norm(self, p: Optional[Union[int, float]] = None, **kwargs) -> torch.Tensor:
        """Returns the norm of the data."""
        return torch.linalg.norm(self.data, ord=p, **kwargs)

    @property
    def L2norm(self) -> torch.Tensor:
        """returns the batched norm, shaped (bsz, )"""
        dims = range(-self.grid.ndim, 0)
        return self.norm(dim=tuple(dims)) * (self.grid.step[0] * self.grid.step[1]) ** (
            1 / self.grid.ndim
        )

    def clone(self):
        return GridVariable(self.data.clone(), self.offset, self.grid, self.bc)

    def to(self, *args, **kwargs):
        return GridVariable(
            self.data.to(*args, **kwargs), self.offset, self.grid, self.bc
        )

    def __getitem__(self, index):
        """Allows indexing into the GridVariable like a tensor."""
        # This is necessary to ensure that the offset and grid are preserved
        # when slicing the data.
        new_data = self.data[index]
        if isinstance(new_data, torch.Tensor):
            return GridVariable(new_data, self.offset, self.grid, self.bc)
        return new_data

    @staticmethod
    def is_torch_fft_func(func):
        return getattr(func, "__module__", "").startswith("torch._C._fft")

    @staticmethod
    def is_torch_linalg_func(func):
        return getattr(func, "__module__", "").startswith("torch._C._linalg")

    @classmethod
    def __torch_function__(cls, ufunc, types, args=(), kwargs=None):
        """Define arithmetic on GridVariable using an implementation similar to NumPy's NDArrayOperationsMixin."""
        if kwargs is None:
            kwargs = {}
        if not all(issubclass(t, _HANDLED_TYPES + (GridVariable,)) for t in types):
            return NotImplemented
        try:
            # get the corresponding torch function similar to numpy ufunc
            if cls.is_torch_fft_func(ufunc):
                # For FFT functions, we can use the original function
                processed_args = [
                    x.data if isinstance(x, GridVariable) else x for x in args
                ]
                result = ufunc(*processed_args, **kwargs)
                offset = consistent_offset_arrays(
                    *[x for x in args if (type(x) is GridVariable)]
                )
                grid = consistent_grid_arrays(
                    *[x for x in args if isinstance(x, GridVariable)]
                )
                bc = consistent_bc_arrays(
                    *[x for x in args if isinstance(x, GridVariable)]
                )
                return GridVariable(result, offset, grid, bc)
            elif cls.is_torch_linalg_func(ufunc):
                # For linalg functions, we can use the original function
                # acting only on the data
                processed_args = [
                    x.data if isinstance(x, GridVariable) else x for x in args
                ]
                return ufunc(*processed_args, **kwargs)
            else:
                ufunc = getattr(torch, ufunc.__name__)
        except AttributeError as e:
            return NotImplemented

        arrays = [x.data if isinstance(x, GridVariable) else x for x in args]
        result = ufunc(*arrays, **kwargs)

        # If the result is a scalar (0-dim tensor or Python scalar), return it directly
        if isinstance(result, torch.Tensor) and result.ndim == 0:
            return result
        if not isinstance(result, torch.Tensor):
            return result

        offset = consistent_offset_arrays(
            *[x for x in args if isinstance(x, GridVariable)]
        )
        grid = consistent_grid_arrays(*[x for x in args if isinstance(x, GridVariable)])
        bc = consistent_bc_arrays(*[x for x in args if isinstance(x, GridVariable)])
        if isinstance(result, tuple):
            return tuple(GridVariable(r, offset, grid, bc) for r in result)
        else:
            return GridVariable(result, offset, grid, bc)

    def shift(
        self,
        offset: int,
        dim: int,
    ) -> GridVariable:
        """Shift this GridVariable by `offset`.

        Args:
          offset: positive or negative integer offset to shift.
          dim: axis to shift along.

        Returns:
          A copy of the encapsulated GridVariable, shifted by `offset`. The returned
          GridVariable has offset `u.offset + offset`.
          u.shape is (*, n, m)

        Note:
          The implementation in Jax-CFD does not take into account the batch dimension upon calling u.shift(). This implementation fixed the behavior by using negative indexing.
        """
        return shift(
            self,
            offset=offset,
            dim=dim,
        )

    def trim_boundary(self) -> GridVariable:
        return self.bc.trim_boundary(self)

    def _interior_grid(self) -> Grid:
        """Returns only the interior grid points."""
        grid = self.grid
        domain = list(grid.domain)
        shape = list(grid.shape)
        for axis in range(self.grid.ndim):
            # nothing happens in periodic case
            if self.bc.types[axis][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            # this will automatically set the grid to interior.
            if math.isclose(self.offset[axis], 1.0):
                shape[axis] -= 1
                domain[axis] = (domain[axis][0], domain[axis][1] - grid.step[axis])
            elif math.isclose(self.offset[axis], 0.0):
                shape[axis] -= 1
                domain[axis] = (domain[axis][0] + grid.step[axis], domain[axis][1])
        return Grid(shape, domain=tuple(domain))

    def _interior_array(self) -> torch.Tensor:
        """Returns only the interior points of self.data."""
        data = self.data
        for axis in range(self.grid.ndim):
            # nothing happens in periodic case
            if self.bc.types[axis][1] == "periodic":
                continue
            # nothing happens if the offset is not 0.0 or 1.0
            if math.isclose(self.offset[axis], 1.0):
                data, _ = tensor_utils.split_along_axis(data, -1, axis)
            elif math.isclose(self.offset[axis], 0.0):
                _, data = tensor_utils.split_along_axis(data, 1, axis)

        return data

    def interior(self) -> GridVariable:
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
        return GridVariable(interior_array, self.offset, interior_grid)

    def enforce_edge_bc(self, *args) -> GridVariable:
        """Returns the GridVariable with edge BC enforced, if applicable.

        For GridVariables having nonperiodic BC and offset 0 or 1, there are values
        in the array data that are dependent on the boundary condition.
        enforce_edge_bc() changes these boundary values to match the prescribed BC.

        Args:
          *args: any optional values passed into BoundaryConditions values method.
        """
        if self.grid.shape != self.data.shape:
            raise ValueError("Stored array and grid have mismatched sizes.")
        data = torch.as_tensor(self.data)
        for axis in range(self.grid.ndim):
            if "periodic" not in self.bc.types[axis]:
                values = self.bc.values(axis, self.grid, *args)
                for boundary_side in range(2):
                    if torch.isclose(self.offset[axis], boundary_side):
                        # boundary data is set to match self.bc:
                        all_slice = [
                            slice(None, None, None),
                        ] * self.grid.ndim
                        all_slice[axis] = -boundary_side
                        data = data.at[tuple(all_slice)].set(values[boundary_side])
        return GridVariable(data, self.offset, self.grid, self.bc)


class GridVectorBase(tuple, Generic[T]):
    """
    A tuple-like container for GridVariable objects, representing a vector field.
    Supports elementwise addition and scalar multiplication.
    """

    def __new__(cls, elements: Sequence[T]):
        if not all(isinstance(x, cls._element_type()) for x in elements):
            raise TypeError(
                f"All elements must be instances of {cls._element_type().__name__}."
            )
        return super().__new__(cls, elements)

    @classmethod
    def _element_type(cls):
        raise NotImplementedError("Subclasses must override _element_type()")

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length.")
        return self.__class__([a + b for a, b in zip(self, other)])

    __radd__ = __add__

    def __iadd__(self, other):
        # Tuples are immutable, so __iadd__ should return a new object using __add__
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length.")
        return self.__class__([a - b for a, b in zip(self, other)])

    __rsub__ = lambda self, other: self.__class__([b - a for a, b in zip(self, other)])

    def __isub__(self, other):
        # Tuples are immutable, so __isub__ should return a new object using __sub__
        return self.__sub__(other)

    def __mul__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (self._element_type(),)):
            return NotImplemented
        return self.__class__([v * x for v in self])

    __imul__ = __rmul__ = __mul__

    def __truediv__(self, x):
        if not isinstance(x, _HANDLED_TYPES + (self._element_type(),)):
            return NotImplemented
        return self.__class__([v / x for v in self])

    def __rtruediv__(self, x):
        """
        __rdiv__ does not really make sense for GridVariableVector, but is
        implemented for consistency.
        """
        if not isinstance(x, _HANDLED_TYPES + (self._element_type(),)):
            return NotImplemented
        return self.__class__([x / v for v in self])


class GridVariableVector(GridVectorBase[GridVariable]):
    @classmethod
    def _element_type(cls):
        return GridVariable

    def to(self, *args, **kwargs):
        return self.__class__([v.to(*args, **kwargs) for v in self])

    def clone(self, *args, **kwargs):
        return self.__class__([v.clone(*args, **kwargs) for v in self])

    @property
    def shape(self):
        """Return the overall shape of the vector.
        Example:
        v0 = GridVariable(...)
        v1 = GridVariable(...)
        self = GridVariableVector([v0, v1])
        then self.shape will return (2, *shape) where shape = v0.shape.
        """
        if not self:
            return ()
        _shape = self[0].shape
        return torch.Size((len(self),) + _shape)

    @property
    def device(self):
        _device = {v.device for v in self}
        if len(_device) != 1:
            raise ValueError(
                f"All component of {type(self)} must be on the same device."
            )
        return _device.pop()

    @property
    def dtype(self):
        """Return the dtype of the vector."""
        _dtype = {v.dtype for v in self}
        if len(_dtype) != 1:
            raise ValueError(f"All component of {type(self)} must have the same dtype.")
        return _dtype.pop()

    @property
    def data(self) -> Tuple[torch.Tensor, ...]:
        """
        added for compatibility
        """
        return tuple(v.data for v in self)

    @data.setter
    def data(self, values: Tuple[torch.Tensor, ...]):
        for v, value in zip(self, values):
            v.data = value


class GridTensor(torch.Tensor):
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

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        return torch.Tensor.__new__(cls, data, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return super().clone(*args, **kwargs)


def shift(
    u: GridVariable, offset: int, dim: int, bc: Optional[BoundaryConditions] = None
) -> GridVariable:
    """Shift a GridVariable by `offset`.

    Args:
        u: an `GridVariable` object.
        offset: positive or negative integer offset to shift.
        dim: axis to shift along.

    Returns:
        A copy of `u`, shifted by `offset`. The returned `GridVariable` has offset
        `u.offset + offset`.
    """
    dim = -u.grid.ndim + dim if dim >= 0 else dim
    padded = pad(u, offset, dim, bc)
    trimmed = trim(padded, -offset, dim)
    return trimmed


def pad(
    u: GridVariable, width: int, dim: int, bc: Optional[BoundaryConditions] = None
) -> GridVariable:
    """Pad a GridVariable by `padding`.

    Important: the original _pad in jax_cfd makes no sense past 1 ghost cell for nonperiodic boundaries.
    This is sufficient to replicate jax_cfd finite difference code.

    Args:
        u: an `GridVariable` object.
        width: number of elements to pad along axis. Use negative value for lower
            boundary or positive value for upper boundary.
        dim: axis to pad along.

    Returns:
        Padded array, elongated along the indicated axis.

    Note:
        - the padding can be only defined when there is proper BC given. If bc is externally given (such as those cases in boundaryies.py), it will override the one in u.bc (if it is not None).
        - In the original Jax-CFD codes, when the width > 1, the code raises error "Padding past 1 ghost cell is not defined in nonperiodic case.", this is now removed and tested for correctness.
    """
    assert not (u.bc is None and bc is None), "Both u.bc and bc cannot be None"
    bc = bc if bc is not None else u.bc

    if width < 0:  # pad lower boundary
        bc_type = bc.types[dim][0]
        padding = (-width, 0)
    else:  # pad upper boundary
        bc_type = bc.types[dim][1]
        padding = (0, width)

    full_padding = [(0, 0)] * u.grid.ndim
    full_padding[dim] = padding

    new_offset = list(u.offset)
    new_offset[dim] -= padding[0]

    if bc_type == BCType.PERIODIC:
        # self.values are ignored here
        data = expand_dims_pad(u.data, full_padding, mode="circular")
        return GridVariable(data, tuple(new_offset), u.grid, bc)
    elif bc_type == BCType.DIRICHLET:
        if math.isclose(u.offset[dim] % 1, 0.5):  # cell center
            # make the linearly interpolated value equal to the boundary by setting
            # the padded values to the negative symmetric values
            data = 2 * expand_dims_pad(
                u.data, full_padding, mode="constant", constant_values=bc._values
            ) - expand_dims_pad(u.data, full_padding, mode="replicate")
            return GridVariable(data, tuple(new_offset), u.grid, bc)
        elif math.isclose(u.offset[dim] % 1, 0):  # cell edge
            data = expand_dims_pad(
                u.data, full_padding, mode="constant", constant_values=bc._values
            )
            return GridVariable(data, tuple(new_offset), u.grid, bc)
        else:
            raise ValueError(
                "expected the new offset to be an edge or cell center, got "
                f"offset[axis]={u.offset[dim]}"
            )
    elif bc_type == BCType.NEUMANN:
        if not (
            math.isclose(u.offset[dim] % 1, 0) or math.isclose(u.offset[dim] % 1, 0.5)
        ):
            raise ValueError(
                "expected offset to be an edge or cell center, got "
                f"offset[axis]={u.offset[dim]}"
            )
        else:
            # When the data is cell-centered, computes the backward difference.
            # When the data is on cell edges, boundary is set such that
            # (u_boundary - u_last_interior)/grid_step = neumann_bc_value (fixed from Jax-cfd).
            # note: Jax-cfd implementation was wrong, Neumann BC is \nabla u \cdot exterior normal, the order is reversed
            data = expand_dims_pad(
                u.data, full_padding, mode="replicate"
            ) + u.grid.step[dim] * (
                expand_dims_pad(
                    u.data,
                    full_padding,
                    mode="constant",
                    constant_values=bc._values,
                )
                - expand_dims_pad(u.data, full_padding, mode="constant")
            )
            return GridVariable(data, tuple(new_offset), u.grid, bc)

    else:
        raise ValueError("invalid boundary type")


def trim(u: GridVariable, width: int, dim: int) -> GridVariable:
    """Trim padding from a GridVariable.

    Args:
      u: data.
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

    limit_index = u.shape[dim] - padding[1]
    data = u.data.index_select(
        dim=dim, index=torch.arange(padding[0], limit_index, device=u.device)
    )
    new_offset = list(u.offset)
    new_offset[dim] += padding[0]
    return GridVariable(data, tuple(new_offset), u.grid)


def expand_dims_pad(
    inputs: torch.Tensor,
    pad: Sequence[Tuple[int, int]],
    mode: str = "constant",
    constant_values: float = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Porting notes:
    - This function serves as replacement for jax.numpy.pad in the original Jax-CFD implementation, basically a wrapper for F.pad with a dimension checker
    - jnp's pad pad_width starts from the first dimension to the last dimension
    while torch's pad pad_width starts from the last dimension to the previous dimension
    example:
    - for torch (1, 1, 2, 2) means padding last dim by (1, 1) and 2nd to last by (2, 2), the pad arg for the expand_dims_pad function should be ((2, 2), (1, 1))

    Args:
      inputs: torch.Tensor or a tuple of arrays to pad.
      pad_width: padding width for each dimension.
      mode: padding mode, one of 'constant', 'reflect', 'symmetric'.
      values: constant value to pad with.

    Returns:
      Padded `inputs`.
    """
    if len(pad) == inputs.ndim:
        # if pad has the same length as inputs.ndim, add a batch dimension
        unsqueezed = True
        inputs = inputs.unsqueeze(0)
    elif len(pad) <= inputs.ndim - 1:
        # if pad has the same length as inputs.ndim - 1, no need to unsqueeze
        unsqueezed = False
    else:
        raise ValueError(f"pad length must <= {inputs.ndim}, got {len(pad)}")

    flat_pad = reduce(lambda a, x: x + a, pad, ())  # flatten the pad and reverse
    if mode == "constant":
        if isinstance(constant_values, (int, float)):
            array = F.pad(inputs, flat_pad, mode=mode, value=constant_values)
        elif isinstance(constant_values, (tuple, list)):
            array = _constant_pad_tensor(inputs, pad, constant_values, **kwargs)
        else:
            raise NotImplementedError(
                f"constant_values must be a float or a tuple/list, got {type(constant_values)}"
            )
    elif mode in ["circular", "reflect", "replicate"]:
        # periodic boundary condition
        array = F.pad(inputs, flat_pad, mode=mode)
    else:
        raise NotImplementedError(f"invalid mode {mode} for torch.nn.functional.pad")

    return array.squeeze(0) if unsqueezed else array


def _constant_pad_tensor(
    inputs: torch.Tensor,
    pad: Tuple[Tuple[int, int], ...],
    constant_values: Tuple[Tuple[float, float], ...],
    **kwargs,
) -> torch.Tensor:
    """
    Corrected padding function that supports different constant values for each side.
    Pads each dimension from first to last as per the user input, bypassing PyTorch's F.pad behavior of last-to-first padding order.

    Args:
        inputs: torch.Tensor to pad.
        pad: padding width for each dimension, e.g. ((2, 2), (1, 1)) for 2D tensor. (2, 2) means padding the first dimension by 2 on both sides, and (1, 1) means padding the second dimension by 1 on both sides.
        constant_values: constant values to pad with for each dimension, e.g. ((0, 0), (1, 1)).
    """
    # inputs was unsqueezed at dim 0, so actual data dims are shifted by +1
    ndim = len(pad)  # number of dimensions to pad
    result = inputs

    for i in reversed(range(ndim)):
        dim_pad_tensor = i - ndim
        # correct mapping from pad index to tensor dim
        # for example, if ((2, 2), (1, 1)) is given for a 2D tensor of shape (10, 20),
        # the pad[1] corresponds to the padding of the last dimension (20),
        # and pad[0] corresponds to the padding of the second-to-last dimension (10).

        left_pad, right_pad = pad[i]

        if left_pad == 0 and right_pad == 0:
            continue

        # Get constant values
        if len(constant_values) > i:
            if (
                isinstance(constant_values[i], (tuple, list))
                and len(constant_values[i]) == 2
            ):
                left_val, right_val = constant_values[i]
            else:
                left_val = right_val = constant_values[i]
        else:
            left_val = right_val = 0.0

        left_val = (
            float(left_val[0])
            if isinstance(left_val, (list, tuple))
            else float(left_val)
        )
        right_val = (
            float(right_val[0])
            if isinstance(right_val, (list, tuple))
            else float(right_val)
        )

        shape = list(result.shape)

        if left_pad > 0:
            shape[dim_pad_tensor] = left_pad
            left_tensor = torch.full(
                shape, left_val, dtype=result.dtype, device=result.device
            )
            result = torch.cat([left_tensor, result], dim=dim_pad_tensor)

        if right_pad > 0:
            shape[dim_pad_tensor] = right_pad
            right_tensor = torch.full(
                shape, right_val, dtype=result.dtype, device=result.device
            )
            result = torch.cat([result, right_tensor], dim=dim_pad_tensor)

    return result


# def _constant_pad(
#     inputs: torch.Tensor,
#     pad: Tuple[Tuple[int, int], ...],
#     constant_values: Tuple[Tuple[float, float], ...],
#     **kwargs,
# ) -> torch.Tensor:
#     """
#     Corrected padding function that supports different constant values for each side.
#     Pads each dimension from first to last as per the user input, mapping correctly to
#     PyTorch's last-to-first padding order.
#     inputs was unsqueezed at dim 0 as a batch_dim, so actual data dims are shifted by +1
#     """
#     ndim = inputs.ndim - 1 #
#     original_shape = list(inputs.shape)
#     out_shape = [1] + [original_shape[i+1] + pad[i][0] + pad[i][1] for i in range(ndim)] # inputs was unsqueezed at dim 0, so actual data dims are shifted by +1

#     output = torch.empty(out_shape, dtype=inputs.dtype, device=inputs.device)

#     def get_vals(dim):
#         if len(constant_values) > dim:
#             vals = constant_values[dim]
#             if isinstance(vals, (tuple, list)) and len(vals) == 2:
#                 return float(vals[0]), float(vals[1])
#             else:
#                 val = float(vals)
#                 return val, val
#         return 0.0, 0.0

#     # Fill with zeros initially
#     output.fill_(0.0)

#     # Main region
#     slices = (slice(None),) + tuple(slice(pad[i][0], pad[i][0] + original_shape[i+1]) for i in range(ndim))
#     output[slices] = inputs

#     # Apply left/right pad values per dim
#     for i in range(ndim):
#         lpad, rpad = pad[i]
#         lval, rval = get_vals(i)

#         if lpad > 0:
#             left_slices = [slice(None)] * ndim
#             left_slices[i] = slice(0, lpad)
#             left_slides = (slice(None), ) + tuple(left_slices)
#             output[left_slides] = lval

#         if rpad > 0:
#             right_slices = [slice(None)] * ndim
#             right_slices[i] = slice(-rpad, None)
#             right_slices = (slice(None), ) + tuple(right_slices)
#             output[right_slices] = rval

#     return output


def averaged_offset(*offsets: List[Tuple[float, ...]]) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    n = len(offsets)
    assert n > 0, "No offsets provided"
    m = len(offsets[0])
    return tuple(sum([o[i] for o in offsets]) / n for i in range(m))


def averaged_offset_arrays(*arrays: GridVariable) -> Tuple[float, ...]:
    """Returns the averaged offset of the given arrays."""
    offsets = list([array.offset for array in arrays])
    return averaged_offset(*offsets)


def control_volume_offsets(*offsets) -> Tuple[Tuple[float, ...], ...]:
    """Returns offsets for the faces of the control volume centered at `offsets`."""
    return tuple(
        tuple(o + 0.5 if i == j else o for i, o in enumerate(offsets))
        for j in range(len(offsets))
    )


def control_volume_offsets_arrays(
    c: GridVariable,
) -> Tuple[Tuple[float, ...], ...]:
    """Returns offsets for the faces of the control volume centered at `c`."""
    return control_volume_offsets(*c.offset)


def consistent_offset_arrays(*arrays: GridVariable) -> Tuple[float, ...]:
    """Returns the unique offset, or raises InconsistentOffsetError."""
    offsets = {array.offset for array in arrays}
    if len(offsets) != 1:
        raise Exception(f"arrays do not have a unique offset: {offsets}")
    (offset,) = offsets
    return offset


def consistent_grid_arrays(*arrays: GridVariable):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids) != 1:
        raise Exception(f"arrays do not have a unique grid: {grids}")
    (grid,) = grids
    return grid


def consistent_bc_arrays(*arrays: GridVariable):
    """Returns the unique bc, or return None if difference BC."""
    bcs = {array.bc for array in arrays}
    if len(bcs) != 1:
        return None
    (bc,) = bcs
    return bc


def consistent_grid(grid: Grid, *arrays: GridVariable):
    """Returns the unique grid, or raises InconsistentGridError."""
    grids = {array.grid for array in arrays}
    if len(grids.union({grid})) != 1:
        raise Exception(
            f"arrays' grids {grids} are not consistent with the grid {grid}"
        )
    (grid,) = grids
    return grid


def domain_interior_masks(grid: Grid):
    """Returns cell face arrays with 1 on the interior, 0 on the boundary."""
    masks = []
    for offset in grid.cell_faces:
        mesh = grid.mesh(offset)
        mask = torch.ones(mesh[0].shape, device=grid.device)
        for i, x in enumerate(mesh):
            lower = (
                ~torch.isclose(x, torch.tensor(grid.domain[i][0], device=grid.device))
            ).int()
            upper = (
                ~torch.isclose(x, torch.tensor(grid.domain[i][1], device=grid.device))
            ).int()
            mask = mask * upper * lower
        masks.append(mask)
    return tuple(masks)


def repeat(x: GridVariable, pattern: str, **kwargs) -> GridVariable:
    data = _repeat(x.data, pattern, **kwargs)
    return GridVariable(data, x.offset, x.grid, x.bc)


def rearrange(x: GridVariable, pattern: str, **kwargs) -> GridVariable:
    data = _rearrange(x.data, pattern, **kwargs)
    return GridVariable(data, x.offset, x.grid, x.bc)


def stack_gridvariables(*arrays: GridVariable, dim: int = 0) -> GridVariable:
    """Stack GridVariable objects by stacking their data and preserving metadata."""

    # Extract data from all tensors
    data = [t.data for t in arrays]
    stacked_data = torch.stack(data, dim=dim)

    # Use the first tensor's metadata (assuming they're compatible)
    return GridVariable(
        stacked_data,
        consistent_offset_arrays(*arrays),
        consistent_grid_arrays(*arrays),
        consistent_bc_arrays(*arrays) if arrays[0].bc is not None else None,
    )


def stack_gridvariable_vectors(
    *vectors: GridVariableVector, dim: int = 0
) -> GridVariableVector:
    """Stack GridVariableVector objects (vector fields) by stacking their components and preserving metadata.

    Args:
        *vectors: GridVariableVector to stack
        dim: dimension along which to stack (default: 0)

    Returns:
        GridVariableVector with stacked components

    Example:
        v1 = GridVariableVector([u1, v1])  # velocity field 1
        v2 = GridVariableVector([u2, v2])  # velocity field 2
        stacked = stack_gridvariable_vectors(v1, v2, dim=0)
        # stacked[0].shape: (2, ...)
    """

    vector_length = len(vectors[0])
    if not all(len(v) == vector_length for v in vectors):
        raise ValueError("All vectors must have the same number of components")

    stacked_vector = []
    for i in range(vector_length):
        cs = [v[i] for v in vectors]
        # Stack these components
        c = stack_gridvariables(*cs, dim=dim)
        stacked_vector.append(c)

    return GridVariableVector(stacked_vector)
