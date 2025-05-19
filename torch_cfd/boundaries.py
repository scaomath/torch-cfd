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
# ported Google's Jax-CFD functional template to PyTorch's tensor ops


import dataclasses
import math
from functools import reduce
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from torch_cfd import grids

BoundaryConditions = grids.BoundaryConditions
Grid = grids.Grid
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


class BCType:
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class Padding:
    MIRROR = "mirror"
    EXTEND = "extend"


@dataclasses.dataclass(init=False, frozen=True)
class ConstantBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a PDE variable that are constant in space and time.

    Example usage:
      grid = Grid((10, 10))
      array = GridArray(torch.zeros((10, 10)), offset=(0.5, 0.5), grid)
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

    def _count_bc_components(self) -> int:
        """Counts the number of components in the boundary conditions.

        Returns:
          The number of components in the boundary conditions.
        """
        count = 0
        ndim = len(self.types)
        for axis in range(ndim): # ndim
            if len(self.types[axis]) != 2:
                raise ValueError(
                    f"Boundary conditions for axis {axis} must have two values got {len(self.types[axis])}."
                )
            count += len(self.types[axis])
        return count

    def _is_aligned(self, u: GridArray, dim: int) -> bool:
        """Checks if array u contains all interior domain information.

        For dirichlet edge aligned boundary, the value that lies exactly on the
        boundary does not have to be specified by u.
        Neumann edge aligned boundary is not defined.

        Args:
        u: torch.Tensor that should contain interior data
        dim: axis along which to check

        Returns:
        True if u is aligned, and raises error otherwise.
        """
        size_diff = u.shape[dim] - u.grid.shape[dim]
        if self.types[dim][0] == BCType.DIRICHLET and math.isclose(u.offset[dim], 1):
            size_diff += 1
        if self.types[dim][1] == BCType.DIRICHLET and math.isclose(u.offset[dim], 1):
            size_diff += 1
        if self.types[dim][0] == BCType.NEUMANN and math.isclose(u.offset[dim] % 1, 0):
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
            if math.isclose(u.offset[dim] % 1, 0.5):  # cell center
                # make the linearly interpolated value equal to the boundary by setting
                # the padded values to the negative symmetric values
                data = 2 * expand_dims_pad(
                    u.data, full_padding, mode="constant", constant_values=self._values
                ) - expand_dims_pad(u.data, full_padding, mode="reflect")
                return GridArray(data, tuple(offset), u.grid)
            elif math.isclose(u.offset[dim] % 1, 0):  # cell edge
                pad_kwargs = dict(mode="constant", constant_values=self._values)
            else:
                raise ValueError(
                    "expected offset to be an edge or cell center, got "
                    f"offset[axis]={u.offset[dim]}"
                )
        elif bc_type == BCType.NEUMANN:
            if not (
                math.isclose(u.offset[dim] % 1, 0)
                or math.isclose(u.offset[dim] % 1, 0.5)
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

    def values(
        self, dim: int, grid: Grid
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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
                elif math.isclose(u.offset[dim] % 1, 0):
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
                        if self.types[dim][1] == BCType.DIRICHLET and math.isclose(
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
                if self.types[axis][0] == BCType.DIRICHLET and math.isclose(
                    u.offset[axis], 1.0
                ):
                    if math.isclose(offset_to_pad_to[axis], 1.0):
                        u = self._pad(u, 1, axis, mode=mode)
                    elif math.isclose(offset_to_pad_to[axis], 0.0):
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
      array = GridArray(torch.zeros((10, 10)), offset=(0.5, 0.5), grid)
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
    elif c.bc.types[dim][0] == BCType.PERIODIC and c.bc.types[dim][0] != c.bc.types[dim][1]:
        raise ValueError(
            "periodic boundary conditions must be the same on both sides of the axis"
        )
    return True


# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> BoundaryConditions:
    """Returns periodic BCs for a variable with `ndim` spatial dimension."""
    return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def consistent_boundary_conditions_grid(grid, *arrays: GridVariable) -> Tuple[int, ...]:
    """Returns the updated boundary condition if the number of components is inconsistent
    with the grid
    """
    bc_counts = []
    for array in arrays:
        bc_counts.append(array.bc._count_bc_components())
    bc_count = bc_counts[0]
    if any(bc_counts[i] != bc_count for i in range(1, len(bc_counts))):
        raise Exception("Boundary condition counts are inconsistent")
    if any(bc_counts[i] != 2 * grid.ndim for i in range(len(bc_counts))):
        raise ValueError(
            f"Boundary condition counts {bc_counts} are inconsistent with grid dimensions {grid.ndim}"
        )
    return arrays


def consistent_boundary_conditions_gridvariable(*arrays: GridVariable) -> Tuple[str, ...]:
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

def get_pressure_bc_from_velocity_bc(bcs: Sequence[BoundaryConditions]) -> HomogeneousBoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity BCs.
    if the velocity BC is periodic, the pressure BC is periodic.
    if the velocity BC is nonperiodic, the pressure BC is zero flux (homogeneous Neumann).
    """
    # assumes that if the boundary is not periodic, pressure BC is zero flux.
    pressure_bc_types = []
    for velocity_bc in bcs:
        if isinstance(velocity_bc, HomogeneousBoundaryConditions):
            velocity_bc_types = velocity_bc.types
        else:
            raise NotImplementedError(
                f"Pressure boundary condition is not implemented for velocity with {velocity_bc}"
            )
        if velocity_bc_types[0][0] == BCType.PERIODIC and velocity_bc_types[1][0] == BCType.PERIODIC:
            pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
        else:
            pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
        
    return HomogeneousBoundaryConditions(pressure_bc_types)
    


def get_pressure_bc_from_velocity(
    v: GridVariableVector,
) -> HomogeneousBoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity."""
    # assumes that if the boundary is not periodic, pressure BC is zero flux.
    velocity_bc_types = consistent_boundary_conditions_gridvariable(*v)
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
                    if not math.isclose(c.bc.bc_values[axis][i], 0.0):
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
      inputs: torch.Tensor or a tuple of arrays to pad.
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
        # periodic boundary condition
        array = F.pad(inputs, pad, mode=mode)
    else:
        raise NotImplementedError(f"invalid mode {mode} for torch.nn.functional.pad")

    return array.squeeze(0) if dim != array.ndim else array
