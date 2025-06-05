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
from typing import Optional, Sequence, Tuple

import torch

from torch_cfd import grids

BoundaryConditions = grids.BoundaryConditions
Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


BCType = grids.BCType()


class Padding:
    MIRROR = "reflect"
    EXTEND = "replicate"


@dataclasses.dataclass(init=False, frozen=True)
class ConstantBoundaryConditions(grids.BoundaryConditions):
    """Boundary conditions for a PDE variable that are constant in space and time.

    Example usage:
      grid = Grid((10, 10))
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)),
                                          ((0.0, 10.0),(1.0, 0.0)))
      v = GridVariable(torch.zeros((10, 10)), offset=(0.5, 0.5), grid, bc)

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
        object.__setattr__(self, "ndim", len(types))

    def shift(
        self,
        u: GridVariable,
        offset: int,
        dim: int,
    ) -> GridVariable:
        """
        A fallback function to make the implementation back-compatible
        see grids.shift
        bc.shift(u, offset, dim) overrides u.bc
        grids.shift(u, offset, dim) keeps u.bc
        """
        return grids.shift(u, offset, dim, self)

    def _is_aligned(self, u: GridVariable, dim: int) -> bool:
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
            raise ValueError(
                "the GridVariable does not contain all interior grid values."
            )
        return True

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

    def _trim_padding(self, u: GridVariable, dim: int = -1, trim_side: str = "both"):
        """Trims padding from a GridVariable along axis and returns the array interior.

        Args:
        u: a `GridVariable` object.
        dim: axis to trim along.
        trim_side: if 'both', trims both sides. If 'right', trims the right side.
            If 'left', the left side.

        Returns:
        Trimmed array, shrunk along the indicated axis side.
        """
        positive_trim = 0
        negative_trim = 0
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
                u = grids.trim(u, negative_trim, dim)
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
            u = grids.trim(u, positive_trim, dim)
        # combining existing padding with new padding
        padding = (-negative_trim, positive_trim)
        return u, padding

    def trim_boundary(self, u: GridVariable) -> GridVariable:
        """Returns GridVariable without the grid points on the boundary.

        Some grid points of GridVariable might coincide with boundary. This trims those
        values. If the array was padded beforehand, removes the padding.

        Args:
        u: a `GridVariable` object.

        Returns:
        A GridVariable shrunk along certain dimensions.
        """
        for axis in range(-u.grid.ndim, 0):
            _ = self._is_aligned(u, axis)
            u, _ = self._trim_padding(u, axis)
        return u

    def pad_and_impose_bc(
        self,
        u: GridVariable,
        offset_to_pad_to: Optional[Tuple[float, ...]] = None,
        mode: Optional[str] = "extend",
    ) -> GridVariable:
        """Returns GridVariable with correct boundary values.

        Some grid points of GridVariable might coincide with boundary. This ensures
        that the GridVariable.array agrees with GridVariable.bc.
        Args:
        u: a `GridVariable` object that specifies only scalar values on the internal
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
        for axis in range(-u.grid.ndim, 0):
            _ = self._is_aligned(u, axis)
            if self.types[axis][0] == BCType.DIRICHLET and math.isclose(
                u.offset[axis], 1.0
            ):
                if math.isclose(offset_to_pad_to[axis], 1.0):
                    u = grids.pad(u, 1, axis, self)
                elif math.isclose(offset_to_pad_to[axis], 0.0):
                    u = grids.pad(u, -1, axis, self)
        return GridVariable(u.data, u.offset, u.grid, self)

    def impose_bc(self, u: GridVariable) -> GridVariable:
        """Returns GridVariable with correct boundary condition.

        Some grid points of GridVariable might coincide with boundary. This ensures
        that the GridVariable.array agrees with GridVariable.bc.
        Args:
        u: a `GridVariable` object.

        Returns:
        A GridVariable that has correct boundary values and is restricted to the
        domain.
        """
        offset = u.offset
        u = self.trim_boundary(u)
        u = self.pad_and_impose_bc(u, offset)
        return u


class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
    """Boundary conditions for a PDE variable.

    Example usage:
      grid = Grid((10, 10))
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)))
      u = GridVariable(torch.zeros((10, 10)), offset=(0.5, 0.5), grid, bc)


    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    def __init__(self, types: Sequence[Tuple[str, str]]):

        ndim = len(types)
        values = ((0.0, 0.0),) * ndim
        super(HomogeneousBoundaryConditions, self).__init__(types, values)


def is_bc_periodic_boundary_conditions(bc: BoundaryConditions, dim: int) -> bool:
    """Returns true if scalar has periodic bc along axis."""
    if bc is None:
        return False
    elif bc.types[dim][0] != BCType.PERIODIC:
        return False
    elif bc.types[dim][0] == BCType.PERIODIC and bc.types[dim][0] != bc.types[dim][1]:
        raise ValueError(
            "periodic boundary conditions must be the same on both sides of the axis"
        )
    return True


def is_periodic_boundary_conditions(c: GridVariable, dim: int) -> bool:
    """Returns true if scalar has periodic bc along axis."""
    return is_bc_periodic_boundary_conditions(c.bc, dim)


# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> BoundaryConditions:
    """Returns periodic BCs for a variable with `ndim` spatial dimension."""
    return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def dirichlet_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> BoundaryConditions:
    """Returns Dirichelt BCs for a variable with `ndim` spatial dimension.

    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim
        )
    else:
        return ConstantBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim, bc_vals
        )


def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> BoundaryConditions:
    """Returns Neumann BCs for a variable with `ndim` spatial dimension.

    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(((BCType.NEUMANN, BCType.NEUMANN),) * ndim)
    else:
        return ConstantBoundaryConditions(
            ((BCType.NEUMANN, BCType.NEUMANN),) * ndim, bc_vals
        )


def dirichlet_and_periodic_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
) -> BoundaryConditions:
    """Returns BCs Dirichlet for dimension 0 and Periodic for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET), (BCType.PERIODIC, BCType.PERIODIC))
        )
    else:
        return ConstantBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET), (BCType.PERIODIC, BCType.PERIODIC)),
            (bc_vals, (None, None)),
        )


def periodic_and_neumann_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
) -> BoundaryConditions:
    """Returns BCs periodic for dimension 0 and Neumann for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN))
        )
    else:
        return ConstantBoundaryConditions(
            ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
            ((None, None), bc_vals),
        )


def _count_bc_components(bc: BoundaryConditions) -> int:
    """Counts the number of components in the boundary conditions.

    Returns:
        The number of components in the boundary conditions.
    """
    count = 0
    ndim = len(bc.types)
    for axis in range(ndim):  # ndim
        if len(bc.types[axis]) != 2:
            raise ValueError(
                f"Boundary conditions for axis {axis} must have two values got {len(bc.types[axis])}."
            )
        count += len(bc.types[axis])
    return count


def consistent_boundary_conditions_grid(
    grid, *arrays: GridVariable
) -> Tuple[GridVariable, ...]:
    """Returns the updated boundary condition if the number of components is inconsistent
    with the grid
    """
    bc_counts = []
    for array in arrays:
        bc_counts.append(_count_bc_components(array.bc))
    bc_count = bc_counts[0]
    if any(bc_counts[i] != bc_count for i in range(1, len(bc_counts))):
        raise Exception("Boundary condition counts are inconsistent")
    if any(bc_counts[i] != 2 * grid.ndim for i in range(len(bc_counts))):
        raise ValueError(
            f"Boundary condition counts {bc_counts} are inconsistent with grid dimensions {grid.ndim}"
        )
    return arrays


def consistent_boundary_conditions_gridvariable(
    *arrays: GridVariable,
) -> Tuple[str, ...]:
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


def get_pressure_bc_from_velocity_bc(
    bcs: Tuple[BoundaryConditions, ...],
) -> HomogeneousBoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity BCs.
    if the velocity BC is periodic, the pressure BC is periodic.
    if the velocity BC is nonperiodic, the pressure BC is zero flux (homogeneous Neumann).
    """
    # assumes that if the boundary is not periodic, pressure BC is zero flux.
    pressure_bc_types = []

    for velocity_bc in bcs:
        if is_bc_periodic_boundary_conditions(velocity_bc, 0):
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


def get_advection_flux_bc_from_velocity_and_scalar_bc(
    u_bc: BoundaryConditions, c_bc: BoundaryConditions, flux_direction: int
) -> ConstantBoundaryConditions:
    """Returns advection flux boundary conditions for the specified velocity.

    Infers advection flux boundary condition in flux direction
    from scalar c and velocity u in direction flux_direction.
    The flux boundary condition should be used only to compute divergence.
    If the boundaries are periodic, flux is periodic.
    In nonperiodic case, flux boundary parallel to flux direction is
    homogeneous dirichlet.
    In nonperiodic case if flux direction is normal to the wall, the
    function supports multiple cases:
      1) Nonporous boundary, corresponding to homogeneous flux bc.
      2) Porous boundary with constant flux, corresponding to
        both the velocity and scalar with Homogeneous Neumann bc.
      3) Non-homogeneous Dirichlet velocity with Dirichlet scalar bc.

    Args:
      u_bc: bc of the velocity component in flux_direction.
      c_bc: bc of the scalar to advect.
      flux_direction: direction of velocity.

    Returns:
      BoundaryCondition instance for advection flux of c in flux_direction.

    Example:
    >>> u_bc = ConstantBoundaryConditions(((BCType.DIRICHLET, BCType.DIRICHLET),),  ((1.0, 2.0),))

    >>> c_bc = ConstantBoundaryConditions(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.5, 1.5),))

    >>> flux_bc = get_advection_flux_bc_from_velocity_and_scalar_bc(u_bc, c_bc,0)
    # flux_bc will have values (0.5, 3.0) = (1.0*0.5, 2.0*1.5)
    """
    flux_bc_types = []
    flux_bc_values = []

    # Handle both homogeneous and non-homogeneous boundary conditions
    if isinstance(u_bc, HomogeneousBoundaryConditions):
        u_values = tuple((0.0, 0.0) for _ in range(u_bc.ndim))
    elif isinstance(u_bc, ConstantBoundaryConditions):
        u_values = u_bc._values
    else:
        raise NotImplementedError(
            f"Flux boundary condition is not implemented for velocity with {type(u_bc)}"
        )

    if isinstance(c_bc, HomogeneousBoundaryConditions):
        c_values = tuple((0.0, 0.0) for _ in range(c_bc.ndim))
    elif isinstance(c_bc, ConstantBoundaryConditions):
        c_values = c_bc._values
    else:
        raise NotImplementedError(
            f"Flux boundary condition is not implemented for scalar with {type(c_bc)}"
        )

    for axis in range(c_bc.ndim):
        if u_bc.types[axis][0] == "periodic":
            flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
            flux_bc_values.append((None, None))
        elif flux_direction != axis:
            # Flux boundary condition parallel to flux direction
            # Set to homogeneous Dirichlet as it doesn't affect divergence computation
            flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
            flux_bc_values.append((0.0, 0.0))
        else:
            # Flux direction is normal to the boundary
            flux_bc_types_ax = []
            flux_bc_values_ax = []

            for i in range(2):  # lower and upper boundary
                u_type = u_bc.types[axis][i]
                c_type = c_bc.types[axis][i]
                u_val = u_values[axis][i] if u_values[axis][i] is not None else 0.0
                c_val = c_values[axis][i] if c_values[axis][i] is not None else 0.0

                # Case 1: Dirichlet velocity with Dirichlet scalar
                if u_type == BCType.DIRICHLET and c_type == BCType.DIRICHLET:
                    # Flux = u * c at the boundary
                    flux_val = u_val * c_val
                    flux_bc_types_ax.append(BCType.DIRICHLET)
                    flux_bc_values_ax.append(flux_val)

                # Case 2: Neumann velocity with Neumann scalar (zero flux condition)
                elif u_type == BCType.NEUMANN and c_type == BCType.NEUMANN:
                    # For zero flux: du/dn = 0 and dc/dn = 0 implies d(uc)/dn = 0
                    if not math.isclose(u_val, 0.0) or not math.isclose(c_val, 0.0):
                        raise NotImplementedError(
                            "Non-homogeneous Neumann boundary conditions for flux "
                            "are not yet implemented"
                        )
                    flux_bc_types_ax.append(BCType.NEUMANN)
                    flux_bc_values_ax.append(0.0)

                # Case 3: Mixed boundary conditions
                elif u_type == BCType.DIRICHLET and c_type == BCType.NEUMANN:
                    # If u is specified and dc/dn is specified, we can compute flux
                    # flux = u * c, but c is not directly specified
                    # This requires more complex handling - for now, raise error
                    raise NotImplementedError(
                        "Mixed Dirichlet velocity and Neumann scalar boundary "
                        "conditions are not yet implemented"
                    )

                elif u_type == BCType.NEUMANN and c_type == BCType.DIRICHLET:
                    # If du/dn is specified and c is specified
                    # This also requires more complex handling
                    raise NotImplementedError(
                        "Mixed Neumann velocity and Dirichlet scalar boundary "
                        "conditions are not yet implemented"
                    )

                else:
                    raise NotImplementedError(
                        f"Flux boundary condition is not implemented for "
                        f"u_bc={u_type} with value={u_val}, "
                        f"c_bc={c_type} with value={c_val}"
                    )

            flux_bc_types.append(tuple(flux_bc_types_ax))
            flux_bc_values.append(tuple(flux_bc_values_ax))

    return ConstantBoundaryConditions(flux_bc_types, flux_bc_values)


def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable, flux_direction: int
) -> ConstantBoundaryConditions:
    return get_advection_flux_bc_from_velocity_and_scalar_bc(u.bc, c.bc, flux_direction)
