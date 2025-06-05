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


import math
from typing import Callable, Optional, Tuple
from functools import partial
import torch
import torch.nn as nn

import torch_cfd.finite_differences as fdm

from torch_cfd import boundaries, grids

Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
FluxInterpFn = Callable[[GridVariable, GridVariableVector, float], GridVariable]

def default(value, d):
    return d if value is None else value

def safe_div(x, y, default_numerator=1):
    """Safe division of `Array`'s."""
    return x / torch.where(y != 0, y, default_numerator)


def van_leer_limiter(r):
    """Van-leer flux limiter."""
    return torch.where(r > 0, safe_div(2 * r, 1 + r), 0.0)


class Upwind(nn.Module):
    """Upwind interpolation module for scalar fields.

    Upwind interpolation of a scalar field `c` to a 
    target offset based on the velocity field `v`. The interpolation is done axis-wise and uses the upwind scheme where values are taken from upstream cells based on the flow direction.

    The module identifies the interpolation axis (must be a single axis) and selects values from the previous cell for positive velocity or the next cell for negative velocity along that axis.

    Args:
        grid: The computational grid on which interpolation is performed (this is only for testing purposes).
        offset: Target offset to which scalar fields will be interpolated during
            forward passes. Must be a tuple of floats with the same length as the grid dimensions.

    Note:
        The target offset should be pre-specified at __init__().

    Example:
        >>> grid = Grid(shape=(64, 64))
        >>> upwind = Upwind(grid=grid, offset=(1.0, 0.5))
        >>> result = upwind(c, v, dt)
    """

    def __init__(
        self,
        grid: Grid,
        target_offset: Tuple[float, ...] = (0.5, 0.5),
    ):
        super().__init__()
        self.grid = grid
        self.target_offset = target_offset # this is the offset to which we will interpolate c

    def forward(
        self,
        c: GridVariable,
        v: GridVariableVector,
        dt: Optional[float] = None,
    ) -> GridVariable:
        """
        Args:
            c: GridVariable representing the scalar field to be interpolated.
            v: GridVariableVector representing the velocity field.
            dt: Time step size (not used in this interpolation).
        Returns:
            A GridVariable containing the values of `c` after interpolation to `self.target_offset`.
        """
        if c.offset == self.target_offset:
            return c
        interpolation_axes = tuple(
            axis
            for axis, (cur, tgt) in enumerate(zip(c.offset, self.target_offset))
            if cur != tgt
        )
        if len(interpolation_axes) != 1:
            raise ValueError(
                f"Offsets must differ in one entry, got {c.offset} vs {self.target_offset}"
            )
        (dim,) = interpolation_axes
        u = v[dim]

        offset_delta = self.target_offset[dim] - c.offset[dim]
        # self.target_offset[axis] should be the same with offset_v[axis][axis], which offset_v = ((1.5, 0.5), (0.5, 1.5)) for aligned_v
        # the original v.offset is ((1.0, 0.5), (0.5, 1.0))
        # which in turn should be the same with the original implementation u.offset[axis] - c.offset[axis]
        if int(offset_delta) == offset_delta:
            return c.shift(int(offset_delta), dim)
        floor = int(math.floor(offset_delta))
        ceil = int(math.ceil(offset_delta))
        c_floor = c.shift(floor, dim).data
        c_ceil = c.shift(ceil, dim).data
        return GridVariable(torch.where(u.data > 0, c_floor, c_ceil), self.target_offset, c.grid, c.bc)


class LaxWendroff(nn.Module):
    """Lax_Wendroff interpolation of scalar field `c` to `offset` based on velocity field `v`.

    Interpolates values of `c` to `offset` in two steps:
    1) Identifies the axis along which `c` is interpolated. (must be single axis)
    2) For positive (negative) velocity along interpolation axis uses value from
       the previous (next) cell along that axis plus a correction originating
       from expansion of the solution at the half step-size.

    This method is second order accurate with fixed coefficients and hence can't
    be monotonic due to Godunov's theorem.
    https://en.wikipedia.org/wiki/Godunov%27s_theorem

    Lax-Wendroff method can be used to form monotonic schemes when augmented with
    a flux limiter. See https://en.wikipedia.org/wiki/Flux_limiter

    Args: 
        grid: The computational grid on which interpolation is performed, only used for step.
        offset: Target offset to which scalar fields will be interpolated during
            forward passes. Target offset have the same length as `c.offset` in forward() and differ in at most one entry.

    Raises:
      ValueError: if `offset` and `c.offset` differ in more than one entry.
    """

    def __init__(
        self,
        grid: Grid,
        target_offset: Tuple[float, ...],
    ):
        super().__init__()
        self.grid = grid  
        self.target_offset = target_offset 

    def forward(
        self,
        c: GridVariable,
        v: GridVariableVector,
        dt: float = 1.0,
    ) -> GridVariable:
        """
        Args:
         c: quantitity to be interpolated.
         v: velocity field with offsets at faces of `c`. One of the components must have the same offset as the target offset.
        dt: size of the time step. Not used.

        Returns:
            A tensor containing the values of `c` after interpolation to `self.target_offset`.
            The returned value will have offset equal to `self.target_offset`.
        """
        if c.offset == self.target_offset:
            return c
        interpolation_axes = tuple(
            axis
            for axis, (cur, tgt) in enumerate(zip(c.offset, self.target_offset))
            if cur != tgt
        )
        if len(interpolation_axes) != 1:
            raise ValueError(
                f"Offsets must differ in one entry, got {c.offset} vs {self.target_offset}"
            )
        (dim,) = interpolation_axes
        u = v[dim]
        offset_delta = self.target_offset[dim] - c.offset[dim]
        # offset_delta = u.offset[axis] - c.offset[axis]
        floor = int(math.floor(offset_delta))
        ceil = int(math.ceil(offset_delta))
        # grid = grids.consistent_grid_arrays(c, u)
        courant = (dt / c.grid.step[dim]) * u.data
        c_floor = c.shift(floor, dim).data
        c_ceil = c.shift(ceil, dim).data
        pos = c_floor + 0.5 * (1 - courant) * (c_ceil - c_floor)
        neg = c_ceil - 0.5 * (1 + courant) * (c_ceil - c_floor)
        return GridVariable(torch.where(u.data > 0, pos, neg), 
                            self.target_offset, c.grid, c.bc)

class AdvectAligned(nn.Module):
    """
    Computes fluxes and the associated advection for aligned `cs` and `v`.

    The values `cs` should consist of a single quantity `c` that has been
    interpolated to the offset of the components of `v`. The components of `v` and
    `cs` should be located at the faces of a single (possibly offset) grid cell.
    We compute the advection as the divergence of the flux on this control volume.

    The boundary condition on the flux is inherited from the scalar quantity `c`.

    A typical example in three dimensions would have

    ```
    cs[0].offset == v[0].offset == (1., .5, .5)
    cs[1].offset == v[1].offset == (.5, 1., .5)
    cs[2].offset == v[2].offset == (.5, .5, 1.)
    ```

    In this case, the returned advection term would have offset `(.5, .5, .5)`.

    Args:
        grid: The computational grid (only for testing purposes).
        bcs_c: Boundary conditions for the gradient of scalar field components
        bcs_v: Boundary conditions for each velocity component
        offsets: Offsets for the control volume faces where `cs` and `v` are defined.
    """

    def __init__(
        self,
        grid: Grid,
        bcs_c: Tuple[boundaries.BoundaryConditions, ...],
        bcs_v: Tuple[boundaries.BoundaryConditions, ...],
        offsets: Tuple[Tuple[float, ...], ...] = ((1.5, 0.5), (0.5, 1.5)),
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        self.bcs_c = bcs_c
        self.bcs_v = bcs_v
        self.offsets = offsets

        # Pre-compute whether we have periodic boundary conditions
        self.is_periodic = all(
            boundaries.is_bc_periodic_boundary_conditions(bc, i)
            for i, bc in enumerate(bcs_c)
        )

        # Pre-compute boundary condition functions for flux if not periodic
        if not self.is_periodic:
            self.flux_bcs = tuple(
                boundaries.get_advection_flux_bc_from_velocity_and_scalar_bc(
                    bcs_v[i], bcs_c[i], i
                )
                for i in range(len(bcs_v))
            )
        else:
            self.flux_bcs = tuple(
                boundaries.periodic_boundary_conditions(ndim=grid.ndim)
                for _ in range(len(bcs_c))
            )

    def forward(self, cs: GridVariableVector, v: GridVariableVector) -> GridVariable:
        """
        Compute advection of aligned cs and v.

        Args:
            cs: A sequence of GridVariables; a single value `c` that has been
                interpolated so that it is aligned with each component of `v`.
            v: A sequence of GridVariable describing a velocity field. Should be
               defined on the same Grid as cs.

        Returns:
            An GridVariable containing the time derivative of `c` due to advection by `v`.

        Raises:
            ValueError: `cs` and `v` have different numbers of components.
        """
        if len(cs) != len(v):
            raise ValueError(
                "`cs` and `v` must have the same length; "
                f"got {len(cs)} vs. {len(v)}."
            )

        # Compute flux: cu
        # if cs and v have different boundary conditions, 
        # flux's bc will become None
        flux = GridVariableVector(tuple(c * u for c, u in zip(cs, v)))

        # Apply boundary conditions to flux if not periodic
        flux = GridVariableVector(
                tuple(bc.impose_bc(f) for f, bc in zip(flux, self.flux_bcs))
            )

        # Return negative divergence of flux
        # after taking divergence the bc becomes None
        return -fdm.divergence(flux)


class LinearInterpolation(nn.Module):
    """Multi-linear interpolation of `c` to `offset`.

    Args:
        - offset: Target offset to which scalar fields will be interpolated during forward passes.
        - grid: The computational grid on which interpolation is performed (this is only for testing purposes).
    """

    def __init__(
        self,
        grid: Grid,
        target_offset: Tuple[float, ...] = (0.5, 0.5),
        bc: Optional[boundaries.BoundaryConditions] = None,
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        self.bc = bc
        self.target_offset = target_offset
        self.interpolant = partial(fdm.linear, offset=target_offset)

    def forward(self, c: GridVariable, *args, **kwargs) -> GridVariable:
        """
        Args:
            c: quantitity to be interpolated.

        Returns:
            An `GridVariable` containing the values of `c` after linear interpolation to `offset`. The returned value will have offset equal to `offset`.
        """
        return self.interpolant(c)


class TVDInterpolation(nn.Module):
    """Combines low and high accuracy interpolators to get Total Variation Diminishing (TVD) Interpolator.

    Generates high accuracy interpolator by combining stable lwo accuracy `upwind`
    interpolation and high accuracy (but not guaranteed to be stable)
    `interpolation_fn` to obtain stable higher order method. This implementation
    follows the procedure outined in:
    http://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2012/Chapter_4.pdf

    Args:
        target_offset: offset to which we will interpolate `c`. 
        Must have the same length as `c.offset` and differ in at most one entry. This offset should interface as other interpolation methods (take `c`, `v` and `dt` arguments and return value of `c` at offset `offset`).
        limiter: flux limiter function that evaluates the portion of the correction (high_accuracy - low_accuracy) to add to low_accuracy solution based on the ratio of the consecutive gradients. 
        Takes array as input and return array of weights. For more details see:
        https://en.wikipedia.org/wiki/Flux_limiter
        
    """

    def __init__(
        self,
        grid: Grid,
        target_offset: Tuple[float, ...],
        low_interp: FluxInterpFn = None,
        high_interp: FluxInterpFn = None,
        limiter: Callable = van_leer_limiter,
    ):
        super().__init__()
        self.grid = grid
        self.low_interp = Upwind(grid, target_offset=target_offset) if low_interp is None else low_interp
        self.high_interp = LaxWendroff(grid, target_offset=target_offset) if high_interp is None else high_interp
        self.limiter = limiter
        self.target_offset = target_offset

    def forward(
        self,
        c: GridVariable,
        v: GridVariableVector,
        dt: float,
    ) -> GridVariable:
        """
        Args:
            c: GridVariable representing the scalar field to be interpolated.
            v: GridVariableVector representing the velocity field.
            dt: Time step size (not used in this interpolation).

        Returns: 
            Interpolated scalar field c to a target offset using Van Leer flux limiting, which uses a combination of high and low order methods to produce monotonic interpolation method."""
        for axis, axis_offset in enumerate(self.target_offset):
            interpolation_offset = tuple(
                [
                    c_offset if i != axis else axis_offset
                    for i, c_offset in enumerate(c.offset)
                ]
            )
            if interpolation_offset != c.offset:
                if interpolation_offset[axis] - c.offset[axis] != 0.5:
                    raise NotImplementedError(
                        "Only forward interpolation to control volume faces is supported."
                    )
                c_low = self.low_interp(c, v, dt)
                c_high = self.high_interp(c, v, dt)
                c_left = c.shift(-1, axis).data
                c_right = c.shift(1, axis).data
                c_next_right = c.shift(2, axis).data
                pos_r = safe_div(c - c_left, c_right - c)
                neg_r = safe_div(c_next_right - c_right, c_right - c)
                pos_phi = self.limiter(pos_r).data
                neg_phi = self.limiter(neg_r).data
                u = v[axis]
                phi = torch.where(u > 0, pos_phi, neg_phi)
                interpolated = c_low - (c_low - c_high) * phi
                c = GridVariable(interpolated.data, interpolation_offset, c.grid, c.bc)
        return c


class AdvectionBase(nn.Module):
    """
    Base class for advection modules.

    Ported from Jax-CFD advect_general and _advect_aligned function
    The user need to implement specifies
    - _flux_interp
    - _velocity_interp

    Computes advection of a scalar quantity `c` by the velocity field `v`.

    This function follows the following procedure:

    1. Interpolate each component of `v` to the corresponding face of the
        control volume centered on `c`.
    2. Interpolate `c` to the same control volume faces.
    3. Compute the flux `cu` using the aligned values.
    4. Set the boundary condition on flux, which is inhereited from `c`.
    5. Return the negative divergence of the flux.

    Args: 
        grid: Grid.
        offset: the current scalar field `c` to be advected.
        bc_c: Boundary conditions for the scalar field `c`.
        bc_v: Boundary conditions for each component of the velocity field `v`.
        limiter: Optional flux limiter function to be used in the interpolation.

    """

    def __init__(self, 
                 grid: Grid,
                 offset: Tuple[float, ...],
                 bc_c: boundaries.BoundaryConditions,
                 bc_v: Tuple[boundaries.BoundaryConditions, ...],
                 limiter: Optional[Callable] = None,
                 ) -> None:
        super().__init__()
        self.grid = grid
        self.offset = offset if offset is not None else (0.5,) * grid.ndim
        self.limiter = limiter
        self.target_offsets = grids.control_volume_offsets(*offset)
        bc_c = default(bc_c, boundaries.periodic_boundary_conditions(ndim=grid.ndim))
        bc_v = default(
            bc_v,
            tuple(
                boundaries.periodic_boundary_conditions(ndim=grid.ndim)
                for _ in range(grid.ndim)
            ),
        )
        self.advect_aligned = AdvectAligned(
            grid=grid, bcs_c=(bc_c, bc_c), bcs_v=bc_v, offsets=self.target_offsets)
        self._flux_interp = nn.ModuleList() # placeholder
        self._velocity_interp = nn.ModuleList() # placeholder

    def __post_init__(self):
        assert len(self._flux_interp) == len(self.target_offsets)
        assert len(self._velocity_interp) == len(self.target_offsets)

        for dim, interp in enumerate(self._flux_interp):
            assert interp.target_offset == self.target_offsets[dim], f"Expected flux interpolation for dimension {dim} to have target offset {self.target_offsets[dim]}, but got {interp.target_offset}."
        
        for dim, interp in enumerate(self._velocity_interp):
            assert interp.target_offset == self.target_offsets[dim], f"Expected velocity interpolation for dimension {dim} to have target offset {self.target_offsets[dim]}, but got {interp.target_offset}."

    def flux_interp(
        self,
        c: GridVariable,
        v: GridVariableVector,
        dt: float,
    ) -> GridVariableVector:
        return GridVariableVector(
            tuple(interp(c, v, dt) for interp in self._flux_interp)
        )

    def velocity_interp(
        self, v: GridVariableVector, *args, **kwargs
    ) -> GridVariableVector:
        """Interpolate the velocity field `v` to the target offsets."""
        return GridVariableVector(tuple(interp(u) for interp, u in zip(self._velocity_interp, v)))

    def forward(
        self,
        c: GridVariable,
        v: GridVariableVector,
        dt: float = 1.0,
    ):
        """
        Args:
            c: the scalar field to be advected.
            v: representing the velocity field.
        
        Returns:
            An GridVariable containing the time derivative of `c` due to advection by `v`.
        
        """
        aligned_v = self.velocity_interp(v)

        aligned_c = self.flux_interp(c, aligned_v, dt)

        return self.advect_aligned(aligned_c, aligned_v)


class AdvectionLinear(AdvectionBase):

    def __init__(
        self,
        grid: Grid,
        offset = (0.5, 0.5),
        bc_c: boundaries.BoundaryConditions = boundaries.periodic_boundary_conditions(
            ndim=2
        ),
        bc_v: Tuple[boundaries.BoundaryConditions, ...] = (
            boundaries.periodic_boundary_conditions(ndim=2),
            boundaries.periodic_boundary_conditions(ndim=2),
        ),
        **kwargs,
    ):
        super().__init__(grid, offset, bc_c, bc_v)
        self._flux_interp = nn.ModuleList(
            LinearInterpolation(grid, target_offset=offset)
            for offset in self.target_offsets)

        self._velocity_interp = nn.ModuleList(
            LinearInterpolation(grid, target_offset=offset)
            for offset in self.target_offsets)
        

class AdvectionUpwind(AdvectionBase):
    """
    Ported from Jax-CFD advect_general and _advect_aligned function
    using upwind flux interpolation.
    The initialization specifies
    - flux_interp: a Upwind interpolation for each component of the velocity field `v`.
    - velocity_interp: a LinearInterpolation for each component of the velocity field `v`.

    Args: 
        - offset: current offset of the scalar field `c` to be advected.
    
    Returns:
        Aligned advection of the scalar field `c` by the velocity field `v` using the target offsets on the control volume faces.
    """

    def __init__(
        self,
        grid: Grid,
        offset: Tuple[float, ...] = (0.5, 0.5),
        bc_c: boundaries.BoundaryConditions = boundaries.periodic_boundary_conditions(
            ndim=2
        ),
        bc_v: Tuple[boundaries.BoundaryConditions, ...] = (
            boundaries.periodic_boundary_conditions(ndim=2),
            boundaries.periodic_boundary_conditions(ndim=2),
        ),
        **kwargs,
    ):
        super().__init__(grid, offset, bc_c, bc_v)
        self._flux_interp = nn.ModuleList(
            Upwind(grid, target_offset=offset)
            for offset in self.target_offsets
        )

        self._velocity_interp = nn.ModuleList(
            LinearInterpolation(grid, target_offset=offset)
            for offset in self.target_offsets
        )


class AdvectionVanLeer(AdvectionBase):
    """
    Ported from Jax-CFD advect_general and _advect_aligned function
    using van_leer flux limiter.
    The initialization specifies
    - flux_interp: a TVDInterpolation with Upwind and LaxWendroff methods
    - velocity_interp: a LinearInterpolation for each component of the velocity field `v`.

    Args: 
        - offset: current offset of the scalar field `c` to be advected.
    
    Returns:
        Aligned advection of the scalar field `c` by the velocity field `v` using the target offsets on the control volume faces.
    """

    def __init__(
        self,
        grid: Grid,
        offset: Tuple[float, ...] = (0.5, 0.5),
        bc_c: boundaries.BoundaryConditions = boundaries.periodic_boundary_conditions(
            ndim=2
        ),
        bc_v: Tuple[boundaries.BoundaryConditions, ...] = (
            boundaries.periodic_boundary_conditions(ndim=2),
            boundaries.periodic_boundary_conditions(ndim=2),
        ),
        limiter: Callable = van_leer_limiter,
        **kwargs,
    ):
        super().__init__(grid, offset, bc_c, bc_v, limiter)
        self._flux_interp = nn.ModuleList(
            TVDInterpolation(
                grid,
                target_offset=offset,
                limiter=limiter,
            )
            for offset in self.target_offsets
        )

        self._velocity_interp = nn.ModuleList(
            LinearInterpolation(grid, target_offset=offset, bc=bc)
            for offset, bc in zip(self.target_offsets, bc_v)
        )

class ConvectionVector(nn.Module):
    """Computes convection of a vector field `v` by the velocity field `u`.


    This module follows the following procedure:

      1. Interpolate each component of `u` to the corresponding face of the
         control volume centered on `v`.
      2. Interpolate `v` to the same control volume faces.
      3. Compute the flux `vu` using the aligned values.
      4. Set the boundary condition on flux, which is inhereited from `v`.
      5. Return the negative divergence of the flux.

    Notes:
      - In FVM of 2D NSE, v = u.
      - The velocity field `u` is assumed to be defined on a staggered grid
            (MAC grid) with the same offsets as `v`.
      - before the computation, the grid information is completely know, so one does not have to check dimensions or perform manipulation on-the-fly.

    Args:
        offsets: the offsets of the velocity field (MAC grid), should be the same for both u and v.
        bcs: boundary conditions for the velocity field to be convected. (u's boundary conditions do not matter)

    """

    def __init__(
        self,
        grid: Grid,
        offsets: Tuple[Tuple[float, ...], ...] = ((1.0, 0.5), (0.5, 1.0)),
        bcs: Tuple[boundaries.BoundaryConditions, ...] = (
            boundaries.periodic_boundary_conditions(ndim=2),
            boundaries.periodic_boundary_conditions(ndim=2),
        ),
        limiter: Callable = van_leer_limiter,
        **kwargs,
    ):
        super().__init__()

        self.advect = nn.ModuleList(
            AdvectionVanLeer(
                grid=grid,
                offset=offset,
                bc_c=bc,
                bc_v=bcs,
                limiter=limiter,
            )
            for bc, offset in zip(bcs, offsets)
        )

    def forward(
        self, v: GridVariableVector, u: GridVariableVector, dt: Optional[float] = None
    ) -> GridVariableVector:
        r"""
        Compute the convection of a vector field v with u.
        Basically computes (u \cdot \nabla) v  as
        the divergence of the flux on the control volume.

        Args:
            v: GridVariableVector (also velocity) to be advected/transported.
            u: GridVariableVector velocity field
            dt: Time step (overrides the instance dt if provided)

        Returns:
            The directional derivative `v` due to advection by `u`.
        """

        return GridVariableVector(
            tuple(self.advect[i](c, u, dt) for i, c in enumerate(v))
        )
