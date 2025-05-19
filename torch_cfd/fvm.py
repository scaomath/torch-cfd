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


from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import torch_cfd.finite_differences as fdm
import torch_cfd.interpolation as interpolation
from torch_cfd import boundaries, forcings, grids, pressure


Grid = grids.Grid
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = Callable[
    [GridVariable, Tuple[float, ...], GridVariableVector, float], GridVariable
]
ForcingFn = forcings.ForcingFn


def _advect_aligned(cs: GridVariableVector, v: GridVariableVector) -> GridArray:
    """Computes fluxes and the associated advection for aligned `cs` and `v`.

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
      cs: a sequence of `GridArray`s; a single value `c` that has been
        interpolated so that it is aligned with each component of `v`.
      v: a sequence of `GridArrays` describing a velocity field. Should be defined
        on the same Grid as cs.

    Returns:
      An `GridArray` containing the time derivative of `c` due to advection by
      `v`.

    Raises:
      ValueError: `cs` and `v` have different numbers of components.
      AlignmentError: if the components of `cs` are not aligned with those of `v`.
    """
    # TODO(jamieas): add more sophisticated alignment checks, ensuring that the
    # values are located on the faces of a control volume.
    if len(cs) != len(v):
        raise ValueError(
            "`cs` and `v` must have the same length;" f"got {len(cs)} vs. {len(v)}."
        )
    flux = GridArrayVector(tuple(c.array * u.array for c, u in zip(cs, v)))
    bcs = tuple(
        boundaries.get_advection_flux_bc_from_velocity_and_scalar(v[i], cs[i], i)
        for i in range(len(v))
    )
    flux = GridVariableVector(tuple(bc.impose_bc(f) for f, bc in zip(flux, bcs)))
    return -fdm.divergence(flux)


def advect_general(
    c: GridVariable,
    v: GridVariableVector,
    u_interpolation_fn: InterpolationFn,
    c_interpolation_fn: InterpolationFn,
    dt: Optional[float] = None,
) -> GridArray:
    """Computes advection of a scalar quantity `c` by the velocity field `v`.

    This function follows the following procedure:

      1. Interpolate each component of `v` to the corresponding face of the
         control volume centered on `c`.
      2. Interpolate `c` to the same control volume faces.
      3. Compute the flux `cu` using the aligned values.
      4. Set the boundary condition on flux, which is inhereited from `c`.
      5. Return the negative divergence of the flux.

    Args:
      c: the quantity to be transported.
      v: a velocity field. Should be defined on the same Grid as c.
      u_interpolation_fn: method for interpolating velocity field `v`.
      c_interpolation_fn: method for interpolating scalar field `c`.
      dt: unused time-step.

    Returns:
      The time derivative of `c` due to advection by `v`.
    """
    if not boundaries.has_all_periodic_boundary_conditions(c):
        raise NotImplementedError(
            "Non-periodic boundary conditions are not implemented."
        )
    target_offsets = grids.control_volume_offsets(c)
    aligned_v = GridVariableVector(
        tuple(
            u_interpolation_fn(u, target_offset, v, dt)
            for u, target_offset in zip(v, target_offsets)
        )
    )
    aligned_c = GridVariableVector(
        tuple(
            c_interpolation_fn(c, target_offset, aligned_v, dt)
            for target_offset in target_offsets
        )
    )
    return _advect_aligned(aligned_c, aligned_v)


def advect_van_leer_using_limiters(
    c: GridVariable, v: GridVariableVector, dt: float
) -> GridArray:
    """Implements Van-Leer advection by applying TVD limiter to Lax-Wendroff."""
    c_interpolation_fn = interpolation.apply_tvd_limiter(
        interpolation.lax_wendroff, limiter=interpolation.van_leer_limiter
    )
    return advect_general(c, v, interpolation.linear, c_interpolation_fn, dt)


def convect(v, dt):
    return GridArrayVector(tuple(advect_van_leer_using_limiters(u, v, dt) for u in v))


def diffuse(w: GridVariable, nu: float) -> GridArray:
    """Returns the rate of change in a concentration `c` due to diffusion."""
    return nu * fdm.laplacian(w)


def diffuse_velocity(v, *args):
    return GridArrayVector(tuple(diffuse(u, *args) for u in v))


def wrap_field_same_bcs(v, field_ref):
    return GridVariableVector(
        tuple(GridVariable(a, w.bc) for a, w in zip(v, field_ref))
    )


class ProjectionExplicitODE(nn.Module):
    r"""Navier-Stokes equation in 2D with explicit stepping and a pressure projection (discrete Helmholtz decomposition by modding the gradient of a Laplacian inverse of the extra divergence).

    \partial u/ \partial t = explicit_terms(u)
    u <- pressure_projection(u)
    """

    def explicit_terms(self, *, u):
        """
        Explicit forcing term as du/dt.
        * allows extra arguments to be passed.
        """
        raise NotImplementedError

    def pressure_projection(self, *, u):
        raise NotImplementedError
    
    def forward(self, u: GridVariableVector, dt: float) -> GridVariableVector:
        """Perform one time step.

        Args:
            u: Initial state (velocity field)
            dt: Time step size

        Returns:
            Updated velocity field after one time step
        """
        raise NotImplementedError


class RKStepper(nn.Module):
    """Base class for Explicit Runge-Kutta stepper.

    Input:
        tableau: Butcher tableau (a, b) for the Runge-Kutta method as a dictionary
        method: String name of built-in RK method if tableau not provided

    Examples:
        stepper = RKStepper.from_name("classic_rk4", equation, ...)
    """

    _METHOD_MAP = {
        "forward_euler": {"a": [], "b": [1.0]},
        "midpoint": {"a": [[1 / 2]], "b": [0, 1.0]},
        "heun_rk2": {"a": [[1.0]], "b": [1 / 2, 1 / 2]},
        "classic_rk4": {
            "a": [[1 / 2], [0.0, 1 / 2], [0.0, 0.0, 1.0]],
            "b": [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        },
    }

    def __init__(
        self,
        tableau: Optional[Dict[str, List]] = None,
        method: str = None,
        dtype: Optional[torch.dtype] = torch.float32,
        requires_grad=False,
        **kwargs,
    ):
        super().__init__()

        self._tableau = None
        self._method = None
        self.dtype = dtype
        self.requires_grad = requires_grad

        # Set the tableau, either directly or from method name
        if tableau is not None:
            self.tableau = tableau
        else:
            self.method = method
        # print("Using Butcher tableau:")
        # print("\n".join([f"{k}: {v}" for k, v in self._tableau.items()]))
        self._set_params(self._tableau)

    @property
    def method(self):
        """Get the current Runge-Kutta method name."""
        return self._method

    @method.setter
    def method(self, name: str):
        """Set the tableau based on the method name."""
        if name not in self._METHOD_MAP:
            raise ValueError(f"Unknown RK method: {name}")
        self._method = name
        self._tableau = self._METHOD_MAP[name]

    @property
    def tableau(self):
        """Get the current tableau."""
        return self._tableau

    @tableau.setter
    def tableau(self, tab: Dict[str, List]):
        """Set the tableau directly."""
        self._tableau = tab
        self._method = None  # Clear method name when setting tableau directly

    def _set_params(self, tableau: Dict[str, List]):
        """Set the parameters of the Butcher tableau."""
        a, b = tableau["a"], tableau["b"]
        if a.__len__() + 1 != b.__len__():
            raise ValueError("Inconsistent Butcher tableau: len(a) + 1 != len(b)")
        self.params = nn.ParameterDict()
        self.params["a"] = nn.ParameterList()
        for a_ in a:
            self.params["a"].append(
                nn.Parameter(
                    torch.tensor(a_, dtype=self.dtype, requires_grad=self.requires_grad)
                )
            )
        self.params["b"] = nn.Parameter(
            torch.tensor(b, dtype=self.dtype, requires_grad=self.requires_grad)
        )

    @classmethod
    def from_method(
        cls, method: str = "forward_euler", requires_grad: bool = False, **kwargs
    ):
        """Factory method to create an RKStepper by name."""
        return cls(method=method, requires_grad=requires_grad, **kwargs)

    def forward(
        self, u0: GridVariableVector, dt: float, equation: ProjectionExplicitODE
    ) -> GridVariableVector:
        """Perform one time step.

        Args:
            u0: Initial state (velocity field)
            dt: Time step size
            equation: The ODE to solve

        Returns:
            Updated velocity field after one time step
        """
        alpha = self.params["a"]
        beta = self.params["b"]
        num_steps = len(beta)

        u = [None] * num_steps
        k = [None] * num_steps

        # First stage
        u[0] = u0
        k[0] = equation.explicit_terms(u0, dt)

        # Intermediate stages
        for i in range(1, num_steps):
            u_star = GridVariableVector(tuple(v.clone() for v in u0))

            for j in range(i):
                if alpha[i - 1][j] != 0:
                    u_star = u_star + dt * alpha[i - 1][j] * k[j]

            u[i] = equation.pressure_projection(u_star)
            k[i] = equation.explicit_terms(u[i], dt)

        u_star = GridVariableVector(tuple(v.clone() for v in u0))
        for j in range(num_steps):
            if beta[j] != 0:
                u_star = u_star + dt * beta[j] * k[j]

        u_final = equation.pressure_projection(u_star)

        return u_final


class NavierStokes2DFVMProjection(ProjectionExplicitODE):
    r"""incompressible Navier-Stokes velocity pressure formulation

    Runge-Kutta time stepper for the NSE discretized using a MAC grid FVM with a pressure projection Chorin's method. The x- and y-dofs of the velocity
    are on a staggered grid, which is reflected in the offset attr.

    Original implementation in Jax-CFD repository:

    - semi_implicit_navier_stokes in jax_cfd.base.fvm which returns a stepper function `time_stepper(ode, dt)` where `ode` specifies the explicit terms and the pressure projection.
    - The time_stepper is a wrapper function by jax.named_call(
      navier_stokes_rk()) that implements the various Runge-Kutta method according to the Butcher tableau.
    - navier_stokes_rk() implements Runge-Kutta time-stepping for the NSE using the explicit terms and pressure projection with equation as an input where user needs to specify the explicit terms and pressure projection.

    (Original reference listed in Jax-CFD)
    This class implements the reference method (equations 16-21) from:
    "Fast-Projection Methods for the Incompressible Navier-Stokes Equations"
    Fluids 2020, 5, 222; doi:10.3390/fluids5040222
    """

    def __init__(
        self,
        viscosity: float,
        grid: Grid,
        bcs: Optional[Sequence[boundaries.BoundaryConditions]] = None,
        drag: float = 0.0,
        density: float = 1.0,
        convect: Callable = convect,
        forcing: Optional[ForcingFn] = None,
        solver: RKStepper = None,
        **kwargs,
    ):
        """
        Args:
            tableau: Tuple (a, b) where a is the coefficient matrix (list of lists of floats)
                    and b is the weight vector (list of floats)
            equation: Navier-Stokes equation to solve
            requires_grad: Whether parameters should be trainable
        """
        super().__init__()
        self.viscosity = viscosity
        self.density = density
        self.grid = grid
        self.bcs = bcs
        self.drag = drag
        self.convect = convect
        self.forcing = forcing
        self.solver = solver
        self._set_pressure_bc()
        self._projection = pressure.PressureProjection(
            grid=grid,
            bc=self.pressure_bc,
        )

    def _set_pressure_bc(self):

        if self.bcs is None:
            self.bcs = [
                boundaries.HomogeneousBoundaryConditions(
                    (
                        (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
                        (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
                    )
                )
            ] * self.grid.ndim
        self.pressure_bc = boundaries.get_pressure_bc_from_velocity_bc(bcs=self.bcs)

    def _explicit_terms(self, v, dt, **kwargs):
        dv_dt = self.convect(v, dt)
        grid = self.grid
        viscosity = self.viscosity
        density = self.density
        forcing = self.forcing
        dv_dt += diffuse_velocity(v, viscosity / density)
        if forcing is not None:
            dv_dt += GridArrayVector(forcing(grid, v)) / density
        dv_dt = wrap_field_same_bcs(dv_dt, v)
        if self.drag > 0.0:
            dv_dt += -self.drag * v
        return dv_dt

    def explicit_terms(self, *args, **kwargs):
        return self._explicit_terms(*args, **kwargs)

    def pressure_projection(self, *args, **kwargs):
        return self._projection(*args, **kwargs)

    def forward(self, u: GridVariableVector, dt: float) -> GridVariableVector:
        """Perform one time step.

        Args:
            u: Initial state (velocity field)
            dt: Time step size

        Returns:
            Updated velocity field after one time step
        """

        return self.solver(u, dt, self)


def advect_van_leer(
    c: GridVariable,
    v: GridVariableVector,
    dt: float,
    mode: str,
) -> GridArray:
    """
    TODO:
    - [ ] NOT YET IMPLEMENTED in Jax_CFD original


    Computes advection of a scalar quantity `c` by the velocity field `v`.

    Implements Van-Leer flux limiting scheme that uses second order accurate
    approximation of fluxes for smooth regions of the solution. This scheme is
    total variation diminishing (TVD). For regions with high gradients flux
    limitor transformes the scheme into a first order method. For [1] for
    reference. This function follows the following procedure:

      1. Shifts c to offset < 1 if necessary.
      2. Scalar c now has a well defined right-hand (upwind) value.
      3. Computes upwind flux for each direction.
      4. Computes van leer flux limiter:
        a. Use the shifted c to interpolate each component of `v` to the
          right-hand (upwind) face of the control volume centered on  `c`.
        b. Compute the ratio of successive gradients:
          In nonperiodic case, the value outside the boundary is not defined.
          Mode is used to interpolate past the boundary.
        c. Compute flux limiter function.
        d. Computes higher order flux correction.
      5. Combines fluxes and assigns flux boundary condition.
      6. Computes the negative divergence of fluxes.
      7. Shifts the computed values back to original offset of c.

    Args:
      c: the quantity to be transported.
      v: a velocity field. Should be defined on the same Grid as c.
      dt: time step for which this scheme is TVD and second order accurate
        in time.
      mode: For non-periodic BC, specifies extrapolation of values beyond the
        boundary, which is used by nonlinear interpolation.

    Returns:
      The time derivative of `c` due to advection by `v`.

    #### References

    [1]:  MIT 18.336 spring 2009 Finite Volume Methods Lecture 19.
          go/mit-18.336-finite_volume_methods-19
    [2]:
      www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2012/Chapter_4.pdf

    """
    # TODO(dkochkov) reimplement this using apply_limiter method.
    c_left_var = c
    # if the offset is 1., shift by 1 to offset 0.
    # otherwise c_right is not defined.
    for ax in range(c.grid.ndim):
        # int(c.offset[ax] % 1 - c.offset[ax]) = -1 if c.offset[ax] is 1 else
        # int(c.offset[ax] % 1 - c.offset[ax]) = 0.
        # i.e. this shifts the 1 aligned data to 0 offset, the rest is unchanged.
        c_left_var = c.bc.impose_bc(
            c_left_var.shift(int(c.offset[ax] % 1 - c.offset[ax]), axis=ax)
        )
    offsets = grids.control_volume_offsets(c_left_var)
    # if c offset is 0, aligned_v is at 0.5.
    # if c offset is at .5, aligned_v is at 1.
    aligned_v = tuple(interpolation.linear(u, offset) for u, offset in zip(v, offsets))
    flux = []
    # Assign flux boundary condition
    flux_bc = [
        grids.get_advection_flux_bc_from_velocity_and_scalar(u, c, direction)
        for direction, u in enumerate(v)
    ]
    # first, compute upwind flux.
    for axis, u in enumerate(aligned_v):
        c_center = c_left_var.data
        # by shifting c_left + 1, c_right is well-defined.
        c_right = c_left_var.shift(+1, axis=axis).data
        upwind_flux = grids.applied(torch.where)(
            u.array > 0, u.array * c_center, u.array * c_right
        )
        flux.append(upwind_flux)
    # next, compute van_leer correction.
    for axis, (u, h) in enumerate(zip(aligned_v, c.grid.step)):
        u = u.bc.shift(u.array, int(u.offset[axis] % 1 - u.offset[axis]), axis=axis)
        # c is put to offset .5 or 1.
        c_center_arr = c.shift(int(1 - c.offset[ax]), axis=ax)
        # if c offset is 1, u offset is .5.
        # if c offset is .5, u offset is 0.
        # u_i is always on the left of c_center_var_i
        c_center = c_center_arr.data
        # shift -1 are well defined now
        # shift +1 is not well defined for c offset 1 because then c(wall + 1) is
        # not defined.
        # However, the flux that uses c(wall + 1) offset gets overridden anyways
        # when flux boundary condition is overridden.
        # Thus, any mode can be used here.
        c_right = c.bc.shift(c_center_arr, +1, axis=axis, mode=mode).data
        c_left = c.bc.shift(c_center_arr, -1, axis=axis).data
        # shift -2 is tricky:
        # It is well defined if c is periodic.
        # Else, c(-1) or c(-1.5) are not defined.
        # Then, mode is used to interpolate the values.
        c_left_left = c.bc.shift(c_center_arr, -2, axis, mode=mode).data

        numerator_positive = c_left - c_left_left
        numerator_negative = c_right - c_center
        numerator = grids.applied(torch.where)(
            u > 0, numerator_positive, numerator_negative
        )
        denominator = grids.GridArray(c_center - c_left, u.offset, u.grid)
        # We want to calculate denominator / (abs(denominator) + abs(numerator))
        # To make it differentiable, it needs to be done in stages.

        # ensures that there is no division by 0
        phi_van_leer_denominator_avoid_nans = grids.applied(torch.where)(
            abs(denominator) > 0, (abs(denominator) + abs(numerator)), 1.0
        )

        phi_van_leer_denominator_inv = denominator / phi_van_leer_denominator_avoid_nans

        phi_van_leer = (
            numerator
            * (
                grids.applied(torch.sign)(denominator)
                + grids.applied(torch.sign)(numerator)
            )
            * phi_van_leer_denominator_inv
        )
        abs_velocity = abs(u)
        courant_numbers = (dt / h) * abs_velocity
        pre_factor = 0.5 * (1 - courant_numbers) * abs_velocity
        flux_correction = pre_factor * phi_van_leer
        # Shift back onto original offset.
        flux_correction = flux_bc[axis].shift(
            flux_correction, int(offsets[axis][axis] - u.offset[axis]), axis=axis
        )
        flux[axis] += flux_correction
    flux = tuple(flux_bc[axis].impose_bc(f) for axis, f in enumerate(flux))
    advection = -fdm.divergence(flux)
    # shift the variable back onto the original offset
    for ax in range(c.grid.ndim):
        advection = c.bc.shift(
            advection, -int(c.offset[ax] % 1 - c.offset[ax]), axis=ax
        )
    return advection
