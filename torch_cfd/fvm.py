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
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import torch_cfd.finite_differences as fdm
from torch_cfd import advection, boundaries, forcings, grids, pressure


Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ForcingFn = forcings.ForcingFn


def wrap_field_same_bcs(v, field_ref):
    return GridVariableVector(
        tuple(
            GridVariable(a.data, a.offset, a.grid, w.bc) for a, w in zip(v, field_ref)
        )
    )


class ProjectionExplicitODE(nn.Module):
    r"""Navier-Stokes equation in 2D with explicit stepping and a pressure projection (discrete Helmholtz decomposition by modding the gradient of a Laplacian inverse of the extra divergence).

    \partial u/ \partial t = explicit_terms(u)
    u <- pressure_projection(u)
    """

    def explicit_terms(self, *args, **kwargs) -> GridVariableVector:
        """
        Explicit forcing term as du/dt.
        """
        raise NotImplementedError

    def pressure_projection(self, *args, **kwargs) -> Tuple[GridVariableVector, GridVariable]:
        """Pressure projection step."""
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
        method: Optional[str] = "forward_euler",
        dtype: Optional[torch.dtype] = torch.float32,
        requires_grad=False,
        **kwargs,
    ):
        super().__init__()

        self._method = None
        self.dtype = dtype
        self.requires_grad = requires_grad

        # Set the tableau first directly, either directly or from method name
        if tableau is not None:
            self.tableau = tableau
        else:
            self.method = method
        self._set_params()

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

    def _set_params(self):
        """Set the parameters of the Butcher tableau."""
        try:
            a, b = self._tableau["a"], self._tableau["b"]
            if a.__len__() + 1 != b.__len__():
                raise ValueError("Inconsistent Butcher tableau: len(a) + 1 != len(b)")
            self.params = nn.ParameterDict()
            self.params["a"] = nn.ParameterList()
            for a_ in a:
                self.params["a"].append(
                    nn.Parameter(
                        torch.tensor(
                            a_, dtype=self.dtype, requires_grad=self.requires_grad
                        )
                    )
                )
            self.params["b"] = nn.Parameter(
                torch.tensor(b, dtype=self.dtype, requires_grad=self.requires_grad)
            )
        except KeyError as e:
            print(f"{e}: Either `tableau` or `method` must be given.")

    @classmethod
    def from_tableau(
        cls,
        tableau: Dict[str, List],
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        """Factory method to create an RKStepper from a Butcher tableau."""
        return cls(tableau=tableau, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def from_method(
        cls, method: str = "forward_euler", requires_grad: bool = False, **kwargs
    ):
        """Factory method to create an RKStepper by name."""
        return cls(method=method, requires_grad=requires_grad, **kwargs)

    def forward(
        self, u0: GridVariableVector, dt: float, equation: ProjectionExplicitODE
    ) -> Tuple[GridVariableVector, GridVariable]:
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

            u[i], _ = equation.pressure_projection(u_star)
            k[i] = equation.explicit_terms(u[i], dt)

        u_star = GridVariableVector(tuple(v.clone() for v in u0))
        for j in range(num_steps):
            if beta[j] != 0:
                u_star = u_star + dt * beta[j] * k[j]

        u_final, p = equation.pressure_projection(u_star)

        return u_final, p


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
        convection: Callable = None,
        pressure_proj: Callable = None,
        forcing: Optional[ForcingFn] = None,
        step_fn: RKStepper = None,
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
        self.forcing = forcing
        self.convection = convection
        self.step_fn = step_fn
        self.pressure_proj = pressure_proj
        self._set_pressure_bc()
        self._set_convect()
        self._set_pressure_projection()

    def _set_convect(self):
        if self.convection is not None:
            self._convect = self.convection
        else:
            self._convect = advection.ConvectionVector(grid=self.grid, bcs=self.bcs)

    def _set_pressure_projection(self):
        if self.pressure_proj is not None:
            self._projection = self.pressure_proj
            return
        self._projection = pressure.PressureProjection(
            grid=self.grid,
            bc=self.pressure_bc,
        )

    def _set_pressure_bc(self):
        if self.bcs is None:
            self.bcs = (
                boundaries.periodic_boundary_conditions(ndim=self.grid.ndim),
                boundaries.periodic_boundary_conditions(ndim=self.grid.ndim),
            )
        self.pressure_bc = boundaries.get_pressure_bc_from_velocity_bc(bcs=self.bcs)

    def _diffusion(self, v: GridVariableVector) -> GridVariableVector:
        """Returns the diffusion term for the velocity field."""
        alpha = self.viscosity / self.density
        lapv = GridVariableVector(tuple(alpha * fdm.laplacian(u) for u in v))
        return lapv

    def _explicit_terms(self, v, dt, **kwargs):
        dv_dt = self._convect(v, v, dt)
        grid = self.grid
        density = self.density
        forcing = self.forcing
        dv_dt += self._diffusion(v)
        if forcing is not None:
            dv_dt += GridVariableVector(forcing(grid, v)) / density
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

        return self.step_fn(u, dt, self)
