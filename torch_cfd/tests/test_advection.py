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
"""Tests for torch_cfd.advection."""

import math
from functools import partial

import torch
from absl.testing import absltest, parameterized

from torch_cfd import advection, boundaries, grids, test_utils

identity = lambda x: x

Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def step_func(c, v, dt, method):
    c_new = c.data + dt * method(c, v, dt)
    return c.bc.impose_bc(c_new)


def _gaussian_concentration(grid, bc):
    offset = tuple(-int(math.ceil(s / 2.0)) for s in grid.shape)

    mesh_coords = grid.mesh(offset=offset)
    squared_sum = torch.zeros_like(mesh_coords[0])
    for m in mesh_coords:
        squared_sum += torch.square(m) * 30.0

    return GridVariable(torch.exp(-squared_sum), (0.5,) * len(grid.shape), grid, bc)


def _square_concentration(grid, bc):
    select_square = lambda x: torch.where(torch.logical_and(x > 0.4, x < 0.6), 1.0, 0.0)
    mesh_coords = grid.mesh()
    concentration = torch.ones_like(mesh_coords[0])
    for m in mesh_coords:
        concentration *= select_square(m)

    return GridVariable(concentration, (0.5,) * len(grid.shape), grid, bc)


def _unit_velocity(grid, velocity_sign=1.0):
    ndim = grid.ndim
    offsets = (torch.eye(ndim) + torch.ones([ndim, ndim])) / 2.0
    offsets = [offsets[i].tolist() for i in range(ndim)]
    return GridVariableVector(
        tuple(
            GridVariable(
                (
                    velocity_sign * torch.ones(grid.shape)
                    if ax == 0
                    else torch.zeros(grid.shape)
                ),
                tuple(offset),
                grid,
            )
            for ax, offset in enumerate(offsets)
        )
    )


def _total_variation(c: GridVariable, dim: int = 0):
    next_values = c.shift(1, dim)
    variation = torch.abs(next_values.data - c.data).sum().item()
    return variation


advect_linear = advection.AdvectionLinear
advect_upwind = advection.AdvectionUpwind
advect_van_leer = partial(advection.AdvectionVanLeer, limiter=identity)
advect_van_leer_using_limiters = advection.AdvectionVanLeer


class AdvectionTestAnalytical(test_utils.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="linear_2D",
            shape=(101, 101),
            dt=1 / 100,
            method=advect_linear,
            num_steps=100,
            cfl_number=0.1,
            atol=5e-2,
            rtol=1e-3,
        ),
        dict(
            testcase_name="upwind_2D",
            shape=(101, 101),
            dt=1 / 100,
            method=advect_upwind,
            num_steps=100,
            cfl_number=0.5,
            atol=7e-2,
            rtol=1e-3,
        ),
        dict(
            testcase_name="van_leer_2D",
            shape=(101, 101),
            dt=1 / 100,
            method=advect_van_leer,
            num_steps=100,
            cfl_number=0.5,
            atol=7e-2,
            rtol=1e-3,
        ),
        dict(
            testcase_name="van_leer_using_limiters_2D",
            shape=(101, 101),
            dt=1 / 100,
            method=advect_van_leer_using_limiters,
            num_steps=100,
            cfl_number=0.1,
            atol=2e-2,
            rtol=1e-3,
        ),
    )
    def test_advection_analytical(
        self, shape, dt, method, num_steps, cfl_number, atol, rtol
    ):
        """Tests advection of a Gaussian concentration on a periodic grid."""
        step = tuple(1.0 / s for s in shape)
        grid = Grid(shape, step)
        v_sign = 1.0
        bc = boundaries.periodic_boundary_conditions(len(shape))
        v = GridVariableVector(tuple(u for u in _unit_velocity(grid, v_sign)))
        c = _gaussian_concentration(grid, bc)
        advect = method(grid, c.offset)
        dt = cfl_number * dt
        ct = c.clone()
        for _ in range(num_steps):
            ct = step_func(ct, v, dt, method=advect)

        expected_shift = int(round(-cfl_number * num_steps * v_sign))
        expected = c.shift(expected_shift, dim=0)

        self.assertAllClose(expected.data, ct.data, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="dirichlet_1d_200_cell_center",
            shape=(200,),
            atol=0.00025,
            rtol=1 / 200,
            offset=0.5,
        ),
        dict(
            testcase_name="dirichlet_1d_400_cell_center",
            shape=(400,),
            atol=0.00007,
            rtol=1 / 400,
            offset=0.5,
        ),
        dict(
            testcase_name="dirichlet_1d_200_cell_edge_0",
            shape=(200,),
            atol=0.0005,
            rtol=1 / 200,
            offset=0.0,
        ),
        dict(
            testcase_name="dirichlet_1d_400_cell_edge_0",
            shape=(400,),
            atol=0.000125,
            rtol=1 / 400,
            offset=0.0,
        ),
        dict(
            testcase_name="dirichlet_1d_200_cell_edge_1",
            shape=(200,),
            atol=0.0005,
            rtol=1 / 200,
            offset=1.0,
        ),
        dict(
            testcase_name="dirichlet_1d_400_cell_edge_1",
            shape=(400,),
            atol=0.000125,
            rtol=1 / 400,
            offset=1.0,
        ),
    )
    def test_burgers_analytical_dirichlet_convergence(
        self,
        shape,
        atol,
        rtol,
        offset,
    ):
        def _step_func(v, dt, method):
            dv_dt = method(c=v[0], v=v, dt=dt) / 2
            return (bc.impose_bc(v[0].data + dt * dv_dt),)

        def _velocity_implicit(grid, offset, u, t):
            """Returns solution of a Burgers equation at time `t`."""
            x = grid.mesh(offset)[0]
            return grids.GridVariable(torch.sin(x - u * t), offset, grid)

        num_steps = 1000
        cfl_number = 0.01
        step = 2 * math.pi / 1000
        offset = (offset,)
        grid = grids.Grid(shape, domain=((0.0, 2 * math.pi),))
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
        v = (bc.impose_bc(_velocity_implicit(grid, offset, 0, 0)),)
        dt = cfl_number * step
        advect = advect_van_leer(grid, offset)

        for _ in range(num_steps):
            """
            dt/2 is used because for Burgers equation
            the flux is u_t + (0.5*u^2)_x = 0
            """
            v = _step_func(v, dt, method=advect)

        expected = bc.impose_bc(
            _velocity_implicit(grid, offset, v[0].data, dt * num_steps)
        ).data
        self.assertAllClose(expected, v[0].data, atol=atol, rtol=rtol)

class AdvectionTestProperties(test_utils.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="upwind_2D",
            shape=(101, 51),
            method=advect_upwind,
        ),
        dict(
            testcase_name="van_leer_2D",
            shape=(101, 51),
            method=advect_van_leer,
        ),
        dict(
            testcase_name="van_leer_using_limiters_2D",
            shape=(101, 101),
            method=advect_van_leer_using_limiters,
        ),
    )
    def test_tvd_property(self, shape, method):
        atol = 1e-6
        step = tuple(1.0 / s for s in shape)
        grid = Grid(shape, step)
        bc = boundaries.periodic_boundary_conditions(grid.ndim)
        v = GridVariableVector(tuple(u for u in _unit_velocity(grid)))
        c = _square_concentration(grid, bc)
        dt = min(step)
        num_steps = 300
        ct = c.clone()

        advect = method(grid, c.offset)

        initial_total_variation = _total_variation(c, 0) + atol

        for _ in range(num_steps):
            ct = step_func(ct, v, dt, method=advect)
            current_total_variation = _total_variation(ct, 0)
            self.assertLessEqual(current_total_variation, initial_total_variation)

    @parameterized.named_parameters(
        dict(
            testcase_name="upwind_2D",
            shape=(201, 101),
            method=advect_upwind,
        ),
        dict(
            testcase_name="van_leer_2D",
            shape=(101, 201),
            method=advect_van_leer,
        ),
        dict(
            testcase_name="van_leer_using_limiters_2D",
            shape=(101, 101),
            method=advect_van_leer_using_limiters,
        ),
    )
    def test_mass_conservation(self, shape, method):
        """
        Note: when the mass integral is close to zero
        the relative error will be big~O(1e1) for fp32
        """
        offset = (0.5, 0.5)
        offsets_v = ((1.0, 0.5), (0.5, 1.0))
        cfl_number = 0.1
        dt = cfl_number / shape[0]
        num_steps = 1000

        grid = Grid(shape, domain=((-1.0, 1.0), (-1.0, 1.0)))
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
        c_bc = boundaries.dirichlet_and_periodic_boundary_conditions(bc_vals=(0.0, 2.0))

        phi = lambda t: torch.sin(math.pi * t)

        def _velocity(grid, offsets):
            x, y = grid.mesh(offsets[0])
            u1 = GridVariable(-phi(x) * phi(y), offsets[0], grid)
            u2 = GridVariable(torch.zeros_like(u1.data), offsets[1], grid)
            return GridVariableVector((u1, u2))

        def c0(grid, offset):
            x = grid.mesh(offset)[0] + 1
            return GridVariable(x, offset, grid)

        v = tuple(bc.impose_bc(u) for u in _velocity(grid, offsets_v))
        c = c_bc.impose_bc(c0(grid, offset))

        ct = c.clone()

        advect = method(grid, c.offset)

        initial_mass = c.data.sum().item()
        for _ in range(num_steps):
            ct = step_func(ct, v, dt, method=advect)
            current_total_mass = ct.data.sum().item()
            self.assertAllClose(current_total_mass, initial_mass, atol=1e-6, rtol=1e-2)


if __name__ == "__main__":
    absltest.main()
