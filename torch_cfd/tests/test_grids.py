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
"""Tests for torch_cfd.grids."""

import math

import torch

from absl.testing import absltest, parameterized

from torch_cfd import boundaries, grids, test_utils


class GridVariableTest(test_utils.TestCase):
    def test_consistent_offset(self):
        data = torch.arange(3)
        grid = grids.Grid((3,))
        array_offset_0 = grids.GridVariable(data, offset=(0,), grid=grid)
        array_offset_1 = grids.GridVariable(data, offset=(1,), grid=grid)

        offset = grids.consistent_offset_arrays(array_offset_0, array_offset_0)
        self.assertEqual(offset, (0,))

        with self.assertRaises(Exception):
            grids.consistent_offset_arrays(array_offset_0, array_offset_1)

    def test_averaged_offset(self):
        data = torch.arange(3)
        grid = grids.Grid((3,))
        array_offset_0 = grids.GridVariable(data, offset=(0,), grid=grid)
        array_offset_1 = grids.GridVariable(data, offset=(1,), grid=grid)

        averaged_offset = grids.averaged_offset_arrays(array_offset_0, array_offset_1)
        self.assertEqual(averaged_offset, (0.5,))

    def test_control_volume_offsets(self):
        data = torch.arange(5, 5)
        grid = grids.Grid((5, 5))
        array = grids.GridVariable(data, offset=(0, 0), grid=grid)
        cv_offset = grids.control_volume_offsets_arrays(array)
        self.assertEqual(cv_offset, ((0.5, 0), (0, 0.5)))

    def test_consistent_grid(self):
        data = torch.arange(3)
        offset = (0,)
        array_grid_3 = grids.GridVariable(data, offset, grid=grids.Grid((3,)))
        array_grid_5 = grids.GridVariable(data, offset, grid=grids.Grid((5,)))

        grid = grids.consistent_grid_arrays(array_grid_3, array_grid_3)
        self.assertEqual(grid, grids.Grid((3,)))

        with self.assertRaises(Exception):
            grids.consistent_grid_arrays(array_grid_3, array_grid_5)

    def test_add_sub_correctness(self):
        values_1 = torch.rand((5, 5))
        values_2 = torch.rand((5, 5))
        offsets = (0.5, 0.5)
        grid = grids.Grid((5, 5))
        input_array_1 = grids.GridVariable(values_1, offsets, grid)
        input_array_2 = grids.GridVariable(values_2, offsets, grid)
        actual_sum = input_array_1 + input_array_2
        actual_sub = input_array_1 - input_array_2
        expected_sum = grids.GridVariable(values_1 + values_2, offsets, grid)
        expected_sub = grids.GridVariable(values_1 - values_2, offsets, grid)
        self.assertAllClose(actual_sum, expected_sum, atol=1e-7, rtol=1e-12)
        self.assertAllClose(actual_sub, expected_sub, atol=1e-7, rtol=1e-12)

    def test_add_sub_inplace_correctness(self):
        values_1 = torch.rand((5, 5))
        values_2 = torch.rand((5, 5))
        values_3 = torch.rand((5, 5))
        offsets = (0.5, 0.5)
        grid = grids.Grid((5, 5))

        # Test in-place addition
        input_array_1 = grids.GridVariable(values_1.clone(), offsets, grid)
        input_array_2 = grids.GridVariable(values_2, offsets, grid)
        input_array_1 += input_array_2
        expected_add = grids.GridVariable(values_1 + values_2, offsets, grid)
        self.assertAllClose(input_array_1, expected_add, atol=1e-7, rtol=1e-12)

        # Test in-place subtraction
        input_array_3 = grids.GridVariable(values_1.clone(), offsets, grid)
        input_array_4 = grids.GridVariable(values_3, offsets, grid)
        input_array_3 -= input_array_4
        expected_sub = grids.GridVariable(values_1 - values_3, offsets, grid)
        self.assertAllClose(input_array_3, expected_sub, atol=1e-7, rtol=1e-12)

    def test_add_sub_offset_raise(self):
        values_1 = torch.rand((5, 5))
        values_2 = torch.rand((5, 5))
        offset_1 = (0.5, 0.5)
        offset_2 = (0.5, 0.0)
        grid = grids.Grid((5, 5))
        input_array_1 = grids.GridVariable(values_1, offset_1, grid)
        input_array_2 = grids.GridVariable(values_2, offset_2, grid)
        with self.assertRaises(ValueError):
            _ = input_array_1 + input_array_2
        with self.assertRaises(ValueError):
            _ = input_array_1 - input_array_2

    def test_add_sub_grid_raise(self):
        values_1 = torch.rand((5, 5))
        values_2 = torch.rand((5, 5))
        offset = (0.5, 0.5)
        grid_1 = grids.Grid((5, 5), domain=((0, 1), (0, 1)))
        grid_2 = grids.Grid((5, 5), domain=((-2, 2), (-2, 2)))
        input_array_1 = grids.GridVariable(values_1, offset, grid_1)
        input_array_2 = grids.GridVariable(values_2, offset, grid_2)
        with self.assertRaises(ValueError):
            _ = input_array_1 + input_array_2
        with self.assertRaises(ValueError):
            _ = input_array_1 - input_array_2

    def test_mul_div_correctness(self):
        values_1 = torch.rand((5, 5))
        values_2 = torch.rand((5, 5))
        scalar = math.pi
        offset = (0.5, 0.5)
        grid = grids.Grid((5, 5))
        input_array_1 = grids.GridVariable(values_1, offset, grid)
        input_array_2 = grids.GridVariable(values_2, offset, grid)
        actual_mul = input_array_1 * input_array_2
        array_1_times_scalar = input_array_1 * scalar
        expected_1_times_scalar = grids.GridVariable(values_1 * scalar, offset, grid)
        actual_div = input_array_1 / 2.5
        expected_div = grids.GridVariable(values_1 / 2.5, offset, grid)
        expected_mul = grids.GridVariable(values_1 * values_2, offset, grid)
        self.assertAllClose(actual_mul, expected_mul, atol=1e-7, rtol=1e-12)
        self.assertAllClose(
            array_1_times_scalar, expected_1_times_scalar, atol=1e-7, rtol=1e-12
        )
        self.assertAllClose(actual_div, expected_div, atol=1e-7, rtol=1e-12)

        # Test in-place scalar multiplication
        input_array_3 = grids.GridVariable(values_1.clone(), offset, grid)
        input_array_3 *= scalar
        expected_inplace_mul = grids.GridVariable(values_1 * scalar, offset, grid)
        self.assertAllClose(input_array_3, expected_inplace_mul, atol=1e-7, rtol=1e-12)

        # Test in-place scalar division
        input_array_4 = grids.GridVariable(values_1.clone(), offset, grid)
        div_scalar = 2.5
        input_array_4 /= div_scalar
        expected_inplace_div = grids.GridVariable(values_1 / div_scalar, offset, grid)
        self.assertAllClose(input_array_4, expected_inplace_div, atol=1e-7, rtol=1e-12)

    def test_unary(self):
        grid = grids.Grid((10, 10))
        offset = (0.5, 0.5)
        u = grids.GridVariable(torch.ones([10, 10]), offset, grid)
        expected = grids.GridVariable(-torch.ones([10, 10]), offset, grid)
        actual = -u
        self.assertAllClose(expected, actual)

    def test_constructor_and_attributes(self):
        with self.subTest("1d"):
            grid = grids.Grid((10,))
            data = torch.zeros((10,), dtype=torch.float32)
            bc = boundaries.periodic_boundary_conditions(grid.ndim)
            variable = grids.GridVariable(data, (0.5,), grid, bc)
            self.assertEqual(variable.bc, bc)
            self.assertEqual(variable.dtype, torch.float32)
            self.assertEqual(variable.shape, (10,))
            self.assertArrayEqual(variable.data, data)
            self.assertEqual(variable.offset, (0.5,))
            self.assertEqual(variable.grid, grid)

        with self.subTest("2d"):
            grid = grids.Grid((10, 10))
            data = torch.zeros((10, 10), dtype=torch.float32)
            bc = boundaries.periodic_boundary_conditions(grid.ndim)
            variable = grids.GridVariable(data, (0.5, 0.5), grid, bc)
            self.assertEqual(variable.bc, bc)
            self.assertEqual(variable.dtype, torch.float32)
            self.assertEqual(variable.shape, (10, 10))
            self.assertArrayEqual(variable.data, data)
            self.assertEqual(variable.offset, (0.5, 0.5))
            self.assertEqual(variable.grid, grid)

        with self.subTest("2d batch dim data"):
            grid = grids.Grid((10, 10))
            data = torch.zeros((5, 10, 10), dtype=torch.float32)
            bc = boundaries.periodic_boundary_conditions(grid.ndim)
            variable = grids.GridVariable(data, (0.5, 0.5), grid, bc)
            self.assertEqual(variable.bc, bc)
            self.assertEqual(variable.dtype, torch.float32)
            self.assertEqual(variable.shape, (5, 10, 10))
            self.assertArrayEqual(variable.data, data)
            self.assertEqual(variable.offset, (0.5, 0.5))
            self.assertEqual(variable.grid, grid)

        with self.subTest("raises exception"):
            with self.assertRaisesRegex(
                ValueError, "Incompatible dimension between grid and bc"
            ):
                grid = grids.Grid((10,))  # 1D
                data = torch.zeros((10,))  # 1D
                bc = boundaries.periodic_boundary_conditions(ndim=2)  # 2D
                grids.GridVariable(data, (0.5,), grid, bc)


class GridVariableBoundaryTest(test_utils.TestCase):
    @parameterized.parameters(
        dict(
            shape=(10,),
            offset=(0.0,),
        ),
        dict(
            shape=(10,),
            offset=(0.5,),
        ),
        dict(
            shape=(10,),
            offset=(1.0,),
        ),
        dict(
            shape=(10, 10),
            offset=(1.0, 0.0),
        ),
    )
    def test_interior_consistency_periodic(self, shape, offset):
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, shape).to(torch.float64)
        bc = boundaries.periodic_boundary_conditions(ndim=len(shape))
        u = grids.GridVariable(data, offset, grid, bc)
        u_interior = u.trim_boundary()
        self.assertArrayEqual(u_interior.data, u.data)

    @parameterized.parameters(
        dict(
            shape=(10,),
            bc=boundaries.dirichlet_boundary_conditions(ndim=1),
        ),
        dict(
            shape=(10,),
            bc=boundaries.neumann_boundary_conditions(ndim=1),
        ),
        dict(
            shape=(10, 10),
            bc=boundaries.dirichlet_boundary_conditions(ndim=2),
        ),
        dict(
            shape=(10, 10),
            bc=boundaries.neumann_boundary_conditions(ndim=2),
        ),
    )
    def test_interior_consistency_no_edge_offsets(self, bc, shape):
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, shape).to(torch.float64)
        u = grids.GridVariable(data, (0.5,) * len(shape), grid, bc)
        u_interior = u.trim_boundary()
        self.assertArrayEqual(u_interior.data, u.data)

    @parameterized.parameters(
        dict(
            shape=(10,),
            bc=boundaries.neumann_boundary_conditions(ndim=1),
            offset=(0.5,),
        ),
        dict(
            shape=(10, 10),
            bc=boundaries.neumann_boundary_conditions(ndim=2),
            offset=(0.5, 0.5),
        ),
    )
    def test_interior_consistency_neumann(self, shape, bc, offset):
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, shape).to(torch.float64)
        u = grids.GridVariable(data, offset, grid, bc)
        u_interior = u.trim_boundary()
        self.assertArrayEqual(u_interior.data, u.data)

    @parameterized.parameters(
        dict(
            shape=(10,),
            bc=boundaries.HomogeneousBoundaryConditions(
                ((grids.BCType.DIRICHLET, grids.BCType.DIRICHLET),)
            ),
            offset=(0.0,),
        ),
        dict(
            shape=(10, 10),
            bc=boundaries.dirichlet_boundary_conditions(ndim=2),
            offset=(0.0, 0.0),
        ),
    )
    def test_interior_consistency_edge_offsets_dirichlet(self, shape, bc, offset):
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, shape).to(torch.float64)
        u = grids.GridVariable(data, offset, grid, bc)
        u_interior = u.trim_boundary()
        self.assertEqual(u_interior.offset, tuple(offset + 1 for offset in u.offset))
        self.assertEqual(u_interior.grid.ndim, u.grid.ndim)
        self.assertEqual(u_interior.grid.step, u.grid.step)

    def test_interior_dirichlet(self):
        data = torch.tensor(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
                [41, 42, 43, 44, 45],
            ], dtype=torch.float64
        )

        grid = grids.Grid(shape=(4, 5), domain=((0, 1), (0, 1)))
        bc = boundaries.dirichlet_boundary_conditions(ndim=2)

        with self.subTest("offset=(1, 0.5)"):
            offset = (1.0, 0.5)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            answer = torch.tensor(
                [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]],
                dtype=torch.float64
            )
            self.assertArrayEqual(u_interior.data, answer)
            self.assertEqual(u_interior.offset, offset)
            self.assertEqual(u.grid, grid)

        with self.subTest("offset=(1, 1)"):
            offset = (1.0, 1.0)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            answer = torch.tensor(
                [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
                dtype=torch.float64
            )
            self.assertArrayEqual(u_interior.data, answer)
            self.assertEqual(u_interior.grid, grid)

        with self.subTest("offset=(0.0, 0.5)"):
            offset = (0.0, 0.5)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            answer = torch.tensor(
                [[21, 22, 23, 24, 25], [31, 32, 33, 34, 35], [41, 42, 43, 44, 45]],
                dtype=torch.float64
            )
            self.assertArrayEqual(u_interior.data, answer)
            self.assertEqual(u_interior.grid, grid)

        with self.subTest("offset=(0.0, 0.0)"):
            offset = (0.0, 0.0)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            answer = torch.tensor(
                [[22, 23, 24, 25], [32, 33, 34, 35], [42, 43, 44, 45]],
                dtype=torch.float64
            )
            self.assertArrayEqual(u_interior.data, answer)
            self.assertEqual(u_interior.grid, grid)

        with self.subTest("offset=(0.5, 0.0)"):
            offset = (0.5, 0.0)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            answer = torch.tensor(
                [[12, 13, 14, 15], [22, 23, 24, 25], [32, 33, 34, 35], [42, 43, 44, 45]],
                dtype=torch.float64
            )
            self.assertArrayEqual(u_interior.data, answer)
            self.assertEqual(u_interior.grid, grid)

        # this is consistent for all offsets, not just edge and center.
        with self.subTest("offset=(0.25, 0.75)"):
            offset = (0.25, 0.75)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            self.assertArrayEqual(u_interior.data, data)
            self.assertEqual(u_interior.grid, grid)

    @parameterized.parameters(
        dict(
            shape=(10,),
            bc=boundaries.periodic_boundary_conditions(ndim=1),
            padding=(1, 1),
            dim=0,
        ),
        dict(
            shape=(10, 10),
            bc=boundaries.dirichlet_boundary_conditions(ndim=2),
            padding=(2, 1),
            dim=1,
        ),
    )
    def test_shift_pad_trim(self, shape, bc, padding, dim):
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, shape).to(torch.float64)
        u = grids.GridVariable(data, (0.5,) * len(shape), grid, bc)

        with self.subTest("shift"):
            self.assertArrayEqual(u.shift(offset=1, dim=dim), bc.shift(u, 1, dim))

        with self.subTest("raises exception"):
            with self.assertRaisesRegex(
                ValueError, "Incompatible dimension between grid and bc"
            ):
                grid = grids.Grid((10,))
                data = torch.zeros((10,))
                array = grids.GridVariable(data, offset=(0.5,), grid=grid)  # 1D
                bc = boundaries.dirichlet_boundary_conditions(ndim=2)  # 2D
                grids.GridVariable(data, (0.5,), grid, bc)

    def test_unique_boundary_conditions(self):
        grid = grids.Grid((5,))
        bc1 = boundaries.periodic_boundary_conditions(ndim=1)
        bc2 = boundaries.dirichlet_boundary_conditions(ndim=1)
        x_bc1 = grids.GridVariable(torch.arange(5), (0.5,), grid, bc1)
        y_bc1 = grids.GridVariable(torch.arange(5), (0.5,), grid, bc1)
        z_bc2 = grids.GridVariable(torch.arange(5), (0.5,), grid, bc2)

        bc = grids.consistent_bc_arrays(x_bc1, y_bc1)
        self.assertEqual(bc, bc1)

        bc = grids.consistent_bc_arrays(x_bc1, y_bc1, z_bc2)
        self.assertIsNone(bc)

    def test_trim_boundary_cell_center(self):
        """Test boundary operations when each batch element has different data."""
        shape = (6, 6)
        grid = grids.Grid(shape)

        # Create different data for each batch element
        data = torch.randn(*shape)

        bc = boundaries.dirichlet_boundary_conditions(ndim=2)

        u = grids.GridVariable(data, (0.5, 0.5), grid, bc)
        u_interior = u.trim_boundary()

        # Check that each batch element is processed independently
        self.assertEqual(
            u_interior.shape, (6, 6)
        )  # cell center data should remain the same shape

class GridVariableBoundaryTestBatch(test_utils.TestCase):
    """Test boundary behavior with batch dimensions in 2D data."""

    @parameterized.parameters(
        dict(
            batch_size=3,
            shape=(10,),
            offset=(1.0,),
        ),
        dict(
            batch_size=2,
            shape=(10, 10),
            offset=(1.0, 0.0),
        ),
    )
    def test_interior_consistency_periodic_batch(self, batch_size, shape, offset):
        """Test that periodic boundary conditions work with batch dimensions."""
        grid = grids.Grid(shape)
        # Create batched data: (batch_size, *shape)
        data = torch.randint(0, 10, (batch_size, *shape)).to(torch.float64)
        bc = boundaries.periodic_boundary_conditions(ndim=len(shape))
        u = grids.GridVariable(data, offset, grid, bc)
        u_interior = u.trim_boundary()

        # For periodic boundaries, interior should equal original data
        self.assertArrayEqual(u_interior.data, u.data)
        self.assertEqual(u_interior.shape, (batch_size, *shape))

    @parameterized.parameters(
        dict(
            batch_size=3,
            shape=(10,),
            bc=boundaries.neumann_boundary_conditions(ndim=1),
        ),
        dict(
            batch_size=2,
            shape=(10, 10),
            bc=boundaries.dirichlet_boundary_conditions(ndim=2),
        ),
        dict(
            batch_size=4,
            shape=(10, 10),
            bc=boundaries.neumann_boundary_conditions(ndim=2),
        ),
    )
    def test_interior_consistency_no_edge_offsets_batch(self, batch_size, bc, shape):
        """Test boundary behavior with non-edge offsets and batch dimensions."""
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, (batch_size, *shape)).to(torch.float64)
        u = grids.GridVariable(data, (0.5,) * len(shape), grid, bc)
        u_interior = u.trim_boundary()

        # For non-edge offsets, interior should equal original data
        self.assertArrayEqual(u_interior.data, u.data)
        self.assertEqual(u_interior.shape, (batch_size, *shape))

    @parameterized.parameters(
        dict(
            batch_size=2,
            shape=(10,),
            bc=boundaries.neumann_boundary_conditions(ndim=1),
            offset=(0.5,),
        ),
        dict(
            batch_size=3,
            shape=(10, 10),
            bc=boundaries.neumann_boundary_conditions(ndim=2),
            offset=(0.5, 0.5),
        ),
    )
    def test_interior_consistency_neumann_batch(self, batch_size, shape, bc, offset):
        """Test Neumann boundary conditions with batch dimensions."""
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, (batch_size, *shape)).to(torch.float64)
        u = grids.GridVariable(data, offset, grid, bc)
        u_interior = u.trim_boundary()

        self.assertArrayEqual(u_interior.data, u.data)
        self.assertEqual(u_interior.shape, (batch_size, *shape))

    @parameterized.parameters(
        dict(
            batch_size=2,
            shape=(10,),
            bc=boundaries.dirichlet_boundary_conditions(ndim=1),
            offset=(0.0,),
        ),
        dict(
            batch_size=3,
            shape=(10, 10),
            bc=boundaries.dirichlet_boundary_conditions(ndim=2),
            offset=(0.0, 0.0),
        ),
    )
    def test_interior_consistency_edge_offsets_dirichlet_batch(
        self, batch_size, shape, bc, offset
    ):
        """Test Dirichlet boundary conditions with edge offsets and batch dimensions."""
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, (batch_size, *shape)).to(torch.float64)
        u = grids.GridVariable(data, offset, grid, bc)
        u_interior = u.trim_boundary()

        # Check offset adjustment
        self.assertEqual(u_interior.offset, tuple(offset + 1 for offset in u.offset))
        self.assertEqual(u_interior.grid.ndim, u.grid.ndim)
        self.assertEqual(u_interior.grid.step, u.grid.step)

        # Check that batch dimension is preserved
        self.assertEqual(u_interior.data.shape[0], batch_size)

    def test_interior_dirichlet_batch(self):
        """Test Dirichlet boundary trimming with batch dimensions using specific data."""
        batch_size = 2

        # Create identical data for each batch element for easier testing
        single_data = torch.tensor(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
                [41, 42, 43, 44, 45],
            ],
            dtype=torch.float64
        )
        data = single_data.unsqueeze(0).repeat(batch_size, 1, 1)

        grid = grids.Grid(shape=(4, 5), domain=((0, 1), (0, 1)))
        bc = boundaries.dirichlet_boundary_conditions(ndim=2)

        with self.subTest("offset=(1, 0.5)"):
            offset = (1.0, 0.5)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            expected_single = torch.tensor(
                [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]],
                dtype=torch.float64
            )
            expected_batch = expected_single.unsqueeze(0).repeat(batch_size, 1, 1)
            self.assertArrayEqual(u_interior.data, expected_batch)
            self.assertEqual(u_interior.offset, offset)
            self.assertEqual(u_interior.shape, (batch_size, 3, 5))

        with self.subTest("offset=(1, 1)"):
            offset = (1.0, 1.0)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            expected_single = torch.tensor(
                [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
                dtype=torch.float64
            )
            expected_batch = expected_single.unsqueeze(0).repeat(batch_size, 1, 1)
            self.assertArrayEqual(u_interior.data, expected_batch)
            self.assertEqual(u_interior.shape, (batch_size, 3, 4))

        with self.subTest("offset=(0.0, 0.5)"):
            offset = (0.0, 0.5)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            expected_single = torch.tensor(
                [[21, 22, 23, 24, 25], [31, 32, 33, 34, 35], [41, 42, 43, 44, 45]],
                dtype=torch.float64
            )
            expected_batch = expected_single.unsqueeze(0).repeat(batch_size, 1, 1)
            self.assertArrayEqual(u_interior.data, expected_batch)
            self.assertEqual(u_interior.shape, (batch_size, 3, 5))

        with self.subTest("offset=(0.0, 0.0)"):
            offset = (0.0, 0.0)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            expected_single = torch.tensor(
                [[22, 23, 24, 25], [32, 33, 34, 35], [42, 43, 44, 45]],
                dtype=torch.float64
            )
            expected_batch = expected_single.unsqueeze(0).repeat(batch_size, 1, 1)
            self.assertArrayEqual(u_interior.data, expected_batch)
            self.assertEqual(u_interior.shape, (batch_size, 3, 4))

        with self.subTest("offset=(0.5, 0.0)"):
            offset = (0.5, 0.0)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            expected_single = torch.tensor(
                [[12, 13, 14, 15], [22, 23, 24, 25], [32, 33, 34, 35], [42, 43, 44, 45]],
                dtype=torch.float64
            )
            expected_batch = expected_single.unsqueeze(0).repeat(batch_size, 1, 1)
            self.assertArrayEqual(u_interior.data, expected_batch)
            self.assertEqual(u_interior.shape, (batch_size, 4, 4))

        # Test with non-edge offset - should be unchanged
        with self.subTest("offset=(0.25, 0.75)"):
            offset = (0.25, 0.75)
            u = grids.GridVariable(data, offset, grid, bc)
            u_interior = u.trim_boundary()
            self.assertArrayEqual(u_interior.data, data)
            self.assertEqual(u_interior.shape, (batch_size, 4, 5))

    @parameterized.parameters(
        dict(
            batch_size=2,
            shape=(10,),
            bc=boundaries.periodic_boundary_conditions(ndim=1),
            padding=(1, 1),
            dim=0,
        ),
        dict(
            batch_size=3,
            shape=(10, 10),
            bc=boundaries.dirichlet_boundary_conditions(ndim=2),
            padding=(2, 1),
            dim=1,
        ),
    )
    def test_shift_pad_trim_batch(self, batch_size, shape, bc, padding, dim):
        """Test shift, pad, and trim operations with batch dimensions."""
        grid = grids.Grid(shape)
        data = torch.randint(0, 10, (batch_size, *shape)).to(torch.float64)
        u = grids.GridVariable(data, (0.5,) * len(shape), grid, bc)

        with self.subTest("shift"):
            shifted = u.shift(offset=1, dim=dim)
            expected = bc.shift(u, 1, dim)
            self.assertArrayEqual(shifted.data, expected.data)
            self.assertEqual(shifted.shape, u.shape)
            self.assertEqual(shifted.shape[0], batch_size)

    def test_batch_consistency_across_boundary_operations(self):
        """Test that boundary operations are consistent across batch elements."""
        batch_size = 3
        shape = (8, 8)
        grid = grids.Grid(shape)

        # Create test data where each batch element is identical
        single_data = torch.randint(0, 100, shape).to(torch.float64)
        batched_data = single_data.unsqueeze(0).repeat(batch_size, 1, 1)

        bc = boundaries.dirichlet_boundary_conditions(ndim=2)

        u = grids.GridVariable(batched_data, (0.0, 0.0), grid, bc)
        u_interior = u.trim_boundary()

        # Each batch element should produce identical results
        for i in range(1, batch_size):
            self.assertArrayEqual(u_interior.data[0], u_interior.data[i])

    def test_batch_different_data_per_element(self):
        """Test boundary operations when each batch element has different data."""
        batch_size = 2
        shape = (6, 6)
        grid = grids.Grid(shape)

        # Create different data for each batch element
        data = torch.randn((batch_size, *shape))

        bc = boundaries.dirichlet_boundary_conditions(ndim=2)

        u = grids.GridVariable(data, (0.0, 0.0), grid, bc)
        # (0.0, 0.0) offset means we expect to trim one layer from only the left side even if the trim_boundary default is trimming both boundary
        u_interior = u.trim_boundary()

        # Check that each batch element is processed independently
        self.assertEqual(
            u_interior.shape, (batch_size, 5, 5)
        )  # Trimmed from (6,6) to (5,5)

        # Verify that batch elements are different (as expected)
        self.assertFalse(torch.allclose(u_interior.data[0], u_interior.data[1]))


class GridTest(test_utils.TestCase):

    def test_constructor_and_attributes(self):
        with self.subTest("1d"):
            grid = grids.Grid((10,))
            self.assertEqual(grid.shape, (10,))
            self.assertEqual(grid.step, (1.0,))
            self.assertEqual(grid.domain, ((0, 10.0),))
            self.assertEqual(grid.ndim, 1)
            self.assertEqual(grid.cell_center, (0.5,))
            self.assertEqual(grid.cell_faces, ((1.0,),))

        with self.subTest("1d domain scalar size"):
            grid = grids.Grid((10,), domain=10)
            self.assertEqual(grid.domain, ((0.0, 10.0),))

        with self.subTest("2d"):
            grid = grids.Grid(
                (10, 10),
                step=0.1,
            )
            self.assertEqual(grid.step, (0.1, 0.1))
            self.assertEqual(grid.domain, ((0, 1.0), (0, 1.0)))
            self.assertEqual(grid.ndim, 2)
            self.assertEqual(grid.cell_center, (0.5, 0.5))
            self.assertEqual(grid.cell_faces, ((1.0, 0.5), (0.5, 1.0)))

        with self.subTest("1d domain"):
            grid = grids.Grid((10,), domain=[(-2, 2)])
            self.assertEqual(grid.step, (2 / 5,))
            self.assertEqual(grid.domain, ((-2.0, 2.0),))
            self.assertEqual(grid.ndim, 1)
            self.assertEqual(grid.cell_center, (0.5,))
            self.assertEqual(grid.cell_faces, ((1.0,),))

        with self.subTest("2d domain"):
            grid = grids.Grid((10, 20), domain=[(-2, 2), (0, 3)])
            self.assertEqual(grid.step, (4 / 10, 3 / 20))
            self.assertEqual(grid.domain, ((-2.0, 2.0), (0.0, 3.0)))
            self.assertEqual(grid.ndim, 2)
            self.assertEqual(grid.cell_center, (0.5, 0.5))
            self.assertEqual(grid.cell_faces, ((1.0, 0.5), (0.5, 1.0)))

        with self.subTest("2d periodic"):
            grid = grids.Grid((10, 20), domain=2 * torch.pi)
            self.assertEqual(grid.step, (2 * torch.pi / 10, 2 * torch.pi / 20))
            self.assertEqual(grid.domain, ((0.0, 2 * torch.pi), (0.0, 2 * torch.pi)))
            self.assertEqual(grid.ndim, 2)

        with self.assertRaisesRegex(TypeError, "cannot provide both"):
            grids.Grid((2,), step=(1.0,), domain=[(0, 2.0)])
        with self.assertRaisesRegex(ValueError, "length of domain"):
            grids.Grid((2, 3), domain=[(0, 1)])
        with self.assertRaisesRegex(ValueError, "pairs of numbers"):
            grids.Grid((2,), domain=[(0, 1, 2)])  # type: ignore
        with self.assertRaisesRegex(ValueError, "length of step"):
            grids.Grid((2, 3), step=(1.0,))

    def test_stagger(self):
        grid = grids.Grid((10, 10))
        array_1 = torch.zeros((10, 10))
        array_2 = torch.ones((10, 10))
        u, v = grid.stagger((array_1, array_2))
        self.assertEqual(u.offset, (1.0, 0.5))
        self.assertEqual(v.offset, (0.5, 1.0))

    def test_center(self):
        grid = grids.Grid((10, 10))

        with self.subTest("array ndim same as grid"):
            array_1 = torch.zeros((10, 10))
            array_2 = torch.zeros((20, 30))
            v = (array_1, array_2)  # tuple is a simple pytree
            v_centered = grid.center(v)
            self.assertLen(v_centered, 2)
            self.assertIsInstance(v_centered[0], grids.GridVariable)
            self.assertIsInstance(v_centered[1], grids.GridVariable)
            self.assertEqual(v_centered[0].shape, (10, 10))
            self.assertEqual(v_centered[1].shape, (20, 30))
            self.assertEqual(v_centered[0].offset, (0.5, 0.5))
            self.assertEqual(v_centered[1].offset, (0.5, 0.5))

        with self.subTest("array ndim different than grid"):
            # Assigns offset dimension based on grid.ndim
            array_1 = torch.zeros((10,))
            array_2 = torch.ones((10, 10, 10))
            v = (array_1, array_2)  # tuple is a simple pytree
            v_centered = grid.center(v)
            self.assertLen(v_centered, 2)
            self.assertIsInstance(v_centered[0], grids.GridVariable)
            self.assertIsInstance(v_centered[1], grids.GridVariable)
            self.assertEqual(v_centered[0].shape, (10,))
            self.assertEqual(v_centered[1].shape, (10, 10, 10))
            self.assertEqual(v_centered[0].offset, (0.5, 0.5))
            self.assertEqual(v_centered[1].offset, (0.5, 0.5))

    def test_axes_and_mesh(self):
        with self.subTest("1d"):
            grid = grids.Grid((5,), step=0.1)
            axes = grid.axes()
            self.assertLen(axes, 1)
            self.assertAllClose(axes[0], torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45]))
            mesh = grid.mesh()
            self.assertLen(mesh, 1)
            self.assertAllClose(axes[0], mesh[0])  # in 1d, mesh matches array

        with self.subTest("1d with offset"):
            grid = grids.Grid((5,), step=0.1)
            axes = grid.axes(offset=(0,))
            self.assertLen(axes, 1)
            self.assertAllClose(axes[0], torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]))
            mesh = grid.mesh(offset=(0,))
            self.assertLen(mesh, 1)
            self.assertAllClose(axes[0], mesh[0])  # in 1d, mesh matches array

        with self.subTest("2d"):
            grid = grids.Grid((4, 6), domain=[(-2, 2), (0, 3)])
            axes = grid.axes()
            self.assertLen(axes, 2)
            self.assertAllClose(axes[0], torch.tensor([-1.5, -0.5, 0.5, 1.5]))
            self.assertAllClose(
                axes[1], torch.tensor([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
            )
            mesh = grid.mesh()
            self.assertLen(mesh, 2)
            self.assertEqual(mesh[0].shape, (4, 6))
            self.assertEqual(mesh[1].shape, (4, 6))
            self.assertAllClose(mesh[0][:, 0], axes[0])
            self.assertAllClose(mesh[1][0, :], axes[1])

        with self.subTest("2d with offset"):
            grid = grids.Grid((4, 6), domain=[(-2, 2), (0, 3)])
            axes = grid.axes(offset=(0, 1))
            self.assertLen(axes, 2)
            self.assertAllClose(axes[0], torch.tensor([-2.0, -1.0, 0.0, 1.0]))
            self.assertAllClose(axes[1], torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
            mesh = grid.mesh(offset=(0, 1))
            self.assertLen(mesh, 2)
            self.assertEqual(mesh[0].shape, (4, 6))
            self.assertEqual(mesh[1].shape, (4, 6))
            self.assertAllClose(mesh[0][:, 0], axes[0])
            self.assertAllClose(mesh[1][0, :], axes[1])

    @parameterized.parameters(
        dict(
            shape=(10,),
            fn=lambda x: 2 * torch.ones_like(x),
            offset=None,
            expected_array=2 * torch.ones((10,)),
            expected_offset=(0.5,),
        ),
        dict(
            shape=(10, 10),
            fn=lambda x, y: torch.ones_like(x) + torch.ones_like(y),
            offset=(1, 0.5),
            expected_array=2 * torch.ones((10, 10)),
            expected_offset=(1, 0.5),
        ),
        dict(
            shape=(10, 10, 10),
            fn=lambda x, y, z: torch.ones_like(z),
            offset=None,
            expected_array=torch.ones((10, 10, 10)),
            expected_offset=(0.5, 0.5, 0.5),
        ),
    )
    def test_eval_on_mesh_default_offset(
        self, shape, fn, offset, expected_array, expected_offset
    ):
        grid = grids.Grid(shape, step=0.1)
        expected = grids.GridVariable(expected_array, expected_offset, grid)
        actual = grid.eval_on_mesh(fn, offset)
        self.assertArrayEqual(expected, actual)

    def test_spectral_axes(self):
        length = 42.0
        shape = (64,)
        grid = grids.Grid(shape, domain=((0, length),))

        (xs,) = grid.axes()
        (fft_xs,) = grid.fft_axes()
        fft_xs *= 2 * torch.pi  # convert ordinal to angular frequencies

        # compare the derivative of the sine function (i.e. cosine) with its
        # derivative computed in frequency-space. Note that this derivative involves
        # the computed frequencies so it can serve as a test.
        angular_freq = 2 * torch.pi / length
        ys = torch.sin(angular_freq * xs)
        expected = angular_freq * torch.cos(angular_freq * xs)
        actual = torch.fft.ifft(1j * fft_xs * torch.fft.fft(ys)).real
        self.assertAllClose(expected, actual, atol=1e-4, rtol=1e-8)

    def test_real_spectral_axes_1d(self):
        length = 42.0
        shape = (64,)
        grid = grids.Grid(shape, domain=((0, length),))

        (xs,) = grid.axes()
        (fft_xs,) = grid.rfft_mesh()
        fft_xs *= 2 * torch.pi  # convert ordinal to angular frequencies

        # compare the derivative of the sine function (i.e. cosine) with its
        # derivative computed in frequency-space. Note that this derivative involves
        # the computed frequencies so it can serve as a test.
        angular_freq = 2 * torch.pi / length
        ys = torch.sin(angular_freq * xs)
        expected = angular_freq * torch.cos(angular_freq * xs)
        actual = torch.fft.irfft(1j * fft_xs * torch.fft.rfft(ys))
        self.assertAllClose(expected, actual, atol=1e-4, rtol=1e-8)

    def test_real_spectral_axes_nd_shape(self):
        dim = 3
        grid_size = 64
        shape = (grid_size,) * dim
        fft_shape = shape[:-1] + (grid_size // 2 + 1,)
        domain = ((0, 2 * torch.pi),) * dim
        grid = grids.Grid(shape, domain=(domain))

        xs1, xs2, xs3 = grid.rfft_mesh()
        self.assertEqual(xs1.shape, fft_shape)
        self.assertEqual(xs2.shape, fft_shape)
        self.assertEqual(xs3.shape, fft_shape)

    def test_domain_interior_masks(self):
        with self.subTest("1d"):
            grid = grids.Grid((5,))
            expected = (torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32),)
            self.assertAllClose(expected, grids.domain_interior_masks(grid))

        with self.subTest("2d"):
            grid = grids.Grid((3, 3))
            expected = (
                torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=torch.float32),
                torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.float32),
            )
            self.assertAllClose(expected, grids.domain_interior_masks(grid))


if __name__ == "__main__":
    absltest.main()
