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
"""Tests for torch_cfd.boundaries."""

from functools import partial

import torch
from absl.testing import absltest, parameterized

from torch_cfd import boundaries, grids, test_utils

BCType = boundaries.BCType

tensor = partial(torch.tensor, dtype=torch.float32)


class ConstantBoundaryConditionsTest(test_utils.TestCase):

    def test_bc_utilities(self):
        with self.subTest("periodic"):
            bc = boundaries.periodic_boundary_conditions(ndim=2)
            self.assertEqual(
                bc.types, (("periodic", "periodic"), ("periodic", "periodic"))
            )

        with self.subTest("dirichlet"):
            bc = boundaries.dirichlet_boundary_conditions(ndim=2)
            self.assertEqual(
                bc.types, (("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"))
            )

        with self.subTest("neumann"):
            bc = boundaries.neumann_boundary_conditions(ndim=2)
            self.assertEqual(bc.types, (("neumann", "neumann"), ("neumann", "neumann")))

        with self.subTest("periodic_and_neumann"):
            bc = boundaries.periodic_and_neumann_boundary_conditions()
            self.assertEqual(
                bc.types, (("periodic", "periodic"), ("neumann", "neumann"))
            )

    def test_periodic_shift(self):
        shape = (4,)
        offset = (0,)
        step = (1.0,)
        data = tensor([11, 12, 13, 14])
        grid = grids.Grid(shape, step)
        bc = boundaries.periodic_boundary_conditions(grid.ndim)
        array = grids.GridVariable(data, offset, grid, bc)

        for shift, expected in [
            (-2, tensor([13, 14, 11, 12])),
            (-1, tensor([14, 11, 12, 13])),
            (0, tensor([11, 12, 13, 14])),
            (1, tensor([12, 13, 14, 11])),
            (2, tensor([13, 14, 11, 12])),
        ]:
            with self.subTest(shift=shift):
                shifted = bc.shift(array, shift, dim=0)
                expected_offset = (offset[0] + shift,)
                self.assertArrayEqual(
                    shifted, grids.GridVariable(expected, expected_offset, grid)
                )

    def test_trim_behavior(self):
        shape = (4,)
        offset = (0,)
        step = (1.0,)
        grid = grids.Grid(shape, step)

        base = tensor([11, 12, 13, 14], dtype=torch.float32)

        for width, expected, expected_offset in [
            (-1, tensor([12, 13, 14]), (1,)),
            (0, base, offset),
            (1, tensor([11, 12, 13]), (0,)),
            (2, tensor([11, 12]), (0,)),
        ]:
            with self.subTest(width=width):
                array = grids.GridVariable(base, offset, grid)
                trimmed = grids.trim(array, width, dim=0)
                self.assertArrayEqual(
                    trimmed, grids.GridVariable(expected, expected_offset, grid)
                )

    @parameterized.parameters(
        # Periodic BC
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=-2,
            expected_data=tensor([13, 14, 11, 12]),
            expected_offset=(-2,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=-1,
            expected_data=tensor([14, 11, 12, 13]),
            expected_offset=(-1,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=0,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 11]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=2,
            expected_data=tensor([13, 14, 11, 12]),
            expected_offset=(2,),
        ),
        # Dirichlet BC
        dict(
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            input_data=tensor([11, 12, 13, 0]),
            input_offset=(1,),
            shift_offset=-1,
            expected_data=tensor([0, 11, 12, 13]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            input_data=tensor([0, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=0,
            expected_data=tensor([0, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            input_data=tensor([0, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 0]),
            expected_offset=(1,),
        ),
        # Neumann BC
        dict(
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            shift_offset=-1,
            expected_data=tensor([11, 11, 12, 13]),
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            shift_offset=0,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 14]),
            expected_offset=(1.5,),
        ),
        # Dirichlet / Neumann BC
        dict(
            bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
            input_data=tensor([0, 12, 13, 14, 15]),
            input_offset=(0.5,),
            shift_offset=-1,
            expected_data=tensor([0, 0, 12, 13, 14]),  # this is cell center
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
            input_data=tensor([0, 12, 13, 14, 15]),
            input_offset=(0.5,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 15, 15]),
            expected_offset=(1.5,),
        ),
    )
    def test_shift_1d(
        self,
        bc_types,
        input_data,
        input_offset,
        shift_offset,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.HomogeneousBoundaryConditions(bc_types)
        actual = bc.shift(array, shift_offset, dim=0)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([-13, -12, -11, 1, 12, 13, 14]),
            input_offset=(-3,),
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([1, 12, 13, 14, 2, -12, -11]),
            input_offset=(0,),
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([10, 11, 12, 13, 14]),
            input_offset=(-0.5,),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([12, 13, 14, 15, 13]),
            input_offset=(0.5,),
            expected_data=tensor([12, 13, 14, 15]),
            expected_offset=(0.5,),
        ),
        # Dirichlet / Neumann BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([-11, 12, 13, 14, 12]),
            input_offset=(-0.5,),
            expected_data=tensor([12, 13, 14, 12]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([12, 13, 14, 12]),
            input_offset=(0.5,),
            expected_data=tensor([12, 13, 14, 12]),
            expected_offset=(0.5,),
        ),
        # Periodic BC
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            grid_shape=4,
            input_data=tensor([-12, 11, 12, 13, 14]),
            input_offset=(-1,),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            grid_shape=4,
            input_data=tensor([11, 12, 13, 14, 12]),
            input_offset=(0,),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_trim_padding_1d(
        self,
        grid_shape,
        input_data,
        input_offset,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_shape,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual, _ = bc._trim_padding(array, dim=0)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),  # cell nodes in 1d (cell edge in 2d)
            width=-1,
            expected_data=tensor([1, 1, 12, 13, 14]),
            expected_offset=(-1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),
            width=0,
            expected_data=tensor([1, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),
            width=1,
            expected_data=tensor([1, 12, 13, 14, 2]),
            expected_offset=(0,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=-1,
            expected_data=tensor([12, 11, 12, 13, 14]),
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=0,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=1,
            expected_data=tensor([11, 12, 13, 14, 16]),
            expected_offset=(0.5,),
        ),
        # Dirichlet / Neumann BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=-1,
            expected_data=tensor([-9, 11, 12, 13, 14]),
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=1,
            expected_data=tensor([11, 12, 13, 14, 16]),
            expected_offset=(0.5,),
        ),
    )
    def test_pad_1d_inhomogeneous(
        self, bc_types, input_data, input_offset, width, expected_data, expected_offset
    ):
        grid = grids.Grid((4,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = grids.pad(array, width, dim=0, bc=bc)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            width=-1,
            axis=0,
            expected_data=tensor(
                [
                    [-11, -12, -13, -14],
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            expected_offset=(-0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            width=1,
            axis=1,
            expected_data=tensor(
                [
                    [11, 12, 13, 14, -14],
                    [21, 22, 23, 24, -24],
                    [31, 32, 33, 34, -34],
                ]
            ),
            expected_offset=(0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 0],
                    [21, 22, 23, 0],
                    [31, 32, 33, 0],
                ]
            ),
            input_offset=(0.5, 1),  # edge aligned offset
            width=-1,
            axis=1,
            expected_data=tensor(
                [
                    [0, 11, 12, 13, 0],
                    [0, 21, 22, 23, 0],
                    [0, 31, 32, 33, 0],
                ]
            ),
            expected_offset=(0.5, 0),
        ),
    )
    def test_pad_2d_dirichlet_cell_center(
        self, input_data, input_offset, width, axis, expected_data, expected_offset
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
        actual = grids.pad(array, width, axis, bc=bc)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            values=((1.0, 2.0), (3.0, 4.0)),
            width=-1,
            axis=0,
            expected_data=tensor(
                [
                    [-9, -10, -11, -12],
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            expected_offset=(-0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            values=((1.0, 2.0), (3.0, 4.0)),
            width=1,
            axis=1,
            expected_data=tensor(
                [
                    [11, 12, 13, 14, -6],
                    [21, 22, 23, 24, -16],
                    [31, 32, 33, 34, -26],
                ]
            ),
            expected_offset=(0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 4],
                    [21, 22, 23, 4],
                    [31, 32, 33, 4],
                ]
            ),
            input_offset=(0.5, 1),  # edge aligned offset
            values=((1.0, 2.0), (3.0, 4.0)),
            width=-1,
            axis=1,
            expected_data=tensor(
                [
                    [3, 11, 12, 13, 4],
                    [3, 21, 22, 23, 4],
                    [3, 31, 32, 33, 4],
                ]
            ),
            expected_offset=(0.5, 0),
        ),
    )
    def test_pad_2d_dirichlet_cell_center_inhomogeneous(
        self,
        input_data,
        input_offset,
        values,
        width,
        axis,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
        actual = grids.pad(array, width, axis, bc=bc)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)


class BoundaryConditionsImposingTest(test_utils.TestCase):
    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([-14, -13, -12, 11, 12, 13, 14]),
            input_offset=(-3,),
            grid_size=4,
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, 2, -12, -11]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, 2, -12, -11]),
            input_offset=(1,),
            grid_size=5,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, -12, -11]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, 12]),
            input_offset=(-0.5,),
            grid_size=4,
            expected_data=tensor([12, 13, 14, 12]),
            expected_offset=(0.5,),
        ),
        # Periodic BC
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            input_data=tensor([-12, 11, 12, 13, 14]),
            input_offset=(-1,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            input_data=tensor([11, 12, 13, 14, 12]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_trim_boundary_1d(
        self,
        input_data,
        input_offset,
        grid_size,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_size,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = bc.trim_boundary(array)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([12, 13, 14]),
            input_offset=(1,),
            grid_size=4,
            expected_data=tensor([1, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(1,),
            grid_size=5,
            expected_data=tensor([11, 12, 13, 14, 2]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        # Periodic BC
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_pad_and_impose_bc_1d(
        self,
        input_data,
        input_offset,
        grid_size,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_size,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = bc.pad_and_impose_bc(array, expected_offset)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            input_data=tensor([0, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            expected_data=tensor([1, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            input_data=tensor([11, 12, 13, 14, 11]),
            input_offset=(1,),
            grid_size=5,
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            expected_data=tensor([11, 12, 13, 14, 2]),
            expected_offset=(1,),
        ),
        dict(
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        # Neumann BC
        dict(
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        # Periodic BC
        dict(
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_impose_bc_1d(
        self,
        input_data,
        input_offset,
        grid_size,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_size,))
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        array = grids.GridVariable(input_data, input_offset, grid, bc)
        actual = bc.impose_bc(array)
        expected = grids.GridVariable(expected_data, expected_offset, grid, bc)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            offset=(1.0, 0.5),
            values=((1.0, 2.0), (3.0, 4.0)),
            expected_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [2, 2, 2, 2],
                ]
            ),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            offset=(0.5, 0.0),
            values=((1.0, 2.0), (3.0, 4.0)),
            expected_data=tensor(
                [
                    [3, 12, 13, 14],
                    [3, 22, 23, 24],
                    [3, 32, 33, 34],
                ]
            ),
        ),
    )
    def test_impose_bc_2d_constant_boundary(
        self, input_data, offset, values, expected_data
    ):
        grid = grids.Grid(input_data.shape)
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
        variable = grids.GridVariable(input_data, offset, grid, bc)
        variable = bc.impose_bc(variable)
        expected = grids.GridVariable(expected_data, offset, grid)
        self.assertArrayEqual(variable, expected)


class PressureBoundaryConditionsTest(test_utils.TestCase):
    def test_get_pressure_bc_from_velocity_2d(self):
        grid = grids.Grid((10, 10))
        bc = boundaries.dirichlet_boundary_conditions(ndim=2)
        u_array = grids.GridVariable(torch.zeros(grid.shape), (1, 0.5), grid, bc)
        v_array = grids.GridVariable(torch.zeros(grid.shape), (0.5, 1), grid, bc)
        v = grids.GridVariableVector(tuple([u_array, v_array]))
        pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
        self.assertEqual(
            pressure_bc.types,
            ((BCType.NEUMANN, BCType.NEUMANN), (BCType.NEUMANN, BCType.NEUMANN)),
        )


if __name__ == "__main__":
    absltest.main()
