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

"""Tests for torch_cfd.finite_difference."""
import math
import torch
from einops import repeat
from absl.testing import absltest, parameterized

from torch_cfd import boundaries, finite_differences as fdm, grids, test_utils

def _trim_boundary(array):
    # fixed jax-cfd bug that trims all dimension for a batched GridVariable
    if isinstance(array, grids.GridVariable):
        # Convert tuple of slices to individual slice objects
        trimmed_slices = (slice(1, -1),) * array.grid.ndim
        data = array.data[(..., *trimmed_slices)]
        return grids.GridVariable(data, array.offset, array.grid)
    else:
        tensor = torch.as_tensor(array)
        trimmed_slices = (slice(1, -1),) * tensor.ndim
        return tensor[(..., *trimmed_slices)]

def periodic_grid_variable(data, offset, grid):
    return grids.GridVariable(
        data, offset, grid, bc=boundaries.periodic_boundary_conditions(grid.ndim)
    )


def stack_tensor_matrix(matrix):
    """Stacks a 2D list or tuple of tensors into a rank-4 tensor."""
    return torch.stack([torch.stack(row, dim=0) for row in matrix], dim=0)


class FiniteDifferenceTest(test_utils.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_periodic",
            method=fdm.central_difference,
            shape=(3, 3),
            step=(1.0, 1.0),
            expected_offset=0,
            negative_shift=-1,
            positive_shift=1,
        ),
        dict(
            testcase_name="_backward_difference_periodic",
            method=fdm.backward_difference,
            shape=(2, 3),
            step=(0.1, 0.3),
            expected_offset=-0.5,
            negative_shift=-1,
            positive_shift=0,
        ),
        dict(
            testcase_name="_forward_difference_periodic",
            method=fdm.forward_difference,
            shape=(23, 32),
            step=(0.01, 2.0),
            expected_offset=+0.5,
            negative_shift=0,
            positive_shift=1,
        ),
    )
    def test_finite_difference_indexing(
        self, method, shape, step, expected_offset, negative_shift, positive_shift
    ):
        """Tests finite difference code using explicit indices."""
        grid = grids.Grid(shape, step)
        u = periodic_grid_variable(
            torch.arange(math.prod(shape)).reshape(shape), (0, 0), grid
        )
        actual_x, actual_y = method(u)

        x, y = torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij")

        diff_x = (
            u.data[torch.roll(x, -positive_shift, dims=0), y]
            - u.data[torch.roll(x, -negative_shift, dims=0), y]
        )
        expected_data_x = diff_x / step[0] / (positive_shift - negative_shift)
        expected_x = grids.GridVariable(expected_data_x, (expected_offset, 0), grid)

        diff_y = (
            u.data[x, torch.roll(y, -positive_shift, dims=1)]
            - u.data[x, torch.roll(y, -negative_shift, dims=1)]
        )
        expected_data_y = diff_y / step[1] / (positive_shift - negative_shift)
        expected_y = grids.GridVariable(expected_data_y, (0, expected_offset), grid)

        self.assertArrayEqual(expected_x, actual_x)
        self.assertArrayEqual(expected_y, actual_y)

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_periodic",
            method=fdm.central_difference,
            shape=(100, 100, 100),
            offset=(0, 0, 0),
            f=lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.sin(z),
            gradf=(
                lambda x, y, z: -torch.sin(x) * torch.cos(y) * torch.sin(z),
                lambda x, y, z: -torch.cos(x) * torch.sin(y) * torch.sin(z),
                lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.cos(z),
            ),
            atol=1e-3,
            rtol=1e-3,
        ),
        dict(
            testcase_name="_backward_difference_periodic",
            method=fdm.backward_difference,
            shape=(100, 100, 100),
            offset=(0, 0, 0),
            f=lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.sin(z),
            gradf=(
                lambda x, y, z: -torch.sin(x) * torch.cos(y) * torch.sin(z),
                lambda x, y, z: -torch.cos(x) * torch.sin(y) * torch.sin(z),
                lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.cos(z),
            ),
            atol=5e-2,
            rtol=1e-3,
        ),
        dict(
            testcase_name="_forward_difference_periodic",
            method=fdm.forward_difference,
            shape=(200, 200, 200),
            offset=(0, 0, 0),
            f=lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.sin(z),
            gradf=(
                lambda x, y, z: -torch.sin(x) * torch.cos(y) * torch.sin(z),
                lambda x, y, z: -torch.cos(x) * torch.sin(y) * torch.sin(z),
                lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.cos(z),
            ),
            atol=5e-2,
            rtol=1e-3,
        ),
    )
    def test_finite_difference_analytic(
        self, method, shape, offset, f, gradf, atol, rtol
    ):
        """Tests finite difference code comparing to the analytice solution."""
        step = tuple([2.0 * torch.pi / s for s in shape])
        grid = grids.Grid(shape, step)
        mesh = grid.mesh()
        u = periodic_grid_variable(f(*mesh), offset, grid)
        expected_grad = torch.stack([df(*mesh) for df in gradf])
        actual_grad = torch.stack([array.data for array in method(u)])
        self.assertAllClose(expected_grad, actual_grad, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_2D_constant",
            shape=(20, 20),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-3,
            rtol=1e-8,
        ),
        dict(
            testcase_name="_2D_quadratic",
            shape=(21, 21),
            f=lambda x, y: x * (x - 1.0) + y * (y - 1.0),
            g=lambda x, y: 4 * torch.ones_like(x),
            atol=1e-3,
            rtol=1e-8,
        ),
    )
    def test_laplacian(self, shape, f, g, atol, rtol):
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        offset = (0,) * len(shape)
        mesh = grid.mesh(offset)
        u = periodic_grid_variable(f(*mesh), offset, grid)
        expected_laplacian = _trim_boundary(grids.GridVariable(g(*mesh), offset, grid))
        actual_laplacian = _trim_boundary(fdm.laplacian(u))
        self.assertAllClose(expected_laplacian, actual_laplacian, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_2D_constant",
            shape=(20, 20),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (torch.ones_like(x), torch.ones_like(y)),
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-3,
            rtol=1e-12,
        ),
        dict(
            testcase_name="_2D_quadratic",
            shape=(21, 21),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (x * (x - 1.0), y * (y - 1.0)),
            g=lambda x, y: 2 * x + 2 * y - 2,
            atol=1e-1,
            rtol=1e-3,
        ),
    )
    def test_divergence(self, shape, offsets, f, g, atol, rtol):
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        v = [
            periodic_grid_variable(f(*grid.mesh(offset))[axis], offset, grid)
            for axis, offset in enumerate(offsets)
        ]
        expected_divergence = _trim_boundary(
            grids.GridVariable(g(*grid.mesh()), (0,) * grid.ndim, grid)
        )
        actual_divergence = _trim_boundary(fdm.divergence(v))
        self.assertAllClose(
            expected_divergence, actual_divergence, atol=atol, rtol=rtol
        )

    @parameterized.named_parameters(
        # https://en.wikipedia.org/wiki/Curl_(mathematics)#Examples
        dict(
            testcase_name="_solenoidal",
            shape=(20, 20),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (y, -x),
            g=lambda x, y: -2 * torch.ones_like(x),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_wikipedia_example_2",
            shape=(21, 21),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (torch.ones_like(x), -(x**2)),
            g=lambda x, y: -2 * x,
            atol=1e-3,
            rtol=1e-10,
        ),
    )
    def test_curl_2d(self, shape, offsets, f, g, atol, rtol):
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        v = [
            periodic_grid_variable(f(*grid.mesh(offset))[axis], offset, grid)
            for axis, offset in enumerate(offsets)
        ]
        result_offset = (0.5, 0.5)
        expected_curl = _trim_boundary(
            grids.GridVariable(g(*grid.mesh(result_offset)), result_offset, grid)
        )
        actual_curl = _trim_boundary(fdm.curl_2d(v))
        self.assertAllClose(expected_curl, actual_curl, atol=atol, rtol=rtol)


class FiniteDifferenceBatchTest(test_utils.TestCase):
    """Test finite difference operations with batch dimensions in 2D."""

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_batch",
            method=fdm.central_difference,
            batch_size=4,
            shape=(32, 32),
            step=(0.1, 0.1),
            expected_offset=0,
        ),
        dict(
            testcase_name="_backward_difference_batch",
            method=fdm.backward_difference,
            batch_size=2,
            shape=(16, 24),
            step=(0.2, 0.15),
            expected_offset=-0.5,
        ),
        dict(
            testcase_name="_forward_difference_batch",
            method=fdm.forward_difference,
            batch_size=3,
            shape=(20, 20),
            step=(0.05, 0.05),
            expected_offset=+0.5,
        ),
    )
    def test_finite_difference_batch_preserves_shape(
        self, method, batch_size, shape, step, expected_offset
    ):
        """Test that finite difference operations preserve batch dimensions."""
        grid = grids.Grid(shape, step)
        
        # Create batched data: (batch_size, *shape)
        batched_data = torch.randn(batch_size, *shape)
        u = periodic_grid_variable(batched_data, (0, 0), grid)
        
        # Apply finite difference
        grad_x, grad_y = method(u)
        
        # Check that batch dimension is preserved
        self.assertEqual(grad_x.data.shape, (batch_size, *shape))
        self.assertEqual(grad_y.data.shape, (batch_size, *shape))
        
        # Check offsets are correct
        self.assertEqual(grad_x.offset, (expected_offset, 0))
        self.assertEqual(grad_y.offset, (0, expected_offset))

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_coarse",
            method=fdm.central_difference,
            batch_size=2,
            shape=(40, 60),
            f=lambda x, y: torch.sin(x) * torch.cos(y),
            gradf=(
                lambda x, y: torch.cos(x) * torch.cos(y),
                lambda x, y: -torch.sin(x) * torch.sin(y),
            ),
            atol=2*math.pi/40,
            rtol=1/40,
        ),
        dict(
            testcase_name="_forward_difference",
            method=fdm.forward_difference,
            batch_size=3,
            shape=(128, 256),
            f=lambda x, y: torch.sin(x) * torch.cos(y),
            gradf=(
                lambda x, y: torch.cos(x) * torch.cos(y),
                lambda x, y: -torch.sin(x) * torch.sin(y),
            ),
            atol=2*math.pi/128,
            rtol=1/128,
        ),
        dict(
            testcase_name="_central_difference_coarse_fine",
            method=fdm.central_difference,
            batch_size=8,
            shape=(1024, 2048),
            f=lambda x, y: torch.sin(x) * torch.cos(y),
            gradf=(
                lambda x, y: torch.cos(x) * torch.cos(y),
                lambda x, y: -torch.sin(x) * torch.sin(y),
            ),
            atol=2*math.pi/1024,
            rtol=1/1024,
        ),
    )
    def test_finite_difference_batch_analytic(
        self, method, batch_size, shape, f, gradf, atol, rtol
    ):
        """Test finite difference on batched data against analytical solutions."""
        step = tuple([2.0 * math.pi / s for s in shape])
        grid = grids.Grid(shape, step)
        mesh = grid.mesh()
        
        # Create batched data by repeating the same function
        # In practice, each batch element could be different
        single_data = f(*mesh)
        batched_data = repeat(single_data, 'h w -> b h w', b=batch_size)
        
        u = periodic_grid_variable(batched_data, (0, 0), grid)
        
        # Compute gradients
        actual_grad_x, actual_grad_y = method(u)
        
        # Expected gradients (also batched)
        expected_grad_x = repeat(gradf[0](*mesh), 'h w -> b h w', b=batch_size)
        expected_grad_y = repeat(gradf[1](*mesh), 'h w -> b h w', b=batch_size)
        
        self.assertAllClose(expected_grad_x, actual_grad_x.data, atol=atol, rtol=rtol)
        self.assertAllClose(expected_grad_y, actual_grad_y.data, atol=atol, rtol=rtol)

    def test_laplacian_batch(self):
        """Test Laplacian operator with batch dimensions."""
        batch_size = 2
        shape = (20, 20)
        step = (0.1, 0.1)
        grid = grids.Grid(shape, step)
        offset = (0, 0)
        
        # Test function: f(x,y) = x^2 + y^2, so Laplacian should be 4
        mesh = grid.mesh(offset)
        single_data = mesh[0]**2 + mesh[1]**2
        batched_data = single_data.unsqueeze(0).repeat(batch_size, 1, 1)
        
        u = periodic_grid_variable(batched_data, offset, grid)
        actual_laplacian = fdm.laplacian(u)
        
        # Expected Laplacian is 4 everywhere
        expected_laplacian = 4 * torch.ones(batch_size, *shape)
        
        # Trim boundary for comparison
        trimmed_actual = _trim_boundary(actual_laplacian)
        trimmed_expected = _trim_boundary(
            grids.GridVariable(expected_laplacian, offset, grid)
        )
        
        self.assertAllClose(
            trimmed_expected.data, trimmed_actual.data, atol=1e-2, rtol=1e-8
        )

    def test_laplacian_batch_analytic(self):
        """Test Laplacian operator on batched data against Laplacian on a single data."""
        batch_size = 3
        shape = (128, 128)
        step = (2*math.pi/128, 2*math.pi/128)
        grid = grids.Grid(shape, step)
        
        mesh = grid.mesh()
        single_data = torch.sin(mesh[0]) * torch.cos(mesh[1])
        batched_data = repeat(single_data, 'h w -> b h w', b=batch_size)
        
        u_single = periodic_grid_variable(single_data, (0, 0), grid)
        u_batch = periodic_grid_variable(batched_data, (0, 0), grid)
        single_laplacian = fdm.laplacian(u_single)
        batch_laplacian = fdm.laplacian(u_batch)
        
        # Trim boundary for comparison
        trimmed_single = _trim_boundary(single_laplacian)
        trimmed_batch = _trim_boundary(batch_laplacian)
        
        for i in range(batch_size):
            self.assertAllClose(
                trimmed_single.data, trimmed_batch.data[i], atol=1e-8, rtol=1e-12
            )

    def test_divergence_batch(self):
        """Test divergence operator with batch dimensions."""
        batch_size = 3
        shape = (16, 16)
        step = (0.1, 0.1)
        grid = grids.Grid(shape, step)
        offsets = ((0.5, 0), (0, 0.5))
        
        # Test vector field: v = (x, y), so divergence should be 2
        mesh_x = grid.mesh(offsets[0])
        mesh_y = grid.mesh(offsets[1])
        
        # Create batched vector components
        vx_single = mesh_x[0]  # x component
        vy_single = mesh_y[1]  # y component
        
        vx_batched = repeat(vx_single, 'h w -> b h w', b=batch_size)
        vy_batched = repeat(vy_single, 'h w -> b h w', b=batch_size)
        
        v = [
            periodic_grid_variable(vx_batched, offsets[0], grid),
            periodic_grid_variable(vy_batched, offsets[1], grid),
        ]
        
        actual_divergence = fdm.divergence(v)
        
        # Expected divergence is 2 everywhere
        expected_divergence = 2 * torch.ones(batch_size, *shape)
        
        # Trim boundary for comparison
        trimmed_actual = _trim_boundary(actual_divergence)
        trimmed_expected = _trim_boundary(
            grids.GridVariable(expected_divergence, (0, 0), grid)
        )
        
        self.assertAllClose(
            trimmed_expected.data, trimmed_actual.data, atol=1e-2, rtol=1e-8
        )

    def test_curl_2d_batch(self):
        """Test 2D curl operator with batch dimensions."""
        batch_size = 2
        shape = (20, 20)
        step = (0.1, 0.1)
        grid = grids.Grid(shape, step)
        offsets = ((0.5, 0), (0, 0.5))
        
        # Test vector field: v = (-y, x), so curl should be 2
        mesh_x = grid.mesh(offsets[0])
        mesh_y = grid.mesh(offsets[1])
        
        # Create batched vector components
        vx_single = -mesh_x[1]  # -y component
        vy_single = mesh_y[0]   # x component
        
        vx_batched = repeat(vx_single, 'h w -> b h w', b=batch_size)
        vy_batched = repeat(vy_single, 'h w -> b h w', b=batch_size)
        
        v = [
            periodic_grid_variable(vx_batched, offsets[0], grid),
            periodic_grid_variable(vy_batched, offsets[1], grid),
        ]
        
        actual_curl = fdm.curl_2d(v)
        
        # Expected curl is 2 everywhere
        result_offset = (0.5, 0.5)
        expected_curl = 2 * torch.ones(batch_size, *shape)
        
        # Trim boundary for comparison
        trimmed_actual = _trim_boundary(actual_curl)
        trimmed_expected = _trim_boundary(
            grids.GridVariable(expected_curl, result_offset, grid)
        )
        
        self.assertAllClose(
            trimmed_expected.data, trimmed_actual.data, atol=1e-2, rtol=1e-8
        )

    def test_batch_consistency_across_operations(self):
        """Test that batch operations are consistent across different batch sizes."""
        shape = (24, 24)
        step = (0.05, 0.05)
        grid = grids.Grid(shape, step)
        
        # Create test function
        mesh = grid.mesh()
        single_data = torch.sin(mesh[0]) * torch.cos(mesh[1])
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            batched_data = repeat(single_data, 'h w -> b h w', b=batch_size)
            u = periodic_grid_variable(batched_data, (0, 0), grid)
            
            # Apply central difference
            grad_x = fdm.central_difference(u, dim=0)
            grad_y = fdm.central_difference(u, dim=1)
            
            # Each batch element should be identical
            for i in range(1, batch_size):
                self.assertAllClose(grad_x.data[0], grad_x.data[i])
                self.assertAllClose(grad_y.data[0], grad_y.data[i])

if __name__ == "__main__":
    absltest.main()
