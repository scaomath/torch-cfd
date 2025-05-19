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

# Modifications copyright (C) 2024 S.Cao
# ported Google's Jax-CFD functional template to PyTorch's tensor ops

"""Functions for computing and applying pressure."""

from functools import reduce
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.fft as fft
import torch.nn as nn

from torch_cfd import (
    boundaries,
    finite_differences as fdm,
    grids,
)

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions

def _set_laplacian(module: nn.Module, grid: grids.Grid, bc: Sequence[BoundaryConditions]):
    """Initialize the Laplacian operators."""
    if module.laplacians is None:
        laplacians = fdm.set_laplacian_matrix(grid, bc)
        laplacians = torch.stack(laplacians, dim=0)
        module.register_buffer("laplacians", laplacians, persistent=True)
    else:
        # Check if the provided laplacians are consistent with the grid
        for laplacian in module.laplacians:
            if laplacian.shape != grid.shape:
                raise ValueError("Provided laplacians do not match the grid shape.")

def _set_laplacian(module: nn.Module, laplacians: torch.Tensor, grid: grids.Grid, bc: Sequence[BoundaryConditions]):
        """
        Initialize the Laplacian operators.
        Args:
            laplacians have the shape (ndim, n, n)
        """
        if laplacians is None:
            laplacians = fdm.set_laplacian_matrix(grid, bc)
            laplacians = torch.stack(laplacians, dim=0)
        else:
            # Check if the provided laplacians are consistent with the grid
            for laplacian in laplacians:
                if laplacian.shape != grid.shape:
                    raise ValueError("Provided laplacians do not match the grid shape.")
        module.register_buffer("laplacians", laplacians, persistent=True)


class PressureProjection(nn.Module):
    def __init__(
        self,
        grid: grids.Grid,
        bc: Sequence[BoundaryConditions],
        dtype: Optional[torch.dtype] = torch.float32,
        implementation: Optional[str] = None,
        laplacians: Optional[torch.Tensor] = None,
        initial_guess_pressure: Optional[GridArray] = None,
    ):
        """
        Args:
            grid: Grid object describing the spatial domain.
            bc: Boundary conditions for the Laplacian operator (for pressure).
            dtype: Tensor data type. For consistency purpose.
            implementation: One of ['fft', 'rfft', 'matmul'].
            circulant: If True, bc is periodical
            laplacians: Precomputed Laplacian operators. If None, they are computed from the grid during initiliazation.
            initial_guess_pressure: Initial guess for pressure. If None, a zero tensor is used.
        """
        super().__init__()
        self.grid = grid
        self.bc = bc
        self.dtype = dtype
        self.implementation = implementation
        _set_laplacian(self, laplacians, grid, bc)

        self.solver = Pseudoinverse(
            grid=grid,
            bc=bc,
            dtype=dtype,
            hermitian=True,
            implementation=implementation,
            laplacians=self.laplacians
        )
        if initial_guess_pressure is None:
            initial_guess_pressure = GridArray(
                torch.zeros(grid.shape), grid.cell_center, grid
            )
            self.q0 = bc.impose_bc(initial_guess_pressure)

    def forward(self, v: GridVariableVector) -> GridVariableVector:
        """Project velocity to be divergence-free."""
        _ = grids.consistent_grid(self.grid, *v)
        pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

        rhs = fdm.divergence(v)
        rhs_transformed = self.rhs_transform(rhs, pressure_bc)
        rhs_inv = self.solver(rhs_transformed)
        q = GridArray(rhs_inv, rhs.offset, rhs.grid)
        q = pressure_bc.impose_bc(q)
        q_grad = fdm.forward_difference(q)
        v_projected = GridVariableVector(
            tuple(u.bc.impose_bc(u.array - q_g) for u, q_g in zip(v, q_grad))
        )
        # assert v_projected.__len__() == v.__len__()
        return v_projected
        

    @staticmethod
    def rhs_transform(
        u: GridArray,
        bc: BoundaryConditions,
    ) -> torch.Tensor:
        """Transform the RHS of pressure projection equation for stability."""
        u_data = u.data  # (b, n, m) or (n, m)
        for axis in range(u.grid.ndim):
            if (
                bc.types[axis][0] == boundaries.BCType.NEUMANN
                and bc.types[axis][1] == boundaries.BCType.NEUMANN
            ):
                # Check if we have batched data
                if u_data.ndim > u.grid.ndim:
                    # For batched data, calculate mean separately for each batch
                    # Keep the batch dimension, reduce over grid dimensions
                    dims = tuple(range(1, u_data.ndim))
                    mean = torch.mean(u_data, dim=dims, keepdim=True)
                else:
                    # For non-batched data, calculate global mean
                    mean = torch.mean(u_data)

                u_data = u_data - mean
        return u_data


class Pseudoinverse(nn.Module):
    def __init__(
        self,
        grid: grids.Grid,
        bc: Optional[Sequence[boundaries.BoundaryConditions]] = None,
        dtype: torch.dtype = torch.float32,
        hermitian: bool = True,
        circulant: bool = True,
        implementation: Optional[str] = None,
        laplacians: Optional[torch.Tensor] = None,
        cutoff: Optional[float] = None,
    ):
        r"""
        This class applies the pseudoinverse of the Laplacian operator on a given Grid.
        This class re-implements to Jax-cfd's function_call type implementations
            - _hermitian_matmul_transform()
            - _circulant_fft_transform()
            - _circulant_rfft_transform()
        in the fast_diagonalization.py:
        https://github.com/google/jax-cfd/blob/main/jax_cfd/base/fast_diagonalization.py
        to PyTorch's tensor ops using nn.Module.

        The application of a linear operator (the inverse of Laplacian)
        can be written as a sum of operators on each axis.
        Such linear operators are *separable*, and can be written as a sum of tensor
        products, e.g., `operators = [A, B]` corresponds to the linear operator
        A ⊗ I + I ⊗ B, where the tensor product ⊗ indicates a separation between
        operators applied along the first and second axis.

        This function computes matrix-valued functions of such linear operators via
        the "fast diagonalization method" [1]:
        F(A ⊗ I + I ⊗ B)
        = (X(A) ⊗ X(B)) F(Λ(A) ⊗ I + I ⊗ Λ(B)) (X(A)^{-1} ⊗ X(B)^{-1})

        where X(A) denotes the matrix of eigenvectors of A and Λ(A) denotes the
        (diagonal) matrix of eigenvalues. The function `F` is easy to compute in
        this basis, because matrix Λ(A) ⊗ I + I ⊗ Λ(B) is diagonal.

        The current implementation directly diagonalizes dense matrices for each
        linear operator, which limits it's applicability to grids with less than
        1e3-1e4 elements per side (~1 second to several minutes of setup time).

        Example: The Laplacian operator can be written as a sum of 1D Laplacian
        operators along each axis, i.e., as a sum of 1D convolutions along each axis.
        This can be seen mathematically (∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²) or by
        decomposing the 2D kernel:

        [0  1  0]               [ 1]
        [1 -4  1] = [1 -2  1] ⊕ [-2]
        [0  1  0]               [ 1]

        Args:
            grid: Grid object describing the spatial domain.
            bc: Boundary conditions for the Laplacian operator (for pressure).
            dtype: Tensor data type.
            hermitian: hermitian: whether or not all linear operator are Hermitian (i.e., symmetric in the real valued case).
            circulant: If True, bc is periodical
            implementation: One of ['fft', 'rfft', 'matmul'].
            cutoff: Minimum eigenvalue to invert.
            laplacians: Precomputed Laplacian operators. If None, they are computed from the grid during initiliazation.


        implementation: how to implement fast diagonalization. Default uses 'rfft'
            for grid size larger than 1024 and 'matmul' otherwise:
            - 'matmul': scales like O(N**(d+1)) for d N-dimensional operators, but
            makes good use of matmul hardware. Requires hermitian=True.
            - 'fft': scales like O(N**d * log(N)) for d N-dimensional operators.
            Requires circulant=True.
            - 'rfft': use the RFFT instead of the FFT. This is a little faster than
            'fft' but also has slightly larger error. It currently requires an even
            sized last axis and circulant=True.
        precision: numerical precision for matrix multplication. Only relevant on
            TPUs with implementation='matmul'.

        Returns:
            The pseudoinverse of the Laplacian operator acting on the input tensor.

        TODO:
        - [x] change the implementation to tensor2tensor
        - [x] originally the laplacian is implemented as 
            laplacians = array_utils.laplacian_matrix_w_boundaries(rhs.grid, rhs.offset, pressure_bc), needs to add this wrapper to support non-periodic BCs. (May 2025): now this is passed by fdm.set_laplacian_matrix
        - [x] add the precomputation to the eigenvalues

        References:
        [1] Lynch, R. E., Rice, J. R. & Thomas, D. H. Direct solution of partial
            difference equations by tensor product methods. Numer. Math. 6, 185-199
            (1964). https://paperpile.com/app/p/b7fdea4e-b2f7-0ada-b056-a282325c3ecf

        """
        super().__init__()
        self.grid = grid
        self.bc = bc

        if self.bc is None:
            self.bc = boundaries.HomogeneousBoundaryConditions(
                (
                    (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
                    (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
                )
            )

        self.cutoff = cutoff or 10 * torch.finfo(dtype).eps

        self.hermitian = hermitian
        self.circulant = circulant
        self.implementation = implementation
        _set_laplacian(self, laplacians, grid, bc)


        if implementation is None:
            self.implementation = "rfft"
            self.circulant = True
        if implementation == "rfft" and self.laplacians[-1].shape[0] % 2:
            self.implementation = "matmul"
            self.circulant = False

        if self.implementation == "rfft":
            self.ifft = fft.irfftn
            self.fft = fft.rfftn
        elif self.implementation == "fft":
            self.ifft = fft.ifftn
            self.fft = fft.fftn
        if self.implementation not in ("fft", "rfft", "matmul"):
            raise NotImplementedError(f"Unsupported implementation: {implementation}")

        self.eigenvalues = self._compute_eigenvalues()

        if self.implementation in ("fft", "rfft"):
            if not self.circulant:
                raise ValueError(
                    f"non-circulant operators not yet supported with implementation='fft' or 'rfft' "
                )
            self._forward = self._apply_in_frequency_space
        elif self.implementation == "matmul":
            if not self.hermitian:
                raise ValueError(
                    "matmul implementation requires hermitian=True. "
                    "Use fft or rfft for non-hermitian operators."
                )
            self._forward = self._apply_in_svd_space

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Apply the pseudoinverse (with a cutoff) Laplacian operator to the input tensor.

        Args:
            value: right-hand-side of the linear operator. This is a tensor with `len(operators)` dimensions, where each dimension corresponds to one of the linear operators.
        """
        return self._forward(value, self.inverse)

    @staticmethod
    def outer_sum(x: Union[List[torch.Tensor], Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Returns the outer sum of a list of one dimensional arrays
        Example:
        x = [a, b, c]
        out = a[..., None, None] + b[..., None] + c

        The full outer sum is equivalent to:
        def _sum(a, b):
            return a[..., None] + b
        return reduce(_sum, x)
        """

        return reduce(lambda a, b: a[..., None] + b, x)

    def _compute_eigenvalues(self):
        """
        Precompute the Laplacian eigenvalues on the Grid mesh.
        """
        eigenvalues = torch.tensor([1.0] * self.grid.ndim)
        eigenvectors = torch.tensor([1.0] * self.grid.ndim)
        if self.implementation == "fft":
            eigenvalues = [fft.fft(op[:, 0]) for op in self.laplacians]
        elif self.implementation == "rfft":
            eigenvalues = [fft.fft(op[:, 0]) for op in self.laplacians[:-1]] + [
                fft.rfft(self.laplacians[-1][:, 0])
            ]
        elif self.implementation == "matmul":
            eigenvalues, eigenvectors = zip(*map(torch.linalg.eigh, self.laplacians))
        else:
            raise NotImplementedError(
                f"Unsupported implementation: {self.implementation} and eigenvalues are not precomputed."
            )
        summed_eigenvalues = self.outer_sum(eigenvalues)
        inverse_eigvs = torch.asarray(
            self._filter_eigenvalues(summed_eigenvalues)
        )
    

        if inverse_eigvs.shape != summed_eigenvalues.shape:
            raise ValueError(
                "output shape from func() does not match input shape: "
                f"{inverse_eigvs.shape} vs {summed_eigenvalues.shape}"
            )
        self.register_buffer("inverse", inverse_eigvs, persistent=True)
        self.register_buffer("eigenvectors", eigenvectors, persistent=True)

    def _filter_eigenvalues(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Apply a cutoff function to the eigenvalues.
        """
        return torch.where(torch.abs(eigenvalues) > self.cutoff, 1 / eigenvalues, 0)

    def _apply_in_frequency_space(
        self, value: torch.Tensor, multiplier: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the inverse in frequency domain and return to real space.
        """

        return self.ifft(multiplier * self.fft(value), s=self.grid.shape).real

    def _apply_in_svd_space(
        self, value: torch.Tensor, multiplier: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the inverse in SVD space and return to real space.
        """
        assert self.implementation == "matmul"
        out = value
        for vectors in self.eigenvectors:
            out = torch.tensordot(out, vectors, dims=(0, 0))
        out *= multiplier
        for vectors in self.eigenvectors:
            out = torch.tensordot(out, vectors, dims=(0, 1))
        return out
