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

from typing import Optional, Tuple

import torch
import torch.fft as fft

from torch_cfd import grids
from einops import repeat

Grid = grids.Grid


def fft_mesh_2d(n, diam, device=None):
    kx, ky = [fft.fftfreq(n, d=diam / n) for _ in range(2)]
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")
    return kx.to(device), ky.to(device)


def fft_expand_dims(fft_mesh, batch_size):
    kx, ky = fft_mesh
    kx, ky = [repeat(z, "x y -> b x y 1", b=batch_size) for z in [kx, ky]]
    return kx, ky


def spectral_laplacian_2d(fft_mesh, device=None):
    kx, ky = fft_mesh
    lap = -4 * (torch.pi**2) * (abs(kx) ** 2 + abs(ky) ** 2)  
    # (2 * torch.pi * 1j)**2
    lap[..., 0, 0] = 1
    return lap.to(device)


def spectral_curl_2d(vhat, rfft_mesh):
    r"""
    Computes the 2D curl in the Fourier basis.
    det [d_x d_y \\ u v]
    """
    uhat, vhat = vhat
    kx, ky = rfft_mesh
    return 2j * torch.pi * (vhat * kx - uhat * ky)


def spectral_div_2d(vhat, rfft_mesh):
    r"""
    Computes the 2D divergence in the Fourier basis.
    """
    uhat, vhat = vhat
    kx, ky = rfft_mesh
    return 2j * torch.pi * (uhat * kx + vhat * ky)


def spectral_grad_2d(vhat, rfft_mesh):
    kx, ky = rfft_mesh
    return 2j * torch.pi * kx * vhat, 2j * torch.pi * ky * vhat


def spectral_rot_2d(vhat, rfft_mesh):
    vgradx, vgrady = spectral_grad_2d(vhat, rfft_mesh)
    return vgrady, -vgradx


def brick_wall_filter_2d(grid: Grid):
    """Implements the 2/3 rule."""
    n, _ = grid.shape
    filter_ = torch.zeros((n, n // 2 + 1))
    filter_[: int(2 / 3 * n) // 2, : int(2 / 3 * (n // 2 + 1))] = 1
    filter_[-int(2 / 3 * n) // 2 :, : int(2 / 3 * (n // 2 + 1))] = 1
    return filter_


def vorticity_to_velocity(
    grid: Grid, w_hat: torch.Tensor, rfft_mesh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
):
    """Constructs a function for converting vorticity to velocity, both in Fourier domain.

    Solves for the stream function and then uses the stream function to compute
    the velocity. This is the standard approach. A quick sketch can be found in
    [1].

    Args:
        grid: the grid underlying the vorticity field.

    Returns:
        A function that takes a vorticity (rfftn) and returns a velocity vector
        field.

    Reference:
        [1] Z. Yin, H.J.H. Clercx, D.C. Montgomery, An easily implemented task-based
        parallel scheme for the Fourier pseudospectral solver applied to 2D
        Navier-Stokes turbulence, Computers & Fluids, Volume 33, Issue 4, 2004,
        Pages 509-520, ISSN 0045-7930,
        https://doi.org/10.1016/j.compfluid.2003.06.003.
    """
    kx, ky = rfft_mesh if rfft_mesh is not None else grid.rfft_mesh()
    assert kx.shape[-2:] == w_hat.shape[-2:]
    laplace = spectral_laplacian_2d((kx, ky))
    psi_hat = -1 / laplace * w_hat
    u_hat, v_hat = spectral_rot_2d(psi_hat, (kx, ky))
    return (u_hat, v_hat), psi_hat