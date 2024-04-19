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

from typing import Tuple, Optional
import torch
from . import grids

Array = torch.Tensor
Grid = grids.Grid
GridArray = grids.GridArray

def kolmogorov_forcing(
    grid: Grid,
    v: Tuple[Array, Array],
    scale: float = 1,
    k: int = 2,
    swap_xy: bool = False,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
    device: Optional[torch.device] = None,
) -> Array:
    """Returns the Kolmogorov forcing function for turbulence in 2D."""
    if offsets is None:
        offsets = grid.cell_faces
    if grid.device is None and device is not None:
        grid.device = device
    if swap_xy:
        x = grid.mesh(offsets[1])[0]
        v = GridArray(scale * torch.sin(k * x), offsets[1], grid)
        u = GridArray(torch.zeros_like(v.data), (1, 1 / 2), grid)
        f = (u, v)
    else:
        y = grid.mesh(offsets[0])[1]
        u = GridArray(scale * torch.sin(k * y), offsets[0], grid)
        v = GridArray(torch.zeros_like(u.data), (1 / 2, 1), grid)
        f = (u, v)
    return f