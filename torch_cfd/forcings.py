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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from . import grids

Array = torch.Tensor
Grid = grids.Grid
GridArray = grids.GridArray


def forcing_eval(eval_func):
    def wrapper(
        cls, grid: Grid, field: Optional[Union[Tuple[Array, Array], Array]]
    ) -> Array:
        return eval_func(grid, field)

    return wrapper


class ForcingFn(nn.Module):
    """
    A meta class for forcing functions

    Args:
    vorticity: whether the forcing function is a vorticity forcing
    """

    def __init__(
        self,
        grid: Grid,
        scale: float = 1,
        k: int = 1,
        diam: float = 1.0,
        swap_xy: bool = False,
        vorticity: bool = False,
        offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        self.scale = scale
        self.k = k
        self.diam = diam
        self.swap_xy = swap_xy
        self.vorticity = vorticity
        self.offsets = grid.cell_faces if offsets is None else offsets
        self.device = grid.device if device is None else device

    @forcing_eval
    def velocity_eval(grid: Grid, velocity: Optional[Tuple[Array, Array]]) -> Array:
        raise NotImplementedError

    @forcing_eval
    def vorticity_eval(grid: Grid, vorticity: Optional[Array]) -> Array:
        raise NotImplementedError

    def forward(
        self,
        grid: Optional[Grid],
        velocity: Optional[Tuple[Array, Array]] = None,
        vorticity: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        if not self.vorticity:
            return self.velocity_eval(grid, velocity)
        else:
            return self.vorticity_eval(grid, vorticity)


class KolmogorovForcing(ForcingFn):
    """
    The Kolmogorov forcing function used in
    Sets up the flow that is used in Kochkov et al. [1].
    which is based on Boffetta et al. [2].

    Note in the port: this forcing belongs a larger class
    of isotropic turbulence. See [3].

    References:
    [1] Machine learning-accelerated computational fluid dynamics. Dmitrii
    Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan
    Hoyer Proceedings of the National Academy of Sciences May 2021, 118 (21)
    e2101784118; DOI: 10.1073/pnas.2101784118.
    https://doi.org/10.1073/pnas.2101784118

    [2] Boffetta, Guido, and Robert E. Ecke. "Two-dimensional turbulence."
    Annual review of fluid mechanics 44 (2012): 427-451.
    https://doi.org/10.1146/annurev-fluid-120710-101240

    [3] McWilliams, J. C. (1984). "The emergence of isolated coherent vortices
    in turbulent flow". Journal of Fluid Mechanics, 146, 21-43.
    """

    def __init__(
        self,
        diam=2 * torch.pi,
        offsets=((0, 0), (0, 0)),
        vorticity=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            diam=diam,
            offsets=offsets,
            vorticity=vorticity,
            **kwargs,
        )

    def velocity_eval(
        self,
        grid: Optional[Grid],
        velocity: Optional[Tuple[Array, Array]] = None,
    ) -> Tuple[Array, Array]:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            v = GridArray(
                self.scale * torch.sin(self.k * domain_factor * x), offsets[1], grid
            )
            u = GridArray(torch.zeros_like(v.data), (1, 1 / 2), grid)
            f = (u, v)
        else:
            y = grid.mesh(offsets[0])[1]
            u = GridArray(
                self.scale * torch.sin(self.k * domain_factor * y), offsets[0], grid
            )
            v = GridArray(torch.zeros_like(u.data), (1 / 2, 1), grid)
            f = (u, v)
        return f

    def vorticity_eval(
        self,
        grid: Optional[Grid],
        vorticity: Optional[Array] = None,
    ) -> Array:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            w = GridArray(
                -self.scale
                * self.k
                * domain_factor
                * torch.cos(self.k * domain_factor * x),
                offsets[1],
                grid,
            )
        else:
            y = grid.mesh(offsets[0])[1]
            w = GridArray(
                -self.scale
                * self.k
                * domain_factor
                * torch.cos(self.k * domain_factor * y),
                offsets[0],
                grid,
            )
        return w


def scalar_potential(potential_func):
    def wrapper(cls, x: Array, y: Array, s: float, k: float) -> Array:
        return potential_func(x, y, s, k)

    return wrapper


class SimpleSolenoidalForcing(ForcingFn):
    """
    A simple solenoidal (rotating, divergence free) forcing function template.
    The template forcing is F = (-psi, psi) such that

    Args:
    grid: grid on which to simulate the flow
    scale: a in the equation above, amplitude of the forcing
    k: k in the equation above, wavenumber of the forcing
    """

    def __init__(
        self,
        scale=1,
        diam=1.0,
        k=1.0,
        offsets=((0, 0), (0, 0)),
        vorticity=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            scale=scale,
            diam=diam,
            k=k,
            offsets=offsets,
            vorticity=vorticity,
            **kwargs,
        )

    @scalar_potential
    def potential(*args, **kwargs) -> Array:
        raise NotImplementedError

    @scalar_potential
    def vort_potential(*args, **kwargs) -> Array:
        raise NotImplementedError

    def velocity_eval(
        self,
        grid: Optional[Grid],
        velocity: Optional[Tuple[Array, Array]] = None,
    ) -> Tuple[Array, Array]:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam
        k = self.k * domain_factor
        scale = 0.5 * self.scale / (2 * torch.pi) / self.k

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            y = grid.mesh(offsets[0])[1]
            rot = self.potential(x, y, scale, k)
            v = GridArray(rot, offsets[1], grid)
            u = GridArray(-rot, (1, 1 / 2), grid)
            f = (u, v)
        else:
            x = grid.mesh(offsets[0])[0]
            y = grid.mesh(offsets[1])[1]
            rot = self.potential(x, y, scale, k)
            u = GridArray(rot, offsets[0], grid)
            v = GridArray(-rot, (1 / 2, 1), grid)
            f = (u, v)
        return f

    def vorticity_eval(
        self,
        grid: Optional[Grid],
        vorticity: Optional[Array] = None,
    ) -> Array:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam
        k = self.k * domain_factor
        scale = self.scale

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            y = grid.mesh(offsets[0])[1]
            return self.vort_potential(x, y, scale, k)
        else:
            x = grid.mesh(offsets[0])[0]
            y = grid.mesh(offsets[1])[1]
            return self.vort_potential(x, y, scale, k)


class SinCosForcing(SimpleSolenoidalForcing):
    """
    The solenoidal (divergence free) forcing function used in [4].

    Note: in the vorticity-streamfunction formulation, the forcing
    is actually the curl of the velocity field, which
    is a*(sin(2*pi*k*(x+y)) + cos(2*pi*k*(x+y)))
    a=0.1, k=1 in [4]

    References:
    [4] Li, Zongyi, et al. "Fourier Neural Operator for
    Parametric Partial Differential Equations."
    ICLR. 2020.

    Args:
    grid: grid on which to simulate the flow
    scale: a in the equation above, amplitude of the forcing
    k: k in the equation above, wavenumber of the forcing
    """

    def __init__(
        self,
        scale=0.1,
        diam=1.0,
        k=1.0,
        offsets=((0, 0), (0, 0)),
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            scale=scale,
            diam=diam,
            k=k,
            offsets=offsets,
            **kwargs,
        )

    @scalar_potential
    def potential(x: Array, y: Array, s: float, k: float) -> Array:
        return s * (torch.sin(k * (x + y)) - torch.cos(k * (x + y)))

    @scalar_potential
    def vort_potential(x: Array, y: Array, s: float, k: float) -> Array:
        return s * (torch.cos(k * (x + y)) + torch.sin(k * (x + y)))
