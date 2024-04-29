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

from typing import Tuple, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.fft as fft
from . import grids
from tqdm.auto import tqdm

TQDM_ITERS = 100

Array = torch.Tensor
Grid = grids.Grid


class ImplicitExplicitODE(nn.Module):
    """Describes a set of ODEs with implicit & explicit terms.

    The equation is given by:

      $\partial x/ \partial t = explicit_terms(x) + implicit_terms(x)$

    `explicit_terms(x)` includes terms that should use explicit time-stepping and
    `implicit_terms(x)` includes terms that should be modeled implicitly.

    Typically the explicit terms are non-linear and the implicit terms are linear.
    This simplifies solves but isn't strictly necessary.
    """

    def explicit_terms(self, *, x):
        """Evaluates explicit terms in the ODE."""
        raise NotImplementedError

    def implicit_terms(self, *, x):
        """Evaluates implicit terms in the ODE."""
        raise NotImplementedError

    def implicit_solve(
        self,
        *,
        x: Array,
        step_size: float,
    ):
        """Solves `y - step_size * implicit_terms(y) = x` for y."""
        raise NotImplementedError


def low_storage_runge_kutta_crank_nicolson(
    u: torch.Tensor,
    dt: float,
    params: Dict,
    equation: ImplicitExplicitODE,
) -> Array:
    """
    ported from jax functional programming to be tensor2tensor
    Time stepping via "low-storage" Runge-Kutta and Crank-Nicolson steps.

    These scheme are second order accurate for the implicit terms, but potentially
    higher order accurate for the explicit terms. This seems to be a favorable
    tradeoff when the explicit terms dominate, e.g., for modeling turbulent
    fluids.

    Per Canuto: "[these methods] have been widely used for the time-discretization
    in applications of spectral methods."

    Args:
      alphas: alpha coefficients.
      betas: beta coefficients.
      gammas: gamma coefficients.
      equation.F: explicit terms (convection, rhs, drag).
      equation.G: implicit terms (diffusion).
      equation.implicit_solve: implicit solver, when evaluates at an input (B, n, n), outputs (B, n, n).
      dt: time step.

    Input: w^{t_i} (B, n, n)
    Returns: w^{t_{i+1}} (B, n, n)

    Reference:
      Canuto, C., Yousuff Hussaini, M., Quarteroni, A. & Zang, T. A.
      Spectral Methods: Evolution to Complex Geometries and Applications to
      Fluid Dynamics. (Springer Berlin Heidelberg, 2007).
      https://doi.org/10.1007/978-3-540-30728-0 (Appendix D.3)
    """
    dt = dt
    alphas = params["alphas"]
    betas = params["betas"]
    gammas = params["gammas"]
    F = equation.explicit_terms
    G = equation.implicit_terms
    G_inv = equation.implicit_solve

    if len(alphas) - 1 != len(betas) != len(gammas):
        raise ValueError("number of RK coefficients does not match")

    h = 0
    for k in range(len(betas)):
        h = F(u) + betas[k] * h
        mu = 0.5 * dt * (alphas[k + 1] - alphas[k])
        u = G_inv(u + gammas[k] * dt * h + mu * G(u), mu)
    return u


def crank_nicolson_rk4(
    u: Array,
    dt: float,
    equation: ImplicitExplicitODE,
) -> Array:
    """Time stepping via Crank-Nicolson and RK4 ("Carpenter-Kennedy")."""
    params = dict(
        alphas=[
            0,
            0.1496590219993,
            0.3704009573644,
            0.6222557631345,
            0.9582821306748,
            1,
        ],
        betas=[0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257],
        gammas=[
            0.1496590219993,
            0.3792103129999,
            0.8229550293869,
            0.6994504559488,
            0.1530572479681,
        ],
    )
    return low_storage_runge_kutta_crank_nicolson(
        u,
        dt=dt,
        params=params,
        equation=equation,
    )


class NavierStokes2DSpectral(nn.Module):
    """Breaks the Navier-Stokes equation into implicit and explicit parts.

    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.

    Attributes:
      viscosity: strength of the diffusion term
      grid: underlying grid of the process
      smooth: smooth the advection term using the 2/3-rule.
      forcing_fn: forcing function, if None then no forcing is used.
      drag: strength of the drag. Set to zero for no drag.
    """

    def __init__(
        self,
        viscosity: float,
        grid: Grid,
        drag: float = 0.0,
        smooth: bool = True,
        forcing_fn: Optional[Callable] = None,
        solver: Optional[Callable] = crank_nicolson_rk4,
    ):
        super().__init__()
        self.viscosity = viscosity
        self.grid = grid
        self.drag = drag
        self.smooth = smooth
        self.forcing_fn = forcing_fn
        self.solver = solver
        self._initialize()

    def _initialize(self):
        kx, ky = self.grid.rfft_mesh()
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        laplace = (torch.pi * 2j) ** 2 * (self.kx**2 + self.ky**2)
        self.register_buffer("laplace", laplace)
        filter_ = self.brick_wall_filter_2d(self.grid)
        linear_term = self.viscosity * self.laplace - self.drag
        self.register_buffer("linear_term", linear_term)
        self.register_buffer("filter", filter_)

    @staticmethod
    def brick_wall_filter_2d(grid: Grid):
        """Implements the 2/3 rule."""
        n, _ = grid.shape
        filter_ = torch.zeros((n, n // 2 + 1))
        filter_[: int(2 / 3 * n) // 2, : int(2 / 3 * (n // 2 + 1))] = 1
        filter_[-int(2 / 3 * n) // 2 :, : int(2 / 3 * (n // 2 + 1))] = 1
        return filter_

    @staticmethod
    def spectral_curl_2d(vhat, rfft_mesh):
        r"""
        Computes the 2D curl in the Fourier basis.
        det [d_x d_y \\ u v]
        """
        uhat, vhat = vhat
        kx, ky = rfft_mesh
        return 2j * torch.pi * (vhat * kx - uhat * ky)

    @staticmethod
    def spectral_grad_2d(vhat, rfft_mesh):
        kx, ky = rfft_mesh
        return 2j * torch.pi * kx * vhat, 2j * torch.pi * ky * vhat

    @staticmethod
    def vorticity_to_velocity(
        grid: Grid, w_hat: Array, rfft_mesh: Optional[Tuple[Array, Array]] = None
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
        two_pi_i = 2 * torch.pi * 1j
        laplace = two_pi_i**2 * (abs(kx) ** 2 + abs(ky) ** 2)
        laplace[0, 0] = 1
        psi_hat = -1 / laplace * w_hat
        vxhat = two_pi_i * ky * psi_hat
        vyhat = -two_pi_i * kx * psi_hat
        return vxhat, vyhat
    
    def residual(self,
        vort_hat: Array,
        vort_t_hat: Array,
    ):
        residual = vort_t_hat -  self.explicit_terms(vort_hat) - self.viscosity *  self.implicit_terms(vort_hat)
        return residual

    def _explicit_terms(self, vort_hat):
        vxhat, vyhat = self.vorticity_to_velocity(self.grid, vort_hat, (self.kx, self.ky))
        vx, vy = fft.irfft2(vxhat), fft.irfft2(vyhat)

        grad_x_hat = 2j * torch.pi * self.kx * vort_hat
        grad_y_hat = 2j * torch.pi * self.ky * vort_hat
        grad_x, grad_y = fft.irfft2(grad_x_hat), fft.irfft2(grad_y_hat)

        advection = -(grad_x * vx + grad_y * vy)
        advection_hat = fft.rfft2(advection)

        if self.smooth:
            advection_hat *= self.filter

        terms = advection_hat

        if self.forcing_fn is not None:
            fx, fy = self.forcing_fn(self.grid, (vx, vy))
            fx_hat, fy_hat = fft.rfft2(fx.data), fft.rfft2(fy.data)
            terms += self.spectral_curl_2d((fx_hat, fy_hat), (self.kx, self.ky))

        return terms

    def explicit_terms(self, vort_hat):
        return self._explicit_terms(vort_hat)

    def implicit_terms(self, vort_hat):
        return self.linear_term * vort_hat

    def implicit_solve(self, vort_hat, dt):
        return 1 / (1 - dt * self.linear_term) * vort_hat

    def get_trajectory(
        self,
        w0: Array,
        dt: float,
        T: float,
        record_every_steps=1,
        pbar=False,
        pbar_desc="",
        require_grad=False,
    ):
        """
        vorticity stacked in the time dimension
        """
        w_all = []
        v_all = []
        dwdt_all = []
        res_all = []
        w = w0
        time_steps = int(T / dt)
        update_iters = time_steps // TQDM_ITERS
        with tqdm(total=time_steps) as pbar:
            for t in range(time_steps):
                w, dwdt = self.forward(w, dt=dt)
                w.requires_grad_(require_grad)
                dwdt.requires_grad_(require_grad)

                if t % update_iters == 0:
                    pbar.set_description(pbar_desc)
                    pbar.update(update_iters)

                if t % record_every_steps == 0:
                    w_ = w.detach().clone()
                    dwdt_ = dwdt.detach().clone()
                    v = self.vorticity_to_velocity(self.grid, w_)
                    res = self.residual(w_, dwdt_)

                    v = torch.stack(v, dim=0)
                    w_all.append(w_)
                    v_all.append(v)
                    dwdt_all.append(dwdt_)
                    res_all.append(res)

        result = {
            var_name: torch.stack(var, dim=0)
            for var_name, var in zip(
                ["vorticity", "velocity", "vort_t", "residual"], [w_all, v_all, dwdt_all, res_all]
            )
        }
        return result

    def step(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, vort_hat, dt):
        vort_hat_new = self.solver(vort_hat, dt, self)
        dvortdt_hat = 1 / dt * (vort_hat_new - vort_hat)
        return vort_hat_new, dvortdt_hat
