# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from typing import Tuple

import torch
import torch.fft as fft
import torch.nn as nn

try:
    from .sfno import OutConv, SpectralConvT
except:
    from sfno import OutConv, SpectralConvT
from data.solvers import *
from einops import rearrange, repeat


class OutConvFT(OutConv):
    def __init__(
        self,
        modes_x,
        modes_y,
        modes_t,
        batch_size: int = 1,
        diam=1.0,
        n_grid: int = 256,
        out_steps=None,
        spatial_padding: int = 0,
        temporal_padding: bool = True,
        norm="backward",
        finetune=True,
        dealias=True,
        delta=5e-2,
        visc=1e-3,
        dt=1e-6,  # marching step for the solver
        bdf_weight=(0, 1),
        dtype=torch.float64,
        debug=False,
    ) -> None:
        super().__init__(
            modes_x=modes_x,
            modes_y=modes_y,
            modes_t=modes_t,
            delta=delta,
            n_grid=n_grid,
            norm=norm,
            out_steps=out_steps,
            spatial_padding=spatial_padding,
            temporal_padding=temporal_padding,
        )
        """
        from latent steps to output steps
        finetuning
        n_grid is only needed for building the spectral "mesh"
        """
        self.finetune = finetune
        self.out_steps = out_steps
        self.batch_size = batch_size
        self.dealias = dealias
        self.diam = diam
        self.dtype = dtype
        self.visc = visc
        self.dt = dt
        self.bdf_weight = bdf_weight
        self._initialize_fftmesh()

    def _initialize_fftmesh(self):

        kx, ky = fft_mesh_2d(self.n_grid, self.diam)
        kmax = self.n_grid // 2
        kx, ky = [repeat(z, "x y -> b x y", b=self.batch_size) for z in [kx, ky]]
        kx = kx[..., : kmax + 1]
        ky = ky[..., : kmax + 1]

        lap = spectral_laplacian_2d(fft_mesh=(kx, ky))

        dealias_filter = (
            torch.logical_and(
                ky.abs() <= (2.0 / 3.0) * kmax,
                kx.abs() <= (2.0 / 3.0) * kmax,
            ).to(self.dtype)
            if self.dealias
            else torch.tensor(True)
        )
        self.register_buffer("lap", lap)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("dealias_filter", dealias_filter)

    def _update_spectral_conv_weights(
        self,
        modes_x,
        modes_y,
        modes_t,
        device: torch.device = None,
        model: nn.Module = None,
        debug=False,
    ):
        """
        update the last spectral conv layer for fine-tuning
        modes_t <= out_steps // 2 + 1 but not explicitly checked
        """
        # self.train()
        model = self if model is None else model
        old_conv = model.conv
        size = [1, 1, modes_x, modes_y, modes_t]
        conv = SpectralConvT(
            *size,
            bias=True,
            delta=self.delta,
            temporal_padding=self.temporal_padding,
            out_steps=self.out_steps,
        ).to(device)
        conv._reset_parameters()

        if not debug:
            mx_ = old_conv.modes_x
            my_ = old_conv.modes_y
            mt_ = old_conv.modes_t

            slice_x = [slice(0, mx_), slice(-mx_, None)]
            slice_y = [slice(0, my_), slice(-my_, None)]
            st = slice(0, mt_)
            for ix, sx in enumerate(slice_x):
                for iy, sy in enumerate(slice_y):
                    old_weights = old_conv.weight[ix + 2 * iy].data
                    old_bias = old_conv.bias[ix + 2 * iy].data
                    conv.weight[ix + 2 * iy].data[..., sx, sy, st, :] = old_weights
                    conv.bias[ix + 2 * iy].data[..., sx, sy, st, :] = old_bias

        self.conv = conv
        self.mode_x = modes_x
        self.mode_y = modes_y
        self.mode_t = modes_t

    @staticmethod
    def get_temporal_derivative(w_h, f_h, dt, weight=(0, 1), **kwargs):
        """
        v: (b, x, y, t)
        kwargs needed
        rfftmesh: (kx, ky)
        laplacian: -4 * (torch.pi**2) * (kx**2 + ky**2)
        dealias_filter
        dealias optional
        """
        w_t = []
        w = []
        for dt in [-dt, dt]:
            w_, w_t_, *_ = imex_crank_nicolson_step(
                w_h,
                f_h,
                delta_t=dt,
                **kwargs,
            )
            w_t.append(w_t_)
            w.append(w_)
        w_t = weight[0] * w_t[0] + weight[1] * w_t[1]
        w = weight[0] * w[0] + weight[1] * w[1]
        return w, w_t

    def _fine_tune(self, w, f, **solver_kws):
        bsz, *s, nt = w.shape  # s = (x, y, t)
        ft_kws = {"s": s, "norm": self.norm}
        dt = self.dt
        w = rearrange(w, "b x y t -> b t x y")
        if f is None:  # for testing
            f = torch.zeros_like(w).to(w.device)
        w_h, f_h = [fft.rfftn(v, **ft_kws) for v in [w, f]]  # f: (b, x, y)

        w_h, w_h_t = self.get_temporal_derivative(w_h, f_h, dt, **solver_kws)

        res_h = update_residual(
            w_h,
            w_h_t,
            f_h,
            **solver_kws,
        )
        w, w_t, res = [fft.irfftn(v, **ft_kws).real for v in [w_h, w_h_t, res_h]]
        w, w_t, res = [rearrange(v, "b t x y -> b x y t") for v in [w, w_t, res]]

        return dict(w=w, w_t=w_t, residual=res)

    def forward(self, v, v_res, f=None, out_steps: int = None, original=False):
        """
        v_latent: (b, 1, x, y, t)
        w_inp: (b, x, y, t)
        f: (b, x, y)
        """
        solver_kws = {
            "visc": self.visc,
            "laplacian": self.lap,
            "dealias_filter": self.dealias_filter,
            "dealias": self.dealias,
            "rfftmesh": (self.kx, self.ky),
            "diam": self.diam,
            "weight": self.bdf_weight,
        }

        v = super().forward(v, v_res, out_steps=out_steps)

        if not self.finetune or original:
            return v
        else:
            return self._fine_tune(v, f, **solver_kws)


if __name__ == "__main__":
    modes = 128
    modes_t = 6
    qft = OutConvFT(modes, modes, modes_t, n_grid=256, delta=1)

    for n_t in [10, 50]:
        v_latent = torch.randn(1, 1, 256, 256, n_t)
        w_res = torch.randn(1, 256, 256, n_t)
        f = torch.randn(1, 256, 256)
        out = qft(v_latent, w_res, f, out_steps=n_t)

        for k in ["w", "w_t", "residual"]:
            print(f"{k:<10} | shape: {list(out[k].shape)}")
