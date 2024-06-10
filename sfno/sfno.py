# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from __future__ import annotations

from functools import partial

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_uniform_
from data import *


class LayerNorm3d(nn.GroupNorm):
    """
    a wrapper for GroupNorm
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    input and output shapes: (bsz, C, *)
    * can be (H, W) or (H, W, T)
    """

    def __init__(
        self, num_channels, eps=1e-06, elementwise_affine=True, device=None, dtype=None
    ) -> None:
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            eps=eps,
            affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor):
        return super().forward(x)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, activation=True):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        for block in [self.mlp1, self.activation, self.mlp2]:
            x = block(x)
        return x


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    standard sinosoidal PE inspired from the Transformers
    input is (batch, 1, x, y, t)
    output is (batch, C, x, y, t)
    1 comes from the input channel
    time_exponential_scale comes from the a priori estimate of Navier-Stokes Eqs
    """

    def __init__(
        self, num_channels=10, max_time_steps=100, time_exponential_scale=1e-2
    ):
        super().__init__()
        self.num_channels = num_channels
        assert num_channels % 2 == 0
        self.max_time_steps = max_time_steps
        self.time_exponential_scale = time_exponential_scale
        self.pe = None

    def forward(self, x):
        if self.pe is None or self.pe.shape[-3:] != x.shape[-3:]:
            *_, nx, ny, nt = x.size()  # (batch, 1, x, y, t)
            gridx = torch.linspace(0, 1, nx)
            gridy = torch.linspace(0, 1, ny)
            gridt = torch.linspace(0, 1, self.max_time_steps + 1)[1 : nt + 1]
            gridx, gridy, _gridt = torch.meshgrid(gridx, gridy, gridt, indexing="ij")
            pe = [gridx, gridy, _gridt]
            for k in range(self.num_channels):
                basis = torch.sin if k % 2 == 0 else torch.cos
                _gridt = torch.exp(self.time_exponential_scale * gridt) * basis(
                    torch.pi * (k + 1) * gridt
                )
                _gridt = repeat(_gridt, "t -> x y t", x=nx, y=ny)
                pe.append(_gridt)
            pe = torch.stack(pe)
            pe = rearrange(
                pe, "c x y t -> 1 c x y t"
            )  # (1, num_channels+3, nx, ny, nt)
            pe = pe.to(x.dtype).to(x.device)
            self.pe = pe
        return x + self.pe


class Helmholtz(nn.Module):
    def __init__(
        self,
        n_grid: int = 64,
        diam: float = 2 * torch.pi,
    ):
        super().__init__()
        """
        Perform Helmholtz decomposition in the frequency domain
        
        Example usage:

        >>> n = 512
        >>> kx, ky = fft_mesh_2d(n, 1)
        >>> lap = spectral_laplacian_2d(fft_mesh=(kx, ky))
        >>> bsz, T = 4, 6
        >>> for t in range(T):
                vhat_ = [fft.fft2(torch.randn(bsz, n, n))/(5e-1+lap) for _ in range(2)]
                vhat_ = torch.stack(vhat_, dim=1)
                vhat.append(vhat_)
            vhat = torch.stack(vhat, dim=-1)
        >>> proj = Helmholtz(n_grid=n)
        >>> w_hat = proj(vhat)
        
        Now w_hat is divergence free
        >>> div_w_hat = proj.div(w_hat, (kx, ky))
        >>> div_w = fft.irfft2(div_w_hat, s=(n, n), dim=(1,2)).real
        >>> print(torch.linalg.norm(div_w)) # should be less than machine epsilon
        """
        self.n_grid = n_grid
        self.diam = diam
        self._update_fft_mesh(n_grid, diam)

    def _update_fft_mesh(self, n, diam=None):
        diam = diam if diam is not None else self.diam
        kx, ky = fft_mesh_2d(n, diam)
        lap = spectral_laplacian_2d(fft_mesh=(kx, ky))
        self.register_buffer("lap", lap)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    @staticmethod
    def div(uhat, fft_mesh):
        """
        uhat: (b, 2, nx, ny, nt//2+1)
        """
        kx, ky = fft_expand_dims(fft_mesh, uhat.size(0))
        return spectral_div_2d([uhat[:, 0], uhat[:, 1]], (kx, ky))

    @staticmethod
    def grad(uhat, fft_mesh):
        """
        uhat: (b, nx, ny, nt//2+1)
        output: (b, 2, nx, ny, nt//2+1)
        """
        kx, ky = fft_expand_dims(fft_mesh, uhat.size(0))
        graduhat = spectral_grad_2d(uhat, (kx, ky))
        return torch.stack(graduhat, dim=1)

    def forward(self, uhat):
        """
        u: (b, 2, nx, ny, nt//2+1)
        """
        bsz, _, nx, ny, nt = uhat.shape
        fft_mesh = (self.kx, self.ky)
        if nx != self.n_grid:
            # this is for evaluation
            self._update_fft_mesh(nx)
        div_u = self.div(uhat, fft_mesh)
        grad_div_u = self.grad(div_u, fft_mesh)
        lap = repeat(self.lap, "x y -> b 2 x y t", b=bsz, t=nt)
        w_hat = uhat - grad_div_u / lap
        return w_hat


class LiftingOperator(nn.Module):
    def __init__(
        self,
        width,
        modes_x,
        modes_y,
        modes_t,
        latent_steps=10,
        norm="backward",
        beta=0.1,
    ) -> None:
        """
        the latent steps: n_t at hidden layers
        """
        super().__init__()
        self.pe = PositionalEncoding(width, time_exponential_scale=beta)
        self.norm = LayerNorm3d(width + 3)
        self.proj = nn.Conv3d(width + 3, width, kernel_size=1)
        self.sconv = SpectralConvT(
            width,
            width,
            modes_x,
            modes_y,
            modes_t,
            out_steps=latent_steps,
            norm=norm,
            bias=False,
        )

    def forward(self, x):
        for block in [self.pe, self.norm, self.proj, self.sconv]:
            x = block(x)
        return x


class OutConv(nn.Module):
    def __init__(
        self,
        modes_x,
        modes_y,
        modes_t,
        delta=0.1,
        dim_reduction: int = 1,
        diam: float = 1,
        n_grid=64,
        out_steps=None,
        spatial_padding: int = 0,
        temporal_padding: bool = True,
        norm="backward",
    ) -> None:
        super().__init__()
        """
        from latent steps to output steps
        """
        self.size = [dim_reduction, dim_reduction, modes_x, modes_y, modes_t]
        if dim_reduction == 2:
            postprocess = Helmholtz(n_grid=n_grid, diam=diam)
        elif dim_reduction == 1:
            postprocess = nn.Identity()
        self.conv = SpectralConvT(
            *self.size,
            norm=norm,
            delta=delta,
            out_steps=out_steps,
            bias=True,
            temporal_padding=temporal_padding,
            postprocess=postprocess,
        )
        self.n_grid = n_grid
        self.norm = norm
        self.delta = delta
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding

    def forward(self, v, v_res, out_steps: int, **kwargs):
        """
        input v: (b, c, x, y, t_latent)
        input v_res: (b, x, y, t_in) or (b, 2, x, y, t_in)
        after channel reduction and padding length
        v: (b, x, y, t_latent) or (b, 2, x, y, t_latent)
        v_res input (b, x, y, t_out) if out_steps is None
        """
        v_res = rearrange(v_res, "b x y t -> b 1 x y t")
        v = torch.cat([v_res[..., -2:], v], dim=-1)
        if self.spatial_padding > 0:
            sp = self.spatial_padding
            padding_kws = {"pad": (0, 0, sp, sp, sp, sp), "mode": "constant"}
            v = F.pad(v, **padding_kws)

        v = self.conv(v, out_steps=out_steps + 2)
        # if dim reduction is 2, then this v is postprocessed to be divergence free
        # the squeeze(1) would do nothing in the case of velocity

        if self.spatial_padding > 0:
            v = v[..., sp:-sp, sp:-sp, :]

        v = v_res[..., -1:] + v[..., -out_steps:]
        return v.squeeze(1)


class SpectralConvS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_x,
        modes_y,
        modes_t,
        bias=False,
        delta: float = 1,
        norm="backward",
    ) -> None:
        super().__init__()

        """
        Spacetime Fourier layer. 
        FFT, linear transform, and Inverse FFT.  
        focusing on space
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        self.bias = bias
        size = [in_channels, out_channels, modes_x, modes_y, modes_t, 2]
        gain = 0.5 / (in_channels * out_channels)
        self._initialize_weights(size, gain)

        self.fft = partial(fft.rfftn, dim=(-3, -2, -1), norm=norm)
        self.ifft = partial(fft.irfftn, dim=(-3, -2, -1), norm=norm)
        self.delta = delta

    def _initialize_weights(self, size, gain=1e-4):
        self.weight = nn.ParameterList(
            [nn.Parameter(gain * torch.rand(*size)) for _ in range(4)]
        )
        if self.bias:
            self.bias = nn.ParameterList(
                [
                    nn.Parameter(
                        gain
                        * torch.zeros(
                            *size[2:],
                        )
                    )
                    for _ in range(4)
                ]
            )

    def _reset_parameters(self, gain=1e-6):
        for name, param in self.named_parameters():
            if "bias" in name:
                constant_(param, 0.0)
            else:
                xavier_uniform_(param, gain)

    @staticmethod
    def complex_matmul_3d(inp, weights):
        # (b, c_i, x,y,t), (c_i, c_o, x,y,t)  -> (b, c_o, x,y,t)
        weights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)

    def spectral_conv(self, vh, nx: int, ny: int, nt: int):
        """
        matmul the weights with the input
        user defined dimensions
        in the space focused conv
        assert nt <= modes_t not explicitly checked
        """
        bsz = vh.size(0)
        sizes = (bsz, self.out_channels, nx, ny, nt)
        out = torch.zeros(
            *sizes,
            dtype=vh.dtype,
            device=vh.device,
        )
        slice_x = [slice(0, self.modes_x), slice(-self.modes_x, None)]
        slice_y = [slice(0, self.modes_y), slice(-self.modes_y, None)]
        st = slice(0, self.modes_t)
        for ix, sx in enumerate(slice_x):
            for iy, sy in enumerate(slice_y):
                out[..., sx, sy, st] = self.complex_matmul_3d(
                    vh[..., sx, sy, st], self.weight[ix + 2 * iy]
                )
                if self.bias:
                    _bias = self.bias[ix + 2 * iy][None, None, ...]
                    out[..., sx, sy, st] += self.delta * torch.view_as_complex(_bias)
        return out

    def forward(self, v):
        *_, nx, ny, nt = v.size()
        v_hat = self.fft(v)
        v_hat = self.spectral_conv(v_hat, nx, ny, nt // 2 + 1)
        v = self.ifft(v_hat, s=(nx, ny, nt))
        return v


class SpectralConvT(SpectralConvS):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_x,
        modes_y,
        modes_t,
        delta=1e-1,
        n_grid: int = 64,
        out_steps: int = None,
        norm="backward",
        bias=True,
        temporal_padding=False,
        postprocess: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            modes_x,
            modes_y,
            modes_t,
            norm=norm,
            delta=delta,
            bias=bias,
        )
        self.n_grid = n_grid
        self.out_steps = out_steps
        self.temporal_padding = temporal_padding
        self.postprocess = postprocess

        """
        Spacetime Fourier layer used in SFNO. 
        FFT, linear transform, and Inverse FFT.
        arbitrary temporal steps focusing on time  
        """

    def forward(self, v, out_steps: int = None):
        """
        when temporal padding is applied
        the outsteps must be given
        not checked explicitly
        """
        nt = v.size(-1)
        if self.temporal_padding:
            # this is for out conv
            t_pad = v.size(-1)
            v = F.pad(v, (t_pad, 0))
        else:
            t_pad = 0
        *_, nx, ny, ntp = v.size()  # (b, c, nx, ny, nt)
        v_hat = self.fft(v)
        v_hat = self.spectral_conv(v_hat, nx, ny, ntp // 2 + 1)

        if out_steps is None and self.out_steps is not None:
            out_steps = self.out_steps  # latent_steps
        v_hat = self.postprocess(v_hat)

        v = self.ifft(v_hat, s=(nx, ny, out_steps + t_pad))
        if self.temporal_padding:
            v = v[..., -out_steps:]
        return v


class SFNO(nn.Module):
    def __init__(
        self,
        modes_x,
        modes_y,
        modes_t,
        width,
        beta=-1e-2,
        delta=1e-1,
        diam: float = 1,
        n_grid: int = 64,
        dim_reduction: int = 1,
        num_spectral_layers: int = 4,
        fft_norm="backward",
        activation: str = "GELU",
        spatial_padding: int = 0,
        temporal_padding: bool = True,
        channel_expansion: int = 128,
        latent_steps: int = 10,
        output_steps: int = None,
        debug=False,
    ):
        super().__init__()

        """
        The overall network reimplemented for scalar field of NSE-like equations

        It contains num_spectral_layers (=4 by default) layers of the Fourier layer.

        Major architectural differences:

        1. New lifting operator
            - new PE: since the treatment of grid is different from FNO official code, which give my autograd trouble, new PE is similar to the one used in the Transformers, the time dimension's PE is according to the NSE. The PE occupies the extra channels.
            - new LayerNorm3d: instead of normalizing the input/output pointwisely when preparing the data like the original FNO did. Tthe normalization prevents to predict arbitrary time steps.
            - the channel lifting now works pretty much like the depth-wise conv but uses the globally spectral as FNO does. Since there is no need to treat the time steps as channels now it can accept arbitrary time steps in the input.
        2. new out projection: it maps the latent time steps to a given output time steps using FFT's natural super-resolution.
            - output arbitrary steps.
            - aliasing error handled by zero padding
            - the spectral bias works like a source term in the Fredholm integral operator.
        3. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.

        Several key hyper-params that is different from FNO3d:
        
        - beta: the exponential scaling factor for the time PE, ideally it should match the a priori estimate the energy of the NSE
        - delta: the strength of the final skip-connection.
        - latent steps: the number of time steps in the hidden layers, this is independent of the input/output steps; chosing it >= 3/2 of input length is similar to zero padding of FFT to avoid aliasing due to non-periodic in the temporal dimension
        - n_grid: the grid size of the training data, only needed for building the fft mesh for the Helmholtz decompostion, in the forward pass the size is arbitrary (if different from the n_grid, Helmholtz layer will re-build the fft mesh, which introduces a tiny overhead)
        - dim_reduction: 1 for vorticity, 2 for velocity

        input: w(x, y, t) in the shape of (bsz, x, y, t)
        output: w(x, y, t) in the shape of (bsz, x, y, t)
        """

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        self.width = width
        self.spatial_padding = (
            spatial_padding  # pad the domain if input is non-periodic in space
        )
        assert num_spectral_layers > 1
        num_spectral_layers -= 1  # the lifting operator has already an sconv

        self.p = LiftingOperator(
            width,
            modes_x,
            modes_y,
            modes_t,
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
        )
        self.spectral_conv = nn.ModuleList(
            [
                SpectralConvS(width, width, modes_x, modes_y, modes_t)
                for _ in range(num_spectral_layers)
            ]
        )

        self.mlp = nn.ModuleList(
            [MLP(width, width, channel_expansion) for _ in range(num_spectral_layers)]
        )

        self.w = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(num_spectral_layers)]
        )

        act_func = getattr(nn, activation)
        self.activations = nn.ModuleList(
            [act_func() for _ in range(num_spectral_layers)]
        )

        self.r = nn.Conv3d(width, dim_reduction, kernel_size=1)
        self.q = OutConv(
            modes_x,
            modes_y,
            modes_t,
            delta=delta,
            diam=diam,
            n_grid=n_grid,
            out_steps=output_steps,
            dim_reduction=dim_reduction,
            spatial_padding=spatial_padding,
            temporal_padding=temporal_padding,
            norm=fft_norm,
        )
        self.out_steps = output_steps
        self.debug = debug

    latent_tensors = {}

    def add_latent_hook(self, layer_name: str):
        def _get_latent_tensors(name):
            def hook(model, input, output):
                self.latent_tensors[name] = output.detach()

            return hook

        module = getattr(self, layer_name)

        if hasattr(module, "__iter__"):
            for k, b in enumerate(module):
                b.register_forward_hook(_get_latent_tensors(f"{layer_name}_{k}"))
        else:
            module.register_forward_hook(_get_latent_tensors(layer_name))

    def double(self):
        for param in self.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.float64)
            elif param.dtype == torch.complex64:
                param.data = param.data.to(torch.complex128)
        return self

    def forward(self, v, out_steps=None):
        """
        if out_steps is None, it will try to use self.out_steps
        if self.out_steps is None, it will use the temporal dimension of the input x
        """
        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)
        v_res = v  # save skip connection
        v = rearrange(v, "b x y t -> b 1 x y t")
        v = self.p(v)  # [b, 1, n, n, T] -> [b, H, n, n, T]

        for conv, mlp, w, nonlinear in zip(
            self.spectral_conv, self.mlp, self.w, self.activations
        ):
            x1 = conv(v)  # (b,C,x,y,t)
            x1 = mlp(x1)  # conv3d (N, C_{in}, D, H, W) -> (N, C_{out}, D, H, W)
            x2 = w(v)
            v = x1 + x2
            v = nonlinear(v)

        v = self.r(v)  # (b, c, x, y, t) -> (b, 1, x, y, t)
        v = self.q(v, v_res, out_steps=out_steps)  # (b,1,x,y,t) -> (b,x,y,t)
        return v


if __name__ == "__main__":
    modes = 8
    modes_t = 2
    width = 10
    bsz = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sizes = [(n, n, n_t) for (n, n_t) in zip([64, 128, 256], [5, 10, 20])]
    model = SFNO(modes, modes, modes_t, width).to(device)
    x = torch.randn(bsz, *sizes[0]).to(device)
    _ = model(x)

    try:
        from torchinfo import summary

        """
        torchinfo has not resolve the complex number problem
        """
        summary(model, input_size=(bsz, *sizes[-1]))
    except:
        from utils import get_num_params

        print(get_num_params(model))
    del model

    print("\n" * 3)
    for k, size in enumerate(sizes[:-1]):
        torch.cuda.empty_cache()
        model = SFNO(modes, modes, modes_t, width).to(device)
        model.add_latent_hook("activations")
        x = torch.randn(bsz, *size).to(device)
        pred, *_ = model(x)
        print(f"\n\noutput shape: {list(pred.size())}")
        for k, v in model.latent_tensors.items():
            print(k, list(v.shape))
        del model
