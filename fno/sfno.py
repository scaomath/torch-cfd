# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .base import *
from torch_cfd.spectral import (
    fft_expand_dims,
    fft_mesh_2d,
    spectral_div_2d,
    spectral_grad_2d,
    spectral_laplacian_2d,
)


class SpaceTimePositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    a modified sinosoidal PE inspired from the Transformers
    input is (batch, *, nx, ny, t)
    output is (batch, C, nx, ny, t)
    1 comes from the input
    time_exponential_scale comes from the a priori estimate of Navier-Stokes Eqs
    the random feature basis are added with a pointwise conv3d
    """

    def __init__(
        self,
        modes_x: int = 16,
        modes_y: int = 16,
        modes_t: int = 5,
        num_channels: int = 20,
        input_shape: Union[List, Tuple] = (64, 64, 10),
        spatial_random_feats: bool = False,
        max_time_steps: int = 100,
        time_exponential_scale: float = 1e-2,
        **kwargs,
    ):
        super().__init__()
        assert num_channels % 2 == 0 and num_channels > 3
        self.num_channels = num_channels  # the Euclidean coords
        self.max_time_steps = max_time_steps
        self.time_exponential_scale = time_exponential_scale
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t

        self._pe = self._pe_expanded if spatial_random_feats else self._pe
        self._pe(*input_shape)
        if spatial_random_feats:
            in_chan = modes_x * modes_y * modes_t + 3  # 3 is spatial temporal coords
            self.proj = nn.Conv3d(in_chan, num_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def _pe_expanded(self, *shape):
        nx, ny, nt = shape
        gridx = torch.linspace(0, 1, nx)
        gridy = torch.linspace(0, 1, ny)
        gridt = torch.linspace(0, 1, self.max_time_steps + 1)[1 : nt + 1]
        gridx, gridy, gridt = torch.meshgrid(gridx, gridy, gridt, indexing="ij")
        pe = [gridx, gridy, gridt]

        for i in range(1, self.modes_x + 1):
            basis_x = torch.sin if i % 2 == 0 else torch.cos
            for j in range(1, self.modes_y + 1):
                basis_y = torch.sin if j % 2 == 0 else torch.cos
                for k in range(1, self.modes_t + 1):
                    basis_t = torch.sin if k % 2 == 0 else torch.cos
                    basis = (
                        1
                        / (i * j * k)
                        * torch.exp(self.time_exponential_scale * gridt)
                        * basis_x(torch.pi * i * gridx)
                        * basis_y(torch.pi * j * gridy)
                        * basis_t(torch.pi * k * gridt)
                    )
                    pe.append(basis)
        pe = torch.stack(pe).unsqueeze(0)  # (1, num_channels+3, nx, ny, nt)
        self.pe = pe

    def _pe(self, *shape):
        nx, ny, nt = shape
        gridx = torch.linspace(0, 1, nx)
        gridy = torch.linspace(0, 1, ny)
        gridt = torch.linspace(0, 1, self.max_time_steps + 1)[1 : nt + 1]
        gridx, gridy, _gridt = torch.meshgrid(gridx, gridy, gridt, indexing="ij")
        pe = [gridx, gridy, _gridt]
        for k in range(self.num_channels - 3):
            basis = torch.sin if k % 2 == 0 else torch.cos
            _gridt = torch.exp(self.time_exponential_scale * gridt) * basis(
                torch.pi * (k + 1) * gridt
            )
            _gridt = _gridt.reshape(1, 1, nt).repeat(nx, ny, 1)
            pe.append(_gridt)
        pe = torch.stack(pe).unsqueeze(0)  # (1, num_channels+3, nx, ny, nt)
        self.pe = pe

    def forward(self, v: torch.Tensor):
        if self.pe is None or self.pe.shape[-3:] != v.shape[-3:]:
            *_, nx, ny, nt = v.size()  # (batch, 1, x, y, t)
            self._pe(nx, ny, nt)
        pe = self.pe.to(v.dtype).to(v.device)
        return v + self.proj(pe)


class HelmholtzProjection(nn.Module):
    def __init__(
        self,
        n_grid: int = 64,
        diam: float = 2 * torch.pi,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        """
        Perform Helmholtz decomposition in the frequency domain
        to project any vector field to divergence free
        
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
        >>> print(torch.linalg.norm(div_w)) 
        The result should be less than batch_size*(interval machine epsilon)
        interval machine epsilon is approx 
        1e-6 for float32 and 1e-12 for float64
        """
        self.n_grid = n_grid
        self.diam = diam
        self._update_fft_mesh(n_grid, diam, dtype)

    def _update_fft_mesh(self, n, diam=None, dtype=torch.float32):
        diam = diam if diam is not None else self.diam
        kx, ky = fft_mesh_2d(n, diam)
        lap = spectral_laplacian_2d(fft_mesh=(kx, ky))
        self.register_buffer("lap", lap.to(dtype))
        self.register_buffer("kx", kx.to(dtype))
        self.register_buffer("ky", ky.to(dtype))

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
        width: int,
        modes_x: int,
        modes_y: int,
        modes_t: int,
        latent_steps: int = 10,
        norm: str = "backward",
        activation: ActivationType = "GELU",
        beta: float = 0.1,
        spatial_random_feats: bool = False,
        channel_expansion: int = 4,
        nonlinear: bool = True,
        **kwargs,
    ) -> None:
        """
        the latent steps: n_t at hidden layers
        """
        super().__init__()
        if modes_t % 2 != 0:
            pe_modes_t = modes_t - 1
        else:
            pe_modes_t = modes_t

        self.pe = SpaceTimePositionalEncoding(
            modes_x // 2,
            modes_y // 2,
            pe_modes_t // 2,
            num_channels=width,
            time_exponential_scale=beta,
            spatial_random_feats=spatial_random_feats,
        )

        in_channels = self.pe.num_channels
        self.norm = LayerNormnd(in_channels)
        self.proj = nn.Conv3d(in_channels, width, kernel_size=1)

        conv_size = [width, width, modes_x, modes_y, modes_t]
        self.sconv = SpectralConvT(
            *conv_size,
            out_steps=latent_steps,
            norm=norm,
            bias=False,
        )
        self.latent_steps = latent_steps
        if nonlinear:
            self.activation = getattr(nn, activation)()
            self.mlp = PointwiseFFN(width, width, channel_expansion * width, activation)
        else:
            self.activation = nn.Identity()
            self.mlp = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, v):
        """
        input: (b, 1, x, y, t)
        output: (b, H, x, y, t_latent)
        the t_latent should be <= the input time steps
        """
        assert self.latent_steps <= v.size(-1)
        for b in [self.pe, self.norm, self.proj]:
            v = b(v)  # (b, 1, x, y, t_in) -> (b, H, x, y, t_latent)
        w = self.mlp(self.sconv(v))  # (b, H, x, y, t_latent)
        v = self.activation(v[..., -1:] + w)
        return v


class OutConv(nn.Module):
    def __init__(
        self,
        modes_x: int,
        modes_y: int,
        modes_t: int,
        delta: float = 0.1,
        out_dim: int = 1,
        diam: float = 1,
        n_grid: int = 64,
        out_steps: int = None,
        spatial_padding: int = 0,
        temporal_padding: bool = True,
        norm: str = "backward",
        **kwargs,
    ) -> None:
        super().__init__()
        """
        from latent steps to output steps
        diam and n_grid are only needed for Helmholtz decomposition
        """
        self.size = [out_dim, out_dim, modes_x, modes_y, modes_t]
        if out_dim == 2:
            postprocess = HelmholtzProjection(n_grid=n_grid, diam=diam)
        elif out_dim == 1:
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
        input v: (b, d, x, y, t_latent)
        d = out_dim = 1 or 2
        input v_res: (b, x, y, t_in) or (b, 2, x, y, t_in)
        after channel reduction and padding length
        v: (b, x, y, t_latent) or (b, 2, x, y, t_latent)
        v_res input (b, x, y, t_out) if out_steps is None
        """
        v_res = repeat(v_res, "b x y t -> b d x y t", d=v.size(1))
        v = torch.cat([v_res[..., -1:], v], dim=-1)
        if self.spatial_padding > 0:
            sp = self.spatial_padding
            padding_kws = {"pad": (0, 0, sp, sp, sp, sp), "mode": "constant"}
            v = F.pad(v, **padding_kws)

        v = self.conv(v, out_steps=out_steps + 1)
        # if dim reduction is 2, then this v is postprocessed to be divergence free
        # the squeeze(1) would do nothing in the case of velocity

        if self.spatial_padding > 0:
            v = v[..., sp:-sp, sp:-sp, :]

        v = v_res[..., -1:] + v[..., -out_steps:]
        return v.squeeze(1)


class SpectralConvS(SpectralConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        modes_t: int,
        dim: int = 3,
        bias: bool = False,
        delta: float = 1,
        norm="backward",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=(modes_x, modes_y, modes_t),
            dim=dim,
            bias=bias,
            norm=norm,
        )

        """
        Spacetime Fourier layer. 
        FFT, linear transform, and Inverse FFT.  
        focusing on space
        see base.py for the boilerplate  
        """
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        self.delta = delta

    def spectral_conv(self, vh, kx: int, ky: int, kt: int):
        """
        kx, ky, kt: the number of modes in the input
        matmul the weights with the input
        user defined dimensions
        in the space focused conv
        assert nt <= modes_t not explicitly checked
        """
        bsz = vh.size(0)
        sizes = (bsz, self.out_channels, kx, ky, kt)
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
                out[..., sx, sy, st] = self.complex_matmul(
                    vh[..., sx, sy, st], torch.view_as_complex(self.weight[ix + 2 * iy])
                )
                # (b, c_i, x, y, t), (c_i, c_o, x, y, t)  -> (b, c_o, x, y, t)
                if self.bias:
                    _bias = self.bias[ix + 2 * iy][None, None, ...]
                    out[..., sx, sy, st] += self.delta * torch.view_as_complex(_bias)
        return out

    def forward(self, v, **kwargs):
        return super().forward(v, **kwargs)


class SpectralConvT(SpectralConvS):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        modes_t: int,
        delta: float = 1e-1,
        out_steps: int = None,
        norm: str = "backward",
        bias: bool = True,
        temporal_padding: bool = False,
        postprocess: nn.Module = nn.Identity(),
        **kwargs,
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


class SFNO(FNOBase):
    def __init__(
        self,
        modes_x: int,
        modes_y: int,
        modes_t: int,
        width: int,
        out_dim: int = 1,
        beta: float = -1e-2,
        delta: float = 1e-1,
        num_spectral_layers: int = 4,
        fft_norm: str = "backward",
        activation: ActivationType = "ReLU",
        spatial_padding: int = 0,
        temporal_padding: bool = True,
        channel_expansion: int = 4,
        spatial_random_feats: bool = False,
        lift_activation: bool = True,
        latent_steps: int = 10,
        output_steps: int = None,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            num_spectral_layers=num_spectral_layers,
            fft_norm=fft_norm,
            activation=activation,
            spatial_padding=spatial_padding,
            channel_expansion=channel_expansion,
            spatial_random_feats=spatial_random_feats,
            lift_activation=lift_activation,
            debug=debug,
            **kwargs,
        )

        """
        The overall network reimplemented to model (2+1)D spatiotemporal PDEs of 
        a scalar field/vector fields of NSE-like equations.

        Major architectural differences:

        1. New lifting operator
            - new PE: since the treatment of grid is different from FNO official code, which give my autograd trouble, new PE is similar to the one used in the Transformers, the time dimension's PE is according to the NSE. The PE occupies the extra channels.
            - new LayerNorm3d: instead of normalizing the input/output pointwisely when preparing the data like the original FNO did, this makes an input-steps agnostic normalization. Note that the global normalization by mean/std of (n, n, n_t)-shaped tensor in the original FNO3d prevents to predict arbitrary time steps.
            - the channel lifting now works pretty much like the depth-wise conv but uses the globally spectral as FNO does. Since there is no need to treat the time steps as channels now it can accept arbitrary time steps in the input.
        2. new out projection: it maps the latent time steps to a given output time steps using FFT's natural super-resolution.
            - output arbitrary steps.
            - aliasing error handled by zero padding
            - the spectral bias works like a source term in the Fredholm integral operator.
        3. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.

        Hyper-params:
        - mode_x, mode_y, mode_t: the number of Fourier modes in the x, y, t dimensions
        - width: the number of channels in the latent space   
        - num_spectral_layers: the number of spectral conv layers, the first layer is in the lifting operator  
        - spatial_padding: the padding size in the spatial dimensions
        - temporal_padding: whether to pad the temporal dimension, by default it is True, recommended to keep it True to avoid aliasing error
        - out_steps: the number of output time steps, if None, it will be set to the temporal dimension of the input
        - activation: the activation function, users provide string that directly pulls from nn. default: ReLU
        - lift_activation: whether to use activation in the lifting operator
        - spatial_random_feats: whether to use spatial random features in the lifting operator
        - channel_expansion: the number of channels in the MLP, default: 128

        Grid information:
        - diam: the diameter of the domain, only used in the Helmholtz decomposition
        - n_grid: the grid size of the training data, only needed for building the fft mesh for the Helmholtz decompostion, in the forward pass the size is arbitrary (if different from the n_grid, Helmholtz layer will re-build the fft mesh, which introduces a tiny overhead)
        
        Several key hyper-params that is different from FNO3d:
        - beta: the exponential scaling factor for the time PE, ideally it should match the a priori estimate the energy of the NSE
        - delta: the strength of the final skip-connection.
        - latent steps: the number of time steps in the hidden layers, this is independent of the input/output steps; chosing it >= 3/2 of input length is similar to zero padding of FFT to avoid aliasing due to non-periodic in the temporal dimension
        - dim_reduction: 1 for scalar field such as vorticity, 2 for vector field such as velocity

        input: w(x, y, t) in the shape of (bsz, x, y, t)
        output: w(x, y, t) in the shape of (bsz, x, y, t)
        """

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        self.width = width

        assert num_spectral_layers > 1
        num_spectral_layers -= 1
        # the lifting operator has already an sconv

        self._set_spectral_layers(
            num_spectral_layers,
            [modes_x, modes_y, modes_t],
            width,
            spectral_conv=SpectralConvS,
            mlp=PointwiseFFN,
            linear=nn.Conv3d,
            activation=activation,
            channel_expansion=channel_expansion,
        )

        self.lifting_operator = LiftingOperator(
            width,
            modes_x,
            modes_y,
            modes_t,
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
            activation=activation,
            spatial_random_feats=spatial_random_feats,
            channel_expansion=channel_expansion,
            nonlinear=lift_activation,
        )

        self.output_operator = OutConv(
            modes_x,
            modes_y,
            modes_t,
            out_dim=out_dim,
            delta=delta,
            out_steps=output_steps,
            spatial_padding=spatial_padding,
            temporal_padding=temporal_padding,
            norm=fft_norm,
        )

        self.reduction = nn.Conv3d(width, 1, kernel_size=1)
        self.out_steps = output_steps
        self.debug = debug

    @property
    def set_lifting_operator(self):
        return self.lifting_operator

    @property
    def set_output_operator(self):
        return self.output_operator

    def forward(self, v, out_steps=None):
        """
        if out_steps is None, it will try to use self.out_steps
        if self.out_steps is None, it will use the temporal dimension of the input x
        """
        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)
        v_res = v  # save skip connection
        v = rearrange(v, "b x y t -> b 1 x y t")
        v = self.lifting_operator(v)  # [b, 1, x, y, T] -> [b, H, x, y, T]

        for conv, mlp, w, nonlinear in zip(
            self.spectral_conv, self.mlp, self.w, self.activations
        ):
            x1 = conv(v)  # (b,H,x,y,t)
            x1 = mlp(x1)  # conv3d (b, H, x, y, t) -> (b, H, x, y, t)
            x2 = w(v)
            v = x1 + x2
            v = nonlinear(v)

        v = self.reduction(v)  # (b, H, x, y, t) -> (b, 1, x, y, t)
        v = self.output_operator(
            v, v_res, out_steps=out_steps
        )  # (b,1,x,y,t) -> (b,x,y,t)
        return v
