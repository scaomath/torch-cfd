# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    1 comes from the
    time_exponential_scale comes from the a priori estimate of Navier-Stokes Eqs
    """

    def __init__(
        self, num_channels=10, max_time_steps=100, time_exponential_scale=1e-1
    ):
        super().__init__()
        self.num_channels = num_channels
        assert num_channels % 2 == 0
        self.max_time_steps = max_time_steps
        self.time_exponential_scale = time_exponential_scale
        self.register_buffer("pe", None)

    def forward(self, x):
        if self.pe is None or self.pe.shape[-3:] != x.shape[-3:]:
            *_, nx, ny, nt = x.size()  # (batch, 1, x, y, t)
            gridx = torch.linspace(0, 1, nx)
            gridy = torch.linspace(0, 1, ny)
            gridt = torch.linspace(0, 1, self.max_time_steps)[:nt]
            gridx, gridy, _gridt = torch.meshgrid(gridx, gridy, gridt)
            pe = [gridx, gridy, _gridt]
            for k in range(self.num_channels):
                basis = torch.sin if k % 2 == 0 else torch.sin
                _gridt = torch.exp(self.time_exponential_scale * gridt) * basis(
                    torch.pi * (k + 1) * gridt
                )
                _gridt = _gridt.reshape(1, 1, nt).repeat(nx, ny, 1)
                pe.append(_gridt)
            pe = torch.stack(pe).unsqueeze(0)  # (1, num_channels+3, nx, ny, nt)
            pe = pe.to(x.dtype).to(x.device)
            self.pe = pe
        return x + self.pe

class LiftingOperator(nn.Module):
    def __init__(
        self, width, modes_x, modes_y, modes_t, latent_steps=10, norm="forward"
    ) -> None:
        """
        the latent steps: n_t at hidden layers
        """
        super().__init__()
        self.pe = PositionalEncoding(width)
        self.norm = LayerNorm3d(width + 3)
        self.proj = nn.Conv3d(width + 3, width, kernel_size=1)
        self.sconv = SpectralConvS(
            width,
            width,
            modes_x,
            modes_y,
            modes_t,
            out_steps=latent_steps,
            norm=norm,
        )

    def forward(self, x):
        for block in [self.pe, self.norm, self.proj, self.sconv]:
            x = block(x)
        return x

class OutProjection(nn.Module):
    def __init__(
        self, width, modes_x, modes_y, modes_t, output_steps=None, norm="forward"
    ) -> None:
        super().__init__()
        """
        from latent steps to output steps
        """
        self.out_steps = output_steps
        self.channel_reduction = nn.Conv3d(width, out_channels=1, kernel_size=1)
        self.conv = SpectralConvS(
            1, 1, modes_x, modes_y, modes_t, norm=norm, out_steps=output_steps
        )
        self.out_steps = output_steps

    def forward(self, x, x_res, out_steps: int):
        """
        input x: (b, c, x, y, t, t_latent)
        after channel reduction and padding length
        x: (b, x, y, t_latent)
        x_res input (b, x, y, t_out) if out_steps is None
        """
        x_out = self.channel_reduction(x) # (b, c, x, y, t) -> (b, 1, x, y, t)
        x_res = x_res[..., -1:]  # (b, x, y, t) -> (b, x, y, 1)
        x = x_res + self.conv(x_out, out_steps=out_steps).squeeze()
        return x, x_out.squeeze()


class SpectralConvS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_x,
        modes_y,
        modes_t,
        out_steps=None,
        norm="forward",
    ) -> None:
        super().__init__()

        """
        Spacetime Fourier layer. It does FFT, linear transform, and Inverse FFT.  
        Arbitrary output steps  
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        size = [in_channels, out_channels, modes_x, modes_y, modes_t]
        self.out_steps = out_steps
        self.scale = 1 / (in_channels * out_channels)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    self.scale
                    * torch.rand(
                        *size,
                        dtype=torch.cfloat,
                    )
                )
                for _ in range(4)
            ]
        )
        self.rfftn = partial(fft.rfftn, dim=(-3, -2, -1), norm=norm)
        self.irfftn = partial(fft.irfftn, dim=(-3, -2, -1), norm=norm)

    @staticmethod
    def compl_mul3d(inp, weights):
        # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)

    def forward(self, x, out_steps=None):
        bsz, _, nx, ny, nt = x.size()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            nx,
            ny,
            nt // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        slice_x = [slice(0, self.modes_x), slice(-self.modes_x, None)]
        slice_y = [slice(0, self.modes_y), slice(-self.modes_y, None)]
        st = slice(0, self.modes_t)

        for ix, sx in enumerate(slice_x):
            for iy, sy in enumerate(slice_y):
                out_ft[..., sx, sy, st] = self.compl_mul3d(
                    x_ft[..., sx, sy, st], self.weights[ix + 2 * iy]
                )

        # Return to physical space
        if self.out_steps is None and out_steps is not None:
            # this is for output projection
            nt = out_steps
        elif self.out_steps is not None and out_steps is None:
            # this is for lifting operator
            nt = self.out_steps
        x = self.irfftn(out_ft, s=(nx, ny, nt))
        return x


class SFNO(nn.Module):
    def __init__(
        self,
        modes_x,
        modes_y,
        modes_t,
        width,
        num_spectral_layers: int = 4,
        fft_norm="forward",
        activation:str='GELU',
        padding:int=0,
        channel_expansion:int=128,
        latent_steps:int=10,
        output_steps:int=None,
        debug=False,
    ):
        super().__init__()

        """
        The overall network reimplemented for scalar field of NSE-like equations

        It contains n (=4 by default) layers of the Fourier layer.
        1. New lifting operator, new out projection, now input and output arbitrary steps
        2. new PE since the treatment of grid is different from FNO official code
        which give my autograd trouble, new PE is similar to the one used in the Transformers
        3. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .

        input: w(x, y, t) in the shape of (bsz, x, y, t)
        output: w(x, y, t) in the shape of (bsz, x, y, t)
        """

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        self.width = width
        self.padding = padding  # pad the domain if input is non-periodic
        assert num_spectral_layers > 1
        num_spectral_layers -= 1  # the lifting operator has already an sconv

        self.p = LiftingOperator(
            width, modes_x, modes_y, modes_t, latent_steps=latent_steps, norm=fft_norm
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
        self.activations = nn.ModuleList([act_func() for _ in range(num_spectral_layers)])
        self.q = OutProjection(
            width,
            modes_x,
            modes_y,
            modes_t,
            output_steps=output_steps,
            norm=fft_norm,
        )
        self.out_steps = output_steps
        self.debug = debug

    latent_tensors = {}
    def add_latent_hook(self, layer_name:str=None):
        def _get_latent_tensors(name):
            def hook(model, input, output):
                self.latent_tensors[name] = output.detach()
            return hook
        
        layer_name = "activations" if layer_name is None else layer_name
        blocks = getattr(self, layer_name)

        for k, block in enumerate(blocks):
            block.register_forward_hook(_get_latent_tensors(f"{layer_name}_{k}"))

    def forward(self, x, out_steps=None):
        """

        """
        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else x.size(-1)
        x_res = x  # save skip connection
        x = self.p(x.unsqueeze(1))  # [b, 1, n, n, T] -> [b, H, n, n, T]

        x = F.pad(
            x,
            [0, 0, self.padding, self.padding, self.padding, self.padding],
            mode="circular",
        )

        for conv, mlp, w, nonlinear in zip(
            self.spectral_conv, self.mlp, self.w, self.activations
        ):
            x1 = conv(x)  # (b,C,x,y,t)
            x1 = mlp(x1)  # conv3d (N, C_{in}, D, H, W) -> (N, C_{out}, D, H, W)
            x2 = w(x)
            x = x1 + x2
            x = nonlinear(x)

        if self.padding != 0:
            x = x[..., self.padding : -self.padding, self.padding : -self.padding, :]
        x, x_out = self.q(x, x_res, out_steps=out_steps)  # (b,C,x,y,t) -> (b,1,x,y,t)

        return x, x_out


if __name__ == "__main__":
    modes = 32
    modes_t = 5
    width = 10
    bsz = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sizes = [(n, n, n_t) for (n, n_t) in zip([64, 128, 256] , [10, 20, 20])]
    model = SFNO(modes, modes, modes_t, width).to(device)
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
    for k, size in enumerate(sizes):
        torch.cuda.empty_cache()
        model = SFNO(modes, modes, modes_t, width).to(device)
        model.add_latent_hook()
        x = torch.randn(bsz, *size).to(device)
        pred, pred_latent = model(x)
        print(f"\n\noutput shape: {list(pred.size())}")
        print(f"latent shape: {list(pred_latent.size())}")
        for k, v in model.latent_tensors.items():
            print(k, list(v.shape))
        del model



