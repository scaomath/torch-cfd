"""
minor modification from the original FNO3d code:
https://github.com/neuraloperator/neuraloperator/blob/master/fourier_3d.py
"""

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, inp, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, activation=True):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        for layer in [self.mlp1, self.activation, self.mlp2]:
            x = layer(x)
        return x


class FNO3d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        dim=3,
        input_channel=10,
        num_spectral_layers=4,
        last_activation=False,
        padding=0,
        extra_mlp=True,
        channel_expansion=128,
        debug=False,
    ):
        super().__init__()

        """
        The overall network reimplemented.
        
        It contains n (=4 by default) layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        

        last_activation: if True, then the last spectral layer activation is gelu, otherwise, it's linear
        channel_expansion: the channel expansion of the MLP after the last spectral layer

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.input_channel = input_channel
        self.padding = padding  # pad the domain if input is non-periodic
        self.extra_mlp = extra_mlp
        self.channel_expansion = channel_expansion

        self.p = nn.Linear(
            input_channel + dim, self.width
        )  # input channel is 13: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.spectral_conv = nn.ModuleList(
            [
                SpectralConv3d(
                    self.width, self.width, self.modes1, self.modes2, self.modes3
                )
                for _ in range(num_spectral_layers)
            ]
        )

        self.mlp = nn.ModuleList(
            [
                MLP(self.width, self.width, self.width)
                for _ in range(num_spectral_layers)
            ]
        )

        self.w = nn.ModuleList(
            [nn.Conv3d(self.width, self.width, 1) for _ in range(num_spectral_layers)]
        )

        self.activation = nn.ModuleList(
            [nn.GELU() for _ in range(num_spectral_layers - 1)]
        )
        self.activation.append(nn.GELU() if last_activation else nn.Identity())

        self.q = MLP(
            self.width, 1, self.channel_expansion, activation=last_activation
        )  # output channel is 1: u(x, y)
        self.debug = debug

    def forward(self, x):
        """
        the treatment of grid is different from FNO official code
        which give my autograd trouble
        """
        # bsz = x.size(0)
        # grid_size = self.grid.size()
        # grid = self.grid[None, ...].expand(bsz, *grid_size).to(x.fdevice)
        # x = torch.cat((x, grid), dim=-1)

        x = self.p(x)  # (b,x,y,t,13) -> (b,x,y,t,c)

        x = x.permute(0, 4, 1, 2, 3)  # (b,x,y,t,c) -> (b,c,x,y,t)

        x = F.pad(
            x,
            [0, 0, self.padding, self.padding, self.padding, self.padding],
            mode="circular",
        )  # pad the domain if input is non-periodic

        for conv, mlp, w, nonlinear in zip(
            self.spectral_conv, self.mlp, self.w, self.activation
        ):
            x1 = conv(x)  # (b,C,x,y,t)
            x1 = mlp(x1)  # conv3d (N, C_{in}, D, H, W) -> (N, C_{out}, D, H, W)
            x2 = w(x)
            x = x1 + x2
            x = nonlinear(x)

        if self.padding != 0:
            x = x[..., self.padding : -self.padding, self.padding : -self.padding, :]
        x = self.q(x)  # (b,C,x,y,t) -> (b,1,x,y,t)

        x = x.permute(0, 2, 3, 4, 1)  # (b,1,x,y,t) -> (b,x,y,t,1)
        return x

    @staticmethod
    def get_grid(grid_size):
        size_x, size_y, size_t = grid_size
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(size_x, 1, 1, 1).repeat([1, size_y, size_t, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([size_x, 1, size_t, 1])
        gridt = torch.linspace(0, 1, size_t)
        gridt = gridt.reshape(1, 1, size_t, 1).repeat([size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridt), dim=-1)


if __name__ == "__main__":
    modes = 8
    modes_t = 11
    width = 20
    model = FNO3d(modes, modes, modes_t, width, extra_mlp=True)
    """
    torchinfo has not resolve the complex number problem
    """
    for layer in model.children():
        if hasattr(layer, "out_features"):
            print(layer.out_features)
    try:
        from torchinfo import summary

        summary(model, input_size=(5, 128, 128, 40, 13))
        print("\n" * 3)
        model_orig = FNO3d(modes, modes, modes_t, width)
        summary(
            model_orig, input_size=(5, 64, 64, 40, 13)
        )  # number of parameters is 6563417 which
    except ImportError as e:
        print(e)
