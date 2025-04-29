# The MIT License (MIT)
# Copyright © 2025 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from __future__ import annotations

from abc import abstractmethod

from copy import deepcopy

from functools import partial
from typing import List, Union, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


conv_dict = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}

ACTIVATION_FUNCTIONS = [
    'CELU', 'ELU', 'GELU', 'GLU', 'Hardtanh', 'Hardshrink', 'Hardsigmoid', 
    'Hardswish', 'LeakyReLU', 'LogSigmoid', 'MultiheadAttention', 'PReLU', 
    'ReLU', 'ReLU6', 'RReLU', 'SELU', 'SiLU', 'Sigmoid', 'SoftPlus', 
    'Softmax', 'Softmax2d', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink',
    'Threshold', 'Mish'
]

# Type hint for activation functions
ActivationType = Union[str]


class LayerNormnd(nn.GroupNorm):
    """
    a wrapper for GroupNorm used as LayerNorm
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    input and output shapes: (bsz, C, *)
        * can be (H, W) or (H, W, T)
    Note: by default the layernorm is applied to the last dimension
    """

    def __init__(
        self, num_channels, eps=1e-07, elementwise_affine=True, device=None, dtype=None
    ):
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            eps=eps,
            affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, v: torch.Tensor):
        return super().forward(v)


class PointwiseFFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        activation: ActivationType = "ReLU",
        dim: int = 3,
    ):
        super().__init__()
        """
        Pointwisely-applied 2-layer FFN with a channel expansion
        """

        if dim not in conv_dict:
            raise ValueError(f"Unsupported dimension: {dim}, expected 1, 2, or 3")

        Conv = conv_dict[dim]
        self.linear1 = Conv(in_channels, mid_channels, 1)
        self.linear2 = Conv(mid_channels, out_channels, 1)
        self.activation = getattr(nn, activation)()

    def forward(self, v: torch.Tensor):
        for b in [self.linear1, self.activation, self.linear2]:
            v = b(v)
        return v


class SpectralConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: List[int],
        dim: int,
        bias: bool = False,
        norm: str = "backward",
    ) -> None:
        super().__init__()

        """
        Spacetime Fourier layer template
        FFT, linear transform, and Inverse FFT.  
        focusing on space
        modes: the number of Fourier modes in each dimension
        modes's length needs to be same as the dimension
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.bias = bias
        assert len(modes) == dim, "modes should match the dimension"
        size = [in_channels, out_channels, *modes, 2]
        gain = 0.5 / (in_channels * out_channels)
        dims = tuple(range(-self.dim, 0))
        self.fft = partial(fft.rfftn, dim=dims, norm=norm)
        self.ifft = partial(fft.irfftn, dim=dims, norm=norm)
        self._initialize_weights(size, gain)

    def _initialize_weights(self, size, gain=1e-4):
        """
        # of weight groups = 4 = 2*(ndim - 1)
        """
        self.weight = nn.ParameterList(
            [
                nn.Parameter(gain * torch.rand(*size))
                for _ in range(2 * (self.dim - 1))
            ]  # 2*(ndim - 1)
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
                    for _ in range(2 * (self.dim - 1))  # 2*(ndim - 1)
                ]
            )

    def _reset_parameters(self, gain=1e-6):
        for name, param in self.named_parameters():
            if "bias" in name:
                constant_(param, 0.0)
            else:
                xavier_uniform_(param, gain)

    @staticmethod
    def complex_matmul(x, w, **kwargs):
        """
        Implement this method in subclass to return complex matmul function
        this is a general implmentation of arbitrary dimension
        (b, c_i, *mesh_dims), (c_i, c_o, *mesh_dims)  -> (b, c_o, *mesh_dims)
        for pure einsum benchmark, ellipsis version runs about 30% slower, 
        however, when being implemented in FNO, the performance difference is negligible
        one can implement a more specific einsum for the dimension
        1D: (b, c_i, x), (c_i, c_o, x)  -> (b, c_o, x)
        2D: (b, c_i, x, y), (c_i, c_o, x, y)  -> (b, c_o, x, y)
        (2+1)D: (b, c_i, x, y, t), (c_i, c_o, x, y, t)  -> (b, c_o, x, y, t)
        """
        return torch.einsum("bi...,io...->bo...", x, w)

    @abstractmethod
    def spectral_conv(self, vhat, *fft_mesh_size, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement spectral_conv() to perform spectral convolution"
        )

    def forward(self, v, out_mesh_size=None, **kwargs):
        bsz, _, *mesh_size = v.size()
        out_mesh_size = mesh_size if out_mesh_size is None else out_mesh_size
        fft_mesh_size = mesh_size.copy()
        fft_mesh_size[-1] = mesh_size[-1] // 2 + 1
        v_hat = self.fft(v)
        v_hat = self.spectral_conv(v_hat, *fft_mesh_size)
        v = self.ifft(v_hat, s=out_mesh_size)
        return v


class FNOBase(nn.Module):
    def __init__(
        self,
        *,
        num_spectral_layers: int = 4,
        fft_norm="backward",
        activation: ActivationType = "ReLU",
        spatial_padding: int = 0,
        channel_expansion: int = 4,
        spatial_random_feats: bool = False,
        lift_activation: bool = False,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        """New implementation for the base class for Fourier Neural Operator (FNO) models.
        The users need to implement 
        - the lifting operator
        - the output operator
        - the forward method

        add_latent_hook() is used to register a hook to get the latent tensors
        Example:
        model.add_latent_hook("reduction") # reduction is the name of the layer
        The hook will save the output of the layer to self.latent_tensors["r"]
        which is the output of the layer self.r
        """

        self.spatial_padding = spatial_padding
        self.fft_norm = fft_norm
        self.activation = activation
        self.spatial_random_feats = spatial_random_feats
        self.lift_activation = lift_activation
        self.channel_expansion = channel_expansion
        self.debug = debug
        self.num_spectral_layers = num_spectral_layers
        # These should be implemented by subclasses

    @staticmethod
    def _set_modulelist(module: nn.Module, num_layers, *args):
        return nn.ModuleList([deepcopy(module(*args)) for _ in range(num_layers)])

    @property
    @abstractmethod
    def set_lifting_operator(self, *args, **kwargs):
        """Implement this method in subclass to return the lifting operator"""
        raise NotImplementedError("Subclasses must implement lifting_operator property")

    @property
    @abstractmethod
    def set_output_operator(self, *args, **kwargs):
        """Implement this method in subclass to return the output operator"""
        raise NotImplementedError("Subclasses must implement output_operator property")

    def _set_spectral_layers(
        self,
        num_layers: int,
        modes: List[int],
        width: int,
        activation: ActivationType,
        spectral_conv: SpectralConv,
        mlp: PointwiseFFN,
        linear: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        channel_expansion: int = 4,
    ) -> None:
        """
        In SFNO
        spectral_conv: SpectralConvS
        mlp: MLP with dim=3
        linear: nn.Conv3d
        """
        act_func = getattr(nn, activation)
        for attr, module, args in zip(
            ["spectral_conv", "mlp", "w", "activations"],
            [spectral_conv, mlp, linear, act_func],
            [
                (width, width, *modes),
                (width, width, channel_expansion * width, activation),
                (width, width, 1),
                (),
            ],
        ):
            setattr(
                self,
                attr,
                self._set_modulelist(module, num_layers, *args),
            )

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of FNO must implement the forward method")
