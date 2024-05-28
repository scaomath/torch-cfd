import math
from functools import partial
from typing import Callable, Tuple, Union

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat


def update_residual(
    w_h, w_h_t, f_h, visc, rfftmesh, laplacian, dealias_filter, dealias=True
):
    """
    compute the residual of an input w in the frequency domain
    dw/dt needs to be given
    the shape of the tensors: (B, n, n) or (B, t, n, n)
    """
    _, *size = w_h.shape
    n = size[-2]
    irfft2 = partial(fft.irfft2, s=(n, n))
    kx, ky = rfftmesh
    psi_h = -w_h / laplacian

    u = 2 * math.pi * ky * 1j * psi_h
    v = -2.0 * math.pi * kx * 1j * psi_h
    w_x = 2.0 * math.pi * kx * 1j * w_h
    w_y = 2.0 * math.pi * ky * 1j * w_h

    u, v, w_x, w_y = [irfft2(z).real for z in [u, v, w_x, w_y]]

    convection_h = fft.rfft2(u * w_x + v * w_y)
    if dealias:
        convection_h = dealias_filter * convection_h

    res_h = w_h_t + convection_h - visc * laplacian * w_h - f_h
    return res_h


def imex_crank_nicolson_step(
    w,
    f,
    visc,
    delta_t,
    diam: float = 1,
    rfftmesh: Tuple[torch.Tensor, torch.Tensor] = None,
    laplacian: torch.Tensor = None,
    dealias_filter: torch.Tensor = None,
    debug=False,
):
    """
    inputs:
    w: current vorticities {w(t_i)} in frequency space
    f: forcing term {f(t_{i+1})} in frequency space

    in the spatial domain
    input shape of w: (B, *, kx, ky)
    input shape of f: (B, *, kx, ky)

    outputs:
    - the vorticities {w(t_{i+1})}
    - the streamfunction {psi(t_{i+1})}
    - derivatives of {w(t_{i+1})} with respect to time
    spectral method update for the streamfunction psi
    """
    bsz, *size = w.shape
    assert (size[-1] - 1) * 2 == size[-2]  # check if the input is an rfft2 tensor
    dtype = w.dtype
    device = w.device
    n = size[-2]

    if rfftmesh is None:
        kx = fft.fftfreq(n, d=diam / n)
        ky = fft.fftfreq(n, d=diam / n)
        kx, ky = torch.meshgrid([kx, ky], indexing="ij")
        k_max = math.floor(n / 2.0)
        kx = kx[..., : k_max + 1]
        ky = ky[..., : k_max + 1]
    else:
        kx, ky = rfftmesh

    if kx.ndim == 2:
        kx, ky = [repeat(z, "x y -> b x y", b=bsz) for z in [kx, ky]]
    kx, ky = [z.to(dtype).to(device) for z in [kx, ky]]

    if laplacian is None:
        laplacian = -4 * (math.pi**2) * (kx**2 + ky**2)
        laplacian[..., 0, 0] = 1.0

    if f.ndim < w.ndim:
        f = f.unsqueeze(0)

    psi_h = -w / laplacian
    u = 2 * math.pi * ky * 1j * psi_h
    v = -2.0 * math.pi * kx * 1j * psi_h
    w_x = 2.0 * math.pi * kx * 1j * w
    w_y = 2.0 * math.pi * ky * 1j * w

    u, v, w_x, w_y = [fft.irfft2(z, s=(n, n)).real for z in [u, v, w_x, w_y]]

    convection_h = fft.rfft2(u * w_x + v * w_y)
    if dealias_filter is not None:
        convection_h = dealias_filter * convection_h

    w_next = (
        -delta_t * convection_h
        + delta_t * f
        + (1.0 + 0.5 * delta_t * visc * laplacian) * w
    ) / (1.0 - 0.5 * delta_t * visc * laplacian)

    dwdt = (w_next - w) / delta_t
    res_h = dwdt + convection_h - visc * laplacian * w - f

    if debug:
        for key, var in zip(
            ["w_new", "convection", "psi", "dealias", "w_t", "f", "lap", "residual"],
            [w_next, convection_h, psi_h, dealias_filter, dwdt, f, laplacian, res_h],
        ):
            print(f"{key} shape: {var.shape}")

    return w_next, w, dwdt, psi_h, res_h

if __name__ == "__main__":
    n = 256
    bsz = 4
    w = torch.randn(bsz, n, n//2+1).to(torch.complex128)
    f = torch.randn(n, n//2+1).to(torch.complex128)
    result = imex_crank_nicolson_step(w, f, 1e-3, 1e-3)
    for v in result:
        if isinstance(v, torch.Tensor):
            print(v.shape, v.dtype, v.device)
    