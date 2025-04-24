import math
from functools import partial
from typing import Callable, Tuple, Union

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

def backdiff(x, order:int=3):
    """
    bdf scheme for x: (b, *, x, y, t)
    """
    bdf_weights ={
        1: [1, -1],
        2: [3/2, -2, 0.5],
        3: [11/6, -3, 3/2, -1/3],
        4: [25/12, -4, 3, -4/3, 1/4],
        5: [137/60, -5, 5, -10/3, 5/4, -1/5]
    }
    weights = torch.as_tensor(bdf_weights[order]).to(x.device)
    x_t = x[...,-(order+1):].flip(-1)*weights
    return x_t.sum(-1)
    
def fft_mesh_2d(n, diam, device=None):
        kx, ky = [fft.fftfreq(n, d=diam/n) for _ in range(2)]
        kx, ky = torch.meshgrid([kx, ky], indexing="ij")
        return kx.to(device), ky.to(device)

def fft_expand_dims(fft_mesh, batch_size):
    kx, ky = fft_mesh
    kx, ky = [repeat(z, "x y -> b x y 1", b=batch_size) for z in [kx, ky]]
    return kx, ky

def spectral_div_2d(vhat, fft_mesh):
    r"""
    Computes the 2D divergence in the Fourier basis.
    TODO: this is a dupe function with torch_cfd module
    needed cleaning up and some refactoring
    """
    uhat, vhat = vhat
    kx, ky = fft_mesh
    return 2j * torch.pi * (uhat * kx + vhat * ky)

def spectral_grad_2d(vhat, rfft_mesh):
    kx, ky = rfft_mesh
    return 2j * torch.pi * kx * vhat, 2j * torch.pi * ky * vhat

def spectral_laplacian_2d(fft_mesh, device=None):
    """
    TODO: this is a dupe function with torch_cfd module
    """
    kx, ky = fft_mesh
    lap = -4 * (torch.pi**2) * (abs(kx) ** 2 + abs(ky) ** 2)
    lap[..., 0, 0] = 1
    return lap.to(device)

def get_freq_spacetime(n, n_t=None, delta_t=None, device=None):
    n_t = n if n_t is None else n_t
    delta_t = 1 / n_t if delta_t is None else delta_t
    kx = fft.fftfreq(n, d=1 / n)
    ky = fft.fftfreq(n, d=1 / n)
    kt = fft.fftfreq(n_t, d=delta_t)
    kx, ky, kt = torch.meshgrid([kx, ky, kt], indexing="ij")
    return kx.to(device), ky.to(device), kt.to(device)

def spectral_laplacian_spacetime(n, n_t=None, device=None):
    kx, ky, _ = get_freq_spacetime(n, n_t)
    lap = -4 * (torch.pi**2) * (kx**2 + ky**2)
    lap[0, 0] = 1
    return lap.to(device)

def update_residual(
    w_h, w_h_t, f_h, visc, rfftmesh, laplacian, 
    dealias_filter=None, dealias=True, **kwargs
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

    # Velocity field in x-direction = psi_y
    u = 2 * math.pi * ky * 1j * psi_h
    # Velocity field in y-direction = -psi_x
    v = -2.0 * math.pi * kx * 1j * psi_h
    # Partial x of vorticity
    w_x = 2.0 * math.pi * kx * 1j * w_h
    # Partial y of vorticity
    w_y = 2.0 * math.pi * ky * 1j * w_h

    u, v, w_x, w_y = [irfft2(z).real for z in [u, v, w_x, w_y]]

    # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
    convection_h = fft.rfft2(u * w_x + v * w_y)
    if dealias and dealias_filter is not None:
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
    dealias: bool = False,
    output_rfft: bool = False,
    debug=False,
    **kwargs
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
    k_max = math.floor(n / 2.0)


    if rfftmesh is None:
        kx = fft.fftfreq(n, d=diam / n)
        ky = fft.fftfreq(n, d=diam / n)
        kx, ky = torch.meshgrid([kx, ky], indexing="ij")
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

    if dealias_filter is None:
        dealias_filter = (
            torch.logical_and(
                torch.abs(ky) <= (2.0 / 3.0) * k_max,
                torch.abs(kx) <= (2.0 / 3.0) * k_max,
            ))

    if f.ndim < w.ndim:
        f = f.unsqueeze(0)

    # Stream function in Fourier space: solve Poisson equation
    psi_h = -w / laplacian # valid for w: (b, *, n, n//2+1, n_t) and lap: (n, n//2+1, n_t)

    # Velocity field in x-direction = psi_y
    u = 2 * math.pi * ky * 1j * psi_h
    # Velocity field in y-direction = -psi_x
    v = -2.0 * math.pi * kx * 1j * psi_h
    # Partial x of vorticity
    w_x = 2.0 * math.pi * kx * 1j * w
    # Partial y of vorticity
    w_y = 2.0 * math.pi * ky * 1j * w

    u, v, w_x, w_y = [fft.irfft2(z, s=(n, n)).real for z in [u, v, w_x, w_y]]

    # Non-linear term (velocity dot grad(w)): compute in physical space then back to Fourier space
    convection_h = fft.rfft2(u * w_x + v * w_y)
    if dealias:
        convection_h = dealias_filter * convection_h

    # Crank-Nicolson update in frequency space
    w_next = (
        -delta_t * convection_h
        + delta_t * f
        + (1.0 + 0.5 * delta_t * visc * laplacian) * w
    ) / (1.0 - 0.5 * delta_t * visc * laplacian)

    # Compute gradient of vorticity in frequency space
    dwdt = (w_next - w) / delta_t
    res_h = dwdt + convection_h - visc * laplacian * w - f

    if output_rfft:
        return w_next, dwdt, w, psi_h, res_h, (kx, ky), laplacian, dealias_filter
    else:
        return w_next, dwdt, w, psi_h, res_h

if __name__ == "__main__":
    n = 256
    bsz = 4
    w = torch.randn(bsz, n, n//2+1).to(torch.complex128)
    f = torch.randn(n, n//2+1).to(torch.complex128)
    result = imex_crank_nicolson_step(w, f, 1e-3, 1e-3)
    for v in result:
        if isinstance(v, torch.Tensor):
            print(v.shape, v.dtype, v.device)
    