import math
from datetime import datetime
from functools import partial
from typing import Callable, Tuple, Union

import torch
import torch.fft as fft
import torch.nn.functional as F
from einops import repeat
from torch.linalg import norm
from torch_cfd.equations import *

TQDM_ITERS = 200

# TODO
# [x] removes dupes with torch_cfd module


def backdiff(x, order: int = 3):
    """
    bdf scheme for x: (b, *, x, y, t)
    """
    if order > 5:
        raise NotImplementedError("only bdf order <= 5 is implemented")
    bdf_weights = {
        1: [1, -1],
        2: [3 / 2, -2, 0.5],
        3: [11 / 6, -3, 3 / 2, -1 / 3],
        4: [25 / 12, -4, 3, -4 / 3, 1 / 4],
        5: [137 / 60, -5, 5, -10 / 3, 5 / 4, -1 / 5],
    }
    weights = torch.as_tensor(bdf_weights[order]).to(x.device)
    x_t = x[..., -(order + 1) :].flip(-1) * weights
    return x_t.sum(-1)


def interp2d(x, **kwargs):
    """
    For Python 3.11, the implementation can be implemented as follows
    expand_dims = [None] * (4 - x.ndim)
    x = x[*expand_dims, ...]
    the following implementation creates the required number of dimensions without unpacking
    """
    for _ in range(4 - x.ndim):
        x = x.unsqueeze(0)
    return F.interpolate(x, **kwargs).squeeze()


def update_residual(
    w_h,
    w_h_t,
    f_h,
    visc,
    rfftmesh,
    laplacian,
    dealias_filter=None,
    dealias=True,
    **kwargs,
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
    **kwargs,
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
        dealias_filter = torch.logical_and(
            torch.abs(ky) <= (2.0 / 3.0) * k_max,
            torch.abs(kx) <= (2.0 / 3.0) * k_max,
        )

    if f.ndim < w.ndim:
        f = f.unsqueeze(0)

    # Stream function in Fourier space: solve Poisson equation
    psi_h = (
        -w / laplacian
    )  # valid for w: (b, *, n, n//2+1, n_t) and lap: (n, n//2+1, n_t)

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


def get_trajectory_imex(
    equation: ImplicitExplicitODE,
    w0: torch.Tensor,
    dt: float,
    num_steps: int = 1,
    record_every_steps: int = 1,
    pbar: bool = False,
    pbar_desc: str = "generating trajectories using RK4",
    require_grad: bool = False,
    dtype: torch.dtype = torch.complex64,
):
    """
    vorticity stacked in the time dimension
    all inputs and outputs are in the frequency domain
    input: w0 (*, n, n)
    output:

    vorticity (*, n_t, kx, ky)
    psi: (*, n_t, kx, ky)

    velocity can be computed from psi
    (*, 2, n_t, kx, ky) by calling spectral_rot_2d
    """
    w_all = []
    dwdt_all = []
    res_all = []
    psi_all = []
    w = w0
    n = w0.size(-1)
    tqdm_iters = num_steps if TQDM_ITERS > num_steps else TQDM_ITERS
    update_every_iters = num_steps // tqdm_iters
    with tqdm(total=num_steps, disable=not pbar) as pb:
        for t_step in range(num_steps):
            w, dwdt = equation.forward(w, dt=dt)
            w.requires_grad_(require_grad)
            dwdt.requires_grad_(require_grad)

            if t_step % update_every_iters == 0:
                res = equation.residual(w, dwdt)
                res_ = fft.irfft2(res).real
                w_ = fft.irfft2(w).real
                res_norm = norm(res_, dim=(-1, -2)).mean() / n
                w_norm = norm(w_, dim=(-1, -2)).mean() / n
                res_desc = f" - ||L(w) - f||_2: {res_norm.item():.4e}"
                res_desc += f" | vort norm {w_norm.item():.4e}"
                desc = (
                    datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                    + " - "
                    + pbar_desc
                    + res_desc
                )
                pb.set_description(desc)
                pb.update(update_every_iters)

            if t_step % record_every_steps == 0:
                _, psi = vorticity_to_velocity(equation.grid, w)
                res = equation.residual(w, dwdt)

                w_, dwdt_, psi, res = [
                    var.detach().to(dtype).cpu().clone() for var in [w, dwdt, psi, res]
                ]

                w_all.append(w_)
                psi_all.append(psi)
                dwdt_all.append(dwdt_)
                res_all.append(res)

    result = {
        var_name: torch.stack(var, dim=-3)
        for var_name, var in zip(
            ["vorticity", "stream", "vort_t", "residual"],
            [w_all, psi_all, dwdt_all, res_all],
        )
    }
    return result


def get_trajectory_imex_crank_nicolson(
    w0,
    f,
    visc: float = 1e-3,
    T: float = 1,
    delta_t: float = 1e-3,
    record_steps: int = 1,
    diam: float = 1,
    dealias: bool = True,
    subsample: int = 1,
    dtype: torch.dtype = None,
    pbar: bool = True,
    **kwargs,
):
    """
    w0: initial vorticity
    f: forcing term, fixed for all time-steps
    visc: viscosity (1/Re)
    T: final time
    delta_t: internal time-step for solve (descrease if blow-up)
    record_steps: number of in-time snapshots to record
    diam: diameter of the domain by default the domain is (0, diam) x (0, diam)
    Solving the 2D Navier-Stokes equation
    vorticity-stream function formulation using Crank-Nicolson scheme
    output: all in (B, t, n, n)
        - vorticity, time derivative of vorticity, streamfunction, residual
    """
    # Grid size - must be power of 2
    dtype = w0.dtype if dtype is None else dtype
    device = w0.device
    bsz, n = w0.size(0), w0.size(-1)
    ns = n // subsample

    # Maximum frequency
    k_max = math.floor(n / 2.0)

    # Number of steps to final time
    total_steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = fft.rfft2(w0)

    # Forcing to Fourier space
    f_h = fft.rfft2(f)

    # If same forcing for the whole batch
    if f_h.ndim < w_h.ndim:
        f_h = f_h.unsqueeze(0)

    # Delta_steps = Record solution every this number of steps
    record_every_n_steps = math.floor(total_steps / record_steps)

    # Wavenumbers in y-direction
    kx = fft.fftfreq(n, d=diam / n, dtype=dtype, device=device)
    ky = fft.fftfreq(n, d=diam / n, dtype=dtype, device=device)
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")

    # Truncate redundant modes
    kx = kx[..., : k_max + 1]
    ky = ky[..., : k_max + 1]
    k_max = (1 / diam) * k_max

    # Laplacian in Fourier space
    lap = -4 * (math.pi**2) * (kx**2 + ky**2)
    lap[0, 0] = 1.0
    kx, ky, lap = kx[None, ...], ky[None, ...], lap[None, ...]

    # Dealiasing mask
    dealias_filter = (
        torch.unsqueeze(
            torch.logical_and(
                torch.abs(kx) <= (2.0 / 3.0) * k_max,
                torch.abs(ky) <= (2.0 / 3.0) * k_max,
            )
            .to(dtype)
            .to(device),
            0,
        )
        if dealias
        else 1.0
    )

    # Saving solution and time
    size = bsz, record_steps, ns, ns
    vort, vort_t, stream, residual = [
        torch.empty(*size, device="cpu") for _ in range(4)
    ]
    t_steps = torch.empty(record_steps, device="cpu")

    # Record counter
    c = 0
    # Physical time
    t = 0.0

    # several quantities to track
    enstrophy = norm(w0, dim=(-1, -2)).mean() / n
    res = torch.zeros(n, n)  # residual placeholder
    residualL2 = norm(res, dim=(-1, -2)).mean() / n

    desc = (
        datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        + f" - enstrophy w: {enstrophy:.4f} \ "
        + f"||L(w, psi) - f||_L2: {residualL2:.4e} \ "
    )

    with tqdm(total=total_steps, desc=desc, disable=not pbar) as pb:
        for j in range(total_steps):

            w_h, w_h_t, _, psi_h, res_h = imex_crank_nicolson_step(
                w_h,
                f_h,
                visc,
                delta_t,
                diam=diam,
                rfftmesh=(kx, ky),
                laplacian=lap,
                dealias_filter=dealias_filter,
                dealias=dealias,
                **kwargs,
            )

            if w_h.isnan().any():
                w_h = w_h[~torch.isnan(w_h)]
                raise ValueError(f"Solution diverged with norm {norm(w_h)}")
                # Id_lap_h = 1.0 + 0.5 * delta_t * visc * lap
                # print(f"min of I - 0.5 * dt * nu * \hat(Delta_h): {Id_lap_h.abs().min()}")

            # Update real time (used only for recording)
            t += delta_t

            if (j + 1) % record_every_n_steps == 0:
                # Solution in physical space
                w = fft.irfft2(w_h, s=(n, n)).real
                w_t = fft.irfft2(w_h_t, s=(n, n)).real
                psi = fft.irfft2(psi_h, s=(n, n)).real

                res_h = update_residual(
                    w_h,
                    w_h_t,
                    f_h,
                    visc,
                    (kx, ky),
                    lap,
                    dealias_filter=dealias_filter,
                    dealias=dealias,
                )
                res = fft.irfft2(res_h, s=(n, n)).real

                if subsample > 1:
                    w, w_t, psi, res = (
                        interp2d(w, size=(ns, ns), mode="bilinear"),
                        interp2d(w_t, size=(ns, ns), mode="bilinear"),
                        interp2d(psi, size=(ns, ns), mode="bilinear"),
                        interp2d(res, size=(ns, ns), mode="bilinear"),
                    )
                # Record solution and time
                vort[:, c] = w.detach().cpu()
                vort_t[:, c] = w_t.detach().cpu()
                stream[:, c] = psi.detach().cpu()
                residual[:, c] = res.detach().cpu()
                t_steps[c] = t

                c += 1
                enstrophy = norm(w, dim=(-1, -2)).mean() / n
                residualL2 = norm(res, dim=(-1, -2)).mean() / n
                divider = {0: "|", 1: "/", 2: "-", 3: "\\"}
                desc = (
                    datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                    + f" - enstrophy w: {enstrophy:.4f} {divider[c%4]} "
                    + f" ||L(w, psi) - f||_2: {residualL2:.4e} {divider[c%4]} "
                )
                pb.set_description(desc)
            pb.update()

    return dict(
        vorticity=vort,
        vorticity_t=vort_t,
        stream=stream,
        residual=residual,
        t_steps=t_steps,
    )


if __name__ == "__main__":
    n = 256
    bsz = 4
    w = torch.randn(bsz, n, n // 2 + 1).to(torch.complex128)
    f = torch.randn(n, n // 2 + 1).to(torch.complex128)
    result = imex_crank_nicolson_step(w, f, 1e-3, 1e-3)
    for v in result:
        if isinstance(v, torch.Tensor):
            print(v.shape, v.dtype, v.device)
