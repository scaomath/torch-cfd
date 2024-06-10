import argparse
import logging
import math
import sys
from collections import defaultdict

import dill
import torch
import torch.fft as fft
import torch.nn.functional as F
import xarray

from data.solvers import *
from torch_cfd.equations import *
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from torch.linalg import norm
from tqdm import tqdm

feval = lambda s: eval("lambda x, y:" + s, globals())

TQDM_ITERS = 200


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(filename, tqdm=True):
    stream_handler = TqdmLoggingHandler() if tqdm else logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(filename=filename),
            stream_handler,
        ],
    )
    return logging.getLogger()


def interp2d(x, **kwargs):
    expand_dims = [None] * (4 - x.ndim)
    x = x[*expand_dims, ...]
    return F.interpolate(x, **kwargs).squeeze()


def get_trajectory_rk4(
    equation: ImplicitExplicitODE,
    w0: Array,
    dt: float,
    num_steps: int = 1,
    record_every_steps: int = 1,
    pbar=False,
    pbar_desc="generating trajectories using RK4",
    require_grad=False,
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
    tqdm_iters = num_steps if TQDM_ITERS > num_steps else TQDM_ITERS
    update_iters = num_steps // tqdm_iters
    with tqdm(total=num_steps) as pbar:
        for t_step in range(num_steps):
            w, dwdt = equation.forward(w, dt=dt)
            w.requires_grad_(require_grad)
            dwdt.requires_grad_(require_grad)

            if t_step % update_iters == 0:
                res = equation.residual(w, dwdt)
                res_ = fft.irfft2(res).real
                w_ = fft.irfft2(w).real
                res_norm = norm(res_).item()/w0.size(-1)
                w_norm = norm(w_).item()/w0.size(-1)
                res_desc = f" - ||L(w) - f||_2: {res_norm:.4e}"
                res_desc += f" | vort norm {w_norm:.4e}"
                desc = (
                    datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                    + " - "
                    + pbar_desc
                    + res_desc
                )
                pbar.set_description(desc)
                pbar.update(update_iters)

            if t_step % record_every_steps == 0:
                _, psi = vorticity_to_velocity(equation.grid, w)
                res = equation.residual(w, dwdt)

                w_, dwdt_, psi, res = [
                    var.detach().cpu().clone() for var in [w, dwdt, psi, res]
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
    visc=1e-3,
    T=1,
    delta_t=1e-3,
    record_steps=1,
    diam=1,
    dealias=True,
    subsample=1,
    dtype=None,
    pbar=True,
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


def get_args():
    parser = argparse.ArgumentParser(
        description="Meta parameters for generating Navier-Stokes data and train"
    )
    parser.add_argument(
        "--example",
        type=str,
        default=None,
        metavar="example name",
        help="data name (default: None)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        metavar="n",
        help="grid size (default: 256)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=4,
        metavar="s",
        help="subsample (default: 4)",
    )
    parser.add_argument(
        "--diam",
        default=1.0,
        metavar="diam",
        help="domain is (0,d)x(0,d) (default: 1.0)",
    )
    parser.add_argument(
        "--scale",
        default=1,
        metavar="scale",
        help="spatial scaling of the domain (default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="bsz",
        help="batch size for data generation (default: 8)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1200,
        metavar="N",
        help="number of samples for data generation (default: 1200)",
    )
    parser.add_argument(
        "--visc",
        type=float,
        default=1e-3,
        metavar="viscosity",
        help="viscosity in front of Laplacian, 1/Re (default: 0.001)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=20.0,
        metavar="T",
        help="total time for simulation (default: 20.0)",
    )
    parser.add_argument(
        "--time-warmup",
        type=float,
        default=4.5,
        metavar="T_warmup",
        help="warm up for simulation (default: 4.5)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-4,
        metavar="delta_t",
        help="time step size for simulation (default: 1e-4)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        metavar="nt",
        help="number of recorded snapshots (default: 50)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="use normalized GRF in IV",
    )
    parser.add_argument(
        "--double",
        action="store_true",
        default=False,
        help="use double precision torch",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.5,
        metavar="alpha",
        help="smoothness of the GRF (default: 2.5)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=7.0,
        metavar="tau",
        help="strength of diagonal regularizer in the covariance (default: 7.0)",
    )
    parser.add_argument(
        "--forcing",
        type=feval,
        nargs="?",
        default="0.1*(torch.sin(2*math.pi*(x+y))+torch.cos(2*math.pi*(x+y)))",
        metavar="f",
        help="rhs in vorticity equation in lambda x: f(x) (default: FNO's default)",
    )
    parser.add_argument(
        "--peak-wavenumber",
        type=int,
        default=4,
        metavar="kappa",
        help="wavenumber of the highest energy density for the initial condition (default: 4)",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=5,
        metavar="v_max",
        help="the maximum speed in the init velocity field (default: 5)",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        metavar="file path",
        help="path to save the data (default: None)",
    )
    parser.add_argument(
        "--logpath",
        type=str,
        default=None,
        metavar="log path",
        help="path to save the logs (default: None)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        metavar="file name",
        help="file name for Navier-Stokes data (default: None)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA"
    )
    parser.add_argument(
        "--extra-vars",
        action="store_true",
        default=False,
        help="store extra variables in the data file",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Force regenerate data even if it exists",
    )
    parser.add_argument(
        "--replicable-init",
        action="store_true",
        default=False,
        help="Use the GRF on a reference max mesh size then downsample to get a replicable initial condition",
    )
    parser.add_argument(
        "--no-dealias",
        action="store_true",
        default=False,
        help="Disable the dealias masking to the nonlinear convection term",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        default=False,
        help="Disable program bar for data generation",
    )
    parser.add_argument(
        "--demo-plots",
        action="store_true",
        default=False,
        help="plot several trajectories for the generated data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1127825,
        metavar="Seed",
        help="random seed (default: 1127825)",
    )

    return parser


def save_pickle(data, save_path, append=True):
    mode = "ab" if append else "wb"
    with open(save_path, mode) as f:
        dill.dump(data, f)


def load_pickle(load_path, mode="rb"):
    data = []
    with open(load_path, mode=mode) as f:
        try:
            while True:
                data.append(dill.load(f))
        except EOFError:
            pass
    return data


def pickle_to_pt(data_path, save_path=None):
    """
    convert serialized data from pickle to pytorch pt file
    using dill instead of pickle
    https://stackoverflow.com/a/28745948/622119
    """
    save_path = data_path.replace(".pkl", ".pt") if save_path is None else save_path
    result = load_pickle(data_path)

    data = defaultdict(list)
    for _res in result:
        for field, value in _res.items():
            data[field].append(value)

    for field, value in data.items():
        v = torch.cat(value)
        if v.ndim == 1: # time steps or seed
            v = torch.unique(v)
        data[field] = v

    torch.save(data, data_path)


def verify_trajectories(
    data_path,
    n_samples=5,
    dt=1e-3,
    T_warmup=4.5,
    diam=2 * torch.pi,
):
    data = torch.load(data_path)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape, v.dtype)
    N, T, ns, _ = data["vorticity"].shape
    n_samples = min(n_samples, N // 2)
    idxes = torch.randint(0, N, (n_samples,))
    gridx = gridy = torch.arange(ns) * diam / ns
    coords = {
        "time": dt * torch.arange(T) + T_warmup,
        "x": gridx,
        "y": gridy,
    }

    for idx in idxes:
        w_data = xarray.DataArray(
            data["vorticity"][idx, :T],
            dims=["time", "x", "y"],
            coords=coords,
        ).to_dataset(name="vorticity")

        g = (
            w_data["vorticity"]
            .isel(time=slice(2, None))
            .thin(time=T // 5)
            .plot.imshow(
                col="time",
                col_wrap=5,
                cmap=sns.cm.icefire,
                robust=True,
                xticks=None,
                yticks=None,
                cbar_kwargs={"label": f"Vorticity in Sample {idx}"},
            )
        )

        g.set_xlabels("")
        g.set_ylabels("")
        plt.show()
