import argparse
import logging
import math
import sys
from collections import defaultdict
from functools import partial

import dill
import torch
import torch.fft as fft
import torch.nn.functional as F
import xarray
from .solvers import *
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from torch.linalg import norm
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
DATA_PATH = os.path.join(SRC_ROOT, "data")
LOG_PATH = os.path.join(SRC_ROOT, "logs")
for p in [DATA_PATH, LOG_PATH]:
    if not os.path.exists(p):
        os.makedirs(p)

feval = lambda s: eval("lambda x, y:" + s, globals())


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


def get_trajectory_imex_crank_nicolson(
    w0,
    f,
    visc,
    T,
    delta_t=1e-3,
    record_steps=1,
    diam=1,
    dealias=True,
    subsample=1,
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
    size, device, dtype = w0.size(), w0.device, w0.dtype
    bsz, n = size[0], size[-1]
    interp2d = partial(
        F.interpolate, size=(n // subsample, n // subsample), mode="bilinear"
    )

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
    dealiasing_filter = (
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
    size = bsz, record_steps, n // subsample, n // subsample
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
        f"enstrophy w: {enstrophy:.4f} \ "
        + f"||L(w, psi) - f||_L2: {residualL2:.4e} \ "
    )

    with tqdm(total=total_steps, desc=desc) as pbar:
        for j in range(total_steps):

            w_h, _, w_h_t, psi_h, res_h = imex_crank_nicolson_step(
                w_h,
                f_h,
                visc,
                delta_t,
                diam=diam,
                rfftmesh=(kx, ky),
                laplacian=lap,
                dealias_filter=dealiasing_filter,
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
                    dealiasing_filter,
                    dealias=dealias,
                )
                res = fft.irfft2(res_h, s=(n, n)).real

                if subsample > 1:
                    w, w_t, psi, res = (
                        interp2d(w),
                        interp2d(w_t),
                        interp2d(psi),
                        interp2d(res),
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
                    f"enstrophy w: {enstrophy:.4f} {divider[c%4]} "
                    + f" ||L(w, psi) - f||_2: {residualL2:.4e} {divider[c%4]} "
                )
                pbar.set_description(desc)
            pbar.update()

    return dict(
        vorticity=vort,
        vorticity_t=vort_t,
        stream=stream,
        residual=residual,
        t_steps=t_steps,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation for 2D NSE with FNO right-hand side"
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
        "--batch-size",
        type=int,
        default=8,
        metavar="bsz",
        help="batch size for data generation (default: 8)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1200,
        metavar="N",
        help="sample size for data generation (default: 1200)",
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
        "--filename",
        type=str,
        default=None,
        metavar="filename",
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
        "--seed",
        type=int,
        default=1127825,
        metavar="Seed",
        help="random seed (default: 1127825)",
    )

    return parser.parse_args()


def pickle_to_pt(data_path, save_path=None):
    """
    convert serialized data from pickle to pytorch pt file
    using dill instead of pickle
    https://stackoverflow.com/a/28745948/622119
    """
    save_path = data_path.replace(".pkl", ".pt") if save_path is None else save_path
    result = []
    with open(data_path, "rb") as f:
        while True:
            try:
                result.append(dill.load(f))
            except EOFError:
                break

    data = defaultdict(list)
    for _res in result:
        for field, value in _res.items():
            data[field].append(value)

    for field, value in data.items():
        data[field] = torch.cat(value)

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
