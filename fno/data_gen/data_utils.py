import argparse
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import dill
import h5py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import xarray
from tqdm.auto import tqdm

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


def get_args_2d(desc="Data generation in 2D"):
    parser = argparse.ArgumentParser(description=desc)
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
        help="grid size (including boundary nodes) in a square domain (default: 256)",
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="periodic",
        metavar="a",
        help="boundary type: periodic, dirichlet, neumann",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        metavar="s",
        help="subsample (default: 1)",
    )
    parser.add_argument(
        "--diam",
        default=1.0,
        metavar="diam",
        help="domain is (0,d)x(0,d) (default: 1.0)",
    )
    parser.add_argument(
        "--scale",
        default=1.0,
        type=float,
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
        "--normalize",
        action="store_true",
        default=False,
        help="use normalized GRF in IV to have L2 norm = 1 (default: False)",
    )
    parser.add_argument(
        "--double",
        action="store_true",
        default=False,
        help="use double precision torch to save data",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.5,
        metavar="alpha",
        help="smoothness of the GRF, spatial covariance (default: 2.5)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=7.0,
        metavar="tau",
        help="strength of diagonal regularizer in the covariance (default: 7.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        metavar="eps",
        help="singular coefficient in -eps*\Delta u + gamma*u= f",
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
        "--no-tqdm",
        action="store_true",
        default=False,
        help="Disable program bar for data generation",
    )
    parser.add_argument(
        "--verify-data",
        action="store_true",
        default=False,
        help="verify the generated data shape, device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1127825,
        metavar="Seed",
        help="random seed (default: 1127825)",
    )

    return parser

def get_args_ns2d(desc="Data generation of Navier-Stokes in 2D"):
    parser = get_args_2d(desc=desc)
    parser.add_argument(
        "--visc",
        type=float,
        default=1e-3,
        metavar="viscosity",
        help="viscosity in front of Laplacian, 1/Re (default: 0.001)",
    )
    parser.add_argument(
        "--Re",
        type=float,
        default=None,
        metavar="Reynolds number",
        help="Re (default: None)",
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
        "--gamma",
        type=float,
        default=0.0,
        metavar="gamma",
        help="L2 coefficient in elliptic problem or NSE (drag) (default: 0.0)",
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
        "--demo",
        action="store_true",
        default=False,
        help="Only demo and plot several trajectories for the generated data (not save to disk)",
    )

    return parser


def save_pickle(data, save_path, append=True):
    mode = "ab" if append else "wb"
    with open(save_path, mode) as f:
        dill.dump(data, f)


def load_pickle(load_path, mode="rb"):
    """
    convert serialized data from pickle to pytorch pt file
    using dill instead of pickle
    https://stackoverflow.com/a/28745948/622119
    """
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
    Change: defaultdict or list is deemed not safe for serialization in PyTorch 2.6.0
    a workaround is to create a new dict after serialization
    """
    save_path = data_path.replace(".pkl", ".pt") if save_path is None else save_path
    result = load_pickle(data_path)

    data = defaultdict(list)
    for _res in result:
        for field, value in _res.items():
            data[field].append(value)

    for field, value in data.items():
        v = torch.cat(value)
        if v.ndim == 1:  # time steps or seed
            v = torch.unique(v)
        data[field] = v

    torch.save({k: v for k, v in data.items() if not callable(v)}, data_path)


def matlab_to_pt(data_path, save_path=None):
    """
    Convert MATLAB .mat files to PyTorch .pt files.
    """
    save_path = data_path.replace(".mat", ".pt") if save_path is None else save_path
    with h5py.File(data_path, "r") as f:
        mat_data = {key: np.array(f[key]) for key in f.keys()}

    data = defaultdict(list)
    for key, value in mat_data.items():
        value = np.transpose(value, axes=range(len(value.shape) - 1, -1, -1))
        data[key] = torch.from_numpy(value)

    torch.save({k: v for k, v in data.items() if not callable(v)}, data_path)


def verify_trajectories(
    data: dict,
    n_samples=5,
    dt=1e-3,
    T_warmup=4.5,
    diam=2 * torch.pi,
):
    import matplotlib
    matplotlib.use('TkAgg')
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
