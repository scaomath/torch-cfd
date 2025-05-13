# The MIT License (MIT)
# Copyright © 2025 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import math
import os
from functools import partial

import torch
import torch.fft as fft
import torch.nn.functional as F

from grf import GRF2d
from solvers import get_trajectory_imex
from data_utils import *
from torch_cfd.grids import *
from torch_cfd.equations import *
from torch_cfd.forcings import *
from fno.pipeline import DATA_PATH, LOG_PATH


def main(args):
    """
    Generate the original FNO data
    the right hand side is a fixed forcing
    0.1*(torch.sin(2*math.pi*(x+y))+torch.cos(2*math.pi*(x+y)))

    It stores data after each batch, and will resume using a fixed formula'd seed
    when starting again.
    The default values of the params for the Gaussian Random Field (GRF) are printed.

    Sample usage:

    - Training data for Spectral-Refiner ICLR 2025 paper 'fnodata_extra_64x64_N1280_v1e-3_T50_steps100_alpha2.5_tau7.pt'
    >>> python data_gen_fno.py --num-samples 1280 --batch-size 256 --grid-size 256 --subsample 4 --extra-vars --time 50 --time-warmup 30 --num-steps 100 --dt 1e-3 --visc 1e-3 --scale 0.1

    - Test data
    >>> python data_gen_fno.py --num-samples 16 --batch-size 8 --grid-size 256 --subsample 1 --double --extra-vars --time 50 --time-warmup 30 --num-steps 100 --dt 1e-3 --scale 0.1 --replicable-init --seed 42

    - Test data fine
    >>> python data_gen_fno.py --num-samples 2 --batch-size 1 --grid-size 512 --subsample 1 --double --extra-vars --time 50 --time-warmup 30 --num-steps 200 --dt 5e-4 --scale 0.1 --replicable-init --seed 42

    - Testing if the code works
    >>> python data_gen/data_gen_fno.py --num-samples 4 --batch-size 2 --grid-size 128 --subsample 1 --double --extra-vars --time 2 --time-warmup 1 --num-steps 10 --dt 1e-3 --scale 0.1 --replicable-init --seed 42

    """

    args = args.parse_args()

    current_time = datetime.now().strftime("%d_%b_%Y_%Hh%Mm")
    log_name = "".join(os.path.basename(__file__).split(".")[:-1])
    logpath = args.logpath if args.logpath is not None else LOG_PATH
    log_filename = os.path.join(logpath, f"{current_time}_{log_name}.log")
    logger = get_logger(log_filename)

    logger.info(f"Using the following arguments: ")
    all_args = {k: v for k, v in vars(args).items() if not callable(v)}
    logger.info(" | ".join(f"{k}={v}" for k, v in all_args.items()))

    n_grid_max = 2048
    n = args.grid_size  # 256
    subsample = args.subsample  # 4
    ns = n // subsample
    diam = args.diam  # 1.0
    diam = eval(diam) if isinstance(diam, str) else diam
    if n > n_grid_max:
        raise ValueError(
            f"Grid size {n} is larger than the maximum allowed {n_grid_max}"
        )
    scale = args.scale
    visc = args.visc if args.Re is None else 1 / args.Re  # 1e-3
    T = args.time  # 50
    T_warmup = args.time_warmup  # 30
    T_new = T - T_warmup
    record_steps = args.num_steps
    dt = args.dt  # 1e-4
    logger.info(f"Using dt = {dt}")

    warmup_steps = int(T_warmup / dt)
    total_steps = int(T_new / dt)
    record_every_iters = int(total_steps / record_steps)

    alpha = args.alpha  # 2.5
    tau = args.tau  # 7
    peak_wavenumber = args.peak_wavenumber

    dtype = torch.float64 if args.double else torch.float32
    normalize = args.normalize
    filename = args.filename
    force_rerun = args.force_rerun
    replicate_init = args.replicable_init
    dealias = not args.no_dealias
    pbar = not args.no_tqdm

    # Number of solutions to generate
    total_samples = args.num_samples  # 8

    # Batch size
    batch_size = args.batch_size  # 8

    extra = "_extra" if args.extra_vars else ""
    dtype_str = "_fp64" if args.double else ""
    if filename is None:
        filename = (
            f"fnodata{extra}{dtype_str}_{ns}x{ns}_N{total_samples}"
            + f"_v{visc:.0e}_T{int(T)}_steps{record_steps}_alpha{alpha:.1f}_tau{tau:.0f}.pt"
        ).replace("e-0", "e-")
        args.filename = filename

    filepath = args.filepath if args.filepath is not None else DATA_PATH
    for p in [filepath]:
        if not os.path.exists(p):
            os.makedirs(p)
            logging.info(f"Created directory {p}")
    data_filepath = os.path.join(DATA_PATH, filename)

    data_exist = os.path.exists(data_filepath)
    if data_exist and not force_rerun:
        logger.info(f"File {filename} exists with current data as follows:")
        data = torch.load(data_filepath)

        for key, v in data.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"{key:<12} | {v.shape} | {v.dtype}")
            else:
                logger.info(f"{key:<12} | {v.dtype}")
        if len(data[key]) == total_samples:
            return
        elif len(data[key]) < total_samples:
            total_samples -= len(data[key])
    else:
        logger.info(f"Generating data and saving in {filename}")

    cuda = not args.no_cuda and torch.cuda.is_available()
    no_tqdm = args.no_tqdm
    device = torch.device("cuda:0" if cuda else "cpu")

    torch.set_default_dtype(torch.float64)
    logger.info(
        f"Using device: {device} | save dtype: {dtype} | computge dtype: {torch.get_default_dtype()}"
    )
    # Set up 2d GRF with covariance parameters
    # Parameters of covariance C = tau^0.5*(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
    # Note that we need alpha > d/2 (here d= 2)

    grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)

    forcing_fn = SinCosForcing(
        grid=grid,
        scale=scale,
        diam=diam,
        k=peak_wavenumber,
        vorticity=True,
    )
    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))

    grf = GRF2d(
        n=n,
        alpha=alpha,
        tau=tau,
        normalize=normalize,
        device=device,
        dtype=torch.float64,
    )

    ns2d = NavierStokes2DSpectral(
        viscosity=visc,
        grid=grid,
        smooth=True,
        forcing_fn=forcing_fn,
        solver=IMEXStepper,
        order=2,
    ).to(device)

    if os.path.exists(data_filepath) and not force_rerun:
        logger.info(f"Data already exists at {data_filepath}")
        return
    elif os.path.exists(data_filepath) and force_rerun:
        logger.info(f"Force rerun and save data to {data_filepath}")
        os.remove(data_filepath)
    else:
        logger.info(f"Save data to {data_filepath}")

    num_batches = total_samples // batch_size
    for i, idx in enumerate(range(0, total_samples, batch_size)):
        logger.info(f"Generate trajectory for batch [{i+1}/{num_batches}]")
        logger.info(
            f"random states: {args.seed + idx} to {args.seed + idx + batch_size-1}"
        )

        # Sample random fields
        seeds = [args.seed + idx + k for k in range(batch_size)]
        n0 = n_grid_max if replicate_init else n
        vort_init = [
            grf.sample(1, n0, random_state=s) for _, s in zip(range(batch_size), seeds)
        ]
        vort_init = torch.stack(vort_init)
        if n != n0:
            vort_init = F.interpolate(vort_init, size=(n, n), mode="nearest")
        vort_init = vort_init.squeeze(1)
        vort_hat = fft.rfft2(vort_init).to(device)

        logger.info(f"initial condition {vort_init.shape}")

        if T_warmup > 0:
            with tqdm(total=warmup_steps, disable=no_tqdm) as pbar:
                for j in range(warmup_steps):
                    vort_hat, _ = ns2d.step(vort_hat, dt)
                    if j % 100 == 0:
                        vort_norm = torch.linalg.norm(fft.irfft2(vort_hat)).item() / n
                        desc = (
                            datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                            + f" - Warmup | vort_hat ell2 norm {vort_norm:.4e}"
                        )
                        pbar.set_description(desc)
                        pbar.update(100)

        logger.info(f"generate data from {T_warmup} to {T}")
        result = get_trajectory_imex(
            ns2d,
            vort_hat,
            dt,
            num_steps=total_steps,
            record_every_steps=record_every_iters,
            pbar=not no_tqdm,
        )

        for field, value in result.items():
            value = fft.irfft2(value).real.cpu().to(dtype)
            logger.info(
                f"variable: {field} | shape: {value.shape} | dtype: {value.dtype}"
            )
            if subsample > 1:
                assert (
                    value.ndim == 4
                ), f"Subsampling only works for (b, c, h, w) tensors, current shape: {value.shape}"
                value = F.interpolate(value, size=(ns, ns), mode="bilinear")
            result[field] = value
            logger.info(f"{field:<15} | {value.shape} | {value.dtype}")

        if not extra:
            for key in ["vort_t", "stream", "residual"]:
                result[key] = torch.empty(0, device="cpu")
        result["random_states"] = torch.as_tensor(seeds, dtype=torch.int32)

        logger.info(f"Saving batch [{i+1}/{num_batches}] to {data_filepath}")
        save_pickle(result, data_filepath)
        del result

    pickle_to_pt(data_filepath)
    logger.info(f"Done converting to pt.")
    if args.demo_plots:
        try:
            verify_trajectories(
                data_filepath,
                dt=T_new / record_steps,
                T_warmup=T_warmup,
                n_samples=1,
            )
        except Exception as e:
            logger.error(f"Error in plotting: {e}")
        finally:
            pass
    return


if __name__ == "__main__":
    args = get_args_ns2d("Generate the original FNO data for NSE in 2D")
    main(args)
