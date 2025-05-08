# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os, sys

import dill

import torch
import torch.fft as fft

from torch_cfd.grids import *
from torch_cfd.equations import *
from torch_cfd.initial_conditions import *
from torch_cfd.finite_differences import *
from torch_cfd.forcings import *

from tqdm import tqdm
from data_utils import *
from solvers import *
import logging

from fno.pipeline import DATA_PATH, LOG_PATH


def main(args):
    """
    Generate the isotropic turbulence in [1]

    [1]: McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. Journal of Fluid Mechanics, 146, 21-43.

    Training dataset for the SFNO ICLR 2025 paper:
    >>> python data_gen_McWilliams2d.py --num-samples 1152 --batch-size 128 --grid-size 256 --subsample 4 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"

    Testing dataset for plotting the enstrohpy spectrum:
    >>> python data_gen_McWilliams2d.py --num-samples 16 --batch-size 8 --grid-size 256 --subsample 1 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --double

    Training dataset with Re=5k:
    >>> python data_gen_McWilliams2d.py --num-samples 1152 --batch-size 128 --grid-size 512 --subsample 1 --Re 5e3 --dt 5e-4 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"

    """
    args = args.parse_args()

    current_time = datetime.now().strftime("%d_%b_%Y_%Hh%Mm")
    log_name = "".join(os.path.basename(__file__).split(".")[:-1])

    log_filename = os.path.join(LOG_PATH, f"{current_time}_{log_name}.log")
    logger = get_logger(log_filename)

    total_samples = args.num_samples
    batch_size = args.batch_size  # 128
    assert batch_size <= total_samples, "batch_size <= num_samples"
    assert total_samples % batch_size == 0, "total_samples divisible by batch_size"
    n = args.grid_size  # 256
    viscosity = args.visc if args.Re is None else 1 / args.Re
    Re = 1 / viscosity
    dt = args.dt  # 1e-3
    T = args.time  # 10
    subsample = args.subsample  # 4
    ns = n // subsample
    T_warmup = args.time_warmup  # 4.5
    num_snapshots = args.num_steps  # 100
    random_state = args.seed
    peak_wavenumber = args.peak_wavenumber  # 4
    diam = (
        eval(args.diam) if isinstance(args.diam, str) else args.diam
    )  # "2 * torch.pi"
    force_rerun = args.force_rerun

    logger = logging.getLogger()
    logger.info(f"Generating data for McWilliams2d with {total_samples} samples")

    max_velocity = args.max_velocity  # 5
    dt = stable_time_step(diam / n, dt, max_velocity, viscosity=viscosity)
    logger.info(f"Using dt = {dt}")

    warmup_steps = int(T_warmup / dt)
    total_steps = int((T - T_warmup) / dt)
    record_every_iters = int(total_steps / num_snapshots)

    dtype = torch.float64 if args.double else torch.float32
    cdtype = torch.complex128 if args.double else torch.complex64
    dtype_str = "_fp64" if args.double else ""
    filename = args.filename
    if filename is None:
        filename = f"McWilliams2d{dtype_str}_{ns}x{ns}_N{total_samples}_Re{int(Re)}_T{num_snapshots}.pt"
        args.filename = filename
    data_filepath = os.path.join(DATA_PATH, filename)
    if os.path.exists(data_filepath) and not force_rerun:
        logger.info(f"Data already exists at {data_filepath}")
        return
    elif os.path.exists(data_filepath) and force_rerun:
        logger.info(f"Force rerun and save data to {data_filepath}")
        os.remove(data_filepath)
    else:
        logger.info(f"Save data to {data_filepath}")

    cuda = not args.no_cuda and torch.cuda.is_available()
    no_tqdm = args.no_tqdm
    device = torch.device("cuda:0" if cuda else "cpu")

    torch.set_default_dtype(torch.float64)
    logger.info(
        f"Using device: {device} | save dtype: {dtype} | compute dtype: {torch.get_default_dtype()}"
    )

    grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)

    ns2d = NavierStokes2DSpectral(
        viscosity=viscosity,
        grid=grid,
        drag=0,
        smooth=True,
        forcing_fn=None,
        solver=RK4CrankNicholson,
    ).to(device)

    num_batches = total_samples // batch_size
    for i, idx in enumerate(range(0, total_samples, batch_size)):
        logger.info(f"Generate trajectory for batch [{i+1}/{num_batches}]")
        logger.info(
            f"random states: {random_state + idx} to {random_state + idx + batch_size-1}"
        )

        vort_init = torch.stack(
            [
                vorticity_field(
                    grid, peak_wavenumber, random_state=random_state + idx + k
                ).data
                for k in range(batch_size)
            ]
        )
        vort_hat = fft.rfft2(vort_init).to(device)

        with tqdm(total=warmup_steps, disable=no_tqdm) as pbar:
            for j in range(warmup_steps):
                vort_hat, _ = ns2d.step(vort_hat, dt)
                if j % 100 == 0:
                    desc = datetime.now().strftime("%d-%b-%Y %H:%M:%S") + " - Warmup"
                    pbar.set_description(desc)
                    pbar.update(100)

        result = get_trajectory_rk4(
            ns2d,
            vort_hat,
            dt,
            num_steps=total_steps,
            record_every_steps=record_every_iters,
            pbar=not no_tqdm,
            dtype=cdtype,
        )

        for field, value in result.items():
            logger.info(
                f"freq variable:  {field:<12} | shape: {value.shape} | dtype: {value.dtype}"
            )
            value = fft.irfft2(value).real.cpu().to(dtype)
            logger.info(
                f"saved variable: {field:<12} | shape: {value.shape} | dtype: {value.dtype}"
            )
            if subsample > 1:
                result[field] = F.interpolate(value, size=(ns, ns), mode="bilinear")
            else:
                result[field] = value

        result["random_states"] = torch.tensor(
            [random_state + idx + k for k in range(batch_size)], dtype=torch.int32
        )
        logger.info(f"Saving batch [{i+1}/{num_batches}] to {data_filepath}")
        save_pickle(result, data_filepath)
        del result

    pickle_to_pt(data_filepath)
    logger.info(f"Done saving.")
    if args.demo_plots:
        try:
            verify_trajectories(
                data_filepath,
                dt=record_every_iters * dt,
                T_warmup=T_warmup,
                n_samples=1,
            )
        except Exception as e:
            logger.error(f"Error in plotting: {e}")
    return 0


if __name__ == "__main__":
    args = get_args("Meta parameters for generating NSE 2d with McWilliams IV")
    main(args)
