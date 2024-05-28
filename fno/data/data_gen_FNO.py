import argparse
import math
import os
from functools import partial

import torch
import torch.fft as fft
import torch.nn.functional as F
from .grf import GRF2d
from .solvers import *
from .data_gen import *


def main():
    """
    Generate the original FNO data
    the right hand side is a fixed forcing
    0.1*(torch.sin(2*math.pi*(x+y))+torch.cos(2*math.pi*(x+y)))

    It stores data after each batch, and will resume using a fixed formula'd seed
    when starting again.
    The default values of the params for the Gaussian Random Field (GRF) are printed.

    Sample usage:

    - Generate new data with 256 grid size with double and extra variables:
    > python ./src/data_gen_FNO.py --sample-size 8 --batch-size 8 --grid-size 256 --double --extra-vars --time 40 --num-steps 100 --dt 1e-3

    """
    current_time = datetime.now().strftime("%d_%b_%Y_%Hh%Mm")
    log_name = "".join(os.path.basename(__file__).split(".")[:-1])

    log_filename = os.path.join(LOG_PATH, f"{current_time}_{log_name}.log")
    logger = get_logger(log_filename)

    args = get_args()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Using the following arguments: ")
    all_args = {k: v for k, v in vars(args).items() if not callable(v)}
    logger.info("\n".join(f"{k}={v}" for k, v in all_args.items()))

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
    visc = args.visc  # 1e-3
    T = args.time  # 50
    delta_t = args.dt  # 1e-4
    alpha = args.alpha  # 2.5
    tau = args.tau  # 7
    f = args.forcing  # FNO's default sin+cos
    dtype = torch.float64 if args.double else torch.float32
    normalize = args.normalize
    filename = args.filename
    force_rerun = args.force_rerun
    replicate_init = args.replicable_init
    dealias = not args.no_dealias
    torch.set_default_dtype(dtype)

    # Number of solutions to generate
    N_samples = args.sample_size  # 8

    # Number of snapshots from solution
    record_steps = args.num_steps

    # Batch size
    bsz = args.batch_size  # 8

    extra = "_extra" if args.extra_vars else ""
    dtype_str = "_fp64" if args.double else ""
    if filename is None:
        filename = (
            f"ns_data{extra}{dtype_str}"
            + f"_N{N_samples}_n{ns}"
            + f"_v{visc:.0e}_T{T}"
            + f"_alpha{alpha:.1f}_tau{tau:.0f}.pt"
        )
        args.filename = filename
    filepath = os.path.join(DATA_PATH, filename)

    data_exist = os.path.exists(filepath)
    if data_exist:
        logger.info(f"\nFile {filename} exists with current data as follows:")
        data = torch.load(filepath)
        for key, v in data.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"{key:<12}", "\t", v.shape)
            else:
                logger.info(f"{key:<12}", "\t", v)
        if len(data[key]) == N_samples:
            return
        if force_rerun:
            logger.info(f"\nRegenerating data and saving in {filename}\n")
    else:
        logger.info(f"\nGenerating data and saving in {filename}\n")

    # Set up 2d GRF with covariance parameters
    # Parameters of covariance C = tau^0.5*(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
    # Note that we need alpha > d/2 (here d= 2)
    grf = GRF2d(
        n=n,
        alpha=alpha,
        tau=tau,
        normalize=normalize,
        device=device,
        dtype=dtype,
    )

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    grid = torch.linspace(0, 1, n + 1, device=device)
    grid = grid[0:-1]

    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    # FNO's original implementation
    # fh = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
    fh = f(X, Y)

    if data_exist and not force_rerun:
        w_init = [x for x in data["a"]]
        w = [x for x in data["u"]]
        w_t = [x for x in data["vort_t"]]
        psi = [x for x in data["stream"]]
        res = [x for x in data["residual"]]
        seeds = [x for x in data["seeds"]]
        N_existing = len(w0)
    else:
        w_init = []
        w = []
        w_t = []
        psi = []
        res = []
        seeds = []
        N_existing = 0

    if N_existing >= N_samples:  # No need to generate more data
        return

    for i in range((N_samples - N_existing) // bsz):
        # Sample random fields
        seed = args.seed + N_existing + i * bsz
        seeds.append(seed)
        if replicate_init:
            w0 = grf.sample(bsz, n_grid_max, random_state=seed)
            w0 = F.interpolate(w0.unsqueeze(1), size=(n, n), mode="nearest")
            w0 = w0.squeeze(1)
        else:
            w0 = grf.sample(bsz, n, random_state=seed)

        result = get_trajectory_imex_crank_nicolson(
            w0,
            fh,
            visc=visc,
            T=T,
            delta_t=delta_t,
            record_steps=record_steps,
            diam=diam,
            dealias=dealias,
            subsample=subsample,
        )

        if not extra:
            for key in ["vort_t", "stream", "res"]:
                result[key] = torch.empty(0, device="cpu")

        w_init.append(w0)
        w.append(result["vorticity"])
        w_t.append(result["vorticity_t"])
        psi.append(result["stream"])
        res.append(result["residual"])

    results = {
        "w0": torch.cat(w_init),
        "w": torch.cat(w),
        "dwdt": torch.cat(w_t),
        "stream": torch.cat(psi),
        "residual": torch.cat(res),
        "t": result["t_steps"],
        "f": fh.cpu(),
        "seeds": seeds,
    }
    torch.save(results, filepath)
    return


if __name__ == "__main__":
    main()
