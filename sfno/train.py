# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from pipeline import *
from data import *
from datasets import BochnerDataset
from losses import SobolevLoss
import matplotlib.pyplot as plt
from sfno import SFNO
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_FILES = {
    "fno": {
        "train": "fnodata_extra_64x64_N1280_v1e-3_T50_steps100_alpha2.5_tau7.pt",
        "valid": "fnodata_extra_64x64_N1280_v1e-3_T50_steps100_alpha2.5_tau7.pt",
        "test": "fnodata_extra_fp64_256x256_N16_v1e-3_T50_steps100_alpha2.5_tau7.pt",
    },
    "McWilliams2d": {
        "train": "McWilliams2d_fp32_64x64_N1152_v1e-3_T100.pt",
        "valid": "McWilliams2d_fp32_64x64_N1152_v1e-3_T100.pt",
        "test": "McWilliams2d_fp64_256x256_N16_v1e-3_T100.pt",
    },
}


def main(args):
    
    current_time = datetime.now().strftime("%d_%b_%Y_%Hh%Mm")
    log_name = "".join(os.path.basename(__file__).split(".")[:-1])

    log_filename = os.path.join(LOG_PATH, f"{current_time}_{log_name}.log")
    logger = get_logger(log_filename)
    logger.info(f"Saving log at {log_filename}")


    all_args = {k: v for k, v in vars(args).items() if not callable(v)}
    logger.info("Arguments: "+" | ".join(f"{k}={v}" for k, v in all_args.items()))

    example = args.example
    Ntrain = args.num_samples
    Nval = args.num_val_samples
    Ntest = args.num_test_samples

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    n = args.res
    n_test = args.test_res
    time_steps = args.time_steps
    out_steps = args.out_time_steps
    norm_order = args.norm_order
    fs = args.field

    modes = args.modes
    modes_t = args.modes_t
    width = args.width
    num_layers = args.num_layers
    beta = args.beta
    activation = args.activation
    spatial_padding = args.spatial_padding
    pe_trainable = args.pe_trainable
    spatial_random_feats = args.spatial_random_feats
    lift_activation = not args.lift_linear

    seed = args.seed
    eval_only = args.eval_only
    train_only = args.train_only

    get_seed(seed, quiet=False, logger=logger)

    beta_str = f"{beta:.0e}".replace("e-0", "e-").replace("e+0", "e")
    model_name = f"sfno_ex_{example}_ep{epochs}_m{modes}_w{width}_b{beta_str}.pt"
    path_model = os.path.join(MODEL_PATH, model_name)
    logger.info(f"Save and load model at {path_model}")

    if not eval_only:
        train_file = DATA_FILES[example]["train"]
        val_file = DATA_FILES[example]["valid"]

        train_path = os.path.join(DATA_PATH, train_file)
        val_path = os.path.join(DATA_PATH, val_file)
        logger.info(f"Training: first {Ntrain} samples at {train_path}")
        logger.info(f"Validation: last {Nval} samples at {val_path}")
        logger.info(f"Training and validating on {n}x{n} grid")
        train_dataset = BochnerDataset(
            datapath=train_path,
            n_samples=Ntrain,
            fields=[fs],
            steps=time_steps,
            out_steps=out_steps,
        )
        val_dataset = BochnerDataset(
            datapath=val_path,
            n_samples=Nval,
            fields=[fs],
            steps=time_steps,
            out_steps=out_steps,
            train=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        torch.cuda.empty_cache()
        model = SFNO(modes, modes, modes_t, width, beta,
                     num_spectral_layers=num_layers, 
                     output_steps=out_steps,
                     spatial_padding=spatial_padding,
                     activation=activation,
                     pe_trainable=pe_trainable,
                     spatial_random_feats=spatial_random_feats,
                     lift_activation=lift_activation)
        logger.info(f"Number of parameters: {get_num_params(model)}")
        model.to(device)

        optimizer = get_core_optimizer(args.optimizer)
        optimizer = optimizer(model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            div_factor=1e3,
            final_div_factor=1e4,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )

        loss_func = SobolevLoss(n_grid=n, norm_order=norm_order, relative=True)
        get_config(loss_func, logger=logger)

        for ep in range(epochs):
            model.train()
            train_l2 = 0.0

            with tqdm(train_loader) as pbar:
                t_ep = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                tr_loss_str = f"current train rel L2: 0.0"
                pbar.set_description(f"{t_ep} - Epoch [{ep+1:3d}/{epochs}]  {tr_loss_str:>35}")
                for i, data in enumerate(train_loader):
                    l2 = train_batch_ns(
                        model,
                        loss_func,
                        data,
                        optimizer,
                        device,
                    )
                    train_l2 += l2.item()
                    is_epoch_scheduler = any(
                        s in str(scheduler.__class__) for s in EPOCH_SCHEDULERS
                    )
                    if not is_epoch_scheduler:
                        scheduler.step()

                    if i % 4 == 0:
                        tr_loss_str = f"current train rel L2: {l2.item():.4e}"
                        pbar.set_description(f"{t_ep} - Epoch [{ep+1:3d}/{epochs}]  {tr_loss_str:>35}")
                        pbar.update(4)
            val_l2_min = 1e4
            val_l2 = eval_epoch_ns(
                model,
                loss_func,
                val_loader,
                device,
                out_steps=out_steps,
            )

            if val_l2 < val_l2_min:
                torch.save(model.state_dict(), path_model)
                val_l2_min = val_l2
            tr_loss_str = f"avg train rel L2: {train_l2/len(train_loader):.4e}"
            val_loss_str = f"avg val rel L2: {val_l2:.4e}"
            logger.info(f"Epoch [{ep+1:3d}/{epochs}]  {tr_loss_str:>35}")
            logger.info(f"Epoch [{ep+1:3d}/{epochs}]  {val_loss_str:>35}")

        logger.info(f"{epochs} epochs training complete. Model saved to {path_model}")

    if not train_only:
        test_dtype = torch.float64
        torch.set_default_dtype(test_dtype)
        test_file = DATA_FILES[example]["test"]
        test_path = os.path.join(DATA_PATH, test_file)
        logger.info(f"Testing data: {test_path}")
        logger.info(f"Testing on {n_test}x{n_test} grid")
        logger.info(f"Testing dtype is {torch.get_default_dtype()}")
        test_dataset = BochnerDataset(
            datapath=test_path,
            n_samples=Ntest,
            fields=[fs],
            T_start=30,
            steps=time_steps,
            out_steps=out_steps,
            dtype=test_dtype,
            train=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        torch.cuda.empty_cache()
        model = SFNO(modes, modes, modes_t, width, beta,
                    num_spectral_layers=num_layers, 
                    spatial_padding=spatial_padding,
                    activation=activation,
                    pe_trainable=pe_trainable,
                    spatial_random_feats=spatial_random_feats,
                    lift_activation=lift_activation).to(device)
        model.load_state_dict(torch.load(path_model))
        logger.info(f"Loaded model from {path_model}")
        eval_metric = SobolevLoss(n_grid=n_test, norm_order=norm_order, relative=True)

        test_l2, preds, gt_solns = eval_epoch_ns(
            model,
            eval_metric,
            test_loader,
            device,
            out_steps=out_steps,
            return_output=True,
        )
        logger.info(f"Test L2 on {n_test}x{n_test} grid: {test_l2:.5e}")

        if args.demo_plots > 0:
            try:
                from visualizations import plot_contour_trajectory
                idx = np.random.randint(0, args.num_test_samples)
                im1 = plot_contour_trajectory(
                    preds[idx],
                    num_snapshots=args.demo_plots,
                    T_start=args.time_warmup,
                    dt=args.dt,
                    title="SFNO predictions"
                )
                im2 = plot_contour_trajectory(
                    gt_solns[idx],
                    num_snapshots=args.demo_plots,
                    T_start=args.time_warmup,
                    dt=args.dt,
                    title="Ground truth generated by IMEX"
                )
                plt.show()
            except Exception as e:
                logger.error(f"Error plotting: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFNO")
    parser.add_argument("--example", type=str, default="fno")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-val-samples", type=int, default=64)
    parser.add_argument("--num-test-samples", type=int, default=16)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--test-res", type=int, default=256)
    parser.add_argument("--field", type=str, default="vorticity")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1127825)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--viscosity", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--modes-t", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--spatial-padding", type=int, default=0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--out-time-steps", type=int, default=10)
    parser.add_argument("--time-warmup", type=float, default=4.5)
    parser.add_argument("--dt", type=float, default=5.5 / 100)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="GELU")
    parser.add_argument("--pe-trainable", default=False, action="store_true")
    parser.add_argument("--spatial-random-feats", default=False, action="store_true")
    parser.add_argument("--lift-linear", default=False, action="store_true")
    parser.add_argument("--double", default=False, action="store_true")
    parser.add_argument("--norm-order", type=float, default=0.0)
    parser.add_argument("--eval-only", default=False, action="store_true")
    parser.add_argument("--train-only", default=False, action="store_true")
    parser.add_argument("--demo-plots", type=int, default=0)

    args = parser.parse_args()
    main(args)