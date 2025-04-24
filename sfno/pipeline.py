import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def default(value, d):
    """
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value


current_path = os.path.abspath(__file__)
SRC_ROOT = os.path.dirname(current_path)
ROOT = os.path.dirname(SRC_ROOT)
MODEL_PATH = default(os.environ.get("MODEL_PATH"), os.path.join(SRC_ROOT, "models"))
LOG_PATH = default(os.environ.get("LOG_PATH"), os.path.join(SRC_ROOT, "logs"))
DATA_PATH = default(os.environ.get("DATA_PATH"), os.path.join(ROOT, "data"))
FIG_PATH = default(os.environ.get("FIG_PATH"), os.path.join(ROOT, "figures"))
for p in [MODEL_PATH, LOG_PATH, DATA_PATH, FIG_PATH]:
    if not os.path.exists(p):
        os.makedirs(p)

EPOCH_SCHEDULERS = [
    "ReduceLROnPlateau",
    "StepLR",
    "MultiplicativeLR",
    "MultiStepLR",
    "ExponentialLR",
    "LambdaLR",
]


def train_batch_ns(
    model,
    loss_func,
    data,
    optimizer,
    device,
    grad_clip=0,
    fname="vorticity",
    normalizer=None,
):
    optimizer.zero_grad()
    a = data[0][fname].to(device)
    u = data[1][fname].to(device)
    out = model(a)
    if normalizer is not None:
        out = normalizer[fname].inverse_transform(out)
        u = normalizer[fname].inverse_transform(u)

    loss = loss_func(out, u)

    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    return loss


def eval_epoch_ns(
    model,
    metric_func,
    valid_loader,
    device,
    fname="vorticity",
    out_steps=None,
    normalizer=None,
    return_output=False,
):
    model.eval()
    metric_vals = []
    preds = []
    targets = []

    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            a = data[0][fname].to(device)
            u = data[1][fname].to(device)
            out = model(a, out_steps=out_steps)

            if normalizer is not None:
                out = normalizer[fname].inverse_transform(out)
                u = normalizer[fname].inverse_transform(u)

            if return_output:
                preds.append(out.cpu())
                targets.append(u.cpu())

            metric_val = metric_func(out, u)
            metric_vals.append(metric_val.item())

    metric = np.mean(np.asarray(metric_vals), axis=0)

    if return_output:
        return metric, torch.cat(preds), torch.cat(targets)
    else:
        return metric
