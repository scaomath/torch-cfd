import os
from .utils import default
import numpy as np
import torch
import torch.nn as nn

# from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    model, loss_func, data, optimizer, device, grad_clip=0, fname='vorticity',
    normalizer=None
):
    optimizer.zero_grad()
    a = data[0][fname].to(device)
    u = data[1][fname].to(device)
    out, _ = model(a)

    if normalizer is not None:
        out = normalizer[fname].inverse_transform(out)
        u = normalizer[fname].inverse_transform(u)

    loss = loss_func(out, u)

    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    return loss


def eval_epoch_ns(model, metric_func, valid_loader, device, fname='vorticity', normalizer=None):
    model.eval()
    metric_vals = []

    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            a = data[0][fname].to(device)
            u = data[1][fname].to(device)
            out, _ = model(a)

            if normalizer is not None:
                out = normalizer[fname].inverse_transform(out)
                u = normalizer[fname].inverse_transform(u)
            
            metric_val = metric_func(out, u)

            metric_vals.append(metric_val.item())

    return np.mean(np.asarray(metric_vals), axis=0)