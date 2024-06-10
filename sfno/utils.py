import copy
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from time import ctime, time
from typing import Generator

import numpy as np
import psutil
import torch
import torch.nn as nn


def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    if printout:
        message = f""""""
        message += f"""
        os.environ['PYTHONHASHSEED'] = str({s})
        numpy.random.seed({s})
        torch.manual_seed({s})
        torch.cuda.manual_seed({s})
        """
        if cudnn:
            message += f"""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False"""

        if torch.cuda.is_available():
            message += f"""
        torch.cuda.manual_seed_all({s})"""
        print("\n")
        print(f"The following code snippets have been run.")
        print("=" * 50)
        print(message)
        print("=" * 50)


class Colors:
    """Defining Color Codes to color the text displayed on terminal."""

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer(label: str = "", compact=False) -> Generator[None, None, None]:
    """
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    print
    1. the time the code block takes to run
    2. the memory usage.
    """
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2.0**30
    start = time()  # Setup - __enter__
    if not compact:
        print(color(f"{label}:\nStart at {ctime(start)};", color=Colors.blue))
        try:
            yield  # yield to body of `with` statement
        finally:  # Teardown - __exit__
            m1 = p.memory_info()[0] / 2.0**30
            delta = m1 - m0
            sign = "+" if delta >= 0 else "-"
            delta = math.fabs(delta)
            end = time()
            print(
                color(
                    f"Done  at {ctime(end)} ({end - start:.6f} secs elapsed);",
                    color=Colors.blue,
                )
            )
            print(color(f"\nLocal RAM usage at START: {m0:.2f} GB", color=Colors.green))
            print(
                color(
                    f"Local RAM usage at END:   {m1:.2f}GB ({sign}{delta:.2f}GB)",
                    color=Colors.green,
                )
            )
            print("\n")
    else:
        yield
        print(
            color(
                f"{label} - done in {time() - start:.6f} seconds. \n", color=Colors.blue
            )
        )


def pretty_tensor_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " x ".join(map(str, size))


def get_size(bytes, suffix="B"):
    """
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MiB'
        1253656678 => '1.17GiB'
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.2f} {unit}{suffix}"
        bytes /= 1024.0
    return f"{bytes:3.2f} 'Yi'{suffix}"


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s:%s%s %s"
                        % (
                            type(obj).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.is_pinned else "",
                            pretty_tensor_size(obj.size()),
                        )
                    )
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    info = (
                        f"{type(obj).__name__} → {type(obj.data).__name__}" + " GPU"
                        if obj.is_cuda
                        else (
                            "" + " pinned"
                            if obj.data.is_pinned
                            else (
                                "" + " grad"
                                if obj.requires_grad
                                else (
                                    "" + " volatile"
                                    if obj.volatile
                                    else "" + f" {pretty_tensor_size(obj.data.size())}"
                                )
                            )
                        )
                    )
                    print(info)
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", get_size(total_size))


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += np.prod(p.size() + (2,) if p.is_complex() else p.size())
    return num_params


def default(value, d):
    """
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value


def clones(module, N):
    """
    Input:
        - module: nn.Module obj
    Output:
        - zip identical N layers (not stacking)

    Refs:
        - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def print_config(model: nn.Module) -> None:
    try:
        for a in model.config.keys():
            print(f"{a:<25}: ", getattr(model, a))
    except:
        config = filter(lambda x: not x.startswith("__"), dir(model))
        for a in config:
            print(f"{a:<25}: ", getattr(model, a))

def check_nan(tensor, tensor_name=""):
    if tensor.isnan().any():
        tensor = tensor[~torch.isnan(tensor)]
        raise ValueError(f"{tensor_name} has nan with norm {torch.linalg.norm(tensor)}")

def get_core_optimizer(name: str):
    """
    ASGD Adadelta Adagrad Adam AdamW Adamax LBFGS NAdam Optimizer RAdam RMSprop Rprop SGD
    """
    import torch.optim as optim
    return getattr(optim, name)

if __name__ == "__main__":
    get_seed(42)
else:
    with timer(f"Loading modules for visualization", compact=True):
        try:
            import plotly.express as px
            import plotly.figure_factory as ff
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError as err:
            sys.stderr.write(f"Error: failed to import module ({err})")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
