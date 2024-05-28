from utils import *
import gc
from os import PathLike
from pathlib import Path
from typing import Union

import h5py

import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DATA_PATH
from einops import repeat
from tensordict import TensorDict
from torch.utils.data import Dataset


class UnitGaussianNormalizer(nn.Module):
    def __init__(
        self,
        eps=1e-5,
        data: Union[torch.Tensor, np.ndarray] = None,
    ):
        super().__init__()
        """
        modified from
        https://github.com/neuraloperator/neuraloperator/blob/master/utilities3.py

        - naming convention follows sklearn's Normalizers
        - added different resolution handling
        - implemented as an nn.Module to inherit the .to() method
        """
        self.eps = eps
        self.device = None
        self.data = data
        if data:
            self._fit_transform(data)

    def _set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _fit_transform(self, x):
        mean = torch.as_tensor(x.mean(0))
        std = torch.as_tensor(x.std(0))
        x_transformed = (x - mean) / (std + self.eps)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        return x_transformed

    def fit_transform(self, *args, **kwargs):
        return self._fit_transform(*args, **kwargs)

    def _transform(self, x, align_shapes=False, **kwargs):
        if hasattr(self, "mean"):
            mean, std = self.mean, self.std
            if align_shapes:
                mean, std = self._align_shapes(x, self.mean, self.std, **kwargs)
        else:
            mean, std = 0, 1

        return (x - mean) / (std + self.eps)

    def transform(self, *args, **kwargs):
        return self._transform(*args, **kwargs)

    def inverse_transform(self, x, sample_idx=None, align_shapes=True, **kwargs):
        std = (self.std + self.eps).to(x.device)
        mean = self.mean.to(x.device)
        if align_shapes:
            mean, std = self._align_shapes(x, mean, std, **kwargs)
        if sample_idx is not None:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]
        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def forward(self, *args, **kwargs):
        return self._transform(*args, **kwargs)

    @staticmethod
    def _align_shapes(x, mean, std, **kwargs):
        """
        x: (N, n, n, C) or (N, n, n)
        """
        _, *size = x.shape
        h_x, w_x = size[0], size[1]
        if h_x != mean.shape[0] or w_x != mean.shape[1]:
            mean = mean.permute(2, 0, 1)
            std = std.permute(2, 0, 1)
            mean = F.interpolate(mean[None, ...], size=(h_x, w_x), **kwargs)
            std = F.interpolate(std[None, ...], size=(h_x, w_x), **kwargs)
            mean = mean.permute(0, 2, 3, 1).squeeze(0)
            std = std.permute(0, 2, 3, 1).squeeze(0)
        return mean, std


class SpatialGaussianNormalizer(UnitGaussianNormalizer):
    def __init__(self, eps=1e-7):
        super().__init__(eps=eps)
        """
        normalized by mean and std only in spatial dimensions
        assumes data have shape (N, n, n, T)
        """
        self.device = None

    def _fit_transform(self, x):
        mean = x.mean((0, -1)).unsqueeze(-1)
        std = x.std((0, -1)).unsqueeze(-1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        return (x - mean) / (std + self.eps)


def add_grid_3d(
    data, dim_concat=-1, expand_dim=False, device="cpu", dtype=torch.float  # or 1
):
    """
    pad grid to tensor
    data should have shape
    (N, n, n, T, *)
    the output is
    (N, n, n, T, 3+*)
    *: number of channels (=T_in by default))

    or

    (N, n, n, *)
    the output is
    (N, n, n, *, 3+*)
    """

    N, n, _, T = data.shape[:4]

    if expand_dim:
        dim_repeat = [1 for _ in range(data.ndim + 1)]
        dim_repeat[dim_concat] = T
        data = data.unsqueeze(dim_concat).repeat(dim_repeat)

    gridx = torch.linspace(0, 1, n, device=device, dtype=dtype)
    gridy = torch.linspace(0, 1, n, device=device, dtype=dtype)
    gridt = torch.linspace(0, 1, T, device=device, dtype=dtype)
    gridx, gridy, gridt = torch.meshgrid(gridx, gridy, gridt, indexing="ij")
    grid = torch.stack((gridx, gridy, gridt)).unsqueeze(0)
    if dim_concat == -1:
        grid = grid.permute(0, 2, 3, 4, 1)
    dim_repeat = [1 for _ in range(data.ndim)]
    dim_repeat[0] = N
    data = torch.cat(
        (grid.repeat(dim_repeat), data),
        dim=dim_concat,
    )
    return data


class NavierStokesDataset(Dataset):
    def __init__(
        self,
        data_path,
        N=1000,
        n=64,  # spatial resolution
        subsample=1,  # subsample
        T_init=None,  # input start time steps
        T_in=10,  # input time steps
        T_out=None,  # output start time steps
        T=40,  # output time steps
        dtype=torch.float,
        train=True,
        normalizer_only=False,
        extra_vars=False,
        fix_shapes=False,
        interp_mode="bilinear",
        inp_normalizer_path=None,
        out_normalizer_path=None,
        device="cpu",
        random_state=1127825,
    ):
        """
        PyTorch dataset overhauled for the Navier-Stokes turbulent
        regime data using the vorticity formulation from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        x: input (N, n, n, T_0:T_1)
        pos: x, y coords flattened, (n*n, 2)
        grid: fine grid, x- and y- coords (n, n, 2)
        targets: solution u_h, (N, n, n, T_1:T_2)
        targets_grad: grad_h u_h, (N, n, n, 2, T_1:T_2)

        """
        self.data_path = data_path
        self.n_samples = default(N, 1000 if train else 200)
        self.n_grid = n  # finest resolution along x-, y- dim
        self.subsample = subsample
        self.h = 1 / self.n_grid
        self.train = train
        self.normalizer_only = normalizer_only
        self.extra_vars = extra_vars
        self.fix_shapes = fix_shapes
        self.interp_mode = interp_mode
        self.inp_normalizer_path = inp_normalizer_path
        self.out_normalizer_path = out_normalizer_path
        self.T_init = default(T_init, 0)  # default start time is 0
        self.T_in = T_in
        self.T_out = default(T_out, T_in)
        # default start time for output is the end of input
        self.T = T
        self.random_state = random_state
        self.dtype = dtype
        self.device = device
        self.eps = 1e-8
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        for norm in [self.inp_normalizer_path, self.out_normalizer_path]:
            if norm is not None:
                assert isinstance(norm, os.PathLike), ValueError(
                    "norm path should either be a path to a saved normalizer"
                )
        self._get_data()
        self.a, self.inp_normalizer = self._get_normalizer(
            self.a, self.inp_normalizer_path, inp=True
        )
        self.u, self.out_normalizer = self._get_normalizer(
            self.u, self.out_normalizer_path, inp=False
        )
        self.a = repeat(self.a, "b x y r -> b x y T r", T=self.T)
        self.a = add_grid_3d(
            self.a,
            device=self.device,
            dtype=self.dtype,
        )

    def _load_file(self):
        self.suffix = self.data_path.split(".")[-1]
        if self.suffix == "mat":
            try:
                self.data = sio.loadmat(self.data_path)
                self.old_matlab_flag = True
            except:
                self.data = h5py.File(self.data_path, mode="r")
                self.old_matlab_flag = False
        elif self.suffix == "pt":
            self.data = torch.load(self.data_path)

    def _get_data(self):
        N = self.n_samples
        n = self.n_grid
        s = self.subsample
        T_init = self.T_init
        T_in = self.T_in
        T_out = self.T_out
        T = self.T
        train_flag = "train" if self.train else "test"

        with timer(
            f"Loading NS {train_flag} data from {self.data_path.split('/')[-1]}",
            compact=True,
        ):
            if not self.normalizer_only:
                self._load_file()
                try:
                    self.idx = slice(0, N) if self.train else slice(-N, None)
                    vort = self.read_field("u", idx=self.idx)
                    self.a = vort[:, ::s, ::s, T_init : T_init + T_in]
                    self.u = vort[:, ::s, ::s, T_out : T_out + T]
                    del vort
                    if self.extra_vars:
                        vort_t = self.read_field("vort_t", idx=self.idx)
                        self.at = vort_t[:, ::s, ::s, T_init : T_init + T_in]
                        self.ut = vort_t[:, ::s, ::s, T_out : T_out + T]
                        del vort_t
                        psi = self.read_field("stream", idx=self.idx)
                        self.psi_in = psi[:, ::s, ::s, T_init : T_init + T_in]
                        self.psi_out = psi[:, ::s, ::s, T_out : T_out + T]
                        del psi
                        res = self.read_field("residual", idx=self.idx)
                        self.res = res[:, ::s, ::s, T_out : T_out + T]
                    else:
                        self.at = torch.empty((N,))
                        self.ut = torch.empty((N,))
                        self.psi_in = torch.empty((N,))
                        self.psi_out = torch.empty((N,))
                        self.res = torch.empty((N,))

                except KeyError:
                    raise ValueError(f"Could not find 'u' field in {self.data_path}")
                assert self.n_grid == self.u.shape[-2] == self.u.shape[-3]
                assert self.T == self.u.shape[-1]
            else:
                print("Only loading normalizers, skipping data loading...\n")
                self.a = torch.empty(
                    N, n, n, T_in, device=self.device, dtype=self.dtype
                )
                self.u = torch.empty(N, n, n, T, device=self.device, dtype=self.dtype)
            # self.data = defaultdict(lambda: torch.Tensor())  # release memory
            self.w0 = self.read_field("a", idx=self.idx)
            delattr(self, "data")
            gc.collect()

    def _get_normalizer(self, tensor, normalizer_path, inp=True):
        flag = "inp" if inp else "out"
        train_flag = "train" if self.train else "test"
        with timer(f"Normalizing the {flag} data for {train_flag}", compact=True):
            try:  # isinstance(inp_normalizer_path, os.PathLike):
                normalizer_ = UnitGaussianNormalizer()
                normalizer_._set_params(**torch.load(normalizer_path))
                print(
                    f"{flag} normalizer weights successfully loaded from\n {normalizer_path}\n"
                )

                if self.train or inp:
                    tensor = normalizer_.transform(
                        tensor, align_shapes=self.fix_shapes, mode=self.interp_mode
                    )
            except:  # inp_normalizer_path is None
                normalizer_ = UnitGaussianNormalizer()
                tensor = normalizer_.fit_transform(tensor)

                normalizer_path = (
                    os.path.splitext(os.path.basename(self.data_path))[0]
                    + f"_{flag}_normalizer.pt"
                )
                normalizer_path = Path(os.path.join(DATA_PATH, normalizer_path))
                torch.save(
                    {"mean": normalizer_.mean.cpu(), "std": normalizer_.std.cpu()},
                    normalizer_path,
                )
                print(
                    f"{flag} normalizer weights successfully saved as\n {normalizer_path}\n"
                )
        check_nan(tensor, tensor_name=flag)
        return tensor, normalizer_.to(self.device)

    def read_field(self, field, idx=None):
        x = self.data[field]
        if self.suffix == "mat":
            if not self.old_matlab_flag:
                x = x[()] if idx is None else x[..., idx]
                x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
            else:
                x = x[idx] if idx is not None else x
            x = torch.from_numpy(x)
        elif self.suffix == "pt":
            x = x[idx] if idx is not None else x
        return x.to(self.dtype).to(self.device)

    def __getitem__(self, idx):
        return dict(
            a=self.a[idx].to(self.dtype),
            u=self.u[idx].to(self.dtype),
            at=self.at[idx].to(self.dtype),
            ut=self.ut[idx].to(self.dtype),
            psi_in=self.psi_in[idx].to(self.dtype),
            psi_out=self.psi_out[idx].to(self.dtype),
            res=self.res[idx].to(self.dtype),
        )


class BochnerDataset(Dataset):
    def __init__(
        self,
        datapath: PathLike,
        n_samples: int = 1024,
        train=True,
        fields=["vorticity", "stream"],
        time_last: bool = True,
        steps=10,
        T_start=None,
        normalizer: Union[bool, nn.ModuleDict] = None,
        normalize_space_only: bool = False,
        dtype=torch.float32,
    ):
        """
        data: path to the dictionary
        fieldname: str, vorticity, stream, velocity

        if time_last = True
        input is (N, n, n, T_0:T_1)
        output is (N, n, n, T_1+1:T_2)

        else
        input is (N, T_0:T_1, n, n)
        output is (N, T_1+1:T_2, n, n)
        """
        self.datapath = datapath
        self.n_samples = n_samples
        self.train = train
        self.fields = fields
        self.steps = steps
        self.T_start = T_start
        self.time_last = time_last
        self.normalizer = normalizer
        self.normalize_space_only = normalize_space_only
        self.dtype = dtype
        self._initialize()
        self._normalize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        """
        torch-cfd generates time dimension
        in dim = -3
        """
        data = torch.load(self.datapath)
        N = data[self.fields[0]].size(0)
        self.total_steps = data[self.fields[0]].size(1)
        data = {key: val for key, val in data.items() if key in self.fields}
        data = TensorDict(data, batch_size=N)

        if self.train:
            data = data[: self.n_samples]
        else:
            data = data[-self.n_samples :]

        if self.time_last:
            for key, val in data.items():
                data[key] = val.permute(0, 2, 3, 1)
        self.data = data
        self.data_transformed = data.clone()

    def _normalize(self):
        """
        Normalize the data
        """
        fields = self.fields
        if isinstance(self.normalizer, nn.ModuleDict) and not self.train:
            normalizer = self.normalizer
            for f in fields:
                self.data_transformed[f] = normalizer[f](
                    self.data[f], align_shapes=True
                )
        elif self.train and self.normalizer == True:
            normalizer = nn.ModuleDict()
            for f in fields:
                if self.normalize_space_only:
                    normalizer[f] = SpatialGaussianNormalizer()
                else:
                    normalizer[f] = UnitGaussianNormalizer()
                self.data_transformed[f] = normalizer[f].fit_transform(self.data[f])
            self.normalizer = normalizer
        elif self.normalizer == False:
            self.normalizer = nn.ModuleDict({f: nn.Identity() for f in fields})

    def __getitem__(self, idx, start_idx=None):
        if start_idx is None and self.T_start is None:
            start_idx = np.random.randint(0, self.total_steps - 2 * self.steps - 1)
        elif self.T_start is not None:
            start_idx = self.T_start
        inp_slice = slice(start_idx, start_idx + self.steps)
        out_slice = slice(start_idx + self.steps, start_idx + 2 * self.steps)

        inp = dict()
        out = dict()
        for field in self.fields:
            inp[field] = self.data_transformed[field][idx, ..., inp_slice].to(
                self.dtype
            )
            out[field] = self.data[field][idx, ..., out_slice].to(self.dtype)

        return inp, out


class BochnerDatasetFixed(BochnerDataset):
    def __init__(
        self,
        datapath: PathLike,
        n_samples: int = 1024,
        train=True,
        fields=["vorticity", "stream"],
        time_last: bool = True,
        T_start=None,
        steps=10,
        normalizer: Union[bool, nn.ModuleDict] = None,
        normalize_space_only: bool = False,
        dtype=torch.float32,
    ):
        """
        BochnerDatasetFixed for the Bochner-space like dataset
        but with fixed time steps
        since this pipeline needs to addgrid to the data
        """
        super().__init__(
            datapath=datapath,
            n_samples=n_samples,
            train=train,
            fields=fields,
            time_last=time_last,
            T_start=T_start,
            steps=steps,
            normalizer=normalizer,
            normalize_space_only=normalize_space_only,
            dtype=dtype,
        )
        self._initialize()
        self._normalize()
        self._add_grid()

    def _add_grid(self):
        """
        preset a 3D PE (3, n, n, n_t)
        """
        n, n, n_t = self.data[self.fields[0]].shape[1:]
        gridx = torch.linspace(0, 1, n, dtype=self.dtype)
        gridy = torch.linspace(0, 1, n, dtype=self.dtype)
        gridt = torch.linspace(0, 1, n_t, dtype=self.dtype)
        gridx, gridy, gridt = torch.meshgrid(gridx, gridy, gridt, indexing="ij")
        self.grid = torch.stack((gridx, gridy, gridt))

    def __getitem__(self, idx, start_idx=None):
        if start_idx is None and self.T_start is None:
            start_idx = np.random.randint(0, self.total_steps - 2 * self.steps - 1)
        elif self.T_start is not None:
            start_idx = self.T_start
        inp_slice = slice(start_idx, start_idx + self.steps)
        out_slice = slice(start_idx + self.steps, start_idx + 2 * self.steps)

        inp = dict()
        out = dict()
        for field in self.fields:
            _inp = self.data_transformed[field][idx, ..., inp_slice]
            dim_repeat = [1 for _ in range(_inp.ndim + 1)]
            dim_repeat[0] = self.steps
            _inp = _inp.unsqueeze(0).repeat(dim_repeat)
            inp[field] = torch.cat((self.grid[..., inp_slice], _inp)).to(self.dtype)
            out[field] = self.data[field][idx, ..., out_slice].to(self.dtype)

        return inp, out
