import math
import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _WeightedLoss
from einops import rearrange

def central_diff(
    u, h=None, mode="constant", padding=True, value=None, channel_last=True
):
    """
    mode: see
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    input u:
        - (b, n, n) or (b, C, n, n)
        - if has time dim (b, T, n, n) or (b, T, C, n, n)
        - if channel_last:
            (b, n, n, C) or (b, T, n, n, C)
    """
    bsz, *sizes = u.shape
    n = sizes[1] if channel_last else sizes[-1]
    h = 1 / n if h is None else h

    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)

    if channel_last:
        u = u.transpose(-1, -3)

    if padding:
        padding = (1, 1, 1, 1)
        u = F.pad(u, padding, mode=mode, value=value)
    d, s = 2, 1  # dilation and stride
    gradx = (u[..., d:, s:-s] - u[..., :-d, s:-s]) / d  # (*, S_x, S_y)
    grady = (u[..., s:-s, d:] - u[..., s:-s, :-d]) / d  # (*, S_x, S_y)

    if channel_last:
        gradx = gradx.transpose(-3, -1)
        grady = grady.transpose(-3, -1)

    return gradx / h, grady / h

class L2Loss2d(_WeightedLoss):
    def __init__(
        self,
        regularizer=False,
        h=1 / 512,  # mesh size
        beta=1.0,  # L2 u
        gamma=1e-1,  # \|D(N(f)) - Du\|,
        metric_reduction="L1",
        noise=0.0,
        eps=1e-3,
        weighted=False,
        debug=False,
    ):
        super(L2Loss2d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.weighted = weighted
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise * torch.rand_like(targets))
        return targets

    def forward(self, preds, targets, targets_grad=None, K=None, weights=None):
        r"""
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        """
        batch_size = targets.size(0)
        K = torch.tensor(1) if K is None else K ** (0.5)
        # diffusion constant (N, n, n) or just a constant

        h = self.h if weights is None else weights
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        target_norm = targets.pow(2).sum(dim=(1, 2, 3)) + self.eps

        if weights is None and self.weighted:
            inv_L2_norm = 1 / target_norm.sqrt()
            weights = inv_L2_norm / inv_L2_norm.mean()
        elif not self.weighted:
            weights = 1

        loss = (
            self.beta
            * weights
            * ((preds - targets).pow(2)).sum(dim=(1, 2, 3))
            / target_norm
        )

        if targets_grad is not None:
            targets_prime_norm = (
                2 * (K * targets_grad.pow(2)).mean(dim=(1, 2, 3)) + self.eps
            )
        else:
            targets_prime_norm = 1

        if targets_grad is not None and self.gamma > 0:
            preds_grad = central_diff(preds)
            preds_grad = torch.cat(preds_grad, dim=-1)

            grad_diff = (K * (preds_grad - targets_grad)).pow(2)
            loss_prime = self.gamma * grad_diff.mean(dim=(1, 2, 3)) / targets_prime_norm
            loss += loss_prime

        if self.metric_reduction == "L2":
            loss = loss.mean().sqrt()
        elif self.metric_reduction == "L1":  # Li et al paper: first norm then average
            loss = loss.sqrt().mean()
        elif (
            self.metric_reduction == "Linf"
        ):  # sup norm in a batch to approx negative functional norm
            loss = loss.sqrt().max()

        return loss


class LpLoss(_WeightedLoss):
    def __init__(
        self, d=2, p=2, h=None, size_average=True, reduction=True, relative=False
    ):
        super().__init__()

        """
        this is the original loss function used in FNO
        loss function with rel/abs Lp loss
        Dimension and Lp-norm type are postive
        """
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.h = h
        self.reduction = reduction
        self.size_average = size_average
        self.relative = relative

    def abs(self, x, y):
        bsz = x.size(0)

        # Assume uniform mesh
        h = 1.0 / (x.size(1) - 1.0) if self.h is None else self.h
        diff_norms = torch.linalg.norm(x.view(bsz, -1) - y.view(bsz, -1), self.p, 1)
        all_norms = (h ** (self.d / self.p)) * diff_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        bsz = x.size(0)

        diff_norms = torch.linalg.norm(
            x.reshape(bsz, -1) - y.reshape(bsz, -1), self.p, 1
        )
        y_norms = torch.linalg.norm(y.reshape(bsz, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        if self.relative:
            return self.rel(x, y)
        else:
            return self.abs(x, y)


class SobolevLoss(_WeightedLoss):
    def __init__(
        self,
        n_grid=256,
        time_average=False,  # this is averaging in the Time dimension
        reduction=True,  # this is averaging in the batch dimension
        mesh_weighted=True,  # if True, compute L2 otherwise ell2
        relative=False,
        time_last=True,
        freq_cutoff=None,
        norm_order: float = -1,  # this now can be fractional
        diam=1,
        debug=False,
    ):
        super().__init__()
        self.n_grid = n_grid
        k_x = fft.fftfreq(n_grid, d=diam / n_grid)
        k_y = fft.fftfreq(n_grid, d=diam / n_grid)
        k_x, k_y = torch.meshgrid([k_x, k_y], indexing="ij")
        k_x, k_y = [rearrange(z, "x y -> 1 x y 1") for z in (k_x, k_y)]
        if freq_cutoff is not None:
            k_x, k_y = [
                (torch.abs(z) * (torch.abs(z) < freq_cutoff)) * z for z in (k_x, k_y)
            ]
        self.register_buffer("k_x", k_x)
        self.register_buffer("k_y", k_y)
        weight = 4 * (torch.pi) ** 2 * (self.k_x**2 + self.k_y**2)
        weight[:, 0, 0, :] = 1.0
        self.register_buffer("weight", weight)

        self.relative = relative
        self.time_average = time_average
        self.reduction = reduction
        self.mesh_weighted = mesh_weighted
        self.norm_order = norm_order
        self.time_last = time_last
        self.debug = debug

    def forward(self, x, y=None):
        """
        x: (bsz, n_grid, n_grid, T)
        y: (bsz, n_grid, n_grid, T)
        """

        bsz = x.size(0)
        n = self.n_grid

        if not self.time_last:
            x = rearrange(x, "b t x y -> b x y t")
            if y is not None:
                y = rearrange(y, "b t x y -> b x y t")

        x = torch.fft.fftn(x, dim=(1, 2), norm="ortho")
        x = x.reshape(bsz, n, n, -1)
        T = x.size(-1)
        weight = self.weight
        weight = torch.sqrt(weight).to(x.device)

        if y is None:
            y = torch.zeros_like(x, device=x.device)
        else:
            y = torch.fft.fftn(y, dim=(1, 2), norm="ortho")
        y = y.reshape(bsz, n, n, -1)

        w = (weight) ** (self.norm_order / 2) if self.norm_order != 0 else weight
        # when order = 0, this is just L2 norm
        # when order = -1 # dual H^1 norm = \|grad (inv Lap) u\|
        # when order = 1, this is H^1 norm = \|grad u\|
        # when order = 2, this is H^2 norm = \|(Lap)^2 u\|
        # when order = -2, this is H^-2 norm = \|(inv Lap) u\|

        x, y = [z * w for z in (x, y)]

        diff_freq = torch.linalg.norm(x - y, dim=(1, 2))  # (bsz, T)
        if self.relative:
            y2_norms = torch.linalg.norm(y, dim=(1, 2))  # (bsz, T)
            y2_norms = (
                (y2_norms**2).sum(dim=-1).sqrt()
            )  # (bsz,) = (int_0^T |y(t)|^2 dt)^{1/2}
        else:
            y2_norms = torch.ones((bsz,), device=x.device)

        loss = (
            (diff_freq**2).sum(dim=-1).sqrt()
        )  # (bsz,) = (int_0^T |x(t) - y(t)|^2 dt)^{1/2}
        y2_norms = y2_norms / n if self.mesh_weighted else y2_norms
        loss = loss / y2_norms
        loss = loss / math.sqrt(T) if self.time_average else loss
        loss = loss.mean(0) if self.reduction else loss.sum(0)
        loss = loss / n if self.mesh_weighted else loss
        return loss


class BochnerNorm(SobolevLoss):
    """
    computes approx.
    (\int_T \| u \|_{p}^2 dt)**(0.5)
    """

    def __init__(
        self,
        n_grid=256,
        dt: float = None,
        p: int = 2,
        relative=True,
        mesh_weighted=True,
        reduction=True,
        time_average=False,
        time_last=False,
    ):
        super().__init__(
            n_grid=n_grid,
            relative=relative,
            time_last=time_last,
            reduction=reduction,
            mesh_weighted=mesh_weighted,
            time_average=time_average,
        )
        self.dt = dt
        self.p = p

    def forward(self, u):
        """
        x: (bsz, n_grid, n_grid, T)
        """
        n = self.n_grid
        if u.ndim == 3:
            u = u.unsqueeze(0)

        if not self.time_last:
            u = rearrange(u, "b t x y -> b x y t")

        norm_space = u.abs().pow(self.p).sum(dim=(1, 2)) ** (1 / self.p)
        norm_space = norm_space / n if self.mesh_weighted else norm_space
        if self.time_average and self.dt is None:
            norm = ((norm_space**2).mean(dim=-1)).sqrt()
        elif self.dt is not None:
            norm = ((norm_space**2).sum(dim=-1) * self.dt).sqrt()
        norm = norm.mean() if self.reduction else norm.sum()
        return norm