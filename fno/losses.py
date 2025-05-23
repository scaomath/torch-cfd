import math

import torch
import torch.fft as fft
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.modules.loss import _WeightedLoss


def central_diff(
    u: torch.Tensor,
    h: float = None,
    mode="constant",
    padding=True,
    value=None,
    channel_last=False,
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
        channel_last=False,
        debug=False,
    ):
        super().__init__()
        self.noise = noise
        self.regularizer = regularizer
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.weighted = weighted
        self.channel_last = channel_last
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise * torch.rand_like(targets))
        return targets

    def forward(self, preds, targets, targets_grad=None, K=None, weights=None):
        r"""
        preds: (N, *, n, n)
        targets: (N, *, n, n)
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        """
        batch_size = targets.size(0)
        K = torch.tensor(1) if K is None else K ** (0.5)
        # diffusion constant (N, n, n) or just a constant

        h = self.h if weights is None else weights
        if self.noise > 0:
            targets = self._noise(targets, self.noise)

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
            preds_grad = central_diff(preds, channel_last=self.channel_last)
            preds_grad = torch.cat(preds_grad, dim=1)

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
        n_grid: int = 256,
        time_average: bool = True,  # this is averaging in the Time dimension
        reduction: bool = True,  # this is averaging in the batch dimension
        mesh_weighted: bool = True,  # if True, compute L2 otherwise ell2
        relative: bool = False,
        inp_time_last: bool = True,
        freq_cutoff: int = None,  # max frequency cutoff
        norm_order: float = -1,  # this now can be fractional
        alpha: float = 0.1,
        fft_norm: str = "backward",
        diam: float = 1,
        debug: bool = False,
    ):
        super().__init__()

        self.relative = relative
        self.time_average = time_average
        self.reduction = reduction
        self.mesh_weighted = mesh_weighted
        self.norm_order = norm_order
        self.alpha = alpha
        self.fft_norm = fft_norm
        self.inp_time_last = inp_time_last
        self._fftmesh(n_grid, diam, norm_order, freq_cutoff)

        self.debug = debug

    def __repr__(self):
        if self.norm_order != 0:
            rel_str = (
                f"/||({self.alpha:.1f} - \Delta)^({self.norm_order}/2)u||"
                if self.relative
                else ""
            )
            return (
                f"Sobolev loss in Fourier domain: ||({self.alpha:.1f} - \Delta)^({self.norm_order}/2) (u - v) ||"
                + rel_str
            )
        else:
            rel_str = "/||u||" if self.relative else ""
            return f"Sobolev loss in Fourier domain: ||u - v||" + rel_str

    def _fftmesh(self, n, diam, norm_order, freq_cutoff):
        self.n_grid = n
        kx = fft.fftfreq(n, d=diam / n)
        ky = fft.fftfreq(n, d=diam / n)
        kx, ky = torch.meshgrid([kx, ky], indexing="ij")
        kx, ky = [rearrange(z, "x y -> 1 x y 1") for z in (kx, ky)]
        if freq_cutoff is None:
            freq_cutoff = n // 2 + 1
        freq_cutoff /= diam
        cutoff_val = torch.inf if norm_order < 0 else 0
        kx, ky = [
            z.clone().masked_fill(z.abs() > freq_cutoff, cutoff_val) for z in (kx, ky)
        ]
        weight = self.alpha + 4 * (torch.pi) ** 2 * (kx**2 + ky**2)
        # weight[:, 0, 0, :] = 1.0
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("weight", weight)

    def forward(self, x, y=None):
        """
        x: (bsz, n_grid, n_grid, T)
        y: (bsz, n_grid, n_grid, T)
        """
        fft_kws = {"dim": (1, 2), "norm": self.fft_norm}

        bsz, *_, nt = x.size()
        n = self.n_grid

        if not self.inp_time_last:
            x = rearrange(x, "b t x y -> b x y t")
            if y is not None:
                y = rearrange(y, "b t x y -> b x y t")

        x = torch.fft.fftn(x, **fft_kws)
        x = x.reshape(bsz, n, n, -1)
        weight = self.weight
        weight = torch.sqrt(weight).to(x.device)

        if y is None:
            y = torch.zeros_like(x, device=x.device)
        else:
            y = torch.fft.fftn(y, **fft_kws)
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
        loss = loss / math.sqrt(nt) if self.time_average else loss
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


class ResidualLoss(_WeightedLoss):
    def __init__(
        self,
        batch_size=1,
        alpha=1e-1,
        visc=1e-3,
        n_grid=64,
        n_t=40,
        delta_t=1e-2,
        norm="ortho",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = alpha
        self.visc = visc
        self.n_grid = n_grid
        self.delta_t = delta_t
        self.n_t = n_t
        self.norm = norm
        self._set_spectral_laplacian_spacetime()

    def _set_spectral_laplacian_spacetime(self):
        """
        get the frequency in the spatial and temporal domain
        n: number of grid points in spatial domain
        n_t: number of time steps
        delta_t: time step size
        """
        n, n_t = self.n_grid, self.n_t
        delta_t = self.delta_t
        kx = torch.fft.fftfreq(n, d=1 / n)
        ky = torch.fft.fftfreq(n, d=1 / n)
        kt = torch.fft.fftfreq(n_t, d=delta_t)
        kx, ky, kt = torch.meshgrid([kx, ky, kt], indexing="ij")
        lap = -4 * (torch.pi**2) * (kx**2 + ky**2)
        lap[0, 0] = 1

        self.kx, self.ky, self.kt, self.lap = [
            repeat(z, "x y t -> b x y t", b=self.batch_size) for z in (kx, ky, kt, lap)
        ]

    def forward(self, w, psi=None, f=None):
        """
        inputs are functions defined on spatial grids
        w: (*, n_grid, n_grid, T)
        psi: (*, n_grid, n_grid, T)
        output: same with w
        """

        batch_size, *size = w.shape
        n = size[0]
        n_t = size[-1]
        assert size[0] == size[1]
        visc = self.visc
        kt = self.kt.to(w.device)
        kx = self.kx.to(w.device)
        ky = self.ky.to(w.device)
        lap = self.lap.to(w.device)
        norm = self.norm

        w_h_t = fft.fftn(w, s=size, norm=norm)  # (B, n, n, n_t)
        w_h_t = 2 * torch.pi * kt * 1j * w_h_t
        w_h_t = fft.ifftn(w_h_t, s=size, norm=norm)
        w_h_t = fft.fftn(w_h_t, s=size, norm=norm)

        w_h = fft.fftn(w, s=size, norm=norm)  # (B, n, n, n_t)

        if psi is not None:
            psi_h = fft.fftn(psi, s=size, norm=norm)
        else:
            psi_h = -w_h / lap
            psi = fft.ifftn(psi_h, s=size, norm=norm)
        # Velocity field in x-direction = psi_y
        q = 2 * torch.pi * ky * 1j * psi_h
        q = fft.ifftn(q, s=size, norm=norm)

        # Velocity field in y-direction = -psi_x
        v = -2.0 * torch.pi * kx * 1j * psi_h
        v = fft.ifftn(v, s=size, norm=norm)

        # Partial x of vorticity
        w_x = 2.0 * torch.pi * kx * 1j * w_h
        w_x = fft.ifftn(w_x, s=size, norm=norm)

        # Partial y of vorticity
        w_y = 2.0 * torch.pi * ky * 1j * w_h
        w_y = fft.ifftn(w_y, s=size, norm=norm)
        convection = q * w_x + v * w_y
        convection = fft.fftn(convection, s=size, norm=norm)

        Lap_w = lap * w_h

        if f is None:
            f = ff = torch.zeros_like(w_h, device=w.device)
        else:
            ff = fft.fftn(f, s=size, norm=norm)

        residual = (w_h_t + convection - visc * Lap_w - ff).real
        residual = torch.linalg.norm(residual, dim=(-1, -2)).mean() / n

        return residual
