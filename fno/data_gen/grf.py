import argparse
import math
import os

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRF2d(nn.Module):
    """
    Params: alpha, tau
    Output: Gaussian random field on [0,1]^2 with mean 0
    and covariance operator C = (-Delta + tau^2)^(-alpha)
    where Delta is the Laplacian with zero Neumann boundary conditions.

    Params:
    - alpha: controls the smoothness of the problem
             alpha = 1 is somewhat limiting in 2d
             the closer alpha is to 1, the less smooth the isocurve is

    - tau: controls the high frequency modes of the problem
           the bigger tau is, the more oscillatory the output is
    # TODO: write this as an n-d function
    """

    def __init__(
        self,
        *,
        dim=2,
        n=128,
        alpha=2,
        tau=3,
        device:torch.device=device,
        dtype=torch.float,
        normalize=False,
        smoothing=False,
        **kwargs,
    ):
        self.dim = dim
        self.n = n
        self.device = device
        self.dtype = dtype
        self.normalize = normalize
        self.alpha = alpha
        self.tau = tau
        self.smoothing = smoothing
        self.max_mesh_size = 2048
        self._initialize()

    def _initialize(self, n=None, device=None, alpha=None, tau=None, sigma=None):
        n = self.n if n is None else n
        device = self.device if device is None else device

        alpha = self.alpha if alpha is None else alpha
        tau = self.tau if tau is None else tau
        sigma = tau ** (0.5 * (2 * alpha - self.dim)) if sigma is None else sigma

        k_max = n // 2  # Nyquist freq
        h = 1 / n

        # this is basically fft.fftfreq(n)*n

        kx = fft.fftfreq(n, d=h, device=device)
        ky = fft.fftfreq(n, d=h, device=device)
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")

        sqrt_eig = ((n**self.dim)
            * math.sqrt(2.0)
            * sigma
            * ((4 * (math.pi**2) * (kx**2 + ky**2) + tau**2) ** (-alpha / 2.0))
        )
        sqrt_eig[0, 0] = 0.0
        self.sqrt_eig = sqrt_eig

    def sample(self, bsz, n=None, random_state=0, **kwargs):
        if n is None or n == self.n:
            n = self.n
        elif n != self.n:
            self._initialize(n=n, **kwargs)
        else:
            raise ValueError

        mesh_size = [n for _ in range(self.dim)]
        torch.cuda.manual_seed(random_state)
        torch.random.manual_seed(random_state)
        if self.smoothing:
            # this is smoothing in the frequency domain
            # by interpolating the neighboring frequencies
            max_mesh_size = [self.max_mesh_size for _ in range(self.dim)]
            coeff = torch.randn(bsz, 2, *max_mesh_size, dtype=self.dtype, device=self.device)
        # interpolate needs the channel dimension, and needs real input
        # which we use to represent the real and imaginary parts
            coeff = F.interpolate(coeff, size=mesh_size, mode='bilinear')
        # because coeff is interpolated, need to call contiguous to have stride 1 to use view_as_complex
        # or use the simplified implmentation as follows
        # coeff = coeff.permute(0, 2, 3, 1).contiguous()
        # coeff = torch.view_as_complex(coeff)
        else:
            coeff = torch.randn(bsz, 2, *mesh_size, dtype=self.dtype, device=self.device)
        coeff = coeff[:, 0] + 1j*coeff[:, 1]

        # coeff = fft.fftn(
        #     torch.randn(bsz, *mesh_size, dtype=self.dtype, device=self.device),
        #     dim=list(range(-1, -self.dim - 1, -1)),
        # )

        coeff = self.sqrt_eig * coeff
        s = fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real
        if self.normalize:
            s = s / torch.linalg.norm(s / n, dim=(-1, -2), keepdim=True)
        return s

    def forward(self, x, **kwargs):
        """
        input: (bsz, C, n, n)
        """
        device = x.device
        bsz, _, *mesh_size = x.size()
        n = max(mesh_size)
        return self.sample(bsz, n=n, device=device, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--tau", type=float, default=5)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--smoothing", action="store_true")
    args = parser.parse_args()

    grf = GRF2d(
        n=args.n,
        alpha=args.alpha,
        tau=args.tau,
        device=device,
        normalize=args.normalize,
        smoothing=args.smoothing,
    )

    sample = grf.sample(args.bsz)
    print(sample.shape)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    idxes = torch.randint(0, args.bsz, (min(args.bsz//2, 4),))
    fig, axs = plt.subplots(1, len(idxes), figsize=(5*len(idxes), 5))
    for i, ax in enumerate(axs.flatten()):
        im = ax.imshow(sample[idxes[i]].cpu().numpy(), cmap=sns.cm.icefire)
        ax.set_title(f"GRF sample {idxes[i]}")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.07)
        fig.colorbar(im, cax=cax)
    plt.show()
        
if __name__ == "__main__":
    main()

