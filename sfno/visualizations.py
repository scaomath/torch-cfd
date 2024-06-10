import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.fft as fft

import xarray
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_contour(z, func=plt.imshow, **kwargs):
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    _, ax = plt.subplots(figsize=(3, 3))
    f = func(z, cmap=sns.cm.icefire)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.1)
    cbar = plt.colorbar(f, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.locator_params(nbins=9)
    cbar.update_ticks()


def plot_contour_plotly(
    z,
    colorscale="RdYlBu",
    showscale=False,
    showlabels=False,
    continuous_coloring=False,
    reversescale=True,
    dimensions=(200, 200),
    line_smoothing=0.7,
    ncontours=20,
    **plot_kwargs,
):
    """
    show 2D solution z of its contour
    colorscale: balance (MATLAB new) or Jet (MATLAB old)
    """

    if not plot_kwargs:
        plot_kwargs = dict(
            contour_kwargs=dict(
                colorscale=colorscale,
                line_smoothing=line_smoothing,
                line_width=0.1,
                ncontours=ncontours,
                reversescale=reversescale,
                # )
            ),
            figure_kwargs=dict(
                layout={
                    "xaxis": {
                        "title": "x-label",
                        "visible": False,
                        "showticklabels": False,
                    },
                    "yaxis": {
                        "title": "y-label",
                        "visible": False,
                        "showticklabels": False,
                    },
                }
            ),
            layout_kwargs=dict(
                margin=dict(l=0, r=0, t=0, b=0),
                width=dimensions[0],
                height=dimensions[1],
                template="plotly_white",
            ),
        )

    contour_kwargs = plot_kwargs["contour_kwargs"]
    figure_kwargs = plot_kwargs["figure_kwargs"]
    layout_kwargs = plot_kwargs["layout_kwargs"]
    if showscale:
        contour_kwargs["showscale"] = True
        contour_kwargs["colorbar"] = dict(
            thickness=0.15 * layout_kwargs["height"],
            tickwidth=0.3,
            exponentformat="e",
        )
        layout_kwargs["width"] = 1.32 * layout_kwargs["height"]
    else:
        contour_kwargs["showscale"] = False

    if continuous_coloring:
        contour_kwargs["contours_coloring"] = "heatmap"

    if showlabels:
        contour_kwargs["contours"] = dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="gray",
            ),
        )

    uplot = go.Contour(z=z, **contour_kwargs)
    fig = go.Figure(data=uplot, **figure_kwargs)
    if "template" not in layout_kwargs.keys():
        fig.update_layout(template="plotly_dark", **layout_kwargs)
    else:
        fig.update_layout(**layout_kwargs)
    return fig


def get_enstrophy_spectrum(vorticity, h):
    if isinstance(vorticity, np.ndarray):
        vorticity = torch.from_numpy(vorticity)
    n = vorticity.shape[0]
    kx = fft.fftfreq(n, d=h)
    ky = fft.fftfreq(n, d=h)
    kx, ky = torch.meshgrid([kx, ky], indexing="ij")
    kmax = n // 2
    kx = kx[..., : kmax + 1]
    ky = ky[..., : kmax + 1]
    k2 = (4 * torch.pi**2) * (kx**2 + ky**2)
    k2[0, 0] = 1.0

    wh = fft.rfft2(vorticity)

    tke = (0.5 * wh * wh.conj()).real
    kmod = torch.sqrt(k2)
    k = torch.arange(1, kmax, dtype=torch.float64)  # Nyquist limit for this grid
    Ens = torch.zeros_like(k)
    dk = (torch.max(k) - torch.min(k)) / (2 * n)
    for i in range(len(k)):
        Ens[i] += (tke[(kmod < k[i] + dk) & (kmod >= k[i] - dk)]).sum()

    Ens = Ens / Ens.sum()
    return Ens


def plot_enstrophy_spectrum(
    fields: list,
    h=None,
    slope=5,
    factor=None,
    cutoff=1e-15,
    plot_cutoff_factor=1 / 8,
    labels=None,
    title=None,
    legend_loc="upper right",
    fontsize=15,
    subplot_kw={"figsize": (5, 5), "dpi": 100, "facecolor": "w"},
    **kwargs,
):
    for k, field in enumerate(fields):
        if isinstance(field, np.ndarray):
            fields[k] = torch.from_numpy(field)
    if labels is None:
        labels = [f"Field {i}" for i in range(len(fields))]
    n = fields[0].shape[0]
    if h is None:
        h = 1 / n
    kmax = n // 2
    k = torch.arange(1, kmax, dtype=torch.float64)  # Nyquist limit for this grid
    Es = [get_enstrophy_spectrum(field, h) for field in fields]
    if factor is None:
        factor = Es[-1].quantile(0.8) / (k[-1] ** (-slope))
        # print(factor)

    fig, ax = plt.subplots(**subplot_kw)
    plot_cutoff = int(n * plot_cutoff_factor)
    for i, E in enumerate(Es):
        if cutoff is not None:
            E[E < cutoff] = np.nan
        E[-plot_cutoff:] = np.nan
        plt.loglog(k, E, label=f"{labels[i]}")

    plt.loglog(
        k[:-plot_cutoff],
        (factor * k ** (-slope))[:-plot_cutoff],
        "b--",
        label=f"$O(k^{{{-slope:.3g}}})$",
    )
    plt.grid(True, which="both", ls="--", linewidth=0.4)
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.legend(fontsize=fontsize, loc=legend_loc)
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Wavenumber", fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)


def plot_contour_trajectory(
    field,
    num_snapshots=5,
    contourf=False,
    T_start=4.5,
    dt=1e-1,
    cb_kws=dict(orientation="vertical", pad=0.01, aspect=10),
    subplot_kws=dict(
        xticks=[],
        yticks=[],
        ylabel="",
        xlabel="",
    ),
    plot_kws=dict(
        col_wrap=5,
        cmap=sns.cm.icefire,
        robust=True,
        add_colorbar=True,
        xticks=None,
        yticks=None,
        size=3,
        aspect=1,
    ),
    **kwargs,
):
    """
    plot trajectory using xarray's imshow or contourf wrapper
    """
    field = field.detach().cpu().numpy()
    *size, T = field.shape
    grid = np.linspace(0, 1, size[0] + 1)[:-1]
    time = np.arange(T) * dt + T_start
    coords = {
        "x": grid,
        "y": grid,
        "t": time,
    }
    ds = xarray.DataArray(field, dims=["x", "y", "t"], coords=coords)
    t_steps = T // num_snapshots
    ds = ds.thin({"t": t_steps})
    plot_func = ds.plot.contourf if contourf else ds.plot.imshow
    plot_func(
        col="t",
        subplot_kws=subplot_kws,
        cbar_kwargs=cb_kws,
        **plot_kws,
    )
