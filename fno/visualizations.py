import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_contour(z, 
                    filename="vorticity", 
                    suffix="svg",
                    func=plt.imshow,
                    **kwargs):
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
            exponentformat = 'e',
        )
        layout_kwargs["width"] = 1.32 * layout_kwargs["height"]
        # layout_kwargs['coloraxis_colorbar'] = dict(thickness=0.32*layout_kwargs['height'])
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
