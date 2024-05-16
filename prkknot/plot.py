import numpy as np
import matplotlib.pyplot as plt
from anesthetic import NestedSamples
from fgivenx import plot_contours, plot_lines
from prkknot import prkknot

theory_list = [
    prkknot.Vanilla1,
    prkknot.Vanilla2,
    prkknot.Vanilla3,
    prkknot.Vanilla4,
    prkknot.Vanilla5,
    prkknot.Vanilla6,
    prkknot.Vanilla7,
    prkknot.Vanilla8,
    prkknot.Vanilla9,
    prkknot.Adaptive,
]


def plot(
    samples: NestedSamples,
    ax=None,
    resolution=100,
    xlabel=r"$k$",
    ylabel=r"$\ln{10^{10} \mathcal{P}_\mathcal{R}(k)}$",
    xscale="log",
    ylim=(2.0, 4.0),
    lines=False,
    **kwargs,
):
    """
    Plot functional posterior of P_R(k) of samples.

    Parameters
    ----------
    samples: NestedSamples
        Samples to plot.

    ax: matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    resolution: int, optional
        Number of points to evaluate the theory at.

    xlabel: str, optional
        Label for x-axis.

    ylabel: str, optional
        Label for y-axis.

    xscale: str, optional
        Scale for x-axis.
        Power spectrum is usually plotted on a log scale.

    ylim: tuple, optional
        Limits for y-axis.

    lines: bool, optional
        Plot lines instead of contours.

    """
    if ax is None:
        _, ax = plt.subplots()

    # special case to allow NPRk column to be added to samples, to treat
    # concatenated Vanilla samples to be treated as Adaptive, even if
    # they don't go up to 9 nodes
    if "NPRk" in samples:
        theory = prkknot.Adaptive()
        keys = theory.params.keys()
        keys = list(filter(lambda k: k in samples, keys))
    else:
        for Theory in theory_list[::-1]:
            if all([key in samples for key in Theory.params.keys()]):
                theory = Theory()
                break
        keys = theory.params.keys()

    if lines:
        plot_lines(
            lambda k, theta: theory.flexknot(np.log10(k), theta),
            np.logspace(theory.lgkmin, theory.lgkmax, resolution),
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            **kwargs,
        )
    else:
        plot_contours(
            lambda k, theta: theory.flexknot(np.log10(k), theta),
            np.logspace(theory.lgkmin, theory.lgkmax, resolution),
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            **kwargs,
        )

    ax.set(xscale=xscale, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    return ax
