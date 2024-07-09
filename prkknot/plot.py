import re
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import NestedSamples
from fgivenx import plot_contours, plot_lines
from prkknot import prkknot


def plot(
    samples: NestedSamples,
    ax=None,
    resolution=100,
    xlabel=r"$k/\mathrm{Mpc}^{-1}$",
    ylabel=r"$\ln{10^{10} \mathcal{P}_\mathcal{R}(k)}$",
    xscale="log",
    ylim=(2.0, 4.0),
    contours=True,
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

    contours : bool, optional
        use fgivenx.plot_contours, else fgivenx.plot_lines

    color : str, optional
        Color of lines.

    **kwargs : passed to fgivenx.plot_contours or fgivenx.plot_lines

    Returns
    -------
    ax : matplotlib.Axes

    """
    if ax is None:
        _, ax = plt.subplots()

    pattern = re.compile(r"\b(?:lnPR\d+|lgk\d+|lnPRn|NPRk)\b")
    keys = [
        key for key in list(samples.columns.get_level_values(0)) if pattern.match(key)
    ]
    n = max(int(i) for i in re.findall(r"\d+", "".join(keys))) + 2
    # regex matching may pick up the wrong order of keys, so get the correct
    # order from the relevant theory
    if "NPRk" in samples:
        theory = prkknot.Adaptive({"n": n})
        keys = theory.params.keys()
        keys = list(filter(lambda k: k in samples, keys))
    else:
        theory = prkknot.Vanilla({"n": n})
        keys = theory.params.keys()

    if contours:
        plot_contours(
            lambda k, theta: theory.flexknot(np.log10(k), theta),
            np.logspace(theory.lgkmin, theory.lgkmax, resolution),
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            **kwargs,
        )
    else:
        plot_lines(
            lambda k, theta: theory.flexknot(np.log10(k), theta),
            np.logspace(theory.lgkmin, theory.lgkmax, resolution),
            samples[keys],
            weights=samples.get_weights(),
            ax=ax,
            **kwargs,
        )

    ax.set(xscale=xscale, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    return ax
