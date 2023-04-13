import numpy as np
import matplotlib.pyplot as plt
import anesthetic as ac
from anesthetic import NestedSamples
from fgivenx import plot_contours

from flexknot import FlexKnot, AdaptiveKnot
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


xlabel = r"$k$"
ylabel = r"$\ln{10^{10} \mathcal{P}_\mathcal{R}(k)}$"
xscale = "log"
ylim = (2.0, 4.0)


def plot(
    samples: NestedSamples,
    ax=None,
    resolution=100,
    title=None,
    fig=None,
    **fgivenx_kwargs
):
    if ax is None:
        _, _ax = plt.subplots()
    else:
        _ax = ax

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

    cbar = plot_contours(
        lambda k, theta: theory.flexknot(np.log10(k), theta),
        np.logspace(theory.lgkmin, theory.lgkmax, resolution),
        samples[keys],
        weights=samples.get_weights(),
        ax=_ax,
        **fgivenx_kwargs,
    )

    # cbar = fig.colorbar(cbar, ticks=[0, 1, 2, 3], ax=ax, location="right")
    # cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"], fontsize="large")

    _ax.set(xscale=xscale, ylim=ylim, title=title)
    _ax.set(xlabel=xlabel, ylabel=ylabel)

    return _ax
