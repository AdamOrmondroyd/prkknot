import numpy as np
import matplotlib.pyplot as plt
import anesthetic as ac
from anesthetic import NestedSamples
from fgivenx import plot_contours

from linf import Linf as Knot, AdaptiveLinf as AdaptiveKnot
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
ylabel = r"$\ln 10^{10} \mathcal P \mathcal R(k)$"
xscale = "log"
ylim = (1.9, 4.1)
ax_set_kwargs = {"xlabel": "$k$", "ylabel": r"$\ln 10^{10} \mathcal P \mathcal R(k)$", "xscale": "log", "ylim": (1.9, 4.1)}


def plot(samples: NestedSamples, ax=None, resolution=100, colors="Reds_r", title=None, fig=None):

    if ax is None:
        _, _ax = plt.subplots()
    else:
        _ax = ax

    for theory in theory_list:
        if any([key not in samples for key in theory.params.keys()]):
            break
    print(theory)

    ks = np.logspace(theory.lgkmin, theory.lgkmax, resolution)

    cbar = plot_contours(
        theory.prk,
        ks,
        samples[_theory.params.keys()],
        weights=samples.weights,
        ax=_ax,
        colors=colors,
    )
        
    # cbar = fig.colorbar(cbar, ticks=[0, 1, 2, 3], ax=ax, location="right")
    # cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"], fontsize="large")

    _ax.set(xscale=xscale, ylim=ylim, title=title)

    return _ax




