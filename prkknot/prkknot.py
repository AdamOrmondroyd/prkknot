"""
Primordial matter Power spectrum using a flex-knot.
"""

import numpy as np
from cobaya import Theory
from flexknot import AdaptiveKnot, FlexKnot


class PRkKnot(Theory):
    """
    Abstract base class for P_R(k) flex-knot.

    Requires additional class attribute params, which needs to be
    ordered in the correct structure for a flex-knot, and definition of
    self.flexknot needs to be added to the constructor.
    """

    num_ks = 100
    lgkmin = -4
    lgkmax = -0.3
    n = None
    params = {}

    def initialise(self):
        if self.n >= 2:
            self.params["lnPR0"] = None
        for i in range(1, self.n - 1):
            self.params[f"a{i}"] = None
            self.params[f"lnPR{i}"] = None
        if self.n >= 1:
            self.params["lnPRn"] = None

    def prk(self, theta):
        lgks = np.linspace(self.lgkmin, self.lgkmax, self.num_ks)
        ks = 10**lgks
        Pks = np.exp(self.flexknot(lgks, theta)) * 10**-10

        return ks, Pks

    def calculate(self, state, want_derived=True, **params_values_dict):
        theta = np.array([params_values_dict[p] for p in self.params.keys()])
        ks, Pks = self.prk(theta)
        state["primordial_scalar_pk"] = {
            "kmin": ks[0],
            "kmax": ks[-1],
            "Pk": Pks,
            "log_regular": True,
        }

    def get_primordial_scalar_pk(self):
        return self.current_state["primordial_scalar_pk"]


class Adaptive(PRkKnot):

    n = 9

    def __init__(self, *args, **kwargs):
        self.flexknot = AdaptiveKnot(self.lgkmin, self.lgkmax)
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.params = {"NPRk": None}
        super().initialize()


class VanillaPRk(PRkKnot):
    def __init__(self, *args, **kwargs):
        self.flexknot = FlexKnot(self.lgkmin, self.lgkmax)
        super().__init__(*args, **kwargs)
