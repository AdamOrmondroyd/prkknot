"""
Primordial matter Power spectrum using a flex-knot.
"""

import numpy as np
from cobaya import Theory
from linf import AdaptiveLinf as AdaptiveKnot, Linf as FlexKnot


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

    params = {
        "NPRk": None,
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lgk3": None,
        "lnPR3": None,
        "lgk4": None,
        "lnPR4": None,
        "lgk5": None,
        "lnPR5": None,
        "lgk6": None,
        "lnPR6": None,
        "lgk7": None,
        "lnPR7": None,
        "lnPR8": None,
    }

    def __init__(self, *args, **kwargs):

        self.flexknot = AdaptiveKnot(self.lgkmin, self.lgkmax)
        super().__init__(*args, **kwargs)


class VanillaPRk(PRkKnot):
    def __init__(self, *args, **kwargs):

        self.flexknot = FlexKnot(self.lgkmin, self.lgkmax)
        super().__init__(*args, **kwargs)


# TODO: see if I can just put the params in the constructor to be
# defined with a for loop


class Vanilla1(VanillaPRk):

    params = {
        "lnPR8": None,
    }


class Vanilla2(VanillaPRk):

    params = {
        "lnPR0": None,
        "lnPR8": None,
    }


class Vanilla3(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lnPR8": None,
    }


class Vanilla4(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lnPR8": None,
    }


class Vanilla5(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lgk3": None,
        "lnPR3": None,
        "lnPR8": None,
    }


class Vanilla6(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lgk3": None,
        "lnPR3": None,
        "lgk4": None,
        "lnPR4": None,
        "lnPR8": None,
    }


class Vanilla7(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lgk3": None,
        "lnPR3": None,
        "lgk4": None,
        "lnPR4": None,
        "lgk5": None,
        "lnPR5": None,
        "lnPR8": None,
    }


class Vanilla8(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lgk3": None,
        "lnPR3": None,
        "lgk4": None,
        "lnPR4": None,
        "lgk5": None,
        "lnPR5": None,
        "lgk6": None,
        "lnPR6": None,
        "lnPR8": None,
    }


class Vanilla9(VanillaPRk):

    params = {
        "lnPR0": None,
        "lgk1": None,
        "lnPR1": None,
        "lgk2": None,
        "lnPR2": None,
        "lgk3": None,
        "lnPR3": None,
        "lgk4": None,
        "lnPR4": None,
        "lgk5": None,
        "lnPR5": None,
        "lgk6": None,
        "lnPR6": None,
        "lgk7": None,
        "lnPR7": None,
        "lnPR8": None,
    }
