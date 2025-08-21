# -*- coding: utf-8 -*-
"""
Modèles Lotka–Volterra (M1, M2, M3) en une hiérarchie POO unique.

- M1 : Modèle de base Lotka–Volterra.
- M2 : Variante avec changement de variable / rééchelonnage (alpha) et mise à l'échelle (q/r, p/r).
- M3 : Variante logistique (capacité de charge X) côté proies.

Toutes les classes exposent :
    - .modele_Lotka_Volterra(t_min, t_max, h) -> pandas.DataFrame
    - .affichage(t_min, t_max, h) -> None

Auteur original : yan-s
"""

from __future__ import annotations
from typing import Tuple
from numbers import Real, Integral
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configuration d'affichage
rcParams["figure.dpi"] = 300


def _is_int_like(x) -> bool:
    """Vrai si x est un entier (y compris numpy.integer), mais pas un bool."""
    return isinstance(x, Integral) and not isinstance(x, bool)


def _is_real_like(x) -> bool:
    """Vrai si x est un réel (y compris numpy.floating), mais pas un bool."""
    return isinstance(x, Real) and not isinstance(x, bool)


class Lotka_Volterra:
    """
    Modèle de base Lotka–Volterra.

    dx/dt = r*x - p*x*y
    dy/dt = -m*y + q*x*y

    Paramètres
    ----------
    x0, y0 : int
        Populations initiales (entiers non négatifs).
    τ_reproduc_lapins = r : float
        Taux de reproduction intrinsèque des lapins (>= 0).
    τ_mortalit_lapins = p : float
        Taux de mortalité des lapins dû aux rencontres avec les lynx (>= 0).
    τ_mortalit_lynx = m : float
        Taux de mortalité intrinsèque des lynx (>= 0).
    τ_reproduc_lynx = q : float
        Taux de reproduction des lynx proportionnel aux lapins mangés (>= 0).
    """

    # --- Construction / validation -------------------------------------------------

    def __init__(
        self,
        x0: int,
        y0: int,
        τ_reproduc_lapins: float,
        τ_mortalit_lapins: float,
        τ_mortalit_lynx: float,
        τ_reproduc_lynx: float,
    ) -> None:
        # Validation de base (on garde l'intention du code d'origine)
        if not _is_int_like(x0) or x0 < 0:
            raise ValueError("x0 doit être un entier >= 0")
        if not _is_int_like(y0) or y0 < 0:
            raise ValueError("y0 doit être un entier >= 0")
        for name, val in {
            "r": τ_reproduc_lapins,
            "p": τ_mortalit_lapins,
            "m": τ_mortalit_lynx,
            "q": τ_reproduc_lynx,
        }.items():
            if not _is_real_like(val) or float(val) < 0.0:
                raise ValueError(f"{name} doit être un réel >= 0")

        # Stockage (on n'obscurcit pas via __getattr__/__setattr__)
        self.x0: int = int(x0)
        self.y0: int = int(y0)
        self.r: float = float(τ_reproduc_lapins)
        self.p: float = float(τ_mortalit_lapins)
        self.m: float = float(τ_mortalit_lynx)
        self.q: float = float(τ_reproduc_lynx)

    # --- Variations élémentaires ---------------------------------------------------

    def _variation_lapin(self, x: float, y: float) -> float:
        """Variation du nombre de lapins : r*x - p*x*y."""
        return self.r * x - self.p * x * y

    def _variation_lynx(self, x: float, y: float) -> float:
        """Variation du nombre de lynx : -m*y + q*x*y."""
        return -self.m * y + self.q * x * y

    # --- Simulation (Euler explicite, comme l'original) ----------------------------

    def modele_Lotka_Volterra(self, t_min: float, t_max: float, h: float) -> pd.DataFrame:
        """
        Intègre le système par Euler explicite.

        Paramètres
        ----------
        t_min, t_max : bornes de temps
        h : pas (réel > 0)

        Retour
        ------
        DataFrame avec colonnes ['t','x','y'].
        """
        if not (_is_real_like(t_min) and _is_real_like(t_max) and _is_real_like(h)):
            raise ValueError("t_min, t_max, h doivent être des réels")
        if h <= 0:
            raise ValueError("h doit être > 0")
        if t_max < t_min:
            raise ValueError("t_max doit être >= t_min")

        t = float(t_min)
        x = float(self.x0)
        y = float(self.y0)
        rows = [[t, x, y]]

        # On reproduit la logique d'arrêt d'origine : dernière valeur de t <= t_max
        while rows[-1][0] <= t_max:
            t += h
            # Euler explicite : v_{k+1} = v_k + f(v_k)*h
            x += self._variation_lapin(x, y) * h
            y += self._variation_lynx(x, y) * h
            rows.append([t, x, y])

        return pd.DataFrame(rows, columns=["t", "x", "y"])

    # --- Visualisation -------------------------------------------------------------

    def affichage(self, t_min: float, t_max: float, h: float) -> None:
        """
        Trace la population de lapins et de lynx en fonction du temps.

        Garde l'esprit des paramètres et styles d'origine.
        """
        df = self.modele_Lotka_Volterra(t_min, t_max, h)
        plt.plot(df["t"], df["x"], label="Lapins", linestyle="solid")
        plt.plot(df["t"], df["y"], label="Lynx", linestyle="dashed")
        plt.xlabel("Mois")
        plt.ylabel("Population en unité")
        durée = t_max - t_min
        plt.title(
            f"Population de lapins et de lynx au cours du temps — "
            f"{self.x0} lapins et {self.y0} lynx au départ, durée {durée} mois"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        plt.grid()
        plt.xlim(t_min, t_max)
        plt.show()


# --------------------------------------------------------------------------- #
# M2 — Changement de variable / rééchelonnage
# --------------------------------------------------------------------------- #

class Lotka_Volterra_chngt_var(Lotka_Volterra):
    """
    Variante avec changement de variable (paramètre alpha) et mise à l'échelle.

    Dans le code d'origine M2 :
        dv = v - v*w
        dw = -alpha*w + v*w
    et la mise à jour Euler est effectuée avec des facteurs (q/r) et (p/r).

    On reproduit fidèlement ce comportement en redéfinissant la méthode
    d'intégration, tout en gardant la même signature publique.
    """

    def __init__(
        self,
        x0: int,
        y0: int,
        τ_reproduc_lapins: float,
        τ_mortalit_lapins: float,
        τ_mortalit_lynx: float,
        τ_reproduc_lynx: float,
        alpha: float,
    ) -> None:
        super().__init__(x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx)
        if not _is_real_like(alpha):
            raise ValueError("alpha doit être un réel")
        self.alpha: float = float(alpha)

    def _variation_lapin(self, v: float, w: float) -> float:
        """Ici : v - v*w (avant pondération dans l'intégrateur)."""
        return v - v * w

    def _variation_lynx(self, v: float, w: float) -> float:
        """Ici : -alpha*w + v*w (avant pondération dans l'intégrateur)."""
        return -self.alpha * w + v * w

    def modele_Lotka_Volterra(self, t_min: float, t_max: float, h: float) -> pd.DataFrame:
        """
        Intégration avec le même schéma que M2 :
            s = r*t_min
            dv ← dv + (q/r) * f_lapins(v,w) * h
            dw ← dw + (p/r) * f_lynx(v,w) * h
            s ← s + h       # (commentaire `* r` présent dans le code d'origine)
        On conserve la logique de temps 's' pour rester au plus près du script.
        """
        if not (_is_real_like(t_min) and _is_real_like(t_max) and _is_real_like(h)):
            raise ValueError("t_min, t_max, h doivent être des réels")
        if h <= 0:
            raise ValueError("h doit être > 0")
        if t_max < t_min:
            raise ValueError("t_max doit être >= t_min")

        s = self.r * float(t_min)  # conforme à M2
        v = float(self.x0)
        w = float(self.y0)
        rows = [[s, v, w]]

        while rows[-1][0] <= t_max:
            s += h                       # fidèle à M2 (le commentaire mentionnait * r)
            v += (self.q / self.r) * self._variation_lapin(v, w) * h
            w += (self.p / self.r) * self._variation_lynx(v, w) * h
            rows.append([s, v, w])

        return pd.DataFrame(rows, columns=["t", "x", "y"])


# --------------------------------------------------------------------------- #
# M3 — Capacité de charge (logistique) côté lapins
# --------------------------------------------------------------------------- #

class Lotka_Volterra_limité(Lotka_Volterra):
    """
    Variante logistique : la variation des lapins prend en compte une
    capacité de charge X (>= 0).

    dx/dt = r*x*(1 - x/X) - p*x*y
    dy/dt = -m*y + q*x*y
    """

    def __init__(
        self,
        x0: int,
        y0: int,
        τ_reproduc_lapins: float,
        τ_mortalit_lapins: float,
        τ_mortalit_lynx: float,
        τ_reproduc_lynx: float,
        X: int,
    ) -> None:
        super().__init__(x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx)
        if not _is_int_like(X) or X < 0:
            raise ValueError("X doit être un entier >= 0")
        self.X: int = int(X)

    def _variation_lapin(self, x: float, y: float) -> float:
        """Variation logistique pour les lapins."""
        if self.X == 0:
            # Cas limite : si X=0, la croissance logistique est nulle,
            # il ne reste que l'effet de prédation (cohérent mathématiquement).
            return -self.p * x * y
        return self.r * x * (1.0 - (x / float(self.X))) - self.p * x * y


# --------------------------------------------------------------------------- #
# Démonstrations
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Jeux de paramètres d'origine (avec un petit bloc de overrides comme dans M1)
    x: int = 4          # >= 0 -> nombre de lapins
    y: int = 10         # >= 0 -> nombre de lynx
    r: float = 1.5      # >= 0 -> τ de reproduction intrinsèques des lapins
    p: float = 0.05     # >= 0 -> τ de mortalité des lapins due aux lynx rencontrés
    m: float = 0.48     # >= 0 -> τ de mortalité intrinsèques des lynx
    q: float = 0.05     # >= 0 -> τ de reproduction des lynx en f° des lapins mangés

    # Comme dans M1, on montre un override possible :
    if True:
        x = 2
        y = 2
        r = 1.0
        p = 1.0
        m = 1.0
        q = 1.0

    # --- M1 : modèle de base
    ode = Lotka_Volterra(x, y, r, p, m, q)
    ode.affichage(0, 50, 0.0005)

    # --- M2 : changement de variable (alpha)
    a: float = 1.3
    ode2 = Lotka_Volterra_chngt_var(x, y, r, p, m, q, a)
    ode2.affichage(0, 50, 0.0005)

    # --- M3 : capacité de charge X côté lapins
    X: int = 50
    ode3 = Lotka_Volterra_limité(x, y, r, p, m, q, X)
    ode3.affichage(0, 50, 0.0005)
