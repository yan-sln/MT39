# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:48:45 2024

@author: yan-s
"""
from matplotlib import rcParams
import pandas as pd
import matplotlib.pyplot as plt

rcParams['figure.dpi']:float = 300    # Augmenter la résolution des figures

class Lotka_Volterra:
    """Modèle dynamique par le système Lotka-Volterra."""
    def __init__(self, x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx):
        self.x0 = x0
        self.y0 = y0
        self.r = τ_reproduc_lapins
        self.p = τ_mortalit_lapins
        self.m = τ_mortalit_lynx
        self.q = τ_reproduc_lynx

    def __getattr__(self, name):
        return self.__dict__[f"__{name}"]

    def __setattr__(self, name, value):
        if name in ['x0', 'y0', 'X']:
            if type(value) is not int:
                raise ValueError(f'{name} doit être un entier!')
        else:
            if type(value) is not float:
                raise ValueError(f'{name} doit être un réel!')
        self.__dict__[f"__{name}"] = value

    def _variation_lapin(self, x, y) -> float:
        """Variation du nombre de lapins."""
        return self.r*x - self.p*x*y

    def _variation_lynx(self, x, y) -> float:
        """Variation du nombre de lynx."""
        return -self.m*y + self.q*x*y

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float) -> pd.DataFrame:
        """h: le pas."""
        t = t_min
        dx, dy = self.x0, self.y0
        lst = [[t, dx, dy]]
        while lst[-1][0] <= t_max:
            t += h
            dx += self._variation_lapin(dx, dy)*h
            dy += self._variation_lynx(dx, dy)*h
            lst.append([t, dx, dy])
        return pd.DataFrame(lst, columns=["t", "x", "y"])

    def affichage(self, t_min: int, t_max: int, h: float):
        """Population de lièvres et de lynx en fonction du temps."""
        df = self.modele_Lotka_Volterra(t_min, t_max, h)
        plt.plot(df['t'], df['x'], 'b', label='Lapins', linestyle='solid')
        plt.plot(df['t'], df['y'], 'r', label='Lynx', linestyle='dashed')
        plt.xlabel('Mois'); plt.ylabel('Population en unité')
        plt.title(f'Population de lapins et de lynx au cours du temps\nConditions initiales : {self.x0} lapins pour {self.y0} lynx sur une durée de {t_max-t_min} mois')
        plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        plt.grid(); plt.xlim(t_min, t_max)#; plt.ylim(0,)
        plt.show()


if __name__ == '__main__':
    x: int = 4          # >= 0 -> nombre de lapins
    y: int = 10         # >= 0 -> nombre de lynx
    r: float = 1.5      # >= 0 -> τ de reproduction intrinsèques des lapins
    p: float = 0.05     # >= 0 -> τ de mortalité des lapins due aux lynx rencontrés
    m: float = 0.48     # >= 0 -> τ de mortalité intrinsèques des lynx
    q: float = 0.05     # >= 0 -> τ de reproduction des lynx en f° des lapins mangés

    if True:
        # Data
        x: int = 2           # >= 0 -> nombre de lapins
        y: int = 2           # >= 0 -> nombre de lynx
        r: float = 1.0       # >= 0 -> τ de reproduction intrinsèques des lapins
        p: float = 1.0       # >= 0 -> τ de mortalité des lapins due aux lynx rencontrés
        m: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des lynx
        q: float = 1.0       # >= 0 -> τ de reproduction des lynx en f° des lapins mangés

    ode = Lotka_Volterra(x, y, r, p, m, q)
    ode.affichage(0, 50, 0.0005)
