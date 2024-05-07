# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:42:26 2024

@author: yan-s
"""
from LaTeX3 import Lotka_Volterra

from numpy import array, arange
import pandas as pd
import matplotlib.pyplot as plt

def dataframe_3_especes(func):
    """Permet d'obtenir un DataFrame prêt à être utilisé."""
    def wrapper(*args, **kwargs):
        lst_t, lst_xy = func(*args, **kwargs)
        lst = [[t, xy[0], xy[1], xy[2]] for t, xy in zip(lst_t, lst_xy)]
        return pd.DataFrame(lst, columns=["t", "x", "y", "n"])
    return wrapper

@dataframe_3_especes
def Euler(f, y0, h, N, N_min=0):
    """Approximation par la méthode d'Euler explicite."""
    t = arange(N_min, 1 + N, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(N_min, len(t) - 1):
        y[k + 1] = y[k] + h*f(t[k], y[k])
    return t, y

@dataframe_3_especes
def Heun(f, y0, h, N, N_min=0):
    """Approximation par la méthode d'Heun."""
    t = arange(N_min, 1 + N, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(N_min+1, len(t)):
        y[k] = y[k-1] + (h/2.0)*(f(t[k-1],y[k-1]) + f(t[k],y[k-1] + h*f(t[k-1],y[k-1])))
    return t, y


class Lotka_Volterra_influence_homme(Lotka_Volterra):
    """ """
    def __init__(self, x0, y0, r, p, m, q, h):
        super().__init__(x0, y0, r, p, m, q)
        self.h = h  # τ de prélevement de l'homme
        self.orbite_x = (self.m-self.h) / self.q
        self.orbite_y = (self.r-self.h) / self.p

    def _variation_lapin(self, x, y) -> float:
        """Variation du nombre de lapins."""
        return (self.r-self.h)*x - self.p*x*y

    def _variation_lynx(self, x, y) -> float:
        """Variation du nombre de lynx."""
        return -(self.m-self.h)*y + self.q*x*y


class Lotka_Volterra_trois_especes(Lotka_Volterra):
    """ """
    def __init__(self, x0, y0, n0, r, p, m, q, j, k, e):
        super().__init__(x0, y0, r, p, m, q)
        self.n0 = n0
        self.j = j; self.k = k; self.e = e

    def _variation_lapin(self, x, n) -> float:
        """Variation du nombre de carottes."""
        return self.r*x - self.p*x*n

    def _variation_fouine(self, x, n, y) -> float:
        """Variation du nombre de lapins."""
        return self.k*self.p*x*n - self.j*n - self.e*n*y

    def _variation_lynx(self, n, y) -> float:
        """Variation du nombre de lynx."""
        return  self.q*self.e*n*y - self.m*y

    def fonction(self, t:float, Y:list):
        """Fonction."""#!!!
        return array([self._variation_lapin(Y[0], Y[1]), self._variation_fouine(Y[0], Y[1], Y[2]), self._variation_lynx(Y[1], Y[2])])

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float, ode='ref') -> pd.DataFrame:
        """h: le pas."""
        if ode == 'ref':
            t = t_min
            dx, dy, dn = self.x0, self.y0, self.n0
            lst = [[t, dx, dy, dn]]
            while lst[-1][0] <= t_max:
                t += h
                dx += self._variation_lapin(dx, dn)*h
                dn += self._variation_fouine(dx, dn, dy)*h
                dy += self._variation_lynx(dn, dy)*h
                lst.append([t, dx, dy, dn])
            return pd.DataFrame(lst, columns=["t", "x", "y", "n"])
        if ode == 'Euler':
            return Euler(self.fonction, array([self.x0, self.n0, self.y0]), h, t_max, t_min)
        if ode == 'Heun':
            return Heun(self.fonction, array([self.x0, self.n0, self.y0]), h, t_max, t_min)
        raise ValueError(f"{ode} n'est pas un modèle valide parmis 'ref', 'Euler' et 'Heun'!")

    def affichage(self, t_min: int, t_max: int, h: float):
        """Population de lièvres, de fouines et de lynx en fonction du temps."""
        df = self.modele_Lotka_Volterra(t_min, t_max, h)
        fig, ax = plt.subplots()
        ax.plot(df['t'], df['x'], 'b', label='Lapins', linestyle='dotted')
        ax.plot(df['t'], df['n'], 'g', label='Fouines', linestyle='dashdot')
        ax.plot(df['t'], df['y'], 'r', label='Lynx', linestyle='dashed')
        plt.xlabel('Mois'); plt.ylabel('Population en unité')
        plt.title(f'Population de lapins, de fouines et de lynx au cours du temps.\nConditions initiales : {self.x0} lapins, {self.n0} fouines pour {self.y0} lynx\nsur une durée de {t_max-t_min} mois navec un pas de {h}.')
        # Rajoute un texte avec les conditions intiales à partir des params
        text =''; _dict = dict(self.__dict__)
        for key in self.__dict__.keys():
            if key[2:] not in 'xnyrpjkemq': _dict.pop(key)
            else: _dict[key[2:]] = _dict.pop(key)
        for key, value in _dict.items(): text += f'{key} : {value}\n'
        ax.text(.95, .23, text, ha='left', transform=fig.transFigure, fontsize= 12)
        ax.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        ax.grid(); plt.xlim(t_min, t_max)#; plt.ylim(0,)
        plt.show()


if __name__ == '__main__':
    x: int = 2           # >= 0 -> nombre de lapins
    n: int = 2           # >= 0 -> nombre de fouines
    y: int = 2           # >= 0 -> nombre de lynx
    r: float = 2.0       # >= 0 -> τ de reproduction intrinsèques des lapins
    p: float = 1.0       # >= 0 -> τ de mortalité des lapins due aux fouines rencontrés
    j: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des fouines
    k: float = 1.0       # >= 0 -> τ de reproduction des fouines en f° des lapins mangés
    e: float = 1.0       # >= 0 -> τ de mortalité des fouines due aux lynx rencontrés
    m: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des lynx
    q: float = 1.0       # >= 0 -> τ de reproduction des lynx en f° des fouines mangés

    h: float = 1.0       # >= 0 -> τ de prélevement de l'homme

    ode = Lotka_Volterra_trois_especes(x, y, n, r, p, m, q, j, k, e)
    ode.affichage(0, 10, 0.01)

    ode = Lotka_Volterra(x, y, r, p, m, q)
    ode.affichage(0, 10, 0.01)
    
    print(p)
    ode = Lotka_Volterra_influence_homme(x, y, n, r, p, m, h)
    print(ode.__p)
    ode.affichage(0, 10, 0.01)
