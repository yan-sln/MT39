# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:35:16 2024

@author: yan-s
"""

from M1_pp_oop import Lotka_Volterra
import pandas as pd

class Lotka_Volterra_chngt_var(Lotka_Volterra):
    def __init__(self, x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx, alpha):
        super().__init__(x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx)
        self.alpha = alpha  # cte
        
    def _variation_lapin(self, v, w) -> float:
        """Variation du nombre de lapins."""
        return v - v*w

    def _variation_lynx(self, v, w) -> float:
        """Variation du nombre de lynx."""
        return -self.alpha*w + v*w

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float) -> pd.DataFrame:
        """Avec chgt de var"""
        s = self.r*t_min
        dv, dw = self.x0, self.y0   #'''Partie à revoir'''
        lst = [[s, dv, dw]]
        while lst[-1][0] <= t_max:
            s += h  # *r
            dv += (self.q/self.r) * self._variation_lapin(dv, dw)*h
            dw += (self.p/self.r) * self._variation_lynx(dv, dw)*h
            lst.append([s, dv, dw])
        return pd.DataFrame(lst, columns=["t", "x", "y"])
    
if __name__ == '__main__':
    x: int = 4         # >= 0 -> nombre de lapins
    y: int = 10         # >= 0 -> nombre de lynx
    r: float = 1.5      # >= 0 -> τ de reproduction intrinsèques des lapins
    p: float = 0.05     # >= 0 -> τ de mortalité des lapins due aux lynx rencontrés
    m: float = 0.48     # >= 0 -> τ de mortalité intrinsèques des lynx
    q: float = 0.05     # >= 0 -> τ de reproduction des lynx en f° des lapins mangés
    a: float = 1.3
    
    ode = Lotka_Volterra_chngt_var(x, y, r, p, m, q, a)
    ode.affichage(0, 50, 0.0005)
