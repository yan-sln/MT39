# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:03:39 2024

@author: yan-s
"""
from M1_pp_oop import Lotka_Volterra

class Lotka_Volterra_limité(Lotka_Volterra):
    def __init__(self, x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx, X):
        super().__init__(x0, y0, τ_reproduc_lapins, τ_mortalit_lapins, τ_mortalit_lynx, τ_reproduc_lynx)
        self.X = X
        
    def _variation_lapin(self, x, y) -> float:
        """Variation du nombre de lapins."""
        return self.r*x*(1-(x/self.X)) - self.p*x*y

if __name__ == '__main__':
    x: int = 4          # >= 0 -> nombre de lapins
    y: int = 10         # >= 0 -> nombre de lynx
    r: float = 1.5      # >= 0 -> τ de reproduction intrinsèques des lapins
    p: float = 0.05     # >= 0 -> τ de mortalité des lapins due aux lynx rencontrés
    m: float = 0.48     # >= 0 -> τ de mortalité intrinsèques des lynx
    q: float = 0.05     # >= 0 -> τ de reproduction des lynx en f° des lapins mangés
    X: int = 50         # >= 0 -> limite du nombre de lapins
    
    ode = Lotka_Volterra_limité(x, y, r, p, m, q, X)
    ode.affichage(0, 50, 0.0005)
