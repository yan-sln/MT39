# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:35:48 2024

@author: yan-s
"""
from numpy import arange, exp
import matplotlib.pyplot as plt

def Euler(y0, h, f):
    """Approximation par la méthode d'Euler explicite."""
    t = arange(0, 1 + h, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(0, len(t) - 1):
        y[k + 1] = y[k] + h*f(t[k], y[k])
    return t, y

def Heun(y0, h, f):
    """Approximation par la méthode d'Heun."""
    t = arange(0, 1 + h, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(1, len(t)):
        y[k] = y[k-1] + (h/2.0)*(f(t[k-1],y[k-1]) + f(t[k],y[k-1] + h*f(t[k-1],y[k-1])))
    return t, y


if __name__ == '__main__':
    # Paramètres
    f = lambda t, s: exp(-t)
    h = 1 # Pas
    y0 = -1 # Condition initiale

    # Plot Euler
    t, s = Euler(y0, h, f)
    plt.plot(t, s, linestyle='dotted', label='Approximate Euler')
    # Plot Heun
    t, y = Heun(y0, h, f)
    plt.plot(t, y, linestyle='dashed', label='Approximate Heun')
    # Plot Ref
    plt.plot(t, -exp(-t), linestyle='solid', label='Exact')
    # Plot Euler
    plt.xlabel('t'); plt.ylabel('f(t)')
    plt.grid(); plt.legend(loc='lower right')
    plt.show()
