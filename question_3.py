# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:19:19 2024

@author: riouyans
"""
from cmath import exp
from math import factorial
from matplotlib import rcParams
from warnings import filterwarnings
import pandas as pd
import matplotlib.pyplot as plt

filterwarnings("ignore")

rcParams['figure.dpi'] = 300
plt.rcParams["legend.loc"] = "upper left"

# Paramètre d'entrée
N = [1, 2, 5, 8]
intervalle = (-4, 4)
PAS = 10e-2     # Correspond à 81 points

# Fonction pour approximer func
def exp_approx_func(N:int, t:float, fct:str) -> float:
    somme = 0
    for n in range(0, N+1):
        somme += (eval(fct)**n)/(factorial(n))
    return somme

# Fonction pour créer des t
def sequence(debut, fin, pas=1):
    n = int(round((fin - debut)/float(pas)))
    if n > 1: return [debut + pas*i for i in range(n+1)]
    elif n == 1: return [debut]
    else: return []

# Définie une suite de points espacé de pas dans l'intervalle
sequence_de_t = sequence(intervalle[0], intervalle[1], PAS)

FONCTION = 't'
# On crée un DataFrame pour les x
df_x = pd.DataFrame({key: exp(key) for key in sequence_de_t}.items(), columns=['x', 'y'])
# On crée un DataFrame pour les N
df_N = pd.DataFrame({key: exp(key) for key in N}.items(), columns=['x', 'y'])
# On trace les différentes courbes
plt.plot(df_x['x'],df_x['y'], label="ref", linestyle='dashed')

courbes_N = []
for element in N:
    N_vals = []
    for t in sequence_de_t:
        N_t = exp_approx_func(element, t, FONCTION)
        N_vals.append([element, t, N_t])
    courbes_N.append(pd.DataFrame(N_vals, columns=['t', 'x', 'y']))
#
for df_N in courbes_N:
    plt.plot(df_N['x'],df_N['y'], label=f"{df_N['t'][0]}")

# Différentes configurations pour rendre le graphe plus lisible
plt.legend()
plt.ylim(1)
