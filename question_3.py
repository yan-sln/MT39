# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:19:19 2024

@author: riouyans
"""
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from math import exp

# %% Paramètre d'entrée & fonctions
x = (-4, 4); x_vals = {}
N = {1: None, 2: None, 5: None, 8: None}
intervalle = (-4, 4)
pas = 10e-2     # Correspond à 81 points

# Fonction pour créer des t
def sequence(debut, fin, pas=1):
    n = int(round((fin - debut)/float(pas)))
    if n > 1:
        return([debut + pas*i for i in range(n+1)])
    elif n == 1:
        return([debut])
    else:
        return([])

# %%

# Définie une suite de points espacé de pas dans l'intervalle
sequence_de_t = sequence(intervalle[0], intervalle[1], pas)

#
for element in sequence_de_t:
    x_vals[element] = exp(element)
df_x = pd.DataFrame(x_vals.items())

#
for element in N.keys():
    N[element] = exp(element)
df_N = pd.DataFrame(N.items())

# %%

size = 10
#
if False:
    size = 100
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = (size, size)

sns.lineplot(data=df_x, x=0, y=1, palette="tab10", label="ref", linewidth=1)
sns.lineplot(data=df_N, x=0, y=1, palette="tab10", label="approx", linewidth=1, marker='o')

plt.ylim(1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper left', fontsize=size)
