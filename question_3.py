# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:19:19 2024

@author: riouyans
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cmath import exp

# Paramètre d'entrée
N = [1, 2, 5, 8]
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

# Définie une suite de points espacé de pas dans l'intervalle
sequence_de_t = sequence(intervalle[0], intervalle[1], pas)

# On crée un DataFrame pour les x
df_x = pd.DataFrame({key: exp(key) for key in sequence_de_t}.items())
# On crée un DataFrame pour les N
df_N = pd.DataFrame({key: exp(key) for key in N}.items())
# On trace les différentes courbes
sns.lineplot(data=df_x, x=0, y=1, palette="tab10", label="ref", linewidth=1)
sns.lineplot(data=df_N, x=0, y=1, palette="tab10", label="approx", linewidth=1, marker='o')

# Différentes configurations pour rendre le graphe plus lisible
plt.ylim(1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper left')
