# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:07:55 2024

@author: riouyans
"""
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from math import factorial
from cmath import exp

# %% Paramètre d'entrée & fonctions
intervalle = (-4, 4)
N = (1, 2, 5, 8)
pas = 10e-2     # Correspond à 81 points
# Définit la question
if False:
    # Question 4
    fonction = '(t*1j)'    
else:
    # Question 5
    fonction = '(t + t*1j)'

# Fonction pour approximer exp
def exp_approx(N:int, t:float, fct:str):
    somme = 0
    for n in range(0, N+1):
        somme += (eval(fct)**n)/(factorial(n))
    return somme

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
t_vals = []
for t in sequence_de_t:
    t_t = exp(eval(fonction))
    t_vals.append([t, t_t.real, t_t.imag])
df_t = pd.DataFrame(t_vals, columns=['t', 'x', 'y'])

#
courbes_N = [] 
for element in N:
    N_vals = []
    for t in sequence_de_t:
        N_t = exp_approx(element, t, fonction)
        N_vals.append([element, N_t.real, N_t.imag])
    courbes_N.append(pd.DataFrame(N_vals, columns=['t', 'x', 'y']))
    
# %%
sns.lineplot(data=df_t, x='x', y='y', palette="tab10", label="ref", linewidth=1)
for df_N in courbes_N:  #
    sns.lineplot(data=df_N, x='x', y='y', palette="tab10", label=df_N['t'][0], linewidth=1)

# %%
size = 13.2
#
if False:
    size = 100
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = (size, size)
#plt.xlim(-1,1)
#plt.ylim(-1,1)
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='lower right', fontsize=size*0.75)
