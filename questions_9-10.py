# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:00:25 2024

@author: yan-s
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt


# Les premiers nombres de la suite
cache = {0: 0, 1: 1}

def fibonacci(n:int):
    if n in cache:  # Cas de base
        return cache[n]
    # On calcule et mémorise le nombre de la suite
    cache[n] = fibonacci(n - 1) + fibonacci(n - 2)  # Cas récursif
    return cache[n]

suite = [fibonacci(n) for n in range(36+1)]
# ou
#print([fibonacci(n) for n in range(36+1)][1:])



#plt.xscale('log')
#plt.yscale('log')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='upper left')

#
df = pd.DataFrame(cache.items())
sns.lineplot(data=df, x=0, y=1, palette="tab10", label="couples", linewidth=1)

#%%

rho = (1+sqrt(5))/2
print(f'rho : {rho}')
print(f'Le nombre d\'or approché est {suite[-1]/suite[-2]}')

k_rho = {}
for k in cache.keys():
    k_rho[k] = rho**k
    
df_k = pd.DataFrame(k_rho.items())
sns.lineplot(data=df_k, x=0, y=1, palette="tab10", label="rho^k", linewidth=1)
