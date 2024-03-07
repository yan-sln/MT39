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
#
def fibonacci(n:int):
    if n in cache:  # Cas de base
        return cache[n]
    # On calcule et mémorise le nombre de la suite
    cache[n] = fibonacci(n - 1) + fibonacci(n - 2)  # Cas récursif
    return cache[n]

mois = 36
#
suite = [fibonacci(n) for n in range(mois+1)]

plt.xlim(1, mois)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left')

#
df = pd.DataFrame(cache.items(), columns=['mois', 'couples'])
sns.lineplot(data=df, x='mois', y='couples', palette="tab10", label="couples", linewidth=1)

#
ρ = (1+sqrt(5))/2
print(f'ρ ≃ {ρ} et le nombre d\'or approché vaut {suite[-1]/suite[-2]}')

df_k = pd.DataFrame({k:ρ**k for k in cache.keys()}.items())
sns.lineplot(data=df_k, x=0, y=1, palette="tab10", label="ρ^k", linewidth=1)
