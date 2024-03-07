# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:35:42 2024

@author: yan-s
"""
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cmath import exp
from math import factorial, sqrt
from matplotlib import rcParams

from warnings import filterwarnings
filterwarnings("ignore")

#
size = 10
#
if False:
    size = 100
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = (size, size)

plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["legend.fontsize"] = size

# Paramètre d'entrée
N = [1, 2, 5, 8]
intervalle = (-4, 4)
pas = 10e-2     # Correspond à 81 points

# Fonction approchée de exp
def exp_approx(N:int, z:complex):
    somme = 0
    for n in range(0, N+1):
        somme += (z**n)/(factorial(n))
    return somme

# Fonction pour approximer func
def exp_approx_func(N:int, t:float, fct:str):
    somme = 0
    for n in range(0, N+1):
        somme += (eval(fct)**n)/(factorial(n))
    return somme

# Fonction pour créer des t
def sequence(debut, fin, pas=1):
    n = int(round((fin - debut)/float(pas)))
    if n > 1: return([debut + pas*i for i in range(n+1)])
    elif n == 1: return([debut])
    else: return([])

# Définie une suite de points espacé de pas dans l'intervalle
sequence_de_t = sequence(intervalle[0], intervalle[1], pas)

# %% Question 3

# On crée un DataFrame pour les x
df_x = pd.DataFrame({key: exp(key) for key in sequence_de_t}.items(), columns=['x', 'y'])
# On crée un DataFrame pour les N
df_N = pd.DataFrame({key: exp(key) for key in N}.items(), columns=['x', 'y'])
# On trace les différentes courbes
sns.lineplot(data=df_x, x='x', y='y', palette="tab10", label="ref", linewidth=1)
sns.lineplot(data=df_N, x='x', y='y', palette="tab10", label="approx", linewidth=1, marker='o')

# Différentes configurations pour rendre le graphe plus lisible
plt.ylim(1)
plt.xscale('log')
plt.yscale('log')

# %% Question 4 ou 5

if True:
    # Question 4
    fonction = '(t*1j)'    
else:
    # Question 5
    fonction = '(t + t*1j)'

def tracer(df:pd.DataFrame, label:str):
    sns.lineplot(data=df, x='x', y='y', palette="tab10", label=label, linewidth=1)
    # Affiche le premier point & dernier point
    ppt, dpt = df.loc[1].values.flatten().tolist()[-2:], df.loc[-1:].values.flatten().tolist()[-2:]
    plt.plot(ppt[0], ppt[1], marker='o', color='black')
    plt.plot(dpt[0], dpt[1], marker='^', color='black')
    
#
df_t = pd.DataFrame([[t,(tmp:=exp(eval(fonction))).real,tmp.imag] for t in sequence_de_t], columns=['t', 'x', 'y'])
#    
tracer(df_t, "ref")

#
courbes_N = [pd.DataFrame([[element,(N_t:=exp_approx_func(element,t,fonction)).real,N_t.imag] for t in sequence_de_t], columns=['t', 'x', 'y']) for element in N] 
#
for df_N in courbes_N:
    tracer(df_N, df_N['t'][0])

plt.grid()
#plt.xscale('log')
plt.title('Premier point : •  Dernier point : ▲')

# %% Question 6 et 7

e = lambda x: exp(x); e = e(1)
a = lambda N, z: abs(exp(z)-exp_approx_func(N, None, f'(({e}**n)/factorial(n))'))
b = lambda N, z: ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2))))

N_max = 103
N = [i for i in range(0, N_max+1)]

dct = {}
for i in N:
    dct[i] = [a(i, e), b(i, e)]

df = pd.DataFrame([(key, lst[0], lst[1]) for (key, lst) in dct.items()], columns=['N', 'a', 'b'])

# Question 6
print(df.iloc[:21,1:])

# Question 7
print(df.iloc[-1:,1:])

# %% Question 9 et 10

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

# %% Question 11 à 15

#
def tracer(func):        
    def wrapper(*args, **kwargs):
        var = func(*args, **kwargs)
        df = pd.DataFrame({k: v for k, v in enumerate(var)}.items(), columns=['x', 'y'])
        sns.lineplot(data=df, x='x', y='y', palette="tab10", label=f'{func.__name__} y0={kwargs["y0"]}', linewidth=1)                
    return wrapper


@tracer
def lapins(r, y0:int, h, N:int):
    solutions = []
    for n in range(0, N+1):
        solutions.append(y0*exp(r*n))
    return solutions

@tracer
def lapinsEuler(r, y0:int, h, N):
    solutions = []
    for n in range(0, N+1):
        solutions.append(y0*exp_approx(15, (r*n)))
    return solutions

@tracer
def lapins2(r, Y, y0:int, h, N:int):
    solutions = []
    for t in range(0, N+1):
        solutions.append(Y/(1+((Y/y0)-1)*exp(-r*t)))
    return solutions

@tracer
def lapinsEuler2(r, Y, y0:int, h, N):
    solutions = []
    for t in range(0, N+1):
        solutions.append(Y/(1+((Y/y0)-1)*exp_approx(15, (-r*t))))
    return solutions

@tracer
def lapinsHeun(r, y0:int, h, N:int):
    solutions = []
    for n in range(0, N+1):
        solutions.append(y0*exp(r*n))
    return solutions

@tracer
def lapins2Heun(r, Y, y0:int, h, N):
    solutions = []
    for n in range(0, N+1):
        solutions.append(y0*exp_approx(15, (r*n)))
    return solutions

if __name__ == '__main__':   
    # Question 11
    lapins(r= 0.5, y0=2, h=0, N=36)
    # Question 12
    lapinsEuler(r= 0.5, y0=2, h=0, N=36)
    
    for y0 in [2, 50, 100]:
        # Question 14
        lapins2(r= 0.5, Y=50, y0=y0, h=0, N=36)
        # Question 15
        lapinsEuler2(r= 0.5, Y=50, y0=y0, h=0, N=36)
    
    #
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
    plt.show()
