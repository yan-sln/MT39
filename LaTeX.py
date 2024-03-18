# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:35:42 2024

@author: yan-s
"""
# %% En-tête commun à toutes les questions
from cmath import exp, sin, cos
from math import factorial, sqrt
from matplotlib import rcParams
from warnings import filterwarnings
import pandas as pd
import matplotlib.pyplot as plt

filterwarnings("ignore")    # Retirer les avertissements de la console

rcParams['figure.dpi'] = 300    # Augmenter la résolution des figures
plt.rcParams["legend.loc"] = "upper left"   # Titre en haut à gauche

# Paramètres d'entrée
N = [1, 2, 5, 8]
intervalle = (-4, 4)
PAS = 10e-2     # NotaBene: ici correspond à 81 points

def exp_approx(N:int, z:complex)->complex:
    """Somme de (z^n)/(n!), avec n=E([0,N])."""
    return sum((z**n)/(factorial(n)) for n in range(0, N+1))

def exp_approx_func(N:int, t:float, fct:str)->float:
    """Somme de ('func'^n)/(n!), avec n=E([0,N])."""
    return sum((eval(fct)**n)/(factorial(n)) for n in range(0, N+1))

def sequence(debut:int|float, fin:int|float, pas:int|float=1)->list[float]:
    """Créer une liste de 'n = (fin-début)/pas' points."""
    n = int(round((fin - debut)/float(pas)))
    if n > 1: return [debut + pas*i for i in range(n+1)]
    elif n == 1: return [debut]
    else: return []

# Définie une liste de t à l'aide des paramètres d'entrée
sequence_de_t = sequence(intervalle[0], intervalle[1], PAS)

# %% Question 3
FONCTION = 't'  # A rentrer dans exp_approx_func

# On crée un jeu de données pour la fonction exp de référence
df_ref = pd.DataFrame({key: exp(key) for key in sequence_de_t}.items(), columns=['x', 'y'])
plt.plot(df_ref['x'],df_ref['y'], label="ref", linewidth=2, linestyle='dashed')  # On le trace

for nb in N:    # Pour chaque nombre de N
    N_vals = []
    for t in sequence_de_t: # On crée une liste
        N_vals.append([nb, t, exp_approx_func(nb, t, FONCTION)])
    # On crée un de jeu de données à partir de la liste
    df_N = pd.DataFrame(N_vals, columns=['t', 'x', 'y'])
    plt.plot(df_N['x'],df_N['y'], label=f"{df_N['t'][0]}")  # On le trace

# Différentes configurations pour rendre le graphe plus lisible
plt.grid(); plt.ylim(1)
plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.show()

# %% Question 4 ou 5
if False:
    # Question 4
    FONCTION = '(t*1j)'
else:
    # Question 5
    FONCTION = '(t + t*1j)'

def tracer_avec_points(df:pd.DataFrame, label:str, **kwargs):
    """Trace une courbe en plus du premier et dernier point."""
    plt.plot(df['x'],df['y'], label=label, **kwargs)
    # Trace le premier point & dernier point
    ppt, dpt = df.loc[1].values.flatten().tolist()[-2:], df.loc[-1:].values.flatten().tolist()[-2:]
    plt.plot(ppt[0], ppt[1], marker='o', color='black')
    plt.plot(dpt[0], dpt[1], marker='^', color='black')

# On crée un jeu de données pour la fonction exp de référence
t_vals = [[t,(tmp:=exp(eval(FONCTION))).real,tmp.imag] for t in sequence_de_t]
df_ref = pd.DataFrame(t_vals, columns=['t', 'x', 'y'])
tracer_avec_points(df_ref, "ref", linewidth=3, linestyle='dashed')  # On le trace

#
for nb in N:    # Pour chaque nombre de N
    N_vals = []
    for t in sequence_de_t: # On crée une liste
        N_t = exp_approx_func(nb, t, FONCTION)
        N_vals.append([nb, N_t.real, N_t.imag])
    # On crée un de jeu de données à partir de la liste
    df_N = pd.DataFrame(N_vals, columns=['t', 'x', 'y'])
    tracer_avec_points(df_N, f'{nb}')

# Différentes configurations pour rendre le graphe plus lisible
plt.grid(); plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.title('Premier point : •  Dernier point : ▲')
plt.show()

# %% Question 6 et 7
def puissance_nombre(nombre:float)->float:
    """De 1.0e-145, renvoie 145."""
    nombre = str(nombre)
    index_de_e = nombre.find('e')
    return float(nombre[index_de_e+2:])

z= 1
decimal = 121
e = lambda x: exp(x); e = e(1)
a = lambda N, z: abs(exp(z)-exp_approx_func(N, None, f'{z}'))
b = lambda N, z: ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2))))

N_MAX = 0
dct = {}
while (puissance_nombre(b(N_MAX, z))) <= decimal:
    dct[N_MAX] = [a(N_MAX, z), b(N_MAX, z)]
    N_MAX+=1
dct[N_MAX] = [a(N_MAX, z), b(N_MAX, z)]

df = pd.DataFrame([(key,lst[0],lst[1]) for (key, lst) in dct.items()], columns=['N', 'a', 'b'])

# Question 6
print(df.iloc[:21,1:])

# Question 7
print(df.iloc[-1:,1:])

# %% Question 9 et 10
cache = {0: 0, 1: 1}    # Les premiers nombres de la suite
def fibonacci(n:int)->int:
    """Calcul le n terme de la suite de Fibonacci."""
    if n in cache:  # Cas de base
        return cache[n]
    # On calcule et mémorise le nombre de la suite
    cache[n] = fibonacci(n - 1) + fibonacci(n - 2)  # Cas récursif
    return cache[n]

MOIS = 36   # Le nombre de n
suite = [fibonacci(n) for n in range(MOIS+1)]   # Créer la suite de Fibonacci

# On crée un jeu de données pour la suite k->uk
df = pd.DataFrame(cache.items(), columns=['mois', 'couples'])
plt.plot(df['mois'],df['couples'], label='k->uk', linewidth=3, linestyle='dashed')

ρ = (1+sqrt(5))/2 # On pose ρ
print(f'ρ ≃ {ρ} et le nombre d\'or approché vaut {suite[-1]/suite[-2]}')

# Sans corretion, -1 car u1=u2 : 1^0 = 1^1, avec le bon coeff
ρ_FONCTION = ['ρ**k', 'ρ**(k-1)', '(1/sqrt(5))*ρ**k']
for func in ρ_FONCTION: # Trace les différentes courbes
    df_k = pd.DataFrame({k:eval(func) for k in cache.keys()}.items(), columns=['mois', 'couples'])
    plt.plot(df_k['mois'],df_k['couples'], label=func)

# Différentes configurations pour rendre le graphe plus lisible
plt.legend(); plt.xlim(1, MOIS)
plt.show()

# %% Question 11 à 15

#
def tracer(func):
    def wrapper(*args, **kwargs):
        var = func(*args, **kwargs)
        df = pd.DataFrame(dict(enumerate(var)).items(), columns=['x', 'y'])
        plt.plot(df['x'],df['y'], label=f'{func.__name__} y0={kwargs["y0"]}')

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
    lapins(r= 0.5, y0=2, h=1, N=36)
    # Question 12
    lapinsEuler(r= 0.5, y0=2, h=1, N=36)

    for y0 in [2, 50, 100]:
        # Question 14
        lapins2(r= 0.5, Y=50, y0=y0, h=1, N=36)
        # Question 15
        lapinsEuler2(r= 0.5, Y=50, y0=y0, h=1, N=36)

    # Différentes configurations pour rendre le graphe plus lisible
    plt.yscale('log'); plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
    plt.show()
