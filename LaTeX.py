# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:35:42 2024

@author: yan-s
"""
# %% En-tête commun à toutes les questions
from cmath import exp, sin, cos
from math import factorial, sqrt, floor, log
from matplotlib import rcParams
from warnings import filterwarnings
import pandas as pd
import matplotlib.pyplot as plt

filterwarnings("ignore")    # Retirer les avertissements de la console

rcParams['figure.dpi']:int = 300    # Augmenter la résolution des figures
plt.rcParams["legend.loc"]:str = "upper left"   # Titre en haut à gauche
#rcParams['figure.figsize'] = (size, size)

# Paramètres d'entrée
N:list = [1, 2, 5, 8]
intervalle:tuple = (-4, 4)
PAS:float = 10e-2     # Avec sequence() correspond à 81 points

def exp_approx(N:int, z:complex)->complex:
    """Somme de (z^n)/(n!), avec n=E([0,N])."""
    return sum((z**n)/(factorial(n)) for n in range(0, N+1))

def sequence(debut:int|float, fin:int|float, pas:int|float=1)->list[float]:
    """Créer une liste de 'n = (fin-début)/pas' points."""
    n = int(round((fin - debut)/float(pas)))
    if n > 1: return [debut + pas*i for i in range(n+1)]
    elif n == 1: return [debut]
    else: return []

# %% Question 3
# Définie une liste de x à l'aide des paramètres d'entrée
sequence_de_x:list = sequence(intervalle[0], intervalle[1], PAS)

# On crée un jeu de données pour la fonction exp de référence
#avec une colonne de x de sequence_de_x, et une colonne y exp(x)
df_ref = pd.DataFrame({key: exp(key) for key in sequence_de_x}.items(), columns=['x', 'y'])
plt.plot(df_ref['x'],df_ref['y'], label="ref", linewidth=2, linestyle='dashed')  # On le trace

for nb in N:    # Pour chaque nombre de N = [1, 2, 5, 8], pour faire 4 graphes
    N_vals = []
    for t in sequence_de_x: # On crée une liste de liste [N, x, y]
        N_vals.append([nb, t, exp_approx(nb, t)])
    # On crée un de jeu de données à partir de la liste
    df_N = pd.DataFrame(N_vals, columns=['t', 'x', 'y'])
    plt.plot(df_N['x'],df_N['y'], label=f"{df_N['t'][0]}")  # On le trace

# Différentes configurations pour rendre le graphe plus lisible
plt.grid(); plt.xlim(-4,4); plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.show() # Affiche le graphe

# %% Question 4 ou 5
# Définie une liste de t à l'aide des paramètres d'entrée
sequence_de_t:list = sequence(intervalle[0], intervalle[1], PAS)

# Petit bloc de logique pour choisir entre les questions, changer par False
if True: FONCTION:str = '(t*1j)'    # Question 4
else: FONCTION:str = '(t + t*1j)'   # Question 5

def tracer_avec_points(df:pd.DataFrame, label:str, **kwargs):
    """Trace une courbe en plus du premier et dernier point."""
    plt.plot(df['x'],df['y'], label=label, **kwargs)
    # Trace le premier point & dernier point
    ppt, dpt = df.loc[1].values.flatten().tolist()[-2:], df.loc[-1:].values.flatten().tolist()[-2:]
    plt.plot(ppt[0], ppt[1], marker='o', color='black') # Premier point
    plt.plot(dpt[0], dpt[1], marker='^', color='black') # Dernier point

# On crée une liste, puis un jeu de données pour la fonction exp de référence
t_vals = [[t,(tmp:=exp(eval(FONCTION))).real,tmp.imag] for t in sequence_de_t]
df_ref = pd.DataFrame(t_vals, columns=['t', 'x', 'y'])
tracer_avec_points(df_ref, "ref", linewidth=3, linestyle='dashed')  # On le trace

for nb in N:    # Pour chaque nombre de N = [1, 2, 5, 8], pour faire 4 graphes
    N_vals = []
    for t in sequence_de_t: # On crée une liste de liste [t, x, y]
        N_t = exp_approx(nb, eval(FONCTION))
        N_vals.append([nb, N_t.real, N_t.imag])
    # On crée un de jeu de données à partir de la liste
    df_N = pd.DataFrame(N_vals, columns=['t', 'x', 'y'])
    tracer_avec_points(df_N, f'{nb}')   # On trace avec le premier et dernier pt

# Différentes configurations pour rendre le graphe plus lisible
plt.xlabel('Partie réelle'); plt.ylabel('Partie imaginaire')
plt.grid(); plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.title('Premier point : •  Dernier point : ▲')
plt.show()  # Affiche le graphe

# %% Question 4 cos et sin
# Définie une liste de t à l'aide des paramètres d'entrée
sequence_de_t:list = sequence(intervalle[0], intervalle[1], PAS)
FONCTION = '(t*1j)'     # On reprend la fonction de la question 4
# On fait le choix d'uiliser le même bloc de logique que la question dernière
#plutôt que d'utiliser une classe
if True: func = cos; x, y = 't', 'x'; plt.ylabel('Partie réelle')
else: func = sin; x, y = 't', 'y'; plt.ylabel('Partie imaginaire')

# On crée un jeu de données pour la fonction 'func'
dct = {key: func(key) for key in sequence_de_t}
df = pd.DataFrame(dct.items(), columns=['t', f'{func.__name__}(t)'])
plt.plot(df['t'], df[f'{func.__name__}(t)'], label=f'{func.__name__}', linewidth=4, linestyle='dashed')

# On crée un jeu de données pour la fonction exp de référence
t_vals = [[t,(tmp:=exp(eval(FONCTION))).real,tmp.imag] for t in sequence_de_t]
df_ref = pd.DataFrame(t_vals, columns=['t', 'x', 'y'])
plt.plot(df_ref[x], df_ref[y], label="ref", linewidth=2)    # On le trace

for nb in N:    # Pour chaque nombre de N = [1, 2, 5, 8], pour faire 4 graphes
    N_vals = []
    for t in sequence_de_t: # On crée une liste
        N_t = exp_approx(nb, eval(FONCTION))
        N_vals.append([t, N_t.real, N_t.imag])
    # On crée un de jeu de données à partir de la liste
    df_N = pd.DataFrame(N_vals, columns=['t', 'x', 'y'])
    plt.plot(df_N[x], df_N[y], label=f'{nb}')   # On le trace

# Différentes configurations pour rendre le graphe plus lisible
plt.xlabel('t'); plt.grid(); plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.title(f'{func.__name__}'); plt.show()   # Affiche le graphe

# %% Question 6 et 7
z:float = 1.0   # Exemple d'utilisation : exp(z)
decimale:int = 121   # Décimale cherchée
e = lambda x: exp(x); e = e(1)  # On définit e
a = lambda N, z: abs(exp(z)-exp_approx(N, z))   # Partie droite de la majoration
b = lambda N, z: ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2)))) # Partie gauche de la majoration

def puissance_nombre(nombre:float)->float:
    """De 1.0e-145, renvoie 145."""
    nombre = str(nombre)
    index_de_e = nombre.find('e')
    return float(nombre[index_de_e+2:])

N_MAX = 0   # On initialise le N de la majoration
dct = {}    # Stocker [a, b] en fonction de N pour afficher dans un tableau
while (puissance_nombre(b(N_MAX, z))) <= decimale:  # Permet de trouver le N_MAX
    dct[N_MAX] = [a(N_MAX, z), b(N_MAX, z)]     # En fonction du rang N, a et b
    N_MAX+=1
dct[N_MAX] = [a(N_MAX, z), b(N_MAX, z)] # Une fois supplémentaire pour garantir la majoration
# On convertit le dictionnaire en jeu de données
df = pd.DataFrame([(key,lst[0],lst[1]) for (key, lst) in dct.items()], columns=['N', 'a', 'b'])
print(df.iloc[:21,1:])  # On affiche les 21 premiers termes du jeu de données

N:int = 20
lst_n = list(range(N+1))    # Abscisse de la courbe, implicitement de 0 à N
lst_k = list(floor((-log(b(n, z)))/(log(10))) for n in range(N+1)) # en reprenant l'équation (4)
plt.plot(lst_n, lst_k, label='')    # Trace le nombre décimal k en fonction de N

# Différentes configurations pour rendre le graphe plus lisible
plt.xlabel('N'); plt.ylabel('Nombre de Décimal')
plt.xticks(range(0,N+1,2)); plt.yticks(range(0,N+1,2))
plt.xlim(0, N); plt.ylim(0); plt.grid(); plt.show # Affiche le graphe

print(df.iloc[-1:,1:])  # Affiche a et b au rang N_MAX
print(f"{exp_approx(N_MAX, z):.121f}")  # Affiche les 121 décimales correctes de e

# %% Question 8
N = 50  # Toujours de 0 à 20
y = exp; y0 = y(0); e = y(1)

def euler_approx(y0:float, N:int)->list:
    """ """
    # y0 Vers quoi on veut que la valeur tende # ex si y0=2, ->  exp(1)*2
    cache_yk_rec = {0:y0}   # Créer un cache pour éviter de re-calculer
    def yk_rec(k:float, n:int)->float:
        """Calcul le n terme entré de la suite."""
        if k in cache_yk_rec:  # Cas de base
            return cache_yk_rec[k]
        cache_yk_rec[k] = (1+(y0/n))*yk_rec(k-1, n)  # Cas récursif
        return cache_yk_rec[k]
    # Retourne un tableau de chaque cas
    return list(yk_rec(k, N) for k in range(N+1))

yk_rec = euler_approx(y0, N)[-1]    # Avec la méthode d'Euler récurrente
yn = lambda n: (1+(1/N))**N     # La méthode d'Euler explicite
yn_val = yn(N)  # Autant commencer à 20...

decimale = 10e-5   # 5 au lieu de 4 pour assurer majoration
while abs(e-yn_val) >= decimale:
    yn_val = yn(N); N+=1
print(f'À n = {N-1}, on a e ≃ {yk_rec.real} avec la formule récurrente et e ≃ {yn_val} avec la formule explicite.')

# %% Question 9 et 10
cache = {0: 0, 1: 1}    # Les premiers nombres de la suite
def fibonacci(n:int)->int:
    """Calcul le n terme de la suite de Fibonacci."""
    if n in cache: return cache[n]   # Cas de base
    # Sinon cas récursif, on calcule et mémoïse le nombre de la suite
    cache[n] = fibonacci(n - 1) + fibonacci(n - 2); return cache[n]

MOIS = 36   # Le nombre de n
suite = [fibonacci(n) for n in range(MOIS+1)]   # Créer la suite de Fibonacci
# On crée un jeu de données pour la suite k|->uk
df = pd.DataFrame(cache.items(), columns=['mois', 'couples'])
plt.plot(df['mois'],df['couples'], label='k->uk', linewidth=3, linestyle='dashed')

ρ = (1+sqrt(5))/2 # On définit ρ
print(f'ρ ≃ {ρ} calculé et le nombre d\'or approché vaut {suite[-1]/suite[-2]}')
# [Sans corretion, k-1 car u1=u2 -> 1^0 = 1^1, coeff. avec la formule de Binet]
ρ_FONCTION = ['ρ**k', 'ρ**(k-1)', '(1/sqrt(5))*ρ**k']
for func in ρ_FONCTION: # Trace les différentes courbes
    df_k = pd.DataFrame({k:eval(func) for k in cache.keys()}.items(), columns=['mois', 'couples'])
    plt.plot(df_k['mois'],df_k['couples'], label=func)  # On le trace

# Différentes configurations pour rendre le graphe plus lisible
plt.legend(); plt.xlim(1, MOIS); plt.show()     # Affiche le graphe
