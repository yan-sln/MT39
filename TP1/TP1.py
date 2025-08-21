# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:35:42 2024

@author: yan-s
Ce script illustre plusieurs notions mathématiques :
1. Développement limité et approximation de e^x
2. Comparaison avec cosinus et sinus (via identité d’Euler)
3. Approximation d’Euler et convergence vers e
4. Suite de Fibonacci et rapport d’or
5. Modèles de croissance (logistique, Euler, Heun)

Chaque question correspond à une partie d’exercice illustrée par des tracés.
"""

# %% Import des librairies
from cmath import exp, sin, cos
from math import factorial, sqrt, floor, log
from matplotlib import rcParams
from warnings import filterwarnings
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linspace
from functools import lru_cache

# Paramètres généraux d’affichage
filterwarnings("ignore")    # Retirer les avertissements de la console
rcParams['figure.dpi'] = 300    # Résolution graphique élevée
plt.rcParams["legend.loc"] = "upper left"   # Position par défaut de la légende

# %% Paramètres de simulation
N: list[int] = [1, 2, 5, 8]                # Ordres de développement limité
intervalle: tuple[float, float] = (-4, 4)  # Intervalle d’étude
PAS: float = 1e-2                          # Pas d’échantillonnage

# %% Fonctions utilitaires
def exp_approx(N: int, z: complex) -> complex:
    """Approximation de e^z par somme de la série de Taylor jusqu'à l’ordre N."""
    return sum((z**n)/(factorial(n)) for n in range(0, N+1))

def sequence(debut: float, fin: float, pas: float = 1) -> list[float]:
    """Crée une liste de valeurs de debut à fin avec un pas donné (type 'range' flottant)."""
    n = int(round((fin - debut)/float(pas)))
    if n > 1:
        return [debut + pas*i for i in range(n+1)]
    elif n == 1:
        return [debut]
    else:
        return []

# %% Question 3 : Approximation de e^x
sequence_de_x: list[float] = sequence(intervalle[0], intervalle[1], PAS)

# Valeurs exactes
df_ref = pd.DataFrame({"x": sequence_de_x, "y": [exp(x) for x in sequence_de_x]})
plt.plot(df_ref['x'], df_ref['y'], label="référence", linewidth=2, linestyle='dashed')

# Valeurs approximées par Taylor aux ordres N choisis
for nb in N:
    df_N = pd.DataFrame({"x": sequence_de_x, "y": [exp_approx(nb, t) for t in sequence_de_x]})
    plt.plot(df_N['x'], df_N['y'], label=f"DL ordre {nb}")

plt.grid(); plt.xlim(-4, 4); plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.show()

# %% Question 4 et 5 : Tracé paramétré avec i*t et t+i*t
sequence_de_t: list[float] = sequence(intervalle[0], intervalle[1], PAS)

# Choix des fonctions étudiées
CHOIX_FONCTIONS = {
    "q4": lambda t: t*1j,       # Cas exp(i*t) → cos(t) + i*sin(t)
    "q5": lambda t: t + t*1j    # Cas exp(t+i*t) → croissance + oscillations
}
FONCTION = "q4"   # Sélectionner "q4" ou "q5"

def tracer_avec_points(df: pd.DataFrame, label: str, **kwargs):
    """Trace une courbe avec indication du premier et du dernier point."""
    plt.plot(df['x'], df['y'], label=label, **kwargs)
    # Ajout des marqueurs
    ppt = df.iloc[0][['x', 'y']].tolist()
    dpt = df.iloc[-1][['x', 'y']].tolist()
    plt.plot(ppt[0], ppt[1], marker='o', color='black')  # premier point
    plt.plot(dpt[0], dpt[1], marker='^', color='black')  # dernier point

# Tracé exact
t_vals = [[t, (tmp := exp(CHOIX_FONCTIONS[FONCTION](t))).real, tmp.imag] for t in sequence_de_t]
df_ref = pd.DataFrame(t_vals, columns=['t', 'x', 'y'])
tracer_avec_points(df_ref, "référence", linewidth=3, linestyle='dashed')

# Tracé des approximations
for nb in N:
    N_vals = []
    for t in sequence_de_t:
        N_t = exp_approx(nb, CHOIX_FONCTIONS[FONCTION](t))
        N_vals.append([t, N_t.real, N_t.imag])
    df_N = pd.DataFrame(N_vals, columns=['t', 'x', 'y'])
    tracer_avec_points(df_N, f"ordre {nb}")

plt.xlabel('Partie réelle'); plt.ylabel('Partie imaginaire')
plt.grid(); plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
plt.title('Approximation complexe de exp(z) (• premier, ▲ dernier)')
plt.show()

# %% Question 6 et 7 : Précision et estimation de l’erreur
z: float = 1.0
decimale: int = 121  # nombre de décimales recherchées
e = exp(1)

# a(N,z) = erreur réelle
a = lambda N, z: abs(exp(z)-exp_approx(N, z))
# b(N,z) = majorant théorique de l’erreur (reste de Taylor)
b = lambda N, z: ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2))))

def puissance_nombre(nombre: float) -> float:
    """Retourne l’ordre de grandeur (nombre de décimales exactes)."""
    return -floor(log(nombre, 10))

# Détermination de N optimal pour atteindre la précision demandée
N_MAX = 0
dct = {}
while puissance_nombre(b(N_MAX, z)) <= decimale:
    dct[N_MAX] = [a(N_MAX, z), b(N_MAX, z)]
    N_MAX += 1
dct[N_MAX] = [a(N_MAX, z), b(N_MAX, z)]

df = pd.DataFrame([(key, lst[0], lst[1]) for (key, lst) in dct.items()], columns=['N', 'a', 'b'])
print(df.iloc[:21,1:])

# Tracé du nombre de décimales correctes en fonction de N
N: int = 20
lst_n = list(range(N+1))
lst_k = [floor((-log(b(n, z)))/(log(10))) for n in range(N+1)]
plt.plot(lst_n, lst_k)
plt.xlabel('N'); plt.ylabel('Nombre de décimales correctes')
plt.xticks(range(0,N+1,2)); plt.yticks(range(0,N+1,2))
plt.xlim(0, N); plt.ylim(0); plt.grid(); plt.show()

# Vérification finale
print(df.iloc[-1:,1:])
print(f"{exp_approx(N_MAX, z):.121f}")

# %% Question 8 : Approximation d’Euler pour e
N = 50
y = exp; y0 = y(0); e = y(1)

def euler_approx(y0: float, N: int) -> list[float]:
    """Approximation récurrente d’Euler de e (définition historique)."""
    cache_yk_rec: dict[int,float] = {0:y0}
    def yk_rec(k: int, n: int) -> float:
        if k in cache_yk_rec:
            return cache_yk_rec[k]
        cache_yk_rec[k] = (1+(y0/n))*yk_rec(k-1, n)
        return cache_yk_rec[k]
    return [yk_rec(k, N) for k in range(N+1)]

yk_rec = euler_approx(y0, N)[-1]   # Version récurrente
yn = lambda n: (1+(1/N))**N        # Version explicite
yn_val = yn(N)

# Recherche de la convergence à 1e-5
decimale = 1e-5
while abs(e-yn_val) >= decimale:
    yn_val = yn(N); N+=1
print(f'À n = {N-1}, e ≃ {yk_rec.real} (récurrente) et e ≃ {yn_val} (explicite).')

# %% Question 9 et 10 : Suite de Fibonacci et rapport d’or
@lru_cache
def fibonacci(n: int) -> int:
    """Calcule le n-ième terme de la suite de Fibonacci avec mémoïsation."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

MOIS = 36
suite = [fibonacci(n) for n in range(MOIS+1)]
df = pd.DataFrame({'mois': list(range(MOIS+1)), 'couples': suite})
plt.plot(df['mois'], df['couples'], label='Fibonacci', linewidth=3, linestyle='dashed')

# Vérification du rapport de la suite vers le nombre d’or
ρ = (1+sqrt(5))/2
print(f'ρ ≃ {ρ} et suite[-1]/suite[-2] ≃ {suite[-1]/suite[-2]}')

# Comparaison avec différentes approximations de type ρ^k
ρ_FONCTIONS = {
    "ρ**k": lambda k: ρ**k,
    "ρ**(k-1)": lambda k: ρ**(k-1),
    "(1/sqrt(5))*ρ**k": lambda k: (1/sqrt(5))*ρ**k
}
for name, func in ρ_FONCTIONS.items():
    df_k = pd.DataFrame({"mois": list(range(MOIS+1)), "couples": [func(k) for k in range(MOIS+1)]})
    plt.plot(df_k['mois'], df_k['couples'], label=name)

plt.legend(); plt.xlim(1, MOIS); plt.show()

# %% Question 11 à 16 : Croissance exponentielle et logistique
def euler_explicite(y0: float, n: int) -> float:
    """Formule explicite d’Euler : (1+y0/n)^n."""
    return (1+(y0/n))**n

def tracer(func):
    """Décorateur : trace automatiquement la courbe de sortie d’une fonction."""
    def wrapper(*args, **kwargs):
        x, y = func(*args, **kwargs)
        df = pd.DataFrame({'x':x, 'y':y})
        plt.plot(df['x'],df['y'], label=f'{func.__name__} y0={kwargs.get("y0")}')
    return wrapper

# Croissance exponentielle exacte
def lapins(r: float, y0: int, h: float, N: int):
    t = linspace(0, N*h, N)
    y = [y0*exp(r*h*k) for k in range(len(t))]
    df = pd.DataFrame({'x':t, 'y':y})
    plt.plot(df['x'],df['y'], label=f'exact y0={y0}', linestyle='dashed', linewidth=3)

# Méthode d’Euler
@tracer
def lapinsEuler(r: float, y0: int, h: float, N: int):
    t = linspace(0, N*h, N)
    y = [y0*euler_explicite(r*k, N) for k in t]
    return t, y

# Méthode de Heun
@tracer
def lapinsHeun(r: float, y0: int, h: float, N: int):
    t = linspace(0, N*h, N)
    y = [0]*len(t); y[0] = y0
    for k in range(1, len(t)):
        y[k] = y[k-1]*(1+(r*h)+((1/2)*(r**2)*(h**2)))
    return t, y

# Croissance logistique exacte
def lapins2(r: float, Y: float, y0: int, h: float, N: int):
    t = linspace(0, N*h, N)
    y = [Y/(1+((Y/y0)-1)*exp(-r*h*k)) for k in range(len(t))]
    df = pd.DataFrame({'x':t, 'y':y})
    plt.plot(df['x'],df['y'], label=f'logistique exacte y0={y0}', linestyle='dashed', linewidth=2)

# Logistique Euler
@tracer
def lapinsEuler2(r: float, Y: float, y0: int, h: float, N: int):
    t = linspace(0, N*h, N)
    y = [Y/(1+((Y/y0)-1)*euler_explicite(-r*k, N)) for k in t]
    return t, y

# Logistique Heun
@tracer
def lapins2Heun(r: float, Y: float, y0: int, h: float, N: int):
    t = linspace(0, N*h, N)
    y = [0]*len(t); y[0] = y0
    for k in range(1, len(t)):
        Γ = r*y[k-1]*(1-y[k-1]/Y)
        y[k] = y[k-1]+0.5*h*(Γ+r*(y[k-1]+h*Γ)*(1-(y[k-1]+h*Γ)/Y))
    return t, y

# %% Programme principal
if __name__ == '__main__':
    MOIS = 36; N = 360; h = MOIS/N; y0 = 2; r = 0.5; Y = 50

    # Croissance exponentielle
    if True:
        lapins(r, y0, h, N)
        lapinsEuler(r, y0, h, N)
        lapinsHeun(r, y0, h, N)
        plt.title(f'Croissance exponentielle — Mois: {MOIS}, N: {N}, Pas: {MOIS/N}, y0: {y0}, taux: {r}.')

    # Comparaison avec Fibonacci (désactivée par défaut)
    if False:
        suite = [fibonacci(n) for n in range(MOIS+1)]
        df = pd.DataFrame({'mois':list(range(MOIS+1)), 'couples': suite})
        plt.plot(df['mois'],df['couples'], label='suite de Fibonacci')

    # Croissance logistique
    if False:
        lapins2(r, Y, y0, h, N)
        lapinsEuler2(r, Y, y0, h, N)
        lapins2Heun(r, Y, y0, h, N)
        plt.title(f'Croissance logistique — Mois: {MOIS}, N: {N}, Pas: {MOIS/N}, y0: {y0}, taux: {r}, Seuil: {Y}.')
        plt.ylim(0, 100)

    plt.xlabel("Mois"); plt.ylabel("Nombre d'individus")
    plt.grid(); plt.xlim(1, N*h); plt.legend(); plt.show()
