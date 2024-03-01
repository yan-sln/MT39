# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:03:00 2024

@author: yan-s

majoration2 = abs(exp(z)-exp_approx(N, None, '((z**n)/factorial(n))')) <= ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2))))
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from cmath import exp
from math import factorial

e = lambda x: exp(x)
e = e(1)

def exp_approx(N:int, t:float, fct:str):
    somme = 0
    for n in range(0, N+1):
        somme += (eval(fct)**n)/(factorial(n))
    return somme
  
a = lambda N, z: abs(exp(z)-exp_approx(N, None, f'(({e}**n)/factorial(n))'))
b = lambda N, z: ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2))))

N_max = 121 #question 6 ou 7
N = [i for i in range(0, N_max+1)]

dct = {}
for i in N:
    dct[i] = [a(i, e), b(i, e)]

df = pd.DataFrame([(key, lst[0], lst[1]) for (key, lst) in dct.items()], columns=['N', 'a', 'b'])

# %%

size = 10
#
if False:
    size = 100
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = (size, size)

sns.lineplot(data=df, x='N', y='a', palette="tab10", label="a", linewidth=1)
sns.lineplot(data=df, x='N', y='b', palette="tab10", label="b", linewidth=1)

#plt.xlim(-1,1)
#plt.ylim(-1,1)
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='upper right', fontsize=size)
