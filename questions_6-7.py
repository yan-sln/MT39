# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:03:00 2024

@author: yan-s

majoration2 = abs(exp(z)-exp_approx(N, None, '((z**n)/factorial(n))')) <= ((abs(z)**(N+1))/(factorial(N+1)))*(1/(1-(abs(z)/(N+2))))
"""
import pandas as pd
from cmath import exp
from math import factorial

def exp_approx(N:int, t:float, fct:str):
    somme = 0
    for n in range(0, N+1):
        somme += (eval(fct)**n)/(factorial(n))
    return somme

e = lambda x: exp(x); e = e(1)
a = lambda N, z: abs(exp(z)-exp_approx(N, None, f'(({e}**n)/factorial(n))'))
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
