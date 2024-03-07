# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:32:10 2024

@author: yan-s
"""
from math import exp, factorial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
def tracer(func):        
    def wrapper(*args, **kwargs):
        var = func(*args, **kwargs)
        df = pd.DataFrame({k: v for k, v in enumerate(var)}.items(), columns=['x', 'y'])
        sns.lineplot(data=df, x='x', y='y', palette="tab10", label=f'{func.__name__} y0={kwargs["y0"]}', linewidth=1)                
    return wrapper

def exp_approx(N:int, z:complex):
    somme = 0
    for n in range(0, N+1):
        somme += (z**n)/(factorial(n))
    return somme

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
    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    
    # Question 11
    lapins(r= 0.5, y0=2, h=0, N=36)
    # Question 12
    lapinsEuler(r= 0.5, y0=2, h=0, N=36)
    
    for y0 in [2, 50, 100]:
        # Question 14
        lapins2(r= 0.5, Y=50, y0=y0, h=0, N=36)
        # Question 15
        lapinsEuler2(r= 0.5, Y=50, y0=y0, h=0, N=36)
