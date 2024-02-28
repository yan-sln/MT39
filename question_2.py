# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:07:55 2024

@author: riouyans
"""
from math import factorial

def exp_approx(N:int, z:complex):
    somme = 0
    for n in range(0, N+1):
        somme += (z**n)/(factorial(n))
    return somme

if __name__ == '__main__':
    N = int(input('N : '))
    z = complex(input('z (sous la forme a+bj) : '))
    somme = exp_approx(N, z)
    print(f'{somme.real} + {somme.imag}i')
