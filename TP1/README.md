# Exponential Function, Approximation, and Population Growth

This project explores the exponential function through series expansions, error analysis, and numerical methods. It also connects these concepts with population growth models, Fibonacci sequences, and classical numerical approximations.

---

## Contents

1. **Definition of the Exponential Function**

   * Series definition for complex numbers
   * Proof of convergence
   * Approximation of $e = \exp(1)$

2. **Series Approximation of $\exp(x)$**

   * Approximation for $x > 0$ and $x < 0$
   * Approximation of $\exp(it)$ for $t \in \mathbb{R}$
   * Error bounds using Taylor’s theorem

3. **Euler’s Method**

   * Numerical solution of $y' = y$
   * Approximation of $e = \exp(1)$
   * Application to $\exp(i2\pi) = 1$

4. **Newton’s Method**

   * Approximation of $\pi$ by solving $\cos(x/2) = 0$
   * Convergence speed and sensitivity to initial conditions
   * Computation of logarithms $\ln(x)$ as inverse of $\exp(x)$

5. **Fibonacci Sequence and the Golden Ratio**

   * Recursive definition of Fibonacci numbers
   * Convergence of the ratio $F_{n+1}/F_n$ to the golden ratio $\varphi$
   * Comparison with closed-form approximations

6. **Population Growth Models**

   * Exponential growth with constant reproduction rate
   * Logistic growth with resource limitations
   * Comparison of exact solution, Euler’s method, and Heun’s method

---

## Visual Results

### Taylor approximations of $e^x$

<p align="center">
  <img src="https://github.com/Toppics/MT39/assets/110732997/aa781186-9498-41d4-856f-78009e798be2" width="350"/>
</p>

---

### Approximation of $\exp(x)$ for $x > 0$

<p align="center">
  <img src="https://github.com/Toppics/MT39/assets/110732997/1710d8a1-044f-4154-96ee-9dfafd11207e" width="350"/>
</p>

---

### Complex exponential $e^{it}$ and Euler identity

<table>
<tr>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/a035c944-d3b3-4843-a5f0-b4793cfd46cb" width="300"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/61607862-4aa3-449f-9e50-ee66fff7d5f8" width="300"/></td>
</tr>
<tr>
<td align="center">Approximation of exp(it)</td>
<td align="center">Real and imaginary parts</td>
</tr>
</table>

---

### Cosine and Sine comparison

<table>
<tr>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/ab63edc0-296b-49e1-8971-7e0e9e4de890" width="300"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/df0db915-b799-47e0-9f08-4d74edbf688f" width="300"/></td>
</tr>
<tr>
<td align="center">Cosine curve</td>
<td align="center">Sine curve</td>
</tr>
</table>

---

### Error bound and convergence

<p align="center">
  <img src="https://github.com/Toppics/MT39/assets/110732997/d24a8f87-a75c-49a9-8bc9-36121762caea" width="350"/>
</p>

---

### Fibonacci and the Golden Ratio

<table>
<tr>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/c8b40515-0539-4f7e-97a5-897421b90bad" width="280"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/6dfe9dd4-4d7e-4524-9276-e3096b4af209" width="280"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/b224803f-a070-43c9-af47-257dbf717710" width="280"/></td>
</tr>
<tr>
<td align="center">With coefficients</td>
<td align="center">Corrected version</td>
<td align="center">Uncorrected version</td>
</tr>
</table>

---

### Euler’s method and exponential growth

<table>
<tr>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/ec34a9ba-b432-4fb1-a21d-7cad4cf86d99" width="300"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/abbae8c4-5ead-441d-83c1-e62dbd197602" width="300"/></td>
</tr>
<tr>
<td align="center">Exponential growth (N=36)</td>
<td align="center">Exponential growth (N=360)</td>
</tr>
</table>

---

### Newton’s method for $\pi$

<table>
<tr>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/93f2d40a-a7a4-4bcd-93cd-8ed607ee06e7" width="280"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/9a4e0ea2-af0e-44af-883b-140db6cf8561" width="280"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/f0d1824e-bb4e-4d80-9d48-c434f9c26a70" width="280"/></td>
</tr>
<tr>
<td align="center">Convergence (N=36)</td>
<td align="center">High precision (N=36000)</td>
<td align="center">General approximation</td>
</tr>
</table>

---

### Logistic growth models

<table>
<tr>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/23764935-2da5-4e78-b049-3743bb03b7fc" width="220"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/ef455df4-d218-43ab-851b-4b5d640fb716" width="220"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/930d7e7d-e7ba-479a-b025-ebe1ca509423" width="220"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/959e8d22-8209-42b6-a9e6-3b42563d9540" width="220"/></td>
<td><img src="https://github.com/Toppics/MT39/assets/110732997/24a9c183-04e4-476b-9d8c-42794ee57a59" width="220"/></td>
</tr>
<tr>
<td align="center">N=18, y0=2</td>
<td align="center">N=18, y0=100</td>
<td align="center">N=36, y0=2</td>
<td align="center">N=36, y0=50</td>
<td align="center">N=36, y0=100</td>
</tr>
</table>

---

## References

* Walter Rudin, *Real and Complex Analysis*, McGraw-Hill, 3rd Edition, 1987.
* Course material: *MT39 – Exponential Function and Numerical Approximations*.
