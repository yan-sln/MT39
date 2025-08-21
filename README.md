# Mathematical & Demographic Modeling Project

This repository brings together three complementary projects exploring **mathematical models, numerical methods, and demographic simulations**.
The goal is to study **exponential growth**, **predator-prey dynamics**, and **population evolution** using both analytical and computational approaches.

---

## Contents

1. [Exponential Function, Approximation, and Population Growth](https://github.com/yan-sln/MT39/new/main?filename=README.md#exponential-function-approximation-and-population-growth--tp1py)
2. [Lotka-Volterra Predator-Prey Models and Variants](https://github.com/yan-sln/MT39/new/main?filename=README.md#lotka-volterra-predator-prey-models-and-variants--tp2py)
3. [Demographic Modeling](https://github.com/yan-sln/MT39/new/main?filename=README.md#demographic-modeling--tp3py)

---

## Exponential Function, Approximation, and Population Growth — TP1.py

This module focuses on the **exponential function** and its approximations:

* Series expansions and convergence proofs
* Approximation of \$e = \exp(1)\$
* Euler’s and Newton’s methods for numerical solutions
* Fibonacci sequence and the golden ratio
* Population growth models (exponential vs logistic)

📄 [Read the detailed README](./TP1/README.md)

---

## Lotka-Volterra Predator-Prey Models and Variants — TP2.py

This module explores **predator-prey dynamics** through Lotka-Volterra equations and extensions:

* **M1**: Basic Lotka-Volterra model
* **M2**: Model with scaling and change of variables
* **M3**: Logistic growth model for prey
* Euler’s and Heun’s methods for numerical approximation
* Phase portraits, direction fields, and stability analysis

📄 [Read the detailed README](./TP2/README.md)

---

## Demographic Modeling — TP3.py

This module contains a **demographic simulation script** based on 2016 baseline data:

* Forward and backward simulations (2016 → target year, or inverse)
* Aggregation by age groups and mortality modeling
* Visualizations: 2D overlays, 3D surfaces, bar charts
* Interactive console menu and CLI usage

📄 [Read the detailed README](./TP3/README.md)

---

## Requirements

Each module is Python-based and relies on **NumPy** and **Matplotlib**.
Some parts also use **Pandas** for CSV data handling.

Install dependencies with:

```bash
pip install numpy matplotlib pandas
```

---

## References

* Rudin, *Real and Complex Analysis*, McGraw-Hill, 1987
* Course material: *MT39 – Exponential Function and Numerical Approximations*
* Population and mortality datasets (baseline 2016, aggregated 1986–2019)
