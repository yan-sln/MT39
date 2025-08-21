# **Lotka-Volterra Predator-Prey Models and Variants**

This project explores different variants of the **Lotka-Volterra model** for predator-prey dynamics, including:

* **M1**: Basic Lotka-Volterra model
* **M2**: Model with scaling and change of variables
* **M3**: Logistic growth model for prey

The numerical methods applied include **Euler's method** and **Heun's method** for approximating solutions to these models.

---

## **Contents**

1. **Lotka-Volterra Models**

   * **M1**: Basic Lotka-Volterra model
   * **M2**: Model with change of variables and scaling
   * **M3**: Logistic growth model for prey

2. **Numerical Methods**

   * Euler's method
   * Heun's method

3. **Population Growth Dynamics**

   * Exponential and logistic growth
   * Comparison of numerical methods

4. **Phase Portraits and Direction Fields**

   * Visualizing predator-prey interactions
   * Stability and equilibrium points

5. **Logistic Growth Models**

   * Impact of carrying capacity on population dynamics

---

## **Lotka-Volterra Models**

### **M1: Basic Lotka-Volterra Model**

The **basic Lotka-Volterra model** is defined by the following equations:

$$
\frac{dx}{dt} = r \cdot x - p \cdot x \cdot y
$$

$$
\frac{dy}{dt} = -m \cdot y + q \cdot x \cdot y
$$

Where:

* $x$ is the prey population (rabbits),
* $y$ is the predator population (lynx),
* $r$ is the prey reproduction rate,
* $p$ is the prey mortality rate due to predation,
* $m$ is the predator mortality rate,
* $q$ is the predator reproduction rate.

---

### **M2: Model with Change of Variable**

**M2** introduces a change of variables and scaling:

$$
\frac{dv}{ds} = v - v \cdot w
$$

$$
\frac{dw}{ds} = -\alpha \cdot w + v \cdot w
$$

Where:

* $v = \frac{q}{r} \cdot x(t)$ and $w = \frac{p}{r} \cdot y(t)$,
* $\alpha$ is a scaling parameter influencing the predator population dynamics.

---

### **M3: Logistic Growth Model for Prey**

The **M3** variant incorporates logistic growth for the prey population, where the prey population is limited by a carrying capacity $X$:

$$
\frac{dx}{dt} = r \cdot x \cdot \left( 1 - \frac{x}{X} \right) - p \cdot x \cdot y
$$

$$
\frac{dy}{dt} = -m \cdot y + q \cdot x \cdot y
$$

Where $X$ represents the maximum population of prey that the environment can support.

---

## **Numerical Methods**

### **Euler's Method**

Euler's method is used to numerically solve the system of equations. It is a simple and widely used method for solving differential equations:

$$
x_{k+1} = x_k + h \cdot f(t_k, x_k)
$$

Where $h$ is the time step, and $f(t_k, x_k)$ is the rate of change at time $t_k$.

---

### **Heun's Method**

Heun’s method improves upon Euler’s method by averaging the slopes at the beginning and end of each time step:

$$
x_{k+1} = x_k + \frac{h}{2} \cdot \left( f(t_k, x_k) + f(t_{k+1}, x_{k+1}) \right)
$$

---

## **Population Growth Dynamics**

### **Exponential Growth**

Exponential growth describes a population growing at a constant rate over time:

$$
\frac{dx}{dt} = r \cdot x
$$

This is the growth pattern followed by the prey population in the basic Lotka-Volterra model.

### **Logistic Growth**

Logistic growth incorporates a carrying capacity $X$, where the population grows quickly at first, then slows as it approaches the carrying capacity:

$$
\frac{dx}{dt} = r \cdot x \cdot \left( 1 - \frac{x}{X} \right)
$$

---

## **Visual Results**

### **M1: Basic Lotka-Volterra Model**

| Figure | Description                                                   | Image                                                                                                      |
| ------ | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1.1    | Population dynamics of rabbits and lynx (M1 model) over time. | ![LV\_simple-1-1-1](https://github.com/Toppics/MT39/assets/110732997/deba5807-eeba-4036-902b-3d4089834d56) |
| 1.2    | Another view showing different parameter values for M1.       | ![LV\_simple-1-1-3](https://github.com/Toppics/MT39/assets/110732997/1b0f0556-e209-48ee-8d78-3284b18df87d) |

### **M2: Model with Variable Change**

| Figure | Description                                                                    | Image                                                                                                        |
| ------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| 2.1    | Dynamics of the system after applying a variable change and scaling (M2).      | ![LV\_simple-2-1-1](https://github.com/Toppics/MT39/assets/110732997/42c58c14-10c4-4554-a30c-bb23b4ff8fb9)   |
| 2.2    | Showing the effect of scaling on the population dynamics of prey and predator. | ![LV\_simple-2-1-3](https://github.com/Toppics/MT39/assets/110732997/57fea5dd-1e7c-4168-8468-85d8cdd07f7e)   |
| 2.3    | Another view of M2 dynamics for varying initial conditions.                    | ![LV\_simple-2-1-2-1](https://github.com/Toppics/MT39/assets/110732997/39762e95-dbec-4b54-8e82-75a29b98929f) |

### **M3: Logistic Growth Model for Prey**

| Figure | Description                                                                             | Image                                                                                                      |
| ------ | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 3.1    | Demonstrating logistic growth with a carrying capacity for the prey population.         | ![LV\_limite-1-1-1](https://github.com/Toppics/MT39/assets/110732997/55bf9753-2ba4-484e-aa6f-29761830b914) |
| 3.2    | Population dynamics with logistic growth for prey and predator under limited resources. | ![LV\_limite-1-1-3](https://github.com/Toppics/MT39/assets/110732997/3dfcb446-5931-4663-9f6e-5835bf3f968f) |

### **Phase Portraits and Direction Fields**

| Figure | Description                                                                           | Image                                                                                                                   |
| ------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 4.1    | The phase portrait showing the vector field and trajectories for M1.                  | ![LV\_simple-1-champ\_direction](https://github.com/Toppics/MT39/assets/110732997/2e4ef28b-e4cf-420a-b44b-53ae688e62f1) |
| 4.2    | The phase portrait for M3 showing the effect of carrying capacity on prey population. | ![LV\_limite-2-champ\_direction](https://github.com/Toppics/MT39/assets/110732997/304ede61-cc83-4245-8d09-1f043fb8f5b8) |

### **Comparison of Numerical Methods**

| Figure | Description                                                                  | Image                                                                                                       |
| ------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 5.1    | Comparison of Euler and Heun’s methods in predator-prey population dynamics. | ![LV\_simple-1-2-5B](https://github.com/Toppics/MT39/assets/110732997/611fbc0b-61cb-4f39-bf50-61c9c369c3cc) |
| 5.2    | Euler method with improved step size.                                        | ![LV\_simple-1-2-5](https://github.com/Toppics/MT39/assets/110732997/4ff6f58c-fbc7-428c-b004-9950f2ec4259)  |

### **Logistic Growth Models**

| Figure | Description                                                | Image                                                                                                      |
| ------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 6.1    | Logistic growth with different initial populations for M3. | ![LV\_simple-1-2-4](https://github.com/Toppics/MT39/assets/110732997/53cead37-0a8a-4f8d-afe7-9006ff8bb235) |

### **Additional Figures**

| Figure | Description                                                     | Image                                                                                                              |
| ------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 7.1    | Overview of predator-prey dynamics with new initial conditions. | ![ghjklm](https://github.com/Toppics/MT39/assets/110732997/e91c1427-dd28-49f5-af47-cb1ce85dc8d9)                   |
| 7.2    | Visualizing equilibrium points in the system.                   | ![Figure\_3](https://github.com/Toppics/MT39/assets/110732997/97cbdaac-ebae-4b18-b69f-9090f50b9986)                |
| 7.3    | Population dynamics with varied parameters.                     | ![Figure\_1](https://github.com/Toppics/MT39/assets/110732997/5cf37d7e-743b-4f93-b2f4-f0058b4d41d9)                |
| 7.4    | Results of a sensitivity analysis on the model parameters.      | ![Figure 2024-05-08 234444](https://github.com/Toppics/MT39/assets/110732997/eedcc23a-15cf-4065-bcaf-eafbf3c23b50) |
| 7.5    | Final population sizes for different predator-prey models.      | ![dfghjk](https://github.com/Toppics/MT39/assets/110732997/6909b8c3-d209-4df9-bc0a-dc5892fc8062)                   |

---

## **References**

* Lotka, A. J., *Elements of Physical Biology*, Williams & Wilkins Co, 1925.
* Volterra, V., *Variazioni e fluttuazioni de numero d'individui in specie animali conviventi*, Memorie della R. Accademia delle Scienze di Torino, 1926.
* Murray, J. D., *Mathematical Biology: I. An Introduction*, Springer, 2002.
