# Demographic Modeling — `TP3.py`

This repository contains a refactored, robust demographic simulation and visualization script based on 2016 baseline age-class data.
The script generates or reads CSV datasets, runs year-by-year forward/backward simulations from 2016 to a target year, aggregates results by age groups, and offers several visualizations (2D and 3D).

---

## Contents

1. **Overview**

   * Purpose and high-level behaviour
   * Input / output (CSV files)
2. **Core Components**

   * `Data` — CSV creation & safe reading
   * `ClasseAge` / `GroupAgrege` — domain data structures
   * `Modele` — simulation engine and visualizations
3. **Simulation Logic**

   * Forward simulation (2016 → target year)
   * Backward simulation (2016 → earlier year) — approximate inversion
   * Aggregation into 4 age groups
4. **Visualizations**

   * Aggregated deviation (%) over years
   * Population by age (overlay curves)
   * 3D surface (age × year → population)
   * 3D bar chart (sampled)
   * Direct 2016 vs target-year comparison
   * Mortality quotient per 100k (2016)
5. **CLI / Menu**

   * Interactive console menu: choose year, centenarian rate, visualizations
6. **Examples & Usage**

   * How to run locally; recommended backends
7. **Notes, Troubleshooting and Tips**
8. **Related scripts (same project)**
9. **References & License**

---

# Overview

`TP3.py` is a standalone script that:

* Creates two CSV files if missing:

  * `donnees_main.csv` — yearly aggregated metrics (1986–2019)
  * `donnees_2016.csv` — population and mortality by age for 2016 (ages 0..99)
* Loads the 2016 age-class baseline and simulates the demographic evolution until a target year between **1986 and 2019**.
* Supports both forward simulations (future relative to 2016) and backward simulations (past relative to 2016).
* Produces plots for analysis and comparison.

The simulation stores:

* class-by-class snapshots (year, age, population)
* aggregated snapshots (age groups: `0–19`, `20–59`, `60–64`, `65–99` and `Total`)

---

# Core Components (API summary)

### `Data`

* `Data.creer_donnees(csv_main=CSV_MAIN, csv_2016=CSV_2016)`
  Generates `donnees_main.csv` and `donnees_2016.csv` if they do not exist (safe, non-overwriting).
* `Data.lecture(path)`
  Validated CSV reader (exists + `.csv` suffix).

### `ClasseAge(age:int, nombre:int, taux_morta:float)`

Dataclass for a single age class (age, population count, mortality rate as a probability between 0 and 1).

* `__call__()` returns `(age, nombre, taux_morta)`

### `GroupAgrege(age_range:str, nombre:int)`

Simple container for aggregated groups (label and sum).

### `Modele(annee_visee:int, taux:float = 0.01, output:bool = False)`

Main simulation class.

Key methods:

* `modele()` — run the simulation and populate internal result lists:

  * determines whether to simulate forward (2016 → target) or backward (2016 → target)
  * mutates `self.liste_classe_age` to final state and fills:

    * `self._resultat_classe_age` — list of `(year, age, population)`
    * `self._resultat_agrege` — list of `{"Année": year, "0 - 19 ans": .., ...}`
* `modele_agrege()` — aggregate current `liste_classe_age` into the 4 groups + `Total`
* `main_agrege(annee)` — read aggregated data from `donnees_main.csv` for comparisons
* `comparaison_agrege(annee)` — compute percent deviation (data vs model) for a given year
* `comparaison_classe_age(annee)` — append class-level results to internal storage

Visualization helpers:

* `affichage_deviation_agrege()` — line plot of percent deviations over years
* `affichage_population_age()` — overlay curves of population vs age for each simulated year
* `affichage_population_age_3D(azim=45, elev=50)` — 3D surface (tri-surface)
* `affichage_population_age_bar_3D(azim=45, elev=50, stride=5)` — 3D bar chart, sampled for readability
* `affichage_population_année()` — 2016 vs target-year population (2D comparison)
* `affichage_quotient_mortalité_100_000_age(log=False)` — plot mortality quotient (2016)

---

# How the simulation works (summary)

* Start from baseline `donnees_2016.csv` (100 age classes, 0..99).
* **Forward simulation** (target > 2016):

  * For each year: births (from `donnees_main.csv` if available) are inserted as age-0; existing classes age by +1; mortality applied by per-age mortality probabilities; classes beyond 99 are aggregated into the 99 category.
* **Backward simulation** (target < 2016):

  * The script performs an approximate inversion to estimate previous-year populations: it reverses aging and restores populations before mortality using the local mortality rates and a centenarian-rate heuristic for added 99+ entries.
* Aggregations and snapshots are stored for plotting and comparison with `donnees_main.csv`.

---

# Visual Results (examples)

The repository does not include binary image files in this README; after running the script you will get interactive matplotlib windows. Typical visual outputs are:

* Aggregated deviation over the simulated period (line plot)
* Population by age for each year (overlay 2D plot)
* 3D surface: population as a function of age and year
* 3D sampled bar chart for quick visual comparisons
* 2D comparison: 2016 vs target year
* Mortality quotient per 100k (2016) — optional log scale

If you want to export images from the plotting functions, you can modify the calls to `plt.savefig("myplot.png")` (before `plt.show()`).

---

# Usage / Examples

## Requirements

```text
Python 3.8+ recommended
pip install pandas numpy matplotlib
```

## Run the interactive console menu

```bash
python TP3.py
```

Follow the prompts:

* Target year (integer between 1986 and 2019)
* Centenarian rate (float, default `0.01`)
* Enable verbose output? (y/N)

Choose a visualization option from the menu.

## Run headless in a script

You can import and use `Modele` programmatically:

```python
from TP3 import Data, Modele

# Ensure CSVs are present
Data.creer_donnees()

m = Modele(annee_visee=1995, taux=0.01, output=True)
m.modele()  # run simulation
m.affichage_deviation_agrege()  # show a plot
```

## Troubleshooting plots not showing

* If interactive plot windows open and immediately close, ensure you use an interactive backend. From an IPython shell or Jupyter, run:

  * `%matplotlib qt`  (desktop windows)
  * `%matplotlib inline` (Jupyter inline — images embedded)
* From plain Python, matplotlib will block on `plt.show()` until the window is closed. If you run this inside some GUI frameworks or within certain terminal emulators, try using `python -m pip install pyqt5` and set the backend to `'Qt5Agg'` in your matplotlib configuration.

---

# Notes and Tips

* The backward simulation is an approximation — population inversion is inherently ill-posed. The algorithm tries to estimate prior counts by reversing the effect of mortality (i.e. dividing by `1 - rate`) but this can amplify noise; treat backward results with care.
* The model uses 2016 per-age mortality quotients (from `donnees_2016.csv`) as fixed rates for all years. If you have time-varying mortality tables, replace `_get_taux_for_age` accordingly.
* `Data.creer_donnees()` will not overwrite existing CSVs — remove them first if you want regenerated data.
* For reproducibility, consider saving plotted figures with `plt.savefig(...)`.

