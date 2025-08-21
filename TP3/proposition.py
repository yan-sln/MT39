# -*- coding: utf-8 -*-
"""
demography_model.py

Refactored and corrected demographic modeling script.
Features:
 - creation of CSV datasets (if absent)
 - safe CSV reading
 - data classes (ClasseAge, GroupAgrege)
 - Modele: simulation forward/backward from 2016
 - visualizations: aggregated deviation, population by age (2D), 3D (surface and bars), mortality quotient
 - interactive console menu (robust input validation)

Author: yan-s
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessary for 3D plotting backend)
import sys
import math
import textwrap

# ---------------------------------------------------------------------------
# Matplotlib configuration (adjustable)
# ---------------------------------------------------------------------------
rcParams["figure.dpi"] = 160
rcParams["figure.figsize"] = (12, 7)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
CSV_MAIN = Path("donnees_main.csv")
CSV_2016 = Path("donnees_2016.csv")


def safe_int(x: Any, default: Optional[int] = None) -> int:
    try:
        return int(x)
    except Exception:
        if default is not None:
            return default
        raise


def safe_float(x: Any, default: Optional[float] = None) -> float:
    try:
        return float(x)
    except Exception:
        if default is not None:
            return default
        raise


# ---------------------------------------------------------------------------
# Data helper: create / read CSVs
# ---------------------------------------------------------------------------
class Data:
    """Create and read dataset CSV files."""

    @staticmethod
    def creer_donnees(csv_main: Path = CSV_MAIN, csv_2016: Path = CSV_2016) -> None:
        """
        Create two CSV files (donnees_main.csv, donnees_2016.csv) if they do not exist.
        Data content is derived from the original script but generated safely.
        """
        # Aggregated yearly data (reformatted)
        donnees_main = {
            "Années": [
                1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                2014, 2015, 2016, 2017, 2018, 2019
            ],
            "de_0_à_19_ans": [
                15999391, 15919634, 15852690, 15793067, 15719647, 15605455, 15473515, 15330473,
                15179775, 15084361, 15058227, 15055742, 15026922, 15017908, 15047287, 15067857,
                15091374, 15116574, 15183494, 15242403, 15280401, 15315094, 15337575, 15368840,
                15406592, 15440408, 15457656, 15513096, 15588708, 15651778, 15645526, 15616148,
                15611515, 15584916
            ],
            "de_20_à_59_ans": [
                29296183, 29495669, 29685169, 29875387, 30093767, 30280463, 30506157, 30730777,
                30940136, 31068721, 31160552, 31232352, 31342595, 31455770, 31667286, 31962065,
                32288026, 32560892, 32760470, 32971217, 33193638, 33219983, 33170779, 33107066,
                33023612, 32939901, 32910610, 32853247, 32791541, 32680632, 32562257, 32448507,
                32383772, 32356902
            ],
            "de_60_à_64_ans": [
                2887651, 2870641, 2869965, 2881891, 2892072, 2916128, 2925603, 2941799, 2921205,
                2912957, 2858806, 2816744, 2765130, 2737639, 2712836, 2675214, 2610227, 2606797,
                2640164, 2682701, 2762859, 3051869, 3325595, 3568379, 3795165, 4022812, 4034993,
                4029597, 3998555, 3979695, 3949846, 3953989, 3951924, 3974120
            ],
            "de_65_à_99_ans": [
                7228013, 7395836, 7558318, 7719465, 7871514, 8038615, 8205258, 8366112, 8523892,
                8686496, 8858374, 9011180, 9164315, 9285296, 9430789, 9561436, 9696272, 9817578,
                9921293, 10066943, 10162835, 10208292, 10300917, 10421424, 10539866, 10667223,
                10972712, 11301925, 11649154, 11988716, 12311163, 12620489, 12896826, 13180830
            ],
            "centenaires": [
                2886, 3126, 3352, 3476, 3760, 3885, 4379, 4458, 5014, 5648, 6303, 7107, 7259, 7754,
                8161, 9444, 11031, 11945, 12393, 13182, 13575, 14426, 14940, 15810, 16688, 18221,
                19897, 20887, 22031, 24085, 21044, 19383, 18513, 18520
            ],
            "Naissances": [
                778468, 767828, 771268, 765473, 762407, 759056, 743658, 711610, 710993, 729609,
                734338, 726768, 738080, 744791, 774782, 770945, 761630, 761464, 767816, 774355,
                796896, 785985, 796044, 793420, 802224, 792996, 790290, 781621, 781167, 760421,
                744697, 730242, 719737, 714029
            ],
        }

        donnees_2016 = {
            "Âge": list(range(100)),
            "Population": [
                # Simplified 100 plausible values for the example
                765550, 788646, 792333, 804236, 816258, 836823, 834997, 839809, 836583, 849949,
                833020, 828197, 823916, 828573, 830033, 859176, 822275, 814634, 792026, 787204,
                776849, 737459, 728187, 761786, 685005, 784180, 784047, 792374, 792888, 795458,
                810141, 804440, 805286, 847033, 856278, 867851, 817763, 808050, 816877, 798343,
                820272, 867477, 914322, 937883, 930282, 912733, 897889, 892061, 885433, 809802,
                909893, 920256, 863651, 877540, 876823, 872333, 821833, 845723, 842277, 833099,
                822054, 821117, 799913, 811671, 788895, 828971, 800702, 818084, 776321, 734510,
                653400, 637175, 618715, 578983, 524883, 540005, 401519, 442518, 383665, 423626,
                404609, 398592, 369172, 366529, 338247, 319195, 277738, 251466, 221891, 134597,
                170448, 141408, 86225, 25053, 77793, 60573, 27183, 17293, 11488, 7637
            ],
            "Quotient_de_mortalité_pour_100_000": [
                # Simplified pattern: increasing mortality with age (per 100k)
                *([50 + i * 10 for i in range(50)]),  # young->middle
                *([500 + i * 100 for i in range(50)])  # older ages higher quotient
            ][:100]
        }

        # Write CSVs if missing
        if not csv_main.exists():
            pd.DataFrame(donnees_main).to_csv(csv_main, index=False, encoding="utf-8")
            print(f"Generated file: {csv_main}")
        else:
            print(f"{csv_main} already exists — not overwritten.")

        if not csv_2016.exists():
            pd.DataFrame(donnees_2016).to_csv(csv_2016, index=False, encoding="utf-8")
            print(f"Generated file: {csv_2016}")
        else:
            print(f"{csv_2016} already exists — not overwritten.")

    @staticmethod
    def lecture(chemin: str | Path) -> pd.DataFrame:
        """Read a CSV while verifying existence and extension."""
        p = Path(chemin)
        if not p.exists():
            raise FileNotFoundError(f"The file {p} does not exist.")
        if p.suffix.lower() != ".csv":
            raise ValueError(f"{p} is not a .csv file")
        return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Domain data classes
# ---------------------------------------------------------------------------
@dataclass
class ClasseAge:
    age: int
    nombre: int
    taux_morta: float  # value between 0 and 1

    def __post_init__(self):
        if not isinstance(self.age, int):
            raise TypeError("age must be an integer")
        if not isinstance(self.nombre, int):
            try:
                self.nombre = int(self.nombre)
            except Exception:
                raise TypeError("nombre must be convertible to int")
        if self.nombre < 0:
            raise ValueError("nombre must be >= 0")
        if not isinstance(self.taux_morta, float):
            try:
                self.taux_morta = float(self.taux_morta)
            except Exception:
                raise TypeError("taux_morta must be convertible to float")
        if not (0.0 <= self.taux_morta <= 1.0):
            # allow rates between 0 and 1 (0% to 100%)
            raise ValueError("taux_morta must be between 0 and 1")

    def __call__(self) -> Tuple[int, int, float]:
        return (self.age, self.nombre, self.taux_morta)


@dataclass
class GroupAgrege:
    """Simple struct for aggregated age group (label, count)."""
    age_range: str
    nombre: int

    def __post_init__(self):
        if not isinstance(self.age_range, str):
            raise TypeError("age_range must be a string")
        if not isinstance(self.nombre, int):
            raise TypeError("nombre must be an integer")


# ---------------------------------------------------------------------------
# Model: demographic simulation
# ---------------------------------------------------------------------------
class Modele:
    """
    Simulates population by age classes starting from 2016.
    - annee_visee : target year (1986..2019)
    - taux : parameter for generating centenarians when simulating the past
    - output : bool to enable concise logging
    """

    def __init__(self, annee_visee: int, taux: float = 0.01, output: bool = False):
        # validations
        if not isinstance(annee_visee, int):
            raise TypeError("annee_visee must be an integer")
        if not (1986 <= annee_visee <= 2019):
            raise ValueError("annee_visee must be between 1986 and 2019 inclusive")
        if not isinstance(taux, (float, int)):
            raise TypeError("taux must be a real number")
        if not (taux > 0):
            raise ValueError("taux must be strictly positive")
        self.__annee_visee = annee_visee
        self.__taux = float(taux)
        self.__output = bool(output)

        # Initial load: list of 100 ClasseAge corresponding to 2016
        self.liste_classe_age: List[ClasseAge] = self._build_liste_2016()
        # Stored results
        self._resultat_agrege: List[Dict[str, Any]] = []
        self._resultat_classe_age: List[Tuple[int, int, int]] = []  # (year, age, population)

    @property
    def annee_visee(self) -> int:
        return self.__annee_visee

    @property
    def taux(self) -> float:
        return self.__taux

    @property
    def output(self) -> bool:
        return self.__output

    def _build_liste_2016(self) -> List[ClasseAge]:
        """Build the initial list of 100 age classes from donnees_2016.csv."""
        df = Data.lecture(CSV_2016)
        # Validate / ensure length 100
        if "Population" not in df.columns or "Âge" not in df.columns or "Quotient_de_mortalité_pour_100_000" not in df.columns:
            raise ValueError("donnees_2016.csv malformed (missing columns)")
        df = df.iloc[:100].copy()
        liste = []
        for _, row in df.iterrows():
            age = int(row["Âge"])
            pop = int(row["Population"])
            q = float(row["Quotient_de_mortalité_pour_100_000"]) / 100000.0  # convert to probability
            # clip rate between 0 and 1
            q = max(0.0, min(1.0, q))
            liste.append(ClasseAge(age=age, nombre=pop, taux_morta=q))
        return liste

    # -------------------------
    # Simulation control
    # -------------------------
    def reset_results(self) -> None:
        self._resultat_agrege = []
        self._resultat_classe_age = []

    def modele(self) -> None:
        """
        Run the simulation between 2016 and annee_visee (inclusive).
        - if annee_visee > 2016 : simulate the future (advance, births each year)
        - if annee_visee < 2016 : simulate the past (approximate inversion)
        Note: the simulation mutates self.liste_classe_age (final state).
        """
        self.reset_results()
        df_main = Data.lecture(CSV_MAIN)
        # Prepare births series indexed by year relative to 2016
        births_by_year = {
            int(r["Années"]): int(r["Naissances"]) for _, r in pd.DataFrame(df_main).iterrows()
        }
        # helper to record current snapshot in results
        def record_snapshot(year: int):
            # fill _resultat_classe_age: append 100 classes (year, age, population)
            for ca in self.liste_classe_age:
                self._resultat_classe_age.append((year, ca.age, ca.nombre))
            # record aggregated snapshot
            self._resultat_agrege.append({"Année": int(year), **self.modele_agrege()})

        if self.annee_visee == 2016:
            # record baseline and return
            if self.output:
                print("Target year = 2016: recording the initial state.")
            record_snapshot(2016)
            return

        if self.annee_visee > 2016:
            # simulate forward
            if self.output:
                print(f"Simulating forward: 2016 -> {self.annee_visee}")
            # iterate year by year
            for year in range(2016, self.annee_visee + 1):
                # record current state for this year
                record_snapshot(year)
                # prepare next year unless at last
                if year == self.annee_visee:
                    break
                # births for upcoming year: prefer explicit mapping, fallback to 2016 births
                births = births_by_year.get(year + 1, births_by_year.get(2016, 0))
                # compute new ClasseAge list for next year:
                new_list: List[ClasseAge] = []
                # newborn generation (age 0)
                taux_morta0 = self.liste_classe_age[0].taux_morta if len(self.liste_classe_age) > 0 else 0.0
                new_list.append(ClasseAge(age=0, nombre=int(births), taux_morta=taux_morta0))
                # age everyone by +1 and apply mortality
                for ca in self.liste_classe_age:
                    # apply mortality
                    survivors = int(round(ca.nombre * (1.0 - ca.taux_morta)))
                    new_age = ca.age + 1
                    if new_age <= 99:
                        # determine mortality rate for the new age: use base 2016 table if available
                        next_taux = self._get_taux_for_age(new_age)
                        new_list.append(ClasseAge(age=new_age, nombre=survivors, taux_morta=next_taux))
                    else:
                        # beyond 99, aggregate into the last class (keep classes 0..99)
                        if new_list and new_list[-1].age == 99:
                            new_list[-1].nombre += survivors
                # ensure classes for ages 0..99 (fill missing with zero-population entries)
                filled = [None] * 100
                for ca in new_list:
                    if 0 <= ca.age <= 99:
                        filled[ca.age] = ca
                for age_idx in range(100):
                    if filled[age_idx] is None:
                        filled[age_idx] = ClasseAge(age=age_idx, nombre=0, taux_morta=self._get_taux_for_age(age_idx))
                self.liste_classe_age = filled

        else:
            # simulate backward (approximate inversion)
            if self.output:
                print(f"Simulating backward: 2016 -> {self.annee_visee}")
            # step backward year by year
            for year in range(2016, self.annee_visee - 1, -1):
                record_snapshot(year)
                if year == self.annee_visee:
                    break
                # Inverse aging: estimate previous year's age-class counts
                df_row = Data.lecture(CSV_MAIN)
                births_prev = df_row.loc[df_row["Années"] == (year - 1), "Naissances"]
                if len(births_prev) == 1:
                    births_val = int(births_prev.values[0])
                else:
                    births_val = int(df_row.loc[df_row["Années"] == 2016, "Naissances"].values[0])
                # Build previous year list (0..99)
                prev_list_temp = [None] * 100
                # age 0 previous year = births_prev (approx)
                prev_list_temp[0] = ClasseAge(age=0, nombre=int(births_prev.values[0]) if len(births_prev) == 1 else int(births_val), taux_morta=self._get_taux_for_age(0))
                # for ages 1..99 in previous year, estimate prior counts
                for age in range(1, 100):
                    current = self._get_classe_by_age(age)
                    prev_taux = self._get_taux_for_age(age - 1)
                    denom = max(1e-9, (1.0 - prev_taux))
                    prev_count = int(round(current.nombre / denom)) if denom > 0 else current.nombre
                    prev_list_temp[age] = ClasseAge(age=age, nombre=prev_count, taux_morta=prev_taux)
                # replace any None with zero-pop entries
                for idx in range(100):
                    if prev_list_temp[idx] is None:
                        prev_list_temp[idx] = ClasseAge(age=idx, nombre=0, taux_morta=self._get_taux_for_age(idx))
                self.liste_classe_age = prev_list_temp

    # -------------------------
    # Helpers for simulation
    # -------------------------
    def _get_taux_for_age(self, age: int) -> float:
        """Return mortality rate (prefers the reference 2016 CSV)."""
        df = Data.lecture(CSV_2016)
        if "Quotient_de_mortalité_pour_100_000" in df.columns:
            try:
                q = float(df.loc[df["Âge"] == age, "Quotient_de_mortalité_pour_100_000"].values[0]) / 100000.0
                q = max(0.0, min(1.0, q))
                return q
            except Exception:
                return 0.0
        return 0.0

    def _get_classe_by_age(self, age: int) -> ClasseAge:
        """Find the current ClasseAge object for a given age (0..99)."""
        for ca in self.liste_classe_age:
            if ca.age == age:
                return ca
        return ClasseAge(age=age, nombre=0, taux_morta=self._get_taux_for_age(age))

    # -------------------------
    # Aggregations and comparisons
    # -------------------------
    def modele_agrege(self) -> Dict[str, int]:
        """Aggregate the current list into 4 age groups and return a dict."""
        # sums by index ranges (0-19, 20-59, 60-64, 65-99)
        def somme_range(a: int, b: int) -> int:
            return sum(ca.nombre for ca in self.liste_classe_age[a:b])

        d = {
            "0 - 19 ans": int(somme_range(0, 20)),
            "20 - 59 ans": int(somme_range(20, 60)),
            "60 - 64 ans": int(somme_range(60, 65)),
            "65 - 99 ans": int(somme_range(65, 100))
        }
        d["Total"] = int(sum(d.values()))
        return d

    def main_agrege(self, annee: int) -> Dict[str, int]:
        """Extract aggregated known data for a given year from donnees_main.csv."""
        df = Data.lecture(CSV_MAIN)
        if annee not in df["Années"].values:
            raise ValueError(f"Year {annee} not present in {CSV_MAIN}")
        row = df.loc[df["Années"] == annee].iloc[0]
        d = {
            "0 - 19 ans": int(row["de_0_à_19_ans"]),
            "20 - 59 ans": int(row["de_20_à_59_ans"]),
            "60 - 64 ans": int(row["de_60_à_64_ans"]),
            "65 - 99 ans": int(row["de_65_à_99_ans"])
        }
        d["Total"] = int(sum(d.values()))
        return d

    def comparaison_agrege(self, annee: int) -> None:
        """Compute percent deviation (data / model - 1) * 100 and append it to aggregated results."""
        model = self.modele_agrege()
        main = self.main_agrege(annee)
        dct = {"Année": int(annee)}
        for key in main:
            model_val = model.get(key, 0)
            main_val = main[key]
            if model_val == 0:
                deviation = float("nan")
            else:
                deviation = round((main_val / model_val * 100.0) - 100.0, 2)
            dct[key] = deviation
            if self.output:
                print(f"{key} : deviation {deviation} % (data={main_val} / model={model_val})")
        self._resultat_agrege.append(dct)

    def comparaison_classe_age(self, annee: int) -> None:
        """Append the current state class-by-class into _resultat_classe_age."""
        for ca in self.liste_classe_age:
            self._resultat_classe_age.append((annee, ca.age, ca.nombre))

    # -------------------------
    # Visualizations
    # -------------------------
    def affichage_deviation_agrege(self) -> None:
        """Plot aggregated relative deviation (%) for the period [2016..annee_visee]."""
        if not self._resultat_agrege:
            raise RuntimeError("No results available — run modele() first.")
        df = pd.DataFrame(self._resultat_agrege)
        # sort by year
        df = df.sort_values("Année")
        years = df["Année"].values
        # plot each group column except 'Année'
        for col in df.columns:
            if col == "Année":
                continue
            plt.plot(years, df[col], label=col)
        plt.title(f"Aggregated deviation (data vs model) — 2016 -> {self.annee_visee}")
        plt.xlabel("Year")
        plt.ylabel("Deviation (%)")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()

    def affichage_population_age(self) -> None:
        """Plot evolution of population by age for each simulated year (overlaid series)."""
        if not self._resultat_classe_age:
            raise RuntimeError("No results available — run modele() first.")
        df = pd.DataFrame(self._resultat_classe_age, columns=["Année", "Âge", "Population"])
        years = sorted(df["Année"].unique())
        # pivot DataFrame: index age (0..99), columns years
        pivot = df.pivot(index="Âge", columns="Année", values="Population").fillna(0).astype(int)
        colors = rainbow(np.linspace(0, 1, len(pivot.columns)))
        for i, col in enumerate(pivot.columns):
            plt.plot(pivot.index, pivot[col], label=str(col), alpha=0.8)
        plt.title(f"Population by age — period 2016 -> {self.annee_visee}")
        plt.xlabel("Age")
        plt.ylabel("Population")
        plt.grid(True)
        # put legend outside plot
        plt.legend(title="Years", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xlim(0, 99)
        plt.show()

    def affichage_population_age_3D(self, azim: int = 45, elev: int = 50) -> None:
        """3D surface visualization: population as function of age and year."""
        if not self._resultat_classe_age:
            raise RuntimeError("No results available — run modele() first.")
        df = pd.DataFrame(self._resultat_classe_age, columns=["Année", "Âge", "Population"])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(df["Année"], df["Âge"], df["Population"], linewidth=0, antialiased=True, alpha=0.9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Age")
        ax.set_zlabel("Population")
        plt.title(f"Population by age and year (2016 -> {self.annee_visee})")
        ax.view_init(elev, azim)
        plt.show()

    def affichage_population_age_bar_3D(self, azim: int = 45, elev: int = 50, stride: int = 5) -> None:
        """3D bar chart (sampling stride improves readability)."""
        if not self._resultat_classe_age:
            raise RuntimeError("No results available — run modele() first.")
        df = pd.DataFrame(self._resultat_classe_age, columns=["Année", "Âge", "Population"])
        # sampling for readability: select subset of years and ages
        years_sorted = sorted(df["Année"].unique())
        ages_sorted = sorted(df["Âge"].unique())
        # prepare grid arrays for bar3d
        xx = []
        yy = []
        zz = []
        dx = []
        dy = []
        dz = []
        for i, year in enumerate(years_sorted[::stride]):
            sub = df[df["Année"] == year]
            for j, age in enumerate(ages_sorted[::stride]):
                pop = int(sub.loc[sub["Âge"] == age, "Population"].values[0]) if not sub.loc[sub["Âge"] == age].empty else 0
                xx.append(year)
                yy.append(age)
                zz.append(0)
                dx.append(0.8 * stride)
                dy.append(0.8 * stride)
                dz.append(pop)
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.bar3d(np.array(xx), np.array(yy), np.array(zz), np.array(dx), np.array(dy), np.array(dz))
        ax.set_xlabel("Year")
        ax.set_ylabel("Age")
        ax.set_zlabel("Population")
        plt.title(f"3D bars (sampling stride={stride}) — 2016 -> {self.annee_visee}")
        ax.view_init(elev, azim)
        plt.show()

    def affichage_population_année(self) -> None:
        """Plot population curves for 2016 and the targeted year for direct comparison."""
        if not self._resultat_classe_age:
            raise RuntimeError("No results available — run modele() first.")
        df = pd.DataFrame(self._resultat_classe_age, columns=["Année", "Âge", "Population"])
        df_2016 = df[df["Année"] == 2016].sort_values("Âge")
        df_visee = df[df["Année"] == self.annee_visee].sort_values("Âge")
        plt.plot(df_2016["Âge"], df_2016["Population"], label="2016")
        plt.plot(df_visee["Âge"], df_visee["Population"], label=str(self.annee_visee))
        plt.title(f"Population by age: 2016 vs {self.annee_visee}")
        plt.xlabel("Age")
        plt.ylabel("Population")
        plt.legend()
        plt.grid(True)
        plt.show()

    def affichage_quotient_mortalité_100_000_age(self, log: bool = False) -> None:
        """Plot mortality quotient per 100,000 by age for 2016."""
        df = Data.lecture(CSV_2016)
        if "Quotient_de_mortalité_pour_100_000" not in df.columns:
            raise ValueError("donnees_2016.csv malformed (missing 'Quotient_de_mortalité_pour_100_000' column)")
        plt.plot(df["Âge"], df["Quotient_de_mortalité_pour_100_000"], marker="o")
        plt.title("Mortality quotient per 100,000 (2016)")
        plt.xlabel("Age")
        plt.ylabel("Deaths per 100,000")
        if log:
            plt.yscale("log")
        plt.grid(True)
        plt.show()


# ---------------------------------------------------------------------------
# Improved console menu
# ---------------------------------------------------------------------------
def menu():
    Data.creer_donnees()  # create files if necessary
    print(textwrap.dedent("""
        === Demographic modeling (base 2016) ===
        - The tool simulates population by age classes starting from 2016 data.
        - Choose a target year (between 1986 and 2019 inclusive).
    """))
    # read user choices with validation
    while True:
        try:
            annee = int(input("Target year (1986-2019) -> ").strip())
            if not (1986 <= annee <= 2019):
                print("Invalid input: year must be between 1986 and 2019.")
                continue
            break
        except Exception:
            print("Invalid input — please try again.")

    while True:
        try:
            taux_s = input("Centenarian rate (float, e.g. 0.01) [default 0.01]: ").strip()
            taux = float(taux_s) if taux_s != "" else 0.01
            break
        except Exception:
            print("Invalid input — please try again.")

    while True:
        out_s = input("Print internal steps to console? (y/N): ").strip().lower()
        if out_s in ("y", "yes"):
            output = True
            break
        if out_s in ("n", "no", ""):
            output = False
            break
        print("Answer with y or n.")

    modele = Modele(annee_visee=annee, taux=taux, output=output)
    # run simulation (populates internal results)
    print("Running simulation ... (may take a few seconds)")
    modele.modele()

    # display menu options
    options = {
        "1": "Aggregated deviation (plot)",
        "2": "Population - age curves (2D)",
        "3": "Population - 3D surface visualization",
        "4": "Population - 3D bar chart (sampled)",
        "5": "Population: 2016 vs target year comparison",
        "6": "Mortality quotient 2016 (by age)",
        "0": "Quit"
    }
    while True:
        print("\nChoose a visualization option:")
        for k, v in options.items():
            print(f" {k} - {v}")
        ch = input("Your choice: ").strip()
        if ch == "0":
            print("Exit.")
            break
        try:
            if ch == "1":
                modele.affichage_deviation_agrege()
            elif ch == "2":
                modele.affichage_population_age()
            elif ch == "3":
                az = int(input("Azimuth (e.g. 45): ") or 45)
                el = int(input("Elevation (e.g. 50): ") or 50)
                modele.affichage_population_age_3D(azim=az, elev=el)
            elif ch == "4":
                az = int(input("Azimuth (e.g. 45): ") or 45)
                el = int(input("Elevation (e.g. 50): ") or 50)
                stride = int(input("Sampling stride (e.g. 5): ") or 5)
                modele.affichage_population_age_bar_3D(azim=az, elev=el, stride=stride)
            elif ch == "5":
                modele.affichage_population_année()
            elif ch == "6":
                log_s = input("Use log scale for y axis? (y/N): ").strip().lower()
                modele.affichage_quotient_mortalité_100_000_age(log=(log_s in ("y", "yes")))
            else:
                print("Unknown option.")
        except Exception as e:
            print(f"Error during visualization: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nUser interruption — exiting.")
        sys.exit(0)
