# -*- coding: utf-8 -*-
"""
menu_figures.py

Menu interactif pour lancer et tracer des simulations Lotka–Volterra
(M1, M2, M3) définies dans TP2.py.

- M1 : Modèle de base Lotka–Volterra
- M2 : Variante avec changement de variable (alpha)
- M3 : Variante logistique (capacité de charge X)

Ce script :
 - tente d'utiliser un backend matplotlib interactif (TkAgg, Qt5Agg, ...),
 - si aucun backend interactif n'est disponible (ex. exécution headless),
   sauvegarde les figures dans ./figures/ et informe l'utilisateur.
"""

from __future__ import annotations
from typing import Callable, Dict
import os
import sys
import time
import traceback
import warnings

# --- essayer de choisir un backend interactif avant d'importer pyplot --- #
import matplotlib

# Ordre de préférence : on essaie d'activer un backend interactif adapté.
_interactive_backends = ["TkAgg", "Qt5Agg", "GTK3Agg", "MacOSX", "WebAgg"]
_selected_backend = None
for _b in _interactive_backends:
    try:
        matplotlib.use(_b, force=True)
        _selected_backend = _b
        break
    except Exception:
        _selected_backend = None

# Si aucun backend interactif trouvé, on retient le backend actuel (probablement 'agg').
_backend_name = matplotlib.get_backend()

# Maintenant on peut importer pyplot
import matplotlib.pyplot as plt  # noqa: E402

# Supprimer warnings inutiles (notamment ceux liés à affichage non-interactif)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Import des modèles depuis TP2.py (doit être dans le même dossier ou dans PYTHONPATH)
try:
    from TP2 import (
        Lotka_Volterra,
        Lotka_Volterra_chngt_var,
        Lotka_Volterra_limité,
    )
except Exception as e:
    print("Erreur lors de l'import de TP2.py : ", e)
    raise

# --------------------------------------------------------------------------- #
# Classes de menu
# --------------------------------------------------------------------------- #

class MenuOption:
    """Une option de menu unique (nom, description, action)."""

    def __init__(self, key: str, description: str, action: Callable[[], None]) -> None:
        self.key = key
        self.description = description
        self.action = action

    def run(self) -> None:
        try:
            self.action()
        except Exception as e:
            print("Une erreur est survenue lors de l'exécution de l'option :")
            traceback.print_exc()


class Menu:
    """Gestionnaire de menu textuel."""

    def __init__(self, title: str = "Menu principal") -> None:
        self.title = title
        self.options: Dict[str, MenuOption] = {}

    def add_option(self, key: str, description: str, action: Callable[[], None]) -> None:
        if key in self.options:
            raise ValueError(f"L’option {key} existe déjà.")
        self.options[key] = MenuOption(key, description, action)

    def run(self) -> None:
        while True:
            print("\n" + "=" * 60)
            print(f"{self.title}")
            print("=" * 60)
            for key in sorted(self.options.keys()):
                print(f"[{key}] {self.options[key].description}")
            print("[q] Quitter")
            choix = input("Votre choix : ").strip().lower()
            if choix == "q":
                print("Fin du programme.")
                break
            elif choix in self.options:
                self.options[choix].run()
            else:
                print("Choix invalide, réessayez.")


# --------------------------------------------------------------------------- #
# FigureManager : intègre TP2 mais contrôle l'affichage/sauvegarde
# --------------------------------------------------------------------------- #

class FigureManager:
    def __init__(self) -> None:
        # Valeurs par défaut
        self.x0: int = 4
        self.y0: int = 10
        self.r: float = 1.5
        self.p: float = 0.05
        self.m: float = 0.48
        self.q: float = 0.05
        self.alpha: float = 1.3
        self.X: int = 50
        self.t_min: float = 0.0
        self.t_max: float = 50.0
        self.h: float = 0.0005

        # dossier de sauvegarde si backend non interactif
        self.outdir = os.path.join(os.getcwd(), "figures")
        os.makedirs(self.outdir, exist_ok=True)

    # --- affichage / sauvegarde robuste --- #
    def _show_or_save(self, fig: plt.Figure, name_hint: str = "figure"):
        """
        Si le backend est interactif, on appelle plt.show().
        Sinon, on sauvegarde la figure dans ./figures/ avec horodatage et lève un message.
        """
        backend = matplotlib.get_backend().lower()
        non_interactive_backends = {"agg", "pdf", "ps", "svg", "cairo"}
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{name_hint}_{timestamp}.png"
        outpath = os.path.join(self.outdir, filename)

        try:
            if any(k in backend for k in non_interactive_backends):
                # backend non interactif => sauvegarde
                fig.savefig(outpath, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print(f"Backend matplotlib = '{matplotlib.get_backend()}' (non interactif).")
                print(f"Figure sauvegardée dans : {outpath}")
            else:
                # backend interactif => tenter d'afficher
                # Utiliser fig.show() + plt.show(block=True) pour forcer si possible
                try:
                    fig.canvas.manager.set_window_title(name_hint)
                except Exception:
                    pass
                try:
                    plt.show(block=True)
                except Exception:
                    # fallback : sauvegarde si show échoue
                    fig.savefig(outpath, bbox_inches="tight", dpi=200)
                    plt.close(fig)
                    print("Impossible d'appeler plt.show() malgré backend interactif — figure sauvegardée.")
                    print(f"Figure sauvegardée dans : {outpath}")
        except Exception:
            # en cas d'erreur inattendue, on sauvegarde quand même
            try:
                fig.savefig(outpath, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print("Erreur lors de l'affichage — figure sauvegardée.")
                print(f"Figure sauvegardée dans : {outpath}")
            except Exception:
                print("Erreur fatale : impossible d'enregistrer la figure.")

    # --- fonctions de tracé (utilisent modele_... de TP2 pour obtenir DataFrame) --- #

    def _plot_time_series(self, df, title: str = ""):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax.plot(df["t"], df["x"], label="Lapins", linestyle="solid")
        ax.plot(df["t"], df["y"], label="Lynx", linestyle="dashed")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Population")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best")
        fig.tight_layout()
        self._show_or_save(fig, name_hint=title.replace(" ", "_"))

    def plot_m1(self) -> None:
        """Trace le modèle M1 (Lotka–Volterra de base)."""
        try:
            mdl = Lotka_Volterra(self.x0, self.y0, self.r, self.p, self.m, self.q)
            df = mdl.modele_Lotka_Volterra(self.t_min, self.t_max, self.h)
            title = f"M1 Base — x0={self.x0}, y0={self.y0}, r={self.r}, p={self.p}, m={self.m}, q={self.q}"
            self._plot_time_series(df, title)
        except Exception as e:
            print("Erreur pendant le tracé M1 :", e)
            traceback.print_exc()

    def plot_m2(self) -> None:
        """Trace le modèle M2 (changement de variable)."""
        try:
            mdl = Lotka_Volterra_chngt_var(self.x0, self.y0, self.r, self.p, self.m, self.q, self.alpha)
            # M2 a sa propre intégration : on utilise sa méthode publique
            df = mdl.modele_Lotka_Volterra(self.t_min, self.t_max, self.h)
            title = f"M2 ChgtVar (alpha={self.alpha}) — x0={self.x0}, y0={self.y0}"
            self._plot_time_series(df, title)
        except Exception as e:
            print("Erreur pendant le tracé M2 :", e)
            traceback.print_exc()

    def plot_m3(self) -> None:
        """Trace le modèle M3 (logistique avec capacité de charge)."""
        try:
            mdl = Lotka_Volterra_limité(self.x0, self.y0, self.r, self.p, self.m, self.q, self.X)
            df = mdl.modele_Lotka_Volterra(self.t_min, self.t_max, self.h)
            title = f"M3 Logistique (X={self.X}) — x0={self.x0}, y0={self.y0}"
            self._plot_time_series(df, title)
        except Exception as e:
            print("Erreur pendant le tracé M3 :", e)
            traceback.print_exc()

    # --- modifier les paramètres via input() de façon robuste --- #
    def update_params(self) -> None:
        def _ask(prompt: str, current, caster, allow_empty=True):
            s = input(f"{prompt} [{current}] : ").strip()
            if s == "" and allow_empty:
                return current
            try:
                return caster(s)
            except Exception:
                print("Valeur invalide — on conserve l'ancienne valeur.")
                return current

        try:
            print("Modifier les paramètres (Entrée = conserver la valeur actuelle)")
            self.x0 = _ask("x0 (lapins initiaux)", self.x0, int)
            self.y0 = _ask("y0 (lynx initiaux)", self.y0, int)
            self.r = _ask("r (τ reproduction lapins)", self.r, float)
            self.p = _ask("p (τ mortalité lapins par lynx)", self.p, float)
            self.m = _ask("m (τ mortalité lynx)", self.m, float)
            self.q = _ask("q (τ reproduction lynx)", self.q, float)
            self.alpha = _ask("alpha (M2)", self.alpha, float)
            self.X = _ask("X (capacité de charge M3)", self.X, int)
            self.t_min = _ask("t_min", self.t_min, float)
            self.t_max = _ask("t_max", self.t_max, float)
            self.h = _ask("h (pas de simulation)", self.h, float)
            print("Paramètres mis à jour.")
        except KeyboardInterrupt:
            print("\nModification interrompue, paramètres non modifiés.")
        except Exception as e:
            print("Erreur lors de la modification des paramètres :", e)
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #

def main():
    # Afficher une petite info sur le backend choisi
    print(f"Matplotlib backend utilisé : '{matplotlib.get_backend()}'")
    if _selected_backend:
        print(f"(Tentative d'activation d'un backend interactif : '{_selected_backend}')")

    fig_manager = FigureManager()
    menu = Menu("Simulation Lotka–Volterra")

    menu.add_option("1", "Modèle M1 (base) — séries temporelles", fig_manager.plot_m1)
    menu.add_option("2", "Modèle M2 (changement de variable) — séries temporelles", fig_manager.plot_m2)
    menu.add_option("3", "Modèle M3 (logistique, capacité de charge) — séries temporelles", fig_manager.plot_m3)
    menu.add_option("p", "Modifier les paramètres", fig_manager.update_params)

    try:
        menu.run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur — sortie.")


if __name__ == "__main__":
    main()
