# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:21:15 2024

@author: yan-s
"""
from LaTeX import Lotka_Volterra

from numpy import array, arange
import pandas as pd
import matplotlib.pyplot as plt

def dataframe_3_especes(func):
    """Permet d'obtenir un DataFrame prêt à être utilisé."""
    def wrapper(*args, **kwargs):
        lst_t, lst_xy = func(*args, **kwargs)
        lst = [[t, xy[0], xy[1], xy[2]] for t, xy in zip(lst_t, lst_xy)]
        return pd.DataFrame(lst, columns=["t", "x", "y", "z"])
    return wrapper

@dataframe_3_especes
def Euler(f, y0, h, N, N_min=0):
    """Approximation par la méthode d'Euler explicite."""
    t = arange(N_min, 1 + N, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(N_min, len(t) - 1):
        y[k + 1] = y[k] + h*f(t[k], y[k])
    return t, y

@dataframe_3_especes
def Heun(f, y0, h, N, N_min=0):
    """Approximation par la méthode d'Heun."""
    t = arange(N_min, 1 + N, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(N_min+1, len(t)):
        y[k] = y[k-1] + (h/2.0)*(f(t[k-1],y[k-1]) + f(t[k],y[k-1] + h*f(t[k-1],y[k-1])))
    return t, y


class Lotka_Volterra_influence_homme(Lotka_Volterra):
    """Modèle dynamique par le système Lotka-Volterra avec influence de l'homme."""
    def __init__(self, x0, y0, r, p, m, q, h):
        """Paramètres + init Lotka_Volterra."""
        super().__init__(x0, y0, r, p, m, q)
        self.h = h  # τ de prélevement de l'homme
        self.orbite_x = (self.m-self.h) / self.q
        self.orbite_y = (self.r-self.h) / self.p

    def _variation_lapin(self, x, y) -> float:
        """Variation du nombre de lapins."""
        return (self.r-self.h)*x - self.p*x*y

    def _variation_lynx(self, x, y) -> float:
        """Variation du nombre de lynx."""
        return -(self.m-self.h)*y + self.q*x*y


class Lotka_Volterra_trois_especes_chaine_alimentaire(Lotka_Volterra):
    """Modèle dynamique par le système Lotka-Volterra à 3 espèces en chaîne alimentaire."""
    def __init__(self, x0, y0, z0, r, p, m, q, j, k, e):
        """Paramètres + init Lotka_Volterra."""
        super().__init__(x0, y0, r, p, m, q)
        self.z0 = z0
        self.j = j; self.k = k; self.e = e

    def _variation_souris(self, x, y) -> float:
        """Variation du nombre de souris."""
        return self.r*x - self.p*x*y

    def _variation_serpent(self, x, y, z) -> float:
        """Variation du nombre de lapins."""
        return self.q*self.p*x*y - self.m*y - self.e*z*y

    def _variation_aigle(self, y, z) -> float:
        """Variation du nombre de lynx."""
        return  self.k*self.e*z*y - self.j*z

    def fonction(self, t:float, Y:list):
        """Fonction pour Euler et Heun."""
        return array([self._variation_souris(Y[0], Y[1]), self._variation_serpent(Y[0], Y[1], Y[2]), self._variation_aigle(Y[1], Y[2])])

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float, ode='ref') -> pd.DataFrame:
        """Compute l'ode sélectionnée."""
        if ode == 'ref':
            t = t_min
            dx, dy, dz = self.x0, self.y0, self.z0
            lst = [[t, dx, dy, dz]]
            while lst[-1][0] <= t_max:
                t += h
                dx += self._variation_souris(dx, dy)*h
                dy += self._variation_serpent(dx, dy, dz)*h
                dz += self._variation_aigle(dy, dz)*h
                lst.append([t, dx, dy, dz])
            return pd.DataFrame(lst, columns=["t", "x", "y", "z"])
        if ode == 'Euler':
            return Euler(self.fonction, array([self.x0, self.y0, self.z0]), h, t_max, t_min)
        if ode == 'Heun':
            return Heun(self.fonction, array([self.x0, self.y0, self.z0]), h, t_max, t_min)
        raise ValueError(f"{ode} n'est pas un modèle valide parmis 'ref', 'Euler' et 'Heun'!")

    def affichage(self, t_min: int, t_max: int, h: float):
        """Population de souris, de serpents et d'aigles en fonction du temps."""
        df = self.modele_Lotka_Volterra(t_min, t_max, h)
        fig, ax = plt.subplots()
        ax.plot(df['t'], df['x'], 'b', label='Souris', linestyle='dotted')
        ax.plot(df['t'], df['z'], 'g', label='Serpents', linestyle='dashdot')
        ax.plot(df['t'], df['y'], 'r', label='Aigles', linestyle='dashed')
        plt.xlabel('Mois'); plt.ylabel('Population en unité')
        plt.title(f'Population de souris, de serpents et d\'aigles au cours du temps.\nConditions initiales : {self.x0} souris, {self.y0} serpents pour {self.z0} aigles\nsur une durée de {t_max-t_min} mois avec un pas de {h}.')
        # Rajoute un texte avec les conditions intiales à partir des params
        text =''; _dict = dict(self.__dict__)
        for key in self.__dict__.keys():
            if key[2:] not in 'xnyrpjkemq': _dict.pop(key)
            else: _dict[key[2:]] = _dict.pop(key)
        for key, value in _dict.items(): text += f'{key} : {value}\n'
        ax.text(.95, .23, text, ha='left', transform=fig.transFigure, fontsize= 12)
        ax.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        ax.grid(); plt.xlim(t_min, t_max); plt.ylim(0,)
        plt.show()
    
    def portrait_phase_3D(self, t_min: int, t_max: int, h: float, elev:float = 55, azim:float = 55, lst_condition_initiale:list = []):
        """ """
        if type(elev) is not (float and int):
            raise ValueError(f'{elev} doit être un réel!')
        if type(azim) is not (float and int):
            raise ValueError(f'{azim} doit être un réel!')
        if type(lst_condition_initiale) is not list:
            raise ValueError(f'{lst_condition_initiale} doit être une liste!')
        for element in lst_condition_initiale:
            if type(element) is not (float and int):
                raise ValueError(f'{element} doit être un réel ou un entier!')
        plt.figure(figsize=(16,9))
        ax = plt.axes(projection ='3d')
        ax.view_init(elev, azim)
        if len(lst_condition_initiale) != 0:
            for init in lst_condition_initiale:
                _dict = dict(self.__dict__)
                for key in self.__dict__.keys():
                    if key.startswith('__'): _dict[key[2:]] = _dict.pop(key)
                    else: _dict[key] = _dict.pop(key)
                _dict['x0'] = self.x0+init
                _dict['y0'] = self.y0+init
                _dict['z0'] = self.z0+init
                _dict.pop('orbite_x'); _dict.pop('orbite_y')
                ode = eval(self.__class__.__name__)(**_dict)   # Instancie un objet
                df = ode.modele_Lotka_Volterra(t_min, t_max, h)
                ax.plot3D(df['x'], df['y'], df['z'], label=f'{init}')
                plt.legend()
        else:
            df = self.modele_Lotka_Volterra(t_min, t_max, h)
            ax.plot3D(df['x'], df['y'], df['z'], 'green')
            ax.scatter(df['x'], df['y'], df['z'], s=5, c=arange(len(df['t'])), cmap='cividis')
        ax.set_xlabel('Souris'); ax.set_ylabel('Serpents'); ax.set_zlabel('Aigles')
        plt.title(f'Population de souris, de serpents et d\'aigles au cours du temps.\nConditions initiales : {self.x0} souris, {self.y0} serpents pour {self.z0} aigles\nsur une durée de {t_max-t_min} mois avec un pas de {h}.')
        plt.grid(); plt.show()
        

class Lotka_Volterra_trois_especes_2predateurs_1proie(Lotka_Volterra):
    """Modèle dynamique par le système Lotka-Volterra à 3 dont 2 prédateurs et 1 proie."""
    def __init__(self, x0, y0, z0, r, p, m, q, j, k, s):
        """Paramètres + init Lotka_Volterra."""
        super().__init__(x0, y0, r, p, m, q)
        self.z0 = z0
        self.j = j; self.k = k; self.s = s

    def _variation_lapin(self, x, y, z) -> float:
        """Variation du nombre de lapins."""
        return self.r*x -self.p*x*y -self.s*x*z

    def _variation_aigle(self, x, y) -> float:
        """Variation du nombre d'aigles."""
        return self.q*self.q*x*y - self.m*y
    
    def _variation_renard(self, x, z) -> float:
        """Variation du nombre de renards."""
        return self.k*self.s*x*z - self.j*z

    def fonction(self, t:float, Y:list):
        """Fonction pour Euler et Heun."""
        return array([self._variation_lapin(Y[0], Y[1], Y[2]), self._variation_aigle(Y[0], Y[1]), self._variation_renard(Y[0], Y[2])])

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float, ode='ref') -> pd.DataFrame:
        """Compute l'ode sélectionnée."""
        if ode == 'ref':
            t = t_min
            dx, dy, dz = self.x0, self.y0, self.z0
            lst = [[t, dx, dy, dz]]
            while lst[-1][0] <= t_max:
                t += h
                dx += self._variation_lapin(dx, dy, dz)*h
                dy += self._variation_aigle(dx, dy)*h
                dz += self._variation_renard(dx, dz)*h
                lst.append([t, dx, dy, dz])
            return pd.DataFrame(lst, columns=["t", "x", "y", "z"])
        if ode == 'Euler':
            return Euler(self.fonction, array([self.x0, self.y0, self.z0]), h, t_max, t_min)
        if ode == 'Heun':
            return Heun(self.fonction, array([self.x0, self.y0, self.z0]), h, t_max, t_min)
        raise ValueError(f"{ode} n'est pas un modèle valide parmis 'ref', 'Euler' et 'Heun'!")

    def affichage(self, t_min: int, t_max: int, h: float):
        """Population de lapins, d'aigles et de renards en fonction du temps."""
        df = self.modele_Lotka_Volterra(t_min, t_max, h)
        fig, ax = plt.subplots()
        ax.plot(df['t'], df['x'], 'b', label='Lapins', linestyle='dotted')
        ax.plot(df['t'], df['y'], 'g', label='Aigle', linestyle='dashdot')
        ax.plot(df['t'], df['z'], 'r', label='Renard', linestyle='dashed')
        plt.xlabel('Mois'); plt.ylabel('Population en unité')
        plt.title(f'Population de lapins, d\'aigles et de renards au cours du temps.\nConditions initiales : {self.x0} lapins, {self.y0} aigles pour {self.z0} renard\nsur une durée de {t_max-t_min} mois avec un pas de {h}.')
        # Rajoute un texte avec les conditions intiales à partir des params
        text =''; _dict = dict(self.__dict__)
        for key in self.__dict__.keys():
            if key[2:] not in 'xnyrpjkemqs': _dict.pop(key)
            else: _dict[key[2:]] = _dict.pop(key)
        for key, value in _dict.items(): text += f'{key} : {value}\n'
        ax.text(.95, .23, text, ha='left', transform=fig.transFigure, fontsize= 12)
        ax.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        ax.grid(); plt.xlim(t_min, t_max); plt.ylim(0,)
        plt.show()
    
    def portrait_phase_3D(self, t_min: int, t_max: int, h: float, elev:float = 55, azim:float = 55, lst_condition_initiale:list = []):
        """ """
        if type(elev) is not (float and int):
            raise ValueError(f'{elev} doit être un réel!')
        if type(azim) is not (float and int):
            raise ValueError(f'{azim} doit être un réel!')
        if type(lst_condition_initiale) is not list:
            raise ValueError(f'{lst_condition_initiale} doit être une liste!')
        for element in lst_condition_initiale:
            if type(element) is not (float and int):
                raise ValueError(f'{element} doit être un réel ou un entier!')
        plt.figure(figsize=(16,9))
        ax = plt.axes(projection ='3d')
        ax.view_init(elev, azim)
        if len(lst_condition_initiale) != 0:
            for init in lst_condition_initiale:
                _dict = dict(self.__dict__)
                for key in self.__dict__.keys():
                    if key.startswith('__'): _dict[key[2:]] = _dict.pop(key)
                    else: _dict[key] = _dict.pop(key)
                _dict['x0'] = self.x0+init
                _dict['y0'] = self.y0+init
                _dict['z0'] = self.z0+init
                _dict.pop('orbite_x'); _dict.pop('orbite_y')
                ode = eval(self.__class__.__name__)(**_dict)   # Instancie un objet
                df = ode.modele_Lotka_Volterra(t_min, t_max, h)
                ax.plot3D(df['x'], df['y'], df['z'], label=f'{init}')
                plt.legend()
        else:
            df = self.modele_Lotka_Volterra(t_min, t_max, h)
            ax.plot3D(df['x'], df['y'], df['z'], 'green')
            ax.scatter(df['x'], df['y'], df['z'], s=5, c=arange(len(df['t'])), cmap='cividis')
        ax.set_xlabel('Lapins'); ax.set_ylabel('Aigles'); ax.set_zlabel('Renards')
        plt.title(f'Population de lapins, d\'aigles et de renards au cours du temps.\nConditions initiales : {self.x0} lapins, {self.y0} aigles pour {self.z0} renard\nsur une durée de {t_max-t_min} mois avec un pas de {h}.')
        plt.grid(); plt.show()


if __name__ == '__main__':
    # Affichage 3D: %matplotlib Qt5
    if True:
        x: int = 2           # >= 0 -> nombre de souris
        y: int = 2           # >= 0 -> nombre de serpents
        z: int = 2           # >= 0 -> nombre de aigles
        
        r: float = 1.0       # >= 0 -> τ de reproduction intrinsèques des lapins
        p: float = 1.0       # >= 0 -> τ de mortalité des lapins due aux fouines rencontrés
        
        m: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des serpents
        q: float = 1.0       # >= 0 -> τ de reproduction des serpents en f° des souris mangés
        e: float = 1.0       # >= 0 -> τ de mortalité des serpents due aux aigles rencontrés
        
        j: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des aigles
        k: float = 1.0       # >= 0 -> τ de reproduction des aigles en f° des serpents mangés
        
        ode = Lotka_Volterra_trois_especes_chaine_alimentaire(x, y, z, r, p, m, q, j, k, e)
        ode.affichage(0, 50, 0.01)
        ode.portrait_phase_3D(0, 50, 0.01)
        ode.portrait_phase_3D(0, 50, 0.01, lst_condition_initiale=[x for x in range(-2, 12, 2)])
        
    if False:
        x: int = 2           # >= 0 -> nombre de lapins
        y: int = 2           # >= 0 -> nombre d'aigles
        z: int = 2           # >= 0 -> nombre de renards
        
        r: float = 1.0       # >= 0 -> τ de reproduction intrinsèques des lapins
        p: float = 1.0       # >= 0 -> τ de mortalité des lapins due aux aigles rencontrés
        s: float = 1.0       # >= 0 -> τ de mortalité des lapins due aux renards rencontrés
        
        m: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des aigles
        q: float = 1.0       # >= 0 -> τ de reproduction des aigles en f° des lapins mangés
        
        j: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des renards
        k: float = 1.0       # >= 0 -> τ de reproduction des renards en f° des lapins mangés
        
        ode = Lotka_Volterra_trois_especes_2predateurs_1proie(x, y, z, r, p, m, q, j, k, s)
        ode.affichage(0, 50, 0.01)
        ode.portrait_phase_3D(0, 50, 0.01)
        ode.portrait_phase_3D(0, 50, 0.01, lst_condition_initiale=[x for x in range(-2, 12, 2)])
        
    if False:
        x: int = 2           # >= 0 -> nombre de lapins
        y: int = 2           # >= 0 -> nombre de lynx
        r: float = 1.0       # >= 0 -> τ de reproduction intrinsèques des lapins
        p: float = 1.0       # >= 0 -> τ de mortalité des lapins due aux lynx rencontrés
        m: float = 1.0       # >= 0 -> τ de mortalité intrinsèques des lynx
        q: float = 1.0       # >= 0 -> τ de reproduction des lynx en f° des lapins mangés
        
        h: float = .2       # >= 0 -> τ de prélevement de l'homme

        ode = Lotka_Volterra(x, y, r, p, m, q)
        ode.affichage(0, 36, 0.01)
        
        ode = Lotka_Volterra_influence_homme(x, y, r, p, m, q, h)
        ode.affichage(0, 36, 0.01)
