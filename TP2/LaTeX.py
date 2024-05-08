# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:20:39 2024

@author: yan-s
"""
from time import sleep
from matplotlib import rcParams
from matplotlib.cm import rainbow
from numpy import array, arange, linspace, random, meshgrid
import pandas as pd
import matplotlib.pyplot as plt

rcParams['figure.dpi']:float = 300    # Augmenter la résolution des figures

def dataframe(func):
    """Permet d'obtenir un DataFrame prêt à être utilisé."""
    def wrapper(*args, **kwargs):
        lst_t, lst_xy = func(*args, **kwargs)
        lst = [[t, xy[0], xy[1]] for t, xy in zip(lst_t, lst_xy)]
        return pd.DataFrame(lst, columns=["t", "x", "y"])
    return wrapper

@dataframe
def Euler(f, y0, h, N, N_min=0):
    """Approximation par la méthode d'Euler explicite."""
    t = arange(N_min, 1 + N, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(N_min, len(t) - 1):
        y[k + 1] = y[k] + h*f(t[k], y[k])
    return t, y

@dataframe
def Heun(f, y0, h, N, N_min=0):
    """Approximation par la méthode d'Heun."""
    t = arange(N_min, 1 + N, h)
    y = list(range(len(t)))
    y[0] = y0
    for k in range(N_min+1, len(t)):
        y[k] = y[k-1] + (h/2.0)*(f(t[k-1],y[k-1]) + f(t[k],y[k-1] + h*f(t[k-1],y[k-1])))
    return t, y


class Lotka_Volterra:
    """Modèle dynamique par le système Lotka-Volterra."""
    def __init__(self, x0, y0, r, p, m, q):
        """Paramètres."""
        self.x0 = x0; self.y0 = y0
        self.r = r; self.p = p
        self.m = m; self.q = q
        self.orbite_x = self.m / self.q     # Pas d'orbite trivial
        self.orbite_y = self.r / self.p

    def __getattr__(self, name):
        """Getter pour toutes les variables."""
        return self.__dict__[f"__{name}"]

    def __setattr__(self, name, value):
        """Setter pour toutes les variables."""
        if name in ['x0', 'y0', 'n0', 'X']:
            if type(value) is not int:
                try: value = int(value)
                except ValueError: raise ValueError(f'{name} doit être un entier!')
        else:
            if type(value) is not float:    # Tente de convertir en <float>
                try: value = float(value)
                except ValueError: raise ValueError(f'{name} doit être un réel!')
        if value < 0:
            raise ValueError(f'{name} doit être supérieur ou égal à 0!')
        self.__dict__[f"__{name}"] = value

    def _variation_lapin(self, x, y) -> float:
        """Variation du nombre de lapins."""
        return self.r*x - self.p*x*y

    def _variation_lynx(self, x, y) -> float:
        """Variation du nombre de lynx."""
        return -self.m*y + self.q*x*y

    def fonction(self, t:float, Y:list):
        """Fonction pour Euler et Heun."""
        return array([self._variation_lapin(Y[0], Y[1]), self._variation_lynx(Y[0], Y[1])])

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float, ode='ref') -> pd.DataFrame:
        """Compute l'ode sélectionnée."""
        if ode == 'ref':
            t = t_min
            dx, dy = self.x0, self.y0
            lst = [[t, dx, dy]]
            while lst[-1][0] <= t_max:
                t += h
                dx += self._variation_lapin(dx, dy)*h
                dy += self._variation_lynx(dx, dy)*h
                lst.append([t, dx, dy])
            return pd.DataFrame(lst, columns=["t", "x", "y"])
        if ode == 'Euler':
            return Euler(self.fonction, array([self.x0, self.y0]), h, t_max, t_min)
        if ode == 'Heun':
            return Heun(self.fonction, array([self.x0, self.y0]), h, t_max, t_min)
        raise ValueError(f"{ode} n'est pas un modèle valide parmis 'ref', 'Euler' et 'Heun'!")

    def affichage(self, t_min: int, t_max: int, h: float):
        """Population de lièvres et de lynx en fonction du temps."""
        df = self.modele_Lotka_Volterra(t_min, t_max, h)
        fig, ax = plt.subplots()
        ax.plot(df['t'], df['x'], 'b', label='Lapins', linestyle='dotted')
        ax.plot(df['t'], df['y'], 'r', label='Lynx', linestyle='dashed')
        plt.xlabel('Mois'); plt.ylabel('Population en unité')
        plt.title(f'Population de lapins et de lynx au cours du temps.\nConditions initiales : {self.x0} lapins pour {self.y0} lynx\nsur une durée de {t_max-t_min} mois avec un pas de {h}.')
        # Rajoute un texte avec les conditions intiales à partir des params
        text =''; _dict = dict(self.__dict__)
        for key in self.__dict__.keys():
            if (key[2:] not in 'xnyrpjkemqXVh') and (key[2:] != 'alpha'): _dict.pop(key)
            else: _dict[key[2:]] = _dict.pop(key)
        for key, value in _dict.items(): text += f'{key} : {value}\n'
        ax.text(.92, .23, text, ha='left', transform=fig.transFigure, fontsize= 12)
        ax.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        ax.grid(); plt.xlim(t_min, t_max)#; plt.ylim(0,)
        plt.show()

    ## Méthode d'affichage de comparaison d'ode
    def affichage_comparaison_ode(self, t_min: int, t_max: int, h: float):
        """Nécessite une précision relativement élevé pour ode Euler."""
        fig, ax = plt.subplots(3,1, sharex=True, sharey=True)  # Créer la figure
        # Trace la courbe de référence avec un pas non modifiable de 10e-3
        df = self.modele_Lotka_Volterra(t_min, t_max, 10e-3, ode='ref')
        ax[0].plot(df['t'], df['x'], df['t'], df['y'])
        # Trace la courbe par approximation de Euler
        df = self.modele_Lotka_Volterra(t_min, t_max, h, ode='Euler')
        ax[1].plot(df['t'], df['x'], df['t'], df['y'], linestyle = '-')
        # Trace la courbe par approximation de Heun
        df = self.modele_Lotka_Volterra(t_min, t_max, h, ode='Heun')
        ax[2].plot(df['t'], df['x'], label='Lapins')
        ax[2].plot(df['t'], df['y'], label='Lynx')
        ax[2].legend(title='Courbes', bbox_to_anchor=(1.05, .2), shadow=True)
        # Les noms des axes
        ax[2].set_xlabel('Temps')
        ax[0].set_ylabel('Ref'); ax[1].set_ylabel('Euler'); ax[2].set_ylabel('Heun')
        # Textes sur le coté
        fig.suptitle(f'Population de lapins et de\nlynx au cours du temps.\nConditions initiales :\n{self.x0} lapins pour {self.y0} lynx sur\nune durée de {t_max-t_min} mois\navec un pas de {h}.', x=0.92, y=0.86, ha='left', fontsize= 10)
        if type(self) is (Lotka_Volterra or Lotka_Volterra_limite):
            plt.text(.92, .43, f'r: {self.r}      p: {self.p}\nm: {self.m}    q: {self.q}\nPas: {h}\nN: {t_max}', transform=fig.transFigure, ha='left', fontsize= 10)
        if type(self) is (Lotka_Volterra_limite or Lotka_Volterra_chngt_var_limite):
            plt.text(.92, .41, f'a: {self.alpha}\nPas: {h}\nN: {t_max}', transform=fig.transFigure, ha='left', fontsize= 10)
        # Grid et trace
        ax[0].grid(); ax[1].grid(); ax[2].grid()
        plt.xlim(t_min,t_max); plt.show()

    def affichage_comparaison_parallele_ode(self, t_min: int, t_max: int, h: float):
        """Nécessite une précision relativement élevé pour ode Euler."""
        fig, ax = plt.subplots(2,1, sharex=True, sharey=True)  # Créer la figure
        # Trace la courbe de référence avec un pas non modifiable de 10e-3
        df = self.modele_Lotka_Volterra(t_min, t_max, 10e-3, ode='ref')
        ax[0].plot(df['t'], df['x'], linestyle = ':', label='ref', linewidth=4)
        ax[1].plot(df['t'], df['y'], linestyle = ':', label='ref', linewidth=4)
        # Trace la courbe par approximation de Euler
        df = self.modele_Lotka_Volterra(t_min, t_max, h, ode='Euler')
        ax[0].plot(df['t'], df['x'], linestyle = '--', label='Euler')
        ax[1].plot(df['t'], df['y'], linestyle = '--', label='Euler')
        # Trace la courbe par approximation de Heun
        df = self.modele_Lotka_Volterra(t_min, t_max, h, ode='Heun')
        ax[0].plot(df['t'], df['x'], linestyle = '-.', label='Heun')
        ax[1].plot(df['t'], df['y'], linestyle = '-.', label='Heun')
        ax[1].legend(title='Courbes', bbox_to_anchor=(1.05, .81), shadow=True)
        # Les noms des axes
        ax[1].set_xlabel('Temps')
        ax[0].set_ylabel('Lapins'); ax[1].set_ylabel('Lynx')
        # Textes sur le coté
        fig.suptitle(f'Population de lapins et de\nlynx au cours du temps.\nConditions initiales :\n{self.x0} lapins pour {self.y0} lynx sur\nune durée de {t_max-t_min} mois\navec un pas de {h}.', x=0.92, y=.86, ha='left', fontsize= 10)
        if type(self) is (Lotka_Volterra or Lotka_Volterra_limite):
            plt.text(.92, .41, f'r: {self.r}      p: {self.p}\nm: {self.m}    q: {self.q}\nPas: {h}\nN: {t_max}', transform=fig.transFigure, ha='left', fontsize= 10)
        if type(self) is (Lotka_Volterra_limite or Lotka_Volterra_chngt_var_limite):
            plt.text(.92, .45, f'a: {self.alpha}\nPas: {h}\nN: {t_max}', transform=fig.transFigure, ha='left', fontsize= 10)
        # Grid et trace
        ax[0].grid(); ax[1].grid()
        plt.xlim(t_min,t_max); plt.show()

    ## Méthode d'affichage un peu plus complexes
    def affichage_variation__τ(self, t_min: int, t_max: int, h: float, var:str, Delta:float = 0.5):
        """Affichage des varations des τ.
        var : str -> 'r' | 'p' | 'm' | 'q'.
        Delta : float, optional -> Varations totales avec un pas de 0.1, in fine le nombre de courbes. 0.5 par défaut.
        """
        if f'__{var}' not in self.__dict__:
            raise AttributeError(f'__{var} doit être dans {self.__dict__.keys()}!')
        if type(Delta) is not (float or int):
            raise ValueError(f'{Delta} doit être un réel ou un entier!')
        deltas = arange(eval(f'self.__{var}'), eval(f'self.__{var}') + Delta, 0.1)   # On créer l'intervalle
        colors = rainbow(linspace(0, 1, random.random((10,len(deltas))).shape[0]))  # On génére des couleurs pour chaque courbe
        fig, ax = plt.subplots(2,1, sharex='row', sharey='col') # Créer la figure
        for delta, i in zip(deltas, range(len(deltas))):    # Créer chaque courbe
            setattr(self, var, float(delta))    # Réactualise la variable var de l'instance
            df = self.modele_Lotka_Volterra(t_min, t_max, h)
            ax[0].plot(df['t'], df['x'], color = colors[i],  linestyle = '-', label = r"$\delta = $" + "{0:.2f}".format(delta))
            ax[1].plot(df['t'], df['y'], color = colors[i], linestyle = '-', label = r" $\delta = $" + "{0:.2f}".format(delta))
            ax[1].legend()
        ax[0].set_xlabel('Mois'); ax[1].set_xlabel('Mois')
        ax[0].set_ylabel('Lapins'); ax[1].set_ylabel('Lynx')
        ax[1].legend(title=f'Variation de {var}', bbox_to_anchor=(1.05, 1), shadow=True)
        fig.suptitle(f'Population de lapins et de\nlynx au cours du temps.\nConditions initiales :\n{self.x0} lapins pour {self.y0} lynx sur\nune durée de {t_max-t_min} mois\navec un pas de {h}.', x=0.92, y=0.8, ha='left', fontsize= 10)
        ax[0].grid(); ax[1].grid(); plt.show()

    def portrait_phase(self, t_min: int, t_max: int, h: float, lst_condition_initiale:list = [x for x in range(-2, 12, 2)], orbite:bool = False):
        """ """
        if type(orbite) is not bool:
            raise ValueError(f'{orbite} doit être un booléen!')
        if type(lst_condition_initiale) is not list:
            raise ValueError(f'{lst_condition_initiale} doit être une liste!')
        for element in lst_condition_initiale:
            if type(element) is not (float and int):
                raise ValueError(f'{element} doit être un réel ou un entier!')
        for init in lst_condition_initiale:     # Pas très poo mais fonctionne v(´・∀・｀*)v
            _dict = dict(self.__dict__) # Nouveau dictionnaire avec paramètres modifiés
            for key in self.__dict__.keys():
                if key.startswith('__'): _dict[key[2:]] = _dict.pop(key)
                else: _dict[key] = _dict.pop(key)
            _dict['x0'] = self.x0+init; _dict['y0'] = self.y0+init
            _dict.pop('orbite_x'); _dict.pop('orbite_y')
            ode = eval(self.__class__.__name__)(**_dict)   # Instancie un objet
            df = ode.modele_Lotka_Volterra(t_min, t_max, h)
            plt.plot(df['x'], df['y'], label=f'{init}', linestyle='solid')
        if orbite:  # Trace l'orbite (on se passe de l'obite trivial)
            plt.plot(self.orbite_x, self.orbite_y, 'x', label='Orbite')
        plt.xlabel('Lapins'); plt.ylabel('Lynx')
        plt.title(f'Population de lapins et de lynx au cours du temps\nPour des conditions initiales variantes sur une durée de {t_max-t_min} mois avec un pas de {h}.')
        plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        plt.grid(); plt.show()

    def affichage_champ_direction(self, t_min: int, t_max: int, h: float, lst_condition_initiale:list = [x for x in range(-2, 12, 2)], orbite:bool = False, nb_fleche:int = 30):
        """ """
        if type(orbite) is not bool:
            raise ValueError(f'{orbite} doit être un booléen!')
        if type(lst_condition_initiale) is not list:
            raise ValueError(f'{lst_condition_initiale} doit être une liste!')
        for element in lst_condition_initiale:
            if type(element) is not (float and int):
                raise ValueError(f'{element} doit être un réel ou un entier!')
        for init in lst_condition_initiale:     # Pas très poo mais fonctionne v(´・∀・｀*)v
            _dict = dict(self.__dict__) # Nouveau dictionnaire avec paramètres modifiés
            for key in self.__dict__.keys():
                if key.startswith('__'): _dict[key[2:]] = _dict.pop(key)
                else: _dict[key] = _dict.pop(key)
            _dict['x0'] = self.x0+init; _dict['y0'] = self.y0+init
            _dict.pop('orbite_x'); _dict.pop('orbite_y')
            ode = eval(self.__class__.__name__)(**_dict)   # Instancie un objet
            df = ode.modele_Lotka_Volterra(t_min, t_max, h)
            plt.plot(df['x'], df['y'], label=f'{init}', linestyle='solid')
        if orbite:  # Trace l'orbite (on se passe de l'obite trivial)
            plt.plot(self.orbite_x, self.orbite_y, 'x', label='Orbite')
        plt.xlabel('Lapins'); plt.ylabel('Lynx')
        plt.title(f'Population de lapins et de lynx au cours du temps\nPour des conditions initiales variantes sur une durée de {t_max-t_min} mois avec un pas de {h}.')
        plt.xlim(0,); plt.ylim(0,)
        mat_x, mat_y = meshgrid(linspace(plt.xlim()[0],round(plt.xlim()[1]),nb_fleche), linspace(plt.ylim()[0],round(plt.ylim()[1]),nb_fleche))
        plt.quiver(mat_x,mat_y,self._variation_lapin(mat_x,mat_y),self._variation_lynx(mat_x,mat_y),pivot='mid')
        plt.legend(bbox_to_anchor=(1.05, 1), shadow=True)
        plt.grid(); plt.show()


class Lotka_Volterra_chngt_var(Lotka_Volterra):
    """Modèle dynamique par le système Lotka-Volterra avec changement de variable."""
    def __init__(self, x0, y0, r, p, m, q, alpha):
        """Paramètres + init Lotka_Volterra."""
        super().__init__(x0, y0, r, p, m, q)
        self.alpha = alpha  # Changement de variable
        self.orbite_x = self.m / self.q 
        self.orbite_y = self.r / self.p

    def _variation_lapin(self, v, w) -> float:
        """Variation du nombre de lapins."""
        return v - v*w

    def _variation_lynx(self, v, w) -> float:
        """Variation du nombre de lynx."""
        return -self.alpha*w + v*w

    def modele_Lotka_Volterra(self, t_min: int, t_max: int, h: float, ode='ref') -> pd.DataFrame:
        """Avec changement de variable."""
        if ode == 'ref':
            s = self.r*t_min
            dv, dw = self.x0, self.y0
            lst = [[s, dv, dw]]
            while lst[-1][0] <= t_max:
                s += h  # *r
                dv += (self.q/self.r) * self._variation_lapin(dv, dw)*h
                dw += (self.p/self.r) * self._variation_lynx(dv, dw)*h
                lst.append([s, dv, dw])
            return pd.DataFrame(lst, columns=["t", "x", "y"])
        if ode == 'Euler':
            return Euler(self.fonction, array([self.x0, self.y0]), h, t_max, t_min)
        if ode == 'Heun':
            return Heun(self.fonction, array([self.x0, self.y0]), h, t_max, t_min)
        raise ValueError(f"{ode} n'est pas un modèle valide parmis 'ref', 'Euler' et 'Heun'!")

class Lotka_Volterra_limite(Lotka_Volterra):
    """Modèle dynamique par le système Lotka-Volterra limité."""
    def __init__(self, x0, y0, r, p, m, q, X):
        """Paramètres + init Lotka_Volterra."""
        super().__init__(x0, y0, r, p, m, q)
        self.X = X  # Seuil
        self.orbite_y = (self.r/self.p)*(1-(self.orbite_x/self.X))

    def _variation_lapin(self, x, y) -> float:
        """Variation du nombre de lapins."""
        return self.r*x*(1-(x/self.X)) - self.p*x*y


class Lotka_Volterra_chngt_var_limite(Lotka_Volterra_chngt_var):
    """Modèle dynamique par le système Lotka-Volterra limité avec changement de variable."""
    def __init__(self, x0, y0, r, p, m, q, V, alpha):
        """Paramètres + init Lotka_Volterra."""
        super().__init__(x0, y0, r, p, m, q, alpha)
        self.V = V  # Seuil
        self.orbite_y = (self.r/self.p)*(1-(self.orbite_x/self.V))

    def _variation_lapin(self, v, w) -> float:
        """Variation du nombre de lapins."""
        return v*(1-(v/self.V)) - v*w


## Menu
def menu():
    """Même un p'tit menu!"""
    def menu_choix(choix, possibilite:str)->int:
        for option in choix.keys(): print(f"{option} - {choix[option]}")
        while (entree:=input("Entrée : ")) not in possibilite:
            print('Mauvaise entrée!')
            sleep(1); print("\033[H\033[J", end="")
            menu_choix(choix, possibilite)
        return int(entree)

    def title(text:str):
        """Colore {text} en bleu."""
        if type(text) is not str:
             raise TypeError(f'{text} doit être une chaîne de caractère!')
        return f'\033[1;34m{text}\033[0;0m'

    def choix_modele()->tuple((str,int)):
        print(title('Veuillez choisir le modèle.'))
        match menu_choix({1:"Modèle 1 - LV simple",2:"Modèle 2 - LV avec changement de variable",3:"Modèle 3 - LV simple et limite de lapins ",4:"Modèle 4 - LV avec changement de variable et limite de lapins",0:"Sortie"},'01234'):
            case 0: raise KeyboardInterrupt('-Fin')
            case 1: return 'Lotka_Volterra',1
            case 2: return 'Lotka_Volterra_chngt_var',2
            case 3: return 'Lotka_Volterra_limite',3
            case 4: return 'Lotka_Volterra_chngt_var_limite',4

    def choix_donnee(num)->str:
        print('\n' + title("Veuillez choisir les données du modèle."))
        match menu_choix({1:"Données préchargées (mieux pour les affichages simples)",2:"Données préchargées (mieux pour les affichages d'orbite)",3:"Entrée manuelle",0:"Sortie"},'0123'):
            case 0: raise KeyboardInterrupt('-Fin')
            case 1:
                match num:
                    case 1: return '2,2,1.0,1.0,1.0,1.0'
                    case 2: return '2,2,1.0,1.0,1.0,1.0,1.0'
                    case 3: return '2,2,1.0,1.0,1.0,1.0,50'
                    case 4: return '2,2,1.0,1.0,1.0,1.0,50,1.0'
            case 2:
                match num:
                    case 1: return '4,10,1.5,0.05,0.48,0.05'
                    case 2: return '4,10,1.5,0.05,0.48,0.05,1.0'
                    case 3: return '4,10,1.5,0.05,0.48,0.05,50'
                    case 4: return '4,10,1.5,0.05,0.48,0.05,50,1.0'
            case 3:
                match num:
                    case 1: return f'{input("nombre de lapins (int>=0) :")},{input("nombre de lynx (int>=0) :")},{input("τ de reproduction intrinsèques des lapins (float>=0) :")},{input("τ de mortalité des lapins due aux lynx rencontrés (float>=0) :")},{input("τ de mortalité intrinsèques des lynx (float>=0) :")},{input("τ de reproduction des lynx en f° des lapins mangés (float>=0) :")}'
                    case 2: return f'{input("nombre de lapins (int>=0) :")},{input("nombre de lynx (int>=0) :")},{input("τ de reproduction intrinsèques des lapins (float>=0) :")},{input("τ de mortalité des lapins due aux lynx rencontrés (float>=0) :")},{input("τ de mortalité intrinsèques des lynx (float>=0) :")},{input("τ de reproduction des lynx en f° des lapins mangés (float>=0) :")},{input("a (float>=0) :")}'
                    case 3: return f'{input("nombre de lapins (int>=0) :")},{input("nombre de lynx (int>=0) :")},{input("τ de reproduction intrinsèques des lapins (float>=0) :")},{input("τ de mortalité des lapins due aux lynx rencontrés (float>=0) :")},{input("τ de mortalité intrinsèques des lynx (float>=0) :")},{input("τ de reproduction des lynx en f° des lapins mangés (float>=0) :")},{input("limite du nombre de lapins (int>=0) :")}'
                    case 4: return f'{input("nombre de lapins (int>=0) :")},{input("nombre de lynx (int>=0) :")},{input("τ de reproduction intrinsèques des lapins (float>=0) :")},{input("τ de mortalité des lapins due aux lynx rencontrés (float>=0) :")},{input("τ de mortalité intrinsèques des lynx (float>=0) :")},{input("τ de reproduction des lynx en f° des lapins mangés (float>=0) :")},{input("limite du nombre de lapins (int>=0) :")},{input("a (float>=0) :")}'

    def choix_intervalle()->str:
        print('\n' + title("Veuillez choisir l\'intervalle d\'affichage."))
        match menu_choix({1:"Intervalle prédéfini",2:"Entrée manuelle",0:"Sortie"},'012'):
            case 0: raise KeyboardInterrupt('-Fin')
            case 1: return '0, 50, 0.0005'
            case 2: return f'{input("Début (float>=0): ")},{input("Fin (float>=0 et >=start): ")},{input("Pas (float>0): ")}'

    def choix_affichage(num)->tuple((str,str)):
        print('\n' + title('Veuillez choisir l\'affichage.'))
        match menu_choix({1:"Affichage simple",2:"Affichage avec variations des coefficients",3:"Portrait de phase (visualisation des orbites)",4:"Affichage comparaison des ODEs",5:"Affichage comparaison parallèle des ODEs",0:"Sortie"},'012345'):
            case 0: raise KeyboardInterrupt('-Fin')
            case 1: return 'affichage',None
            case 2:
                match num:
                    case 1 | 3:
                        print('\n' + title('Veuillez choisir le paramètre à faire varier.'))
                        match menu_choix({1:"τ de reproduction intrinsèques des lapins",2:"τ de mortalité des lapins due aux lynx rencontrés",3:"τ de mortalité intrinsèques des lynx",4:"τ de reproduction des lynx en f° des lapins mangés",0:"Sortie"},'01234'):
                            case 0: raise KeyboardInterrupt('-Fin')
                            case 1: return 'affichage_variation__τ',"'r'"  # Petit artifice bien pratique pour le eval
                            case 2: return 'affichage_variation__τ',"'p'"
                            case 3: return 'affichage_variation__τ',"'m'"
                            case 4: return 'affichage_variation__τ',"'q'"
                    case 2 | 4:
                        raise ValueError('-Non défini')
            case 3:
                print(f"\n{title('Variations des conditions initiales.')}\nNombres (int) séparés par <espace>\nNB: <entree> pour liste par défaut.")
                lst = [int(x) for x in input('>>').split()]
                if len(lst) == 0:
                    lst = [x for x in range(-2, 12, 2)]
                print('\n' + title('Activer les points d\'équilibres?'))
                match menu_choix({1:"Activé point d'équilibre",2:"Désactivé point d'équilibre",0:"Sortie"},'01234'):
                    case 0: raise KeyboardInterrupt('-Fin')
                    case 1: return 'portrait_phase',f'lst_condition_initiale={lst},orbite=True'
                    case 2: return 'portrait_phase',f'lst_condition_initiale={lst},orbite=False'
            case 4: return 'affichage_comparaison_ode', None
            case 5: return 'affichage_comparaison_parallele_ode',None
    try:
        modele, num = choix_modele()
        donnee = choix_donnee(num)
        intervalle = choix_intervalle()
        affichage,sup = choix_affichage(num)
        # Dangereux, mais fonctionne... # eval(modele(donnee).affichage(intervalle,sup))
        eval(f'{modele}(*{eval(donnee)}).{affichage}{f'(*{eval(intervalle)}, {sup})' if sup else f'(*{eval(intervalle)})'}')
    except KeyboardInterrupt as err:
        print(err)
    except Exception as err:
        print(err)

if __name__ == '__main__':
    menu()
