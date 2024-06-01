# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:25:06 2024

@author: yan-s
"""
from time import sleep
from os.path import isfile
from matplotlib import rcParams
from matplotlib.cm import rainbow
from numpy import linspace, random, ones
import matplotlib.pyplot as plt
import pandas as pd

rcParams['figure.dpi']:float = 300    # Augmenter la résolution des figures


class Data:
    """Créer les données dans des .csv + méthodes de lecture."""
    @staticmethod
    def creer_donnees():
        """Données provenant de l'INSEE, et assemblées main."""
        donnees_main = {'Années': ['1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'],
                        'de_0_à_19_ans': ['15999391','15919634','15852690','15793067','15719647','15605455','15473515','15330473','15179775','15084361','15058227','15055742','15026922','15017908','15047287','15067857','15091374','15116574','15183494','15242403','15280401','15315094','15337575','15368840','15406592','15440408','15457656','15513096','15588708','15651778','15645526','15616148','15611515','15584916'],
                        'de_20_à_59_ans': ['29296183','29495669','29685169','29875387','30093767','30280463','30506157','30730777','30940136','31068721','31160552','31232352','31342595','31455770','31667286','31962065','32288026','32560892','32760470','32971217','33193638','33219983','33170779','33107066','33023612','32939901','32910610','32853247','32791541','32680632','32562257','32448507','32383772','32356902'],
                        'de_60_à_64_ans': ['2887651','2870641','2869965','2881891','2892072','2916128','2925603','2941799','2921205','2912957','2858806','2816744','2765130','2737639','2712836','2675214','2610227','2606797','2640164','2682701','2762859','3051869','3325595','3568379','3795165','4022812','4034993','4029597','3998555','3979695','3949846','3953989','3951924','3974120'],
                        'de_65_à_99_ans': ['7228013','7395836','7558318','7719465','7871514','8038615','8205258','8366112','8523892','8686496','8858374','9011180','9164315','9285296','9430789','9561436','9696272','9817578','9921293','10066943','10162835','10208292','10300917','10421424','10539866','10667223','10972712','11301925','11649154','11988716','12311163','12620489','12896826','13180830'],
                        'centenaires': ['2886','3126','3352','3476','3760','3885','4379','4458','5014','5648','6303','7107','7259','7754','8161','9444','11031','11945','12393','13182','13575','14426','14940','15810','16688','18221','19897','20887','22031','24085','21044','19383','18513','18520'],
                        'Naissances': ['778468','767828','771268','765473','762407','759056','743658','711610','710993','729609','734338','726768','738080','744791','774782','770945','761630','761464','767816','774355','796896','785985','796044','793420','802224','792996','790290','781621','781167','760421','744697','730242','719737','714029']}
        donnees_2016 = {'Âge': ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99'],
                        'Population': ['765550','788646','792333','804236','816258','836823','834997','839809','836583','849949','833020','828197','823916','828573','430033','859176','422275','814634','792026','787204','376849','737459','728187','761786','385005','784180','784047','792374','792888','395458','410141','804440','405286','847033','856278','867851','417763','808050','816877','798343','820272','867477','914322','937883','930282','912733','897889','892061','885433','459802','909893','920256','463651','877540','876823','872333','421833','845723','842277','833099','822054','821117','799913','811671','788895','428971','800702','418084','776321','734510','553400','537175','518715','478983','424883','440005','201519','442518','183665','243626','404609','398592','369172','366529','338247','319195','277738','251466','221891','134597','170448','141408','86225','25053','77793','60573','27183','17293','11488','7637'],
                        'Quotient_de_mortalité_pour_100_000': ['354','62','24','18','14','12','11','9','8','8','8','8','9','10','13','17','22','30','39','48','55','59','61','64','66','70','71','74','76','78','80','83','86','91','95','100','105','113','125','136','148','161','175','197','215','236','263','293','321','354','389','424','463','512','570','640','694','767','834','886','979','1032','1102','1181','1244','1319','1390','1471','1590','1693','1830','1952','2102','2298','2461','2692','2954','3284','3612','4089','4536','5103','5772','6465','7326','8269','9361','10557','11953','13317','14814','16554','18387','20313','22524','24750','26700','28837','31396','33844']}
        for chemin in ('donnees_main', 'donnees_2016'):
            if not isfile(f'{chemin}.csv'):
                df = pd.DataFrame(eval(chemin))
                df.to_csv(f'{chemin}.csv', sep=',', encoding='utf-8')
    
    @staticmethod
    def lecture(chemin: str) -> pd.DataFrame:
        """Lit le contenu d'un fichier .csv entré."""
        if type(chemin) != str:   # Vérifie que l'argument fourni est correcte
            raise TypeError(f'{chemin} n\'est pas une chaîne de caractère!')
        if not isfile(chemin):   # Vérifie que le fichier existe
            raise FileNotFoundError('Le fichier n\'existe pas!')
        if not chemin.endswith('.csv'): # Vérifie que l'argument fourni est correcte
            raise ValueError(f'{chemin} n\'est pas un fichier .csv!')
        return pd.read_csv(chemin)


class ClasseAge:
    """Simple structure de données."""
    def __init__(self, age:int, nombre:int, taux_morta:float):
        self.age = age
        self.nombre = nombre
        self.taux_morta = taux_morta

    def __getattr__(self, name):
        """Getter pour toutes les variables."""
        return self.__dict__[name]

    def __setattr__(self, name, value):
        """Setter pour toutes les variables."""
        if name in ['age','nombre']:
            if type(value) is not int:
                try: value = int(value)
                except ValueError: raise ValueError(f'{name} doit être un entier!')
            if value < 0:
                raise ValueError(f'{name} doit être supérieur ou égal à 0!')
        if name in ['taux_morta']:
            if type(value) is not float:    # Tente de convertir en <float>
                try: value = float(value)
                except ValueError: raise ValueError(f'{name} doit être un réel!')
        self.__dict__[name] = value

    def __call__(self):
        """Formate object pour la suite de programme."""
        return [int(self.age), int(self.nombre), float(self.taux_morta)]


class GroupAgrege:
    """Simple structure de données."""
    def __init__(self, age:str, nombre:int):
        self.age = age
        self.nombre = nombre

    @property
    def nombre(self):
        return self.__nombre
    @nombre.setter
    def nombre(self, value):
        if not isinstance(value, int):
            raise TypeError(f"{value} n'est pas un entier!")
        if value < 0:
            raise ValueError(f'{value} doit être supérieur ou égal à 0!')
        self.__nombre = value

    @property
    def age(self):
        return self.__age
    @age.setter
    def age(self, value):
        if not isinstance(value, str):
            raise TypeError(f"{value} n'est pas une chaîne de caratèreZ!")
        self.__age = value

    def __call__(self):
        """Formate object pour la suite de programme."""
        return {str(self.age) : int(self.nombre)}


class Modele:
    """Le gros du projet."""
    def __init__(self, annee_visee:int, taux= 0.01, output=False):
        self.annee_visee = annee_visee
        self.liste_classe_age:list(ClasseAge) = self.add_liste_classe_age()
        self.output = output
        self.taux = taux
        # Consigne des résultats
        self.__resultat_agrege = []
        self.__resultat_classe_age = []

    @property
    def annee_visee(self):
        return self.__annee_visee
    @annee_visee.setter
    def annee_visee(self, value):
        if not isinstance(value, int):
            raise TypeError(f"{value} n'est pas un entier!")
        if not (1986 <= value <= 2019):
            raise ValueError(f"{value} doit être compris entre 1986 et 2019 (inclus)!")
        self.__annee_visee = value

    @property
    def taux(self):
        return self.__taux
    @taux.setter
    def taux(self, value):
        if type(value) is not float:    # Tente de convertir en <float>
            try: value = float(value)
            except ValueError: raise ValueError(f'{value} doit être un réel!')
        if not (0 < value):
            raise ValueError(f"{value} doit être strictement positif!")
        self.__taux = value

    @property
    def output(self):
        return self.__output
    @output.setter
    def output(self, value):
        if type(value) != bool:
            raise TypeError(f"{value} n'est pas un booléen!")
        self.__output = value

    def add_liste_classe_age(self):
        """ """
        df = Data.lecture('donnees_2016.csv')
        liste_classe_age = []
        for age, nombre, morta in zip(df['Âge'], df['Population'], df['Quotient_de_mortalité_pour_100_000']):
            liste_classe_age.append(ClasseAge(age, nombre, int(morta)/100000))
        return liste_classe_age

    def modele(self):
        if len(self.liste_classe_age) == 0:
            raise Exception('Les données ne sont pas chargées!')
        df = Data.lecture('donnees_main.csv')

        if (periode:=2016-self.annee_visee) < 0: # Modélise le futur
            for annee in range(abs(periode)+1):
                # On rajoute les nouveaux-nés
                taux_morta = self.liste_classe_age[0].taux_morta
                generation_0 = ClasseAge(0, int(list(df.loc[df['Années'] == 2016+annee]['Naissances'])[0]), taux_morta)
                for idx, classe_age in enumerate((self.liste_classe_age)):
                    # On constate les décès en f° du taux_morta
                    classe_age.nombre *= (1-classe_age.taux_morta)
                    try:
                        classe_age.taux_morta = self.liste_classe_age[idx+1].taux_morta
                    except IndexError:
                        # Supprime gen100
                        del self.liste_classe_age[idx]
                    finally:
                        # On fête les anniversaires
                        classe_age.age += 1
                # rajoute gen0 position 0
                self.liste_classe_age.insert(0, generation_0)
                # Différentes fonctions consignant les résultats
                self.comparaison_agrege(tmp:=(2016+annee))
                self.comparaison_classe_age(tmp)

        elif periode == 0: # Si l'année 2016 est demandée
            self.comparaison_agrege(2016) # On observe une déviation entre les données
        else:   # Modélise le passé
            for annee in range(abs(periode)+1):
                # On rajoute les centenaires
                taux_morta = self.liste_classe_age[-1].taux_morta
                nombre = round(self.taux * int(list(df.loc[df['Années'] == 2016-annee]['Naissances'])[0]))
                generation_99 = ClasseAge(99, nombre, taux_morta)
                self.liste_classe_age.reverse() # Retourne liste
                for idx, classe_age in enumerate(self.liste_classe_age):
                    # On constate les décès en f° du taux_morta, et on les resucite
                    classe_age.nombre /= (1-classe_age.taux_morta)
                    try:
                        classe_age.taux_morta = self.liste_classe_age[idx+1].taux_morta
                        # On fête les anniversaires
                        classe_age.age -= 1
                    except IndexError:
                        # Supprime gen-1
                        self.liste_classe_age.remove(classe_age)
                self.liste_classe_age.reverse() # Retourne liste (à l'endroit)
                # Rajoute gen99 dernière position
                self.liste_classe_age.append(generation_99)
                # Différentes fonctions consignant les résultats
                self.comparaison_agrege(tmp:=(2016-annee))
                self.comparaison_classe_age(tmp)

    ## Fonctions pour agréger
    def modele_agrege(self)-> dict:
        """Récupère données modèles et retourne dict."""
        def agrege(age_min:int, age_max:int)->dict:
            age = f"{age_min} - {age_max-1} ans"
            nombre = sum(obj.nombre for obj in self.liste_classe_age[age_min:age_max])
            tmp = GroupAgrege(age, nombre)
            return {tmp.age: tmp.nombre}
        dct = {**agrege(0, 20),**agrege(20, 60),**agrege(60, 65),**agrege(65, 100)}
        dct.update({'Total': sum(val for val in dct.values())}) # Calcul le total
        return dct

    def main_agrege(self, annee)-> dict:
        """Récupère données .csv et retourne dict."""
        df = Data.lecture('donnees_main.csv')
        row = df.loc[df['Années'] == annee].index[0]
        dct = {
            **GroupAgrege('0 - 19 ans', int(df.iloc[row]['de_0_à_19_ans']))(),
            **GroupAgrege('20 - 59 ans', int(df.iloc[row]['de_20_à_59_ans']))(),
            **GroupAgrege('60 - 64 ans', int(df.iloc[row]['de_60_à_64_ans']))(),
            **GroupAgrege('65 - 99 ans', int(df.iloc[row]['de_65_à_99_ans']))()}
        dct.update({'Total': sum(val for val in dct.values())}) # Calcul le total
        return dct

    ## Différentes méthodes pour pré-formater et comparer le modèle et les données
    def comparaison_agrege(self, annee:int)->pd.DataFrame:
        """ """
        model = self.modele_agrege()    # Agrège les données du modèle dans un dictionnaire
        data = self.main_agrege(annee)   # Agrège les données connus dans un dictionnaire
        dct = {}    # Créer un dictionnaire de comparaison
        #
        if self.output: print(f'Déviation du modèle en {annee}: (données/modèle)')
        #
        dct.update({'Année': int(annee)})
        for key in data.keys(): # data.keys == main.keys
            dct.update(tmp:={key: round((data[key]/model[key] * 100)-100, 2)})
            if self.output: print(f'{list(tmp.keys())[0]} : {list(tmp.values())[0]} %')
        # Ajoute à la liste de dict
        self.__resultat_agrege.append(dct)

    def comparaison_classe_age(self, annee:int)->pd.DataFrame:
        """ """
        #
        if self.output: print(f'Déviation du modèle en {annee}: (données/modèle)')
        #
        for classe_age in self.liste_classe_age:
            self.__resultat_classe_age.append([int(annee), classe_age()[0], classe_age()[1]])

    def affichage_deviation_agrege(self):
        """Déviation relative en fonction de l'année sur la période 2016 à l'année visée."""
        t_min, t_max = min(2016,self.annee_visee), max(2016,self.annee_visee)
        df = pd.DataFrame(self.__resultat_agrege)
        #
        for column in df:
            if not column == 'Année':
                plt.plot(df['Année'], df[column], label=column)
        # Différentes configurations pour rendre le graphe plus lisible
        plt.title(f'Graphique de déviation sur la période 2016 - {self.annee_visee}\navec un taux de centenaires de {self.taux}')
        plt.legend(title='Groupe agrégé', bbox_to_anchor=(1.05, 1), shadow=True)
        plt.xlabel('Années'); plt.ylabel('Déviation en pourcentage')
        plt.grid(); plt.xlim(t_min, t_max)
        plt.show()

    def affichage_population_age(self):
        """Population dans une classe d'âge, en fonction de l'âge."""
        # Récupère la liste sans le deuxième membre (nom de la classe d'âge)
        classe_age = [[lst[0],lst[2]] for lst in self.__resultat_classe_age]
        # Réorganise les données
        output_dict = {}
        for key, *values in classe_age:
            if key not in output_dict:
                output_dict[key] = [key]
            output_dict[key].extend(values)
        resultat = list(output_dict.values())
        # Transforme en DataFrame pour simplifier les manipulations
        df = pd.DataFrame()
        # On génére des couleurs pour chaque courbe
        colors = rainbow(linspace(0,1,random.random((tmp:=abs(2016-self.annee_visee)+1,1)).shape[0]))
        for i in range(tmp):    # Créer chaque courbe
            df[resultat[i][0]] = resultat[i][1:]
            # Trace directement
            plt.plot(df.iloc[:,i], label=df.columns[i], color = colors[i], alpha=.7)
        # Différentes configurations pour rendre le graphe plus lisible
        plt.title(f"Population dans une classe d'âge, en fonction de l'âge\navec un taux de centenaires de {self.taux}")
        plt.legend(title=f'Années : {df.columns[0]} - {df.columns[-1]}', bbox_to_anchor=(1.05, 1), shadow=True, ncol=tmp//8)
        plt.xlabel('Âge'); plt.ylabel('Population')
        plt.grid(); plt.xlim(0,99)
        plt.show()

    def affichage_population_age_3D(self, azim=45, elev=50):
        """Population dans une classe d'âge, en fonction de l'âge et du temps."""
        df = pd.DataFrame(self.__resultat_classe_age, columns=['Années','Âge','Population'])
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
        ax.plot3D(df['Années'], df['Âge'], df['Population'], label='')
        ax.set(xlabel='Années', ylabel='Âge', zlabel='Population')
        plt.title(f'Population dans une classe d\'âge, sur la\npériode 2016 - {self.annee_visee} en fonction de l\'âge\navec un taux de centenaires de {self.taux}')
        ax.view_init(elev, azim); plt.grid()
        plt.show()

    def affichage_population_age_bar_3D(self, azim=45, elev=50):
        """Population dans une classe d'âge, en fonction de l'âge et du temps."""
        df = pd.DataFrame(self.__resultat_classe_age, columns=['Années','Âge','Population'])
        n = len(df['Population'])
        # Organise les données
        x, dx = df['Années'], ones(n)           # x et largeur de la barre
        y, dy = df['Âge'], ones(n)              # y et profondeur de la barre
        z, dz = df['Population'], range(1,n+1)  # z et hauteur de la barre
        # Trace
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
        ax.bar3d(x, y, z, dx, dy, dz)
        # Différentes configurations pour rendre le graphe plus lisible
        plt.title(f'Population dans une classe d\'âge, sur la\npériode 2016 - {self.annee_visee} en fonction de l\'âge\navec un taux de centenaires de {self.taux}')
        ax.set(xlabel='Années', ylabel='Âge', zlabel='Population')
        ax.view_init(35, 10)
        plt.show()

    def affichage_population_année(self):
        """Courbes des populations de 2016 et de l'année visée."""
        # Récupère les 100 classes d'âges de 2016 et de la dernière année
        df_2016 = pd.DataFrame(self.__resultat_classe_age[0:100],columns=['Années','Âge','Population'])
        df_visee = pd.DataFrame(self.__resultat_classe_age[-100:],columns=['Années','Âge','Population'])
        # Trace et label
        plt.plot(df_2016['Âge'],df_2016['Population'],label= (a2016:=df_2016['Années'][0]))
        plt.plot(df_visee['Âge'],df_visee['Population'],label= (aVisee:=df_visee['Années'][0]))
        # Différentes configurations pour rendre le graphe plus lisible
        plt.title(f'Population en fonction de la classe d\'âge des années {a2016} et {aVisee}\navec un taux de centenaires de {self.taux}')
        plt.xlabel('Classe d\'âge'); plt.ylabel('Population')
        plt.grid(); plt.xlim(df_2016.iloc[0,:]['Âge'], df_2016.iloc[-1,:]['Âge'])
        plt.legend(); plt.show()

    def affichage_quotient_mortalité_100_000_age(self, log=False):
        """Affiche le quotient de mortalité pour 100 000 en 2016 par âges de 0 à 99 ans."""
        if type(log) != bool: raise TypeError(f"{log} n'est pas un booléen!")
        df = Data.lecture('donnees_2016.csv') # Récupère les données
        plt.plot(df['Âge'],df['Quotient_de_mortalité_pour_100_000']) # Les trace
        # Différentes configurations pour rendre le graphe plus lisible
        plt.title('Quotient de mortalité pour 100 000 en 2016 par âges de 0 à 99 ans.')
        plt.xlabel('Âge'); plt.ylabel('Décès pour 100 000')
        plt.grid(); plt.xlim(df.iloc[0,:]['Âge'], df.iloc[-1,:]['Âge'])
        if log: plt.yscale("log")   # Éventuellement échelle logarithmique
        plt.show()



def menu():
    """Même un p'tit menu assez succinct!"""
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

    def choix_affichage(m:Modele):
        print(title('Veuillez choisir le modèle.'))
        match menu_choix({1:"Deviation agrege ",2:"Population âge ",3:"Population âge 3D ",4:"Population âge 3D barre ", 5:"Population année ", 6:"Quotient mortalité 100 000 âge ",0:"Sortie"},'0123456'):
            case 0: raise KeyboardInterrupt('-Fin')
            case 1: m.affichage_deviation_agrege()
            case 2: m.affichage_population_age()
            case 3: m.affichage_population_age_3D(int(input('Angle (int): ')), int(input('Hauteur (int): ')))
            case 4: m.affichage_population_age_bar_3D(int(input('Angle (int): ')), int(input('Hauteur (int): ')))
            case 5: m.affichage_population_année()
            case 6: m.affichage_quotient_mortalité_100_000_age(bool(input('Echèlle log (True, rien pour False): ')))
    try:
        #
        print(title("Utiliser %matplotlib avant d'éxécuter ce menu pour afficher les graphes en 3D."))
        Data.creer_donnees()  # Créer les données
        annee = int(input("Année entre 1986 et 2019 (inclus): "))
        taux = float(input("Taux centenaires (bon résultat pour 0.01): "))
        output = bool(input('Affichage (True, rien pour False): '))
        m = Modele(annee, taux=taux, output=output)
        m.modele()  # Exécute le modèle
        choix_affichage(m)  # Propose différents affichages
    except KeyboardInterrupt as err:
        print(err)
    except Exception as err:
        print(err)


if __name__ == '__main__':
    """Si script principal, exécute main."""
    menu()
