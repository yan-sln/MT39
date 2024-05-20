# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:38:00 2024

@author: yan-s
"""
from os.path import isfile
import pandas as pd


class Data:
    """Créer les données dans des .csv + méthodes de lecture."""
    @staticmethod
    def creer_donnees():
        """Sources:
        bdd-donnees-annexes-fr-2023//annexe 1 --> NB total popu + naissances + décès
        fm_dod_struct_pop --> répartition popu par tranches d'âges f° année
        morta_niv_2016 --> table mortalité f° âge (année 2016)
        https://www.insee.fr/fr/statistiques/7234483#tableau-figure6 --> centenaires"""

        donnees_main = {'Années': ['1986',  '1987',  '1988',  '1989',  '1990',  '1991',  '1992',  '1993',  '1994',  '1995',  '1996',  '1997',  '1998',  '1999',  '2000',  '2001',  '2002',  '2003',  '2004',  '2005',  '2006',  '2007',  '2008',  '2009',  '2010',  '2011',  '2012',  '2013',  '2014',  '2015',  '2016',  '2017',  '2018',  '2019'],
                        'Population_totale': ['55411238','55681780','55966142','56269810','56577000','56840661','57110533','57369161','57565008','57752535','57935959','58116018','58298962','58496613','58858198','59266572','59685899','60101841','60505421','60963264','61399733','61795238','62134866','62465709','62765235','63070344','63375971','63697865','64027958','64300821','64468792','64639133','64844037','65096768'],
                        'de_0_à_19_ans': ['15999391','15919634','15852690','15793067','15719647','15605455','15473515','15330473','15179775','15084361','15058227','15055742','15026922','15017908','15047287','15067857','15091374','15116574','15183494','15242403','15280401','15315094','15337575','15368840','15406592','15440408','15457656','15513096','15588708','15651778','15645526','15616148','15611515','15584916'],
                        'de_20_à_59_ans': ['29296183','29495669','29685169','29875387','30093767','30280463','30506157','30730777','30940136','31068721','31160552','31232352','31342595','31455770','31667286','31962065','32288026','32560892','32760470','32971217','33193638','33219983','33170779','33107066','33023612','32939901','32910610','32853247','32791541','32680632','32562257','32448507','32383772','32356902'],
                        'de_60_à_64_ans': ['2887651','2870641','2869965','2881891','2892072','2916128','2925603','2941799','2921205','2912957','2858806','2816744','2765130','2737639','2712836','2675214','2610227','2606797','2640164','2682701','2762859','3051869','3325595','3568379','3795165','4022812','4034993','4029597','3998555','3979695','3949846','3953989','3951924','3974120'],
                        'de_65_à_99_ans': ['7228013','7395836','7558318','7719465','7871514','8038615','8205258','8366112','8523892','8686496','8858374','9011180','9164315','9285296','9430789','9561436','9696272','9817578','9921293','10066943','10162835','10208292','10300917','10421424','10539866','10667223','10972712','11301925','11649154','11988716','12311163','12620489','12896826','13180830'],
                        'centenaires': ['2886',  '3126',  '3352',  '3476',  '3760',  '3885',  '4379',  '4458',  '5014',  '5648',  '6303',  '7107',  '7259',  '7754',  '8161',  '9444',  '11031',  '11945',  '12393',  '13182',  '13575',  '14426',  '14940',  '15810',  '16688',  '18221',  '19897',  '20887',  '22031',  '24085',  '21044',  '19383',  '18513',  '18520'],
                        'Naissances': ['778468','767828','771268','765473','762407','759056','743658','711610','710993','729609','734338','726768','738080','744791','774782','770945','761630','761464','767816','774355','796896','785985','796044','793420','802224','792996','790290','781621','781167','760421','744697','730242','719737','714029'],
                        'Décès': ['546926','527466','524600','529283','526201','524685','521530','532263','519965','531618','535775','530319','534005','537661','530864','531073','535144','552339','509429','527533','516416','521016','532131','538116','540469','534795','559227','558408','547003','581770','581073','593606','596552','599408']}

        donnees_2016 = {'Âge': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'],
                        'Population': ['765550','788646','792333','804236','816258','836823','834997','839809','836583','849949','833020','828197','823916','828573','430033','859176','422275','814634','792026','787204','376849','737459','728187','761786','385005','784180','784047','792374','792888','395458','410141','804440','405286','847033','856278','867851','417763','808050','816877','798343','820272','867477','914322','937883','930282','912733','897889','892061','885433','459802','909893','920256','463651','877540','876823','872333','421833','845723','842277','833099','822054','821117','799913','811671','788895','428971','800702','418084','776321','734510','553400','537175','518715','478983','424883','440005','201519','442518','183665','243626','404609','398592','369172','366529','338247','319195','277738','251466','221891','134597','170448','141408','86225','25053','77793','60573','27183','17293','11488','7637'],
                        'Quotient_de_mortalité_pour_100_000': ['354','62','24','18','14','12','11','9','8','8','8','8','9','10','13','17','22','30','39','48','55','59','61','64','66','70','71','74','76','78','80','83','86','91','95','100','105','113','125','136','148','161','175','197','215','236','263','293','321','354','389','424','463','512','570','640','694','767','834','886','979','1032','1102','1181','1244','1319','1390','1471','1590','1693','1830','1952','2102','2298','2461','2692','2954','3284','3612','4089','4536','5103','5772','6465','7326','8269','9361','10557','11953','13317','14814','16554','18387','20313','22524','24750','26700','28837','31396','33844'],
                        'Survie_à_l_âge_x': ['100000','99646','99584','99561','99543','99529','99517','99507','99497','99489','99481','99473','99465','99457','99446','99433','99416','99394','99365','99326','99278','99223','99165','99104','99041','98976','98907','98836','98763','98688','98612','98533','98451','98367','98277','98184','98085','97982','97871','97748','97615','97471','97314','97144','96953','96744','96516','96262','95980','95672','95333','94963','94561','94122','93640','93106','92511','91869','91165','90404','89603','88726','87811','86843','85817','84749','83631','82469','81255','79963','78610','77171','75665','74075','72372','70591','68691','66661','64472','62144','59603','56899','53996','50879','47590','44104','40457','36669','32798','28878','25032','21324','17794','14522','11572','8966','6747','4945','3519','2414'],
                        'Espérance_de_vie_à_l_âge_x': ['79,0', '78,2', '77,3', '76,3', '75,3', '74,3', '73,3', '72,3', '71,4', '70,4', '69,4', '68,4', '67,4', '66,4', '65,4', '64,4', '63,4', '62,4', '61,4', '60,5', '59,5', '58,5', '57,6', '56,6', '55,6', '54,7', '53,7', '52,7', '51,8', '50,8', '49,9', '48,9', '47,9', '47,0', '46,0', '45,1', '44,1', '43,2', '42,2', '41,3', '40,3', '39,4', '38,4', '37,5', '36,6', '35,7', '34,8', '33,8', '32,9', '32,0', '31,2', '30,3', '29,4', '28,5', '27,7', '26,9', '26,0', '25,2', '24,4', '23,6', '22,8', '22,0', '21,3', '20,5', '19,8', '19,0', '18,3', '17,5', '16,8', '16,1', '15,3', '14,6', '13,9', '13,2', '12,5', '11,8', '11,2', '10,5', '9,9', '9,2', '8,6', '8,0', '7,5', '6,9', '6,4', '5,9', '5,4', '5,0', '4,6', '4,2', '3,8', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']}

        for chemin in ('donnees_main', 'donnees_2016'):
            if not isfile(f'{chemin}.csv'):
                df = pd.DataFrame(eval(chemin))
                df.to_csv(f'{chemin}.csv', sep=',', encoding='utf-8')

    @staticmethod
    def lecture(chemin: str) -> pd.DataFrame:
        """Lit le contenu d'un fichier .pkl entré."""
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
        if name in ['age', 'nombre']:
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


class ClasseAgrege:
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


class Modele:
    """Le gros du projet."""
    def __init__(self, annee_visee:int):
        self.annee_visee = annee_visee
        self.liste_classe_age:list(ClasseAge) = self.add_liste_classe_age()

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
            for annee in range(abs(periode)):
                # On rajoute les nouveaux-nés
                taux_morta = self.liste_classe_age[0].taux_morta
                generation_0 = ClasseAge(0, int(list(df.loc[df['Années'] == 2016+annee]['Naissances'])[0]), taux_morta)
                for idx, classe_age in enumerate((self.liste_classe_age)):
                    # On constate les décèes en f° du taux_morta
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

        else:   # Modélise le passé
            for annee in range(abs(periode)):
                # On rajoute les centenaires
                taux_morta = self.liste_classe_age[-1].taux_morta
                generation_99 = ClasseAge(99, int(list(df.loc[df['Années'] == 2016-annee]['Naissances'])[0]), taux_morta)

                self.liste_classe_age.reverse() # Retourne liste
                for idx, classe_age in enumerate(self.liste_classe_age):
                    # On constate les décèes en f° du taux_morta, et on les resucite
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

    def classe_agrege(self)-> dict:
        """Récupère données modèles et retourne dict."""
        def agrege(age_min:int, age_max:int)->dict:
            age = f"{age_min} - {age_max-1} ans"
            nombre = sum(obj.nombre for obj in self.liste_classe_age[age_min:age_max])
            tmp = ClasseAgrege(age, nombre)
            return {tmp.age: tmp.nombre}
        dct = {}
        dct.update(agrege(0, 20))
        dct.update(agrege(20, 60))
        dct.update(agrege(60, 65))
        dct.update(agrege(65, 100))
        dct.update({'Total':sum(val for val in dct.values())})
        return dct

    def main_agrege(self)-> dict:
        """Récupère données .csv et retourne dict."""
        def lst_to_dict(agrege:ClasseAgrege):
            lst = list(vars(agrege).values())
            return {lst[0]: lst[1]}
        df = Data.lecture('donnees_main.csv')
        row = df.loc[df['Années'] == self.annee_visee].index[0]

        dct = {}
        dct.update(lst_to_dict(ClasseAgrege('0 - 19 ans', int(df.iloc[row]['de_0_à_19_ans']))))
        dct.update(lst_to_dict(ClasseAgrege('20 - 59 ans', int(df.iloc[row]['de_20_à_59_ans']))))
        dct.update(lst_to_dict(ClasseAgrege('60 - 64 ans', int(df.iloc[row]['de_60_à_64_ans']))))
        dct.update(lst_to_dict(ClasseAgrege('65 - 99 ans', int(df.iloc[row]['de_65_à_99_ans']))))
        dct.update({'Total':sum(val for val in dct.values())})
        return dct

    def comparaison_agrege(self):
        model = self.classe_agrege()
        data = self.main_agrege()

        for key in data.keys(): # data.keys == main.keys
            print(f"{key} : {data[key] - model[key]}")

def main():
    """Exemple d'exécution."""
    Data.creer_donnees() # Créer les données
    annee = int(input('Sélectionnez une année cible entre 1986 et 2019 : '))
    m = Modele(annee) # Soit me suis foiré, soit montre bien la couille avec 1986
    print('Popu en 2016:')
    for age in m.liste_classe_age:
        print(vars(age))
    # Execute le modèle à partir de 2016 jusqu'à année sélectionnée en conservant les taux de 2016
    m.modele()

    print('\n'*2, f'Popu en {annee} à partir de 2016:')
    for age in m.liste_classe_age:
        print(vars(age))

    print('\n'*2, 'Agrège les populations en catégories :')
    print('D\'après le modèle :', m.classe_agrege())
    print('D\'après les données :', m.main_agrege())
    print('En comparant : '); m.comparaison_agrege()

    if False:
        """
        Ce que fait:
            -en gros garde taux mortalité 2016, et modèlise à partir de cette date
            en récupérant les naissances et les décès des années en q°
            -de 1986 à 2019 car données existantes
        
        To do :
            -Améliorer f° comparaison_agrege (exemple faire pourcentage déviation)
            -Affichage bar plot -> pyramide âge
            -Affichage simple popu modélisé vs popu réelle (f° temps)
        """


if __name__ == '__main__':
    """Si script principal, exécute main."""
    main()
