from data_prep.CSVCleaner import CSVCleaner

"""
 Caractétisque d'export:
 - Format : Excel 2000
 - Dates: Français Long (JJ/MM/AAAA)
 - Séparateur de champs: tabulation
 - Montants sur : une seule colonne
 - Séparateur décimal : point
 - Inclure vos caractéristiques personnalisés : NON
"""


class CreditMutuelCleaner(CSVCleaner):
    def clean(self):
        self.df.rename(columns={'Date': 'date',
                                'Libell?': 'libelle',
                                'Montant': 'montant'}, inplace=True)

        self.df.drop(columns=['Date de valeur', 'Solde'], inplace=True)
        self.normalize()