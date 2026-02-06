from abc import abstractmethod, ABC

import pandas as pd


class CSVCleaner(ABC):
    df = None

    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path, delimiter='\t', decimal='.')

    def get_df(self) -> pd.DataFrame:
        return self.df

    def normalize(self):
        self.df['libelle'] = (self.df['libelle']
         .str.replace(r'\d+', '', regex=True)
         .str.upper()
         .str.replace(r'\s+', ' ', regex=True)
         .str.strip())

        self.df['categorie'] = 'Autres'

    def extract(self, extract_csv):
        self.df.to_csv(extract_csv, index=False)

    @abstractmethod
    def clean(self):
        pass
