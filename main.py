# Data preparation
from data_prep.CreditMutuelCleaner import CreditMutuelCleaner

if __name__ == '__main__':
    creditMutuelCleaner = CreditMutuelCleaner()

    creditMutuelCleaner.load_csv("data_prep/data/test.csv")
    creditMutuelCleaner.clean()


    print(creditMutuelCleaner.get_df().head(100))

    creditMutuelCleaner.extract("data_prep/export/out.csv")