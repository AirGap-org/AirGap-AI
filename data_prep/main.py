# Data preparation
from data_prep.CreditMutuelCleaner import CreditMutuelCleaner

if __name__ == '__main__':
    creditMutuelCleaner = CreditMutuelCleaner()

    creditMutuelCleaner.load_csv("data/aldwin_in.csv")
    creditMutuelCleaner.clean()

    creditMutuelCleaner.extract("export/aldwin_out.csv")
