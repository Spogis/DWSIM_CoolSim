import numpy as np
import pandas as pd


def CleanDataset(df):
    column_names = list(df)

    # Imprimindo o tipo de dados de cada coluna
    df = df.dropna()
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df = df.dropna()

    print(df.dtypes)

    for names in column_names[0:]:
        for x in [names]:
            q75, q25 = np.percentile(df.loc[:, x], [75, 25])
            intr_qr = q75 - q25

            max = q75 + (1.5 * intr_qr)
            min = q25 - (1.5 * intr_qr)

            df.loc[df[x] < min, x] = np.nan
            df.loc[df[x] > max, x] = np.nan

    df.isnull().sum()

    df = df.dropna(axis=0)

    df.to_excel("datasets/Dataset_No_Outliers.xlsx")

    return df
