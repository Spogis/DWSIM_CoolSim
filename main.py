import pandas as pd
import numpy as np
import math
from scipy.integrate import solve_ivp
from asset.odes import *

t = None
y = None

df = pd.read_excel('datasets/DOE_LHC.xlsx')
NumberOfSimulations = len(df)


ExportDataset = pd.DataFrame(columns=['POX/C',
                                      'C/A',
                                      'POX/M',
                                      'X',
                                      'PDI',
                                      'Mn'])

for i in range(NumberOfSimulations):
    if i % 100 == 0:
        print(f"Running Cases {i+1} to {i+100} of {NumberOfSimulations}")

    Temp_P0XC = df.iloc[i]['POX/C']
    Temp_CA = df.iloc[i]['C/A']
    Temp_POXM = df.iloc[i]['POX/M']

    POX = Temp_POXM*M
    C = POX / Temp_P0XC
    A = C / Temp_CA

    Initial_Conditions = [0, 0, 0,
                          0, 0, 0,
                          0, 0, 0,
                          M,
                          POX, 0,
                          C, 0,
                          A, 0]

    SolveODEs(Initial_Conditions)
    Results = MoreUsableDataset()

    DesignData = np.array([POX/C, C/A, POX/M,
                           Results['X'].iloc[-1],
                           Results['PDI'].iloc[-1],
                           Results['Mn'].iloc[-1]])

    ExportDataset.loc[len(ExportDataset)] = DesignData


ExportFile = 'datasets/ARGET_ATRP_ODEs_Dataset.xlsx'
ExportDataset.to_excel(ExportFile, index=False)

