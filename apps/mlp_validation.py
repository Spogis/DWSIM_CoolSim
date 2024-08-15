import pandas as pd
import numpy as np
import io
from scipy.integrate import solve_ivp
import plotly.graph_objs as go

from tensorflow.keras.models import load_model
import pickle

from apps.run_DWSIM import *


def MLP_Validation(ValidationCases):
    data_describe = pd.read_excel('datasets/Parallel_Filter_Stats.xlsx', index_col=0)

    evaporator_temperature_random = np.random.uniform(data_describe.loc['min', 'Evaporator Temperature'],
                                                      data_describe.loc['max', 'Evaporator Temperature'],
                                                      size=ValidationCases)

    condenser_temperature_random = np.random.uniform(data_describe.loc['min', 'Condenser Temperature'],
                                                     data_describe.loc['max', 'Condenser Temperature'],
                                                     size=ValidationCases)

    adiabatic_efficiency_random = np.random.uniform(data_describe.loc['min', 'Adiabatic Efficiency'],
                                                    data_describe.loc['max', 'Adiabatic Efficiency'],
                                                    size=ValidationCases)

    df = pd.DataFrame({
        'Evaporator Temperature': evaporator_temperature_random,
        'Condenser Temperature': condenser_temperature_random,
        'Adiabatic Efficiency': adiabatic_efficiency_random
    })

    numberofsimulations = len(df)

    exportdataset = pd.DataFrame(columns=[
        'Evaporator Temperature',
        'Condenser Temperature',
        'Adiabatic Efficiency',
        'Compressor Energy',
        'Electric Current',
        'Discharge Temperature',
        'Refrigerant Mass Flow',
        'MLP_Compressor_Energy',
        'MLP_Electric_Current',
        'MLP_Discharge_Temperature',
        'MLP_Refrigerant_Mass_Flow'
    ])

    # Carrega o modelo
    model = load_model('kerasoutput/Keras_MLP_Surrogate.keras')

    # Carrega o scaler dos dados de entrada
    with open('kerasoutput/scalerX.pkl', 'rb') as file:
        scalerX = pickle.load(file)

    # Carrega o scaler dos dados de saída
    with open('kerasoutput/scalerY.pkl', 'rb') as file:
        scalerY = pickle.load(file)

    for i in range(numberofsimulations):
        evaporator_temperature_value = df.iloc[i]['Evaporator Temperature']
        condenser_temperature_value = df.iloc[i]['Condenser Temperature']
        adiabatic_efficiency_value = df.iloc[i]['Adiabatic Efficiency']

        energy, discharge_temperature, mass_flow = run_DWSIM(evaporator_temperature=evaporator_temperature_value,
                                                             condenser_temperature=condenser_temperature_value,
                                                             adiabatic_efficiency=adiabatic_efficiency_value)

        energy = energy * 1000
        discharge_temperature = discharge_temperature - 273.15
        mass_flow = mass_flow * 60
        electric_current = energy / 220

        input_data = np.array([[evaporator_temperature_value, condenser_temperature_value, adiabatic_efficiency_value]])
        X_valid = scalerX.transform(input_data)
        ypred_Scaled = model.predict(X_valid)
        ypred = scalerY.inverse_transform(ypred_Scaled)

        MLP_Compressor_Energy, MLP_Electric_Current, MLP_Discharge_Temperature, MLP_Refrigerant_Mass_Flow = ypred[0]

        designdata = np.array([evaporator_temperature_value,
                               condenser_temperature_value,
                               adiabatic_efficiency_value,
                               energy,
                               electric_current,
                               discharge_temperature,
                               mass_flow,
                               MLP_Compressor_Energy,
                               MLP_Electric_Current,
                               MLP_Discharge_Temperature,
                               MLP_Refrigerant_Mass_Flow])

        exportdataset.loc[len(exportdataset)] = designdata

        if i % int(numberofsimulations / 20) == 0:
            status = round((i / numberofsimulations) * 100, 0)
            with open('assets/status2.txt', 'w') as file:
                file.write(str(status))

    exportfile = 'datasets/MLP_Validation_Dataset.xlsx'
    exportdataset.to_excel(exportfile, index=False)


def Inverse_MLP_Validation(ValidationCases):
    data_describe = pd.read_excel('datasets/Parallel_Filter_Stats.xlsx', index_col=0)

    evaporator_temperature_random = np.random.uniform(data_describe.loc['min', 'Evaporator Temperature'],
                                                      data_describe.loc['max', 'Evaporator Temperature'],
                                                      size=ValidationCases)

    condenser_temperature_random = np.random.uniform(data_describe.loc['min', 'Condenser Temperature'],
                                                     data_describe.loc['max', 'Condenser Temperature'],
                                                     size=ValidationCases)

    adiabatic_efficiency_random = np.random.uniform(data_describe.loc['min', 'Adiabatic Efficiency'],
                                                    data_describe.loc['max', 'Adiabatic Efficiency'],
                                                    size=ValidationCases)

    df = pd.DataFrame({
        'Evaporator Temperature': evaporator_temperature_random,
        'Condenser Temperature': condenser_temperature_random,
        'Adiabatic Efficiency': adiabatic_efficiency_random
    })

    numberofsimulations = len(df)

    exportdataset = pd.DataFrame(columns=[
        'Evaporator Temperature',
        'Condenser Temperature',
        'Adiabatic Efficiency',
        'MLP_Evaporator_Temperature',
        'MLP_Condenser_Temperature',
        'MLP_Adiabatic_Efficiency',
    ])

    # Carrega o modelo
    model = load_model('kerasoutput/Keras_MLP_Surrogate.keras')

    # Carrega o scaler dos dados de entrada
    with open('kerasoutput/scalerX.pkl', 'rb') as file:
        scalerX = pickle.load(file)

    # Carrega o scaler dos dados de saída
    with open('kerasoutput/scalerY.pkl', 'rb') as file:
        scalerY = pickle.load(file)

    for i in range(numberofsimulations):
        evaporator_temperature_value = df.iloc[i]['Evaporator Temperature']
        condenser_temperature_value = df.iloc[i]['Condenser Temperature']
        adiabatic_efficiency_value = df.iloc[i]['Adiabatic Efficiency']

        energy, discharge_temperature, mass_flow = run_DWSIM(evaporator_temperature=evaporator_temperature_value,
                                                             condenser_temperature=condenser_temperature_value,
                                                             adiabatic_efficiency=adiabatic_efficiency_value)

        energy = energy * 1000
        discharge_temperature = discharge_temperature - 273.15
        mass_flow = mass_flow * 60
        electric_current = energy / 220

        input_data = np.array([[energy, electric_current, discharge_temperature, mass_flow]])
        X_valid = scalerX.transform(input_data)
        ypred_Scaled = model.predict(X_valid)
        ypred = scalerY.inverse_transform(ypred_Scaled)

        MLP_Evaporator_Temperature, MLP_Condenser_Temperature, MLP_Adiabatic_Efficiency = ypred[0]

        designdata = np.array([evaporator_temperature_value,
                               condenser_temperature_value,
                               adiabatic_efficiency_value,
                               MLP_Evaporator_Temperature,
                               MLP_Condenser_Temperature,
                               MLP_Adiabatic_Efficiency])

        exportdataset.loc[len(exportdataset)] = designdata

        if i % int(numberofsimulations / 20) == 0:
            status = round((i / numberofsimulations) * 100, 0)
            with open('assets/status2.txt', 'w') as file:
                file.write(str(status))

    exportfile = 'datasets/MLP_Validation_Dataset.xlsx'
    exportdataset.to_excel(exportfile, index=False)

