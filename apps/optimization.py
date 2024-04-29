from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

from keras_files.KerasPredict import *
from apps.odes import *


def objective_function(inputs, desired_values):
    inputs = inputs.reshape(1, -1)
    pred = PredictValues(inputs)
    mape = mean_absolute_percentage_error(pred, desired_values)
    return mape


def callback(x, f, accept):
    if f < 1e-2:
        return True
    return False

def optimize(desired_values, column_names):
    df = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    min_max_values = []
    only_min_values = []
    for column_name in column_names:
        if column_name in df.columns:
            min_value = df[column_name].min()
            max_value = df[column_name].max()
            min_max_values.append((min_value, max_value))
            only_min_values.append(min_value)

    bounds = np.array(min_max_values)
    initial_params = np.array(only_min_values)

    minimizer_kwargs = {'method': 'Powell', 'args': (desired_values), 'bounds': bounds}
    results = basinhopping(objective_function, x0=initial_params, minimizer_kwargs=minimizer_kwargs,
                           callback=callback, seed=42)

    return results.x
