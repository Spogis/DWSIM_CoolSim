from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, minimize
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

from keras_files.KerasPredict import *
from apps.odes import *


def objective_function(model, desired_values):
    inputs = np.array([model.inputs[i].value for i in range(len(model.inputs))]).reshape(1, -1)
    pred = PredictValues(inputs)
    mape = mean_absolute_percentage_error(pred, desired_values)
    return mape


def optimize(desired_values, column_names):
    df = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    min_max_values = []
    for column_name in column_names:
        if column_name in df.columns:
            min_value = df[column_name].min()
            max_value = df[column_name].max()
            min_max_values.append((min_value, max_value))

    model = ConcreteModel()
    model.inputs = Var(range(len(column_names)), bounds=(0, 1))

    # Set bounds for each variable
    for i, bounds in enumerate(min_max_values):
        model.inputs[i].setlb(bounds[0])
        model.inputs[i].setub(bounds[1])

    model.obj = Objective(expr=objective_function(model, desired_values), sense=minimize)

    solver = SolverFactory('ipopt')
    solver.solve(model, tee=True)

    optimized_values = [model.inputs[i].value for i in range(len(column_names))]
    return optimized_values
