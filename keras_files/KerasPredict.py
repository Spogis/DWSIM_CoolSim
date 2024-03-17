from tensorflow.keras.models import load_model
import pickle
import numpy as np

def PredictValues(input_data):
    # Carrega o modelo
    model = load_model('kerasoutput/Keras_MLP_Surrogate.keras')

    # Carrega o scaler dos dados de entrada
    with open('kerasoutput/scalerX.pkl', 'rb') as file:
        scalerX = pickle.load(file)

    # Carrega o scaler dos dados de saída
    with open('kerasoutput/scalerY.pkl', 'rb') as file:
        scalerY = pickle.load(file)

    # Regression Report
    X_valid = scalerX.transform(input_data)
    ypred_Scaled = model.predict(X_valid)
    ypred = scalerY.inverse_transform(ypred_Scaled)

    return ypred[0]