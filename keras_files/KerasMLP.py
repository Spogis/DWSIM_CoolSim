import os
from tensorflow import keras
from keras.models import Sequential
from keras import initializers
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from sklearn.metrics import r2_score

from keras_files.CleanData import *


import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def limpar_nome_arquivo(nome_arquivo):
    caracteres_invalidos = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in caracteres_invalidos:
        nome_arquivo = nome_arquivo.replace(char, '_')
    return nome_arquivo


def Create_Keras_Model(in_features, out_features, n_camadas_ocultas):
    # MLP Regression
    model = Sequential()
    initializer = initializers.GlorotNormal(seed=42)

    # Adicionando a camada de entrada
    model.add(Dense(64, input_dim=in_features, activation="relu", kernel_initializer=initializer))

    # Adicionando camadas ocultas
    for i in range(n_camadas_ocultas):
        model.add(Dense(64, activation="relu", kernel_initializer=initializer))

    # Adicionando a camada de saída
    model.add(Dense(out_features, kernel_initializer=initializer))

    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['mae'])

    model.summary()

    return model


def RunMLP(Dataset, Input_Columns, Output_Columns):
    Input_Plus_Output = Input_Columns + Output_Columns
    Filtered_Dataset = Dataset[Input_Plus_Output]

    df = Filtered_Dataset
    print(df.columns.tolist())
    df = CleanDataset(df)

    # Extraímos os recursos relevantes para nossas matrizes (arrays) numpy X e Y:
    print("Input_Columns: ", Input_Columns)
    print("Output_Columns: ", Output_Columns)

    X = df[Input_Columns].to_numpy()
    Y = df[Output_Columns].to_numpy()
    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)

    in_features = X.shape[1]
    out_features = Y.shape[1]

    # Split Train / Test Data
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, random_state=42)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

    # Dataset Standardization
    scalerX = StandardScaler().fit(X_train)
    # Scale the train set
    X_train = scalerX.transform(X_train)
    # Scale the test set
    X_valid = scalerX.transform(X_valid)

    scalerY = StandardScaler().fit(y_train)
    # Scale the train set
    y_train = scalerY.transform(y_train)
    # Scale the train set
    y_valid = scalerY.transform(y_valid)

    # MLP Regression
    n_camadas_ocultas = 2
    model = Create_Keras_Model(in_features, out_features, n_camadas_ocultas)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=50,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=512,
        epochs=2000,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Regression Report
    ypred_Scaled = model.predict(X_valid)
    ypred = scalerY.inverse_transform(ypred_Scaled)
    y_valid = scalerY.inverse_transform(y_valid)

    # Obtendo os valores de R²
    r2_str = ""
    i = 0
    for i in range(len(Output_Columns)):
        valor_r2 = r2_score(y_valid[:, i], ypred[:, i])
        valor_formatado = f"{valor_r2:.4f}"
        r2_str += f"r² {Output_Columns[i]}:  {valor_formatado}\n"

    model.save('kerasoutput/Keras_MLP_Surrogate.keras')

    # Salva o scaler dos dados de entrada
    with open('kerasoutput/scalerX.pkl', 'wb') as file:
        pickle.dump(scalerX, file)

    # Salva o scaler dos dados de saída
    with open('kerasoutput/scalerY.pkl', 'wb') as file:
        pickle.dump(scalerY, file)

    # Define o caminho do diretório
    directory_path = 'assets/images'

    # Lista todos os arquivos e diretórios no diretório especificado
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Verifica se é um arquivo (e não um diretório)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # Deleta o arquivo
        else:
            print(f"Ignorado: {file_path} não é um arquivo.")

    # Plotando Gráficos
    i = 0
    for output in Output_Columns:
        output = limpar_nome_arquivo(output)
        figure_file = 'assets/images/' + str("%02d" % (i + 1)) + ' - ' + output + '.png'
        plt.figure(1)
        plt.clf()
        plt.scatter(y_valid[:, i], ypred[:, i], s=6, label=output)
        plt.plot(y_valid[:, i], y_valid[:, i])
        plt.legend()
        plt.savefig(figure_file)
        plt.close()
        i = i + 1

    # Avaliação de Desempenho
    history_df = pd.DataFrame(history.history)

    plt.figure(2)
    plt.clf()
    loss_array = history_df.loc[5:, 'loss'].values
    val_loss_array = history_df.loc[5:, 'val_loss'].values

    plt.plot(loss_array, label='Loss')
    plt.plot(val_loss_array, label='Validation Loss')

    # Adicione rótulos e título ao gráfico
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss vs. Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig('assets/images/00 - LossEpoch.png')
    plt.close()

    return r2_str
