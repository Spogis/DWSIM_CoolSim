import os
import shutil
import keras_tuner as kt
from tensorflow import keras
from keras import initializers
from tensorflow.keras import layers
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from keras_files.CleanData import *

from sklearn.metrics import r2_score
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def limpar_nome_arquivo(nome_arquivo):
    caracteres_invalidos = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in caracteres_invalidos:
        nome_arquivo = nome_arquivo.replace(char, '_')
    return nome_arquivo

def limpar_KerasOutput(diretorio):
    for nome in os.listdir(diretorio):
        caminho = os.path.join(diretorio, nome)
        try:
            if os.path.isfile(caminho) or os.path.islink(caminho):
                os.unlink(caminho)
            elif os.path.isdir(caminho):
                shutil.rmtree(caminho)
        except Exception as e:
            print('Falha ao deletar %s. Razão: %s' % (caminho, e))


def model_builder(hp, input_neurons=1, output_neurons=1):
    model = keras.Sequential()
    initializer = initializers.GlorotNormal(seed=42)
    model.add(layers.InputLayer(input_shape=(input_neurons,)))

    # Permite ao Keras Tuner escolher entre várias funções de ativação
    #activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'softplus'])
    activation_choice = hp.Choice('activation', values=['relu', 'tanh'])

    # Permite ao Keras Tuner decidir o número de camadas ocultas e neurônios
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=128,
                                            step=32),
                               activation=activation_choice,
                               kernel_initializer=initializer))

    model.add(layers.Dense(output_neurons, kernel_initializer=initializer))  # Camada de saída

    # Compila o modelo
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3])),
                  loss='mse',
                  metrics=['mae'])

    return model


def RunOptimizedMLP(Dataset, Input_Columns, Output_Columns):
    limpar_KerasOutput('kerasoutput')
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

    # Criando uma versão parcial da função model_builder que inclui os neurônios de entrada e saída
    custom_model_builder = partial(model_builder, input_neurons=in_features, output_neurons=out_features)

    # Instancia o tuner
    tuner = kt.Hyperband(custom_model_builder,
                         objective='val_mae',
                         max_epochs=5000,
                         factor=3,
                         seed=42,
                         directory='kerasoutput',
                         project_name='Hyperband_Tuner')

    stop_early = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=50,
        min_delta=0.001,
        restore_best_weights=True,
    )

    # Inicia a busca
    tuner.search(X_train, y_train, epochs=500,
                 validation_data=(X_valid, y_valid), callbacks=[stop_early],
                 batch_size=512)

    # Obtém o melhor modelo
    best_model = tuner.get_best_models(num_models=1)[0]

    # Treinamos o melhor modelo manualmente
    history = best_model.fit(X_train, y_train,
                             validation_data=(X_valid, y_valid),
                             batch_size=512,
                             epochs=2000,
                             callbacks=[stop_early],
                             verbose=1)

    # Obtém o melhor conjunto de hiperparâmetros
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Gerando a string de melhores hiperparâmetros
    best_hps_str = 'Melhores hiperparâmetros encontrados:\n'
    best_hps_str += f" - Número de camadas ocultas: {best_hps.get('num_layers')}\n"
    for i in range(best_hps.get('num_layers')):
        best_hps_str += f" - Unidades na camada {i + 1}: {best_hps.get('units_' + str(i))}\n"
    best_hps_str += f" - Taxa de aprendizado do otimizador: {best_hps.get('learning_rate')}\n"

    # Gera o resumo da arquitetura
    model_summary_str = []
    best_model.summary(print_fn=lambda x: model_summary_str.append(x))
    model_summary_str = "\n".join(model_summary_str)

    # Regression Report
    ypred_Scaled = best_model.predict(X_valid)
    ypred = scalerY.inverse_transform(ypred_Scaled)
    y_valid = scalerY.inverse_transform(y_valid)

    # Obtendo os valores de R²
    r2_str = ""
    i = 0
    for i in range(len(Output_Columns)):
        valor_r2 = r2_score(y_valid[:, i], ypred[:, i])
        valor_formatado = f"{valor_r2:.4f}"
        r2_str += f"r² {Output_Columns[i]}:  {valor_formatado}\n"

    best_model.save('assets/Keras_MLP_Surrogate.keras')

    # Salva o scaler dos dados de entrada
    with open('assets/scalerX.pkl', 'wb') as file:
        pickle.dump(scalerX, file)

    # Salva o scaler dos dados de saída
    with open('assets/scalerY.pkl', 'wb') as file:
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
    plt.title('Loss vs. Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig('assets/images/00 - LossEpoch.png')
    plt.close()

    return best_hps_str, model_summary_str, r2_str

