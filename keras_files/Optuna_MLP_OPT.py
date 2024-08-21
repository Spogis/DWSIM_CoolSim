import os
import shutil
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import io
from contextlib import redirect_stdout

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


def objective(trial, X_train, y_train, X_valid, y_valid):
    model = keras.Sequential()
    initializer = keras.initializers.GlorotNormal(seed=42)
    model.add(layers.InputLayer(shape=(X_train.shape[1],)))

    activation_choice = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu', 'selu', 'sigmoid'])
    num_layers = trial.suggest_int('num_layers', 1, 3)

    for i in range(num_layers):
        units = trial.suggest_int(f'units_{i}', 32, 128, step=32)
        model.add(layers.Dense(units=units,
                               activation=activation_choice,
                               kernel_initializer=initializer))

    model.add(layers.Dense(y_train.shape[1], kernel_initializer=initializer))

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 512, step=32)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])

    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        batch_size=batch_size,
                        epochs=500,
                        callbacks=[stop_early],
                        verbose=0)

    val_mae = min(history.history['val_mae'])
    return val_mae


def RunOptimizedMLP(Dataset, Input_Columns, Output_Columns):
    limpar_KerasOutput('kerasoutput')
    Input_Plus_Output = Input_Columns + Output_Columns
    Filtered_Dataset = Dataset[Input_Plus_Output]

    df = Filtered_Dataset
    print(df.columns.tolist())
    df = CleanDataset(df)

    X = df[Input_Columns].to_numpy()
    Y = df[Output_Columns].to_numpy()
    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)

    in_features = X.shape[1]
    out_features = Y.shape[1]

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, random_state=42)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

    scalerX = StandardScaler().fit(X_train)
    X_train = scalerX.transform(X_train)
    X_valid = scalerX.transform(X_valid)

    scalerY = StandardScaler().fit(y_train)
    y_train = scalerY.transform(y_train)
    y_valid = scalerY.transform(y_valid)

    # Defina a seed para garantir reprodutibilidade
    SEED = 42

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    n_trials = 50
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=n_trials)

    best_params = study.best_params
    print("Best hyperparameters: ", best_params)

    # Train best model with best hyperparameters
    model = keras.Sequential()
    initializer = keras.initializers.GlorotNormal(seed=42)
    model.add(layers.InputLayer(shape=(in_features,)))

    activation_choice = best_params['activation']
    num_layers = best_params['num_layers']

    for i in range(num_layers):
        units = best_params[f'units_{i}']
        model.add(layers.Dense(units=units,
                               activation=activation_choice,
                               kernel_initializer=initializer))

    model.add(layers.Dense(out_features, kernel_initializer=initializer))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
                  loss='mse',
                  metrics=['mae'])

    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        batch_size=best_params['batch_size'],
                        epochs=2000,
                        callbacks=[stop_early],
                        verbose=1)

    model.save('kerasoutput/Keras_MLP_Surrogate.keras')

    with open('kerasoutput/scalerX.pkl', 'wb') as file:
        pickle.dump(scalerX, file)

    with open('kerasoutput/scalerY.pkl', 'wb') as file:
        pickle.dump(scalerY, file)

    directory_path = 'assets/images'
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)

    i = 0
    for i, output in enumerate(Output_Columns):
        output = limpar_nome_arquivo(output)
        figure_file = 'assets/images/' + str("%02d" % (i + 1)) + ' - ' + output + '.png'

        # Inverter a transformação de padronização nos dados
        y_valid_original = scalerY.inverse_transform(y_valid)[:, i]
        y_pred_original = scalerY.inverse_transform(model.predict(X_valid))[:, i]

        plt.figure(1)
        plt.clf()
        plt.scatter(y_valid_original, y_pred_original, s=6, label=output)
        plt.plot(y_valid_original, y_valid_original)
        plt.legend()
        plt.savefig(figure_file)
        plt.close()

    history_df = pd.DataFrame(history.history)

    plt.figure(2)
    plt.clf()
    loss_array = history_df.loc[5:, 'loss'].values
    val_loss_array = history_df.loc[5:, 'val_loss'].values

    plt.plot(loss_array, label='Loss')
    plt.plot(val_loss_array, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig('assets/images/00 - LossEpoch.png')
    plt.close()


    ypred_Scaled = model.predict(X_valid)
    ypred = scalerY.inverse_transform(ypred_Scaled)
    y_valid = scalerY.inverse_transform(y_valid)

    r2_str = ""
    for i in range(len(Output_Columns)):
        valor_r2 = r2_score(y_valid[:, i], ypred[:, i])
        valor_formatado = f"{valor_r2:.4f}"
        r2_str += f"r² {Output_Columns[i]}:  {valor_formatado}\n"

    model_summary_str = ""
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary_str = f.getvalue()

    best_hps_str = 'Best hyperparameters found:\n'
    best_hps_str += f" - Number of hidden layers: {best_params.get('num_layers')}\n"
    best_hps_str += f" - Activation Function: {best_params.get('activation')}\n"
    for i in range(best_params.get('num_layers')):
        best_hps_str += f" - Neurons in each layer {i + 1}: {best_params.get('units_' + str(i))}\n"
    best_hps_str += f" - Optimizer learning rate: {best_params.get('learning_rate')}\n"
    best_hps_str += f" - Batch size: {best_params.get('batch_size')}\n"

    return best_hps_str, model_summary_str, r2_str
