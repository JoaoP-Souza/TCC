import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch


# Definir uma seed para garantir reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Carregar o CSV
data = pd.read_csv('df_final.csv')

# Extrair os recursos (features) e os alvos (targets)
X = data.drop(['Best_X', 'Best_Y'], axis=1).values
y = data[['Best_X', 'Best_Y']].values

# Normalizar os dados
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Printe o número de características
print(f"Número de características (features): {X_scaled.shape[1]}")

# Definir o número de instantes de tempo (timesteps)
timesteps = 10

# Ajustar o formato dos dados para (amostras, timesteps, características)
n_samples = X_scaled.shape[0] // timesteps
X_reshaped = X_scaled[:n_samples * timesteps].reshape(n_samples, timesteps, X_scaled.shape[1])

print(f"Formato dos dados de entrada após reshape: {X_reshaped.shape}")
print(f"Número de amostras: {X_reshaped.shape[0]}")
print(f"Número de instantes de tempo (timesteps): {X_reshaped.shape[1]}")
print(f"Número de características (features) por instante de tempo: {X_reshaped.shape[2]}")

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled[:n_samples], test_size=0.2, random_state=SEED)

# Definir o modelo usando a função de construção do modelo
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(timesteps, X_train.shape[2])))  # timesteps x características
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   activation='relu'))  # Removido return_sequences=True, pois só há uma LSTM
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))  # Mantendo apenas uma Dropout
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Instanciar o tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='C:/Users/jkspa/Desktop/arquivosTCC/TCC/tuner_results',
    project_name='lstm_vehicle_prediction'
)

# Realizar a busca pelos melhores hiperparâmetros
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)

# Resumo dos melhores resultados
tuner.results_summary()

# Obter os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Construir e treinar o modelo com os melhores hiperparâmetros
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Avaliar o modelo
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Plotar a perda de treino e validação ao longo das épocas
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()