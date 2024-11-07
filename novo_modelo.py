import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd
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
data = pd.read_csv('df_final_K10.csv')

# Extrair os recursos (features) e os alvos (targets)
X = data.drop(['Best_X', 'Best_Y'], axis=1).values
y = data[['Best_X', 'Best_Y']].values

# Normalizar os dados
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Printar o número de características no terminal
print(f"Número de características (features): {X_scaled.shape[1]}")

# Definir o número de instantes de tempo (K)
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
    executions_per_trial=3,
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
plt.plot(history.history['loss'], label='Perda no treinamento')
plt.plot(history.history['val_loss'], label='Perda na validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.title('Perda no Treinamento e na Validação durante as épocas')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
print(f"Previsões: {y_pred[:5]}")  # Verifique as primeiras 5 previsões
print(f"Valores reais: {y_test[:5]}")  # Compare com os valores reais

error_percent = np.mean(np.abs((y_pred - y_test) / y_test)) * 100 
print(f"Erro percentual médio: {error_percent:.2f}%") #printa o erro percentual no terminal

# Parâmetros dos quadrantes
esc = 1000
quad_ranges = {
    'quad1': {'x': (0, 1*esc), 'y': (0, esc)},
    'quad2': {'x': ((1*esc)+1, 2*esc), 'y': (0, esc)},
    'quad3': {'x': ((2*esc)+1, 3*esc), 'y': (0, esc)},
}

# Função para distribuir K dispositivos uniformemente em cada quadrante
def distribui_disp(K, quad):
    x_min, x_max = quad_ranges[quad]['x']
    y_min, y_max = quad_ranges[quad]['y']
    x_coords = np.random.uniform(x_min, x_max, K)
    y_coords = np.random.uniform(y_min, y_max, K)
    return x_coords, y_coords

# Função para prever e ajustar a posição dos PBs em cada quadrante usando o modelo treinado
def prediz_PB(x_coords, y_coords, modelo, quad, esc):
    # Combina as coordenadas dos dispositivos em uma matriz de entrada no formato adequado
    X = np.column_stack((x_coords, y_coords)).reshape(1, 10, 2)  # 10 dispositivos com 2 coordenadas
    X_expanded = np.tile(X, (1, 1, 10))  # Ajustar dimensão para (1, 10, 20) se for o esperado

    # Faz a previsão usando o modelo treinado
    previsao = modelo.predict(X_expanded)
    pb_x, pb_y = previsao[0, 0], previsao[0, 1]

    return pb_x, pb_y

# Número de dispositivos para cada quadrante
K = 10

# Definir as cores para cada quadrante
cores_dispositivos = {
    'quad1': 'purple',   
    'quad2': 'purple',  
    'quad3': 'purple' 
}

cores_pbs = {
    'quad1': 'red', 
    'quad2': 'orange', 
    'quad3': 'green'    
}

plt.figure(figsize=(10, 10))

# Processar cada quadrante e prever PBs
for i, quad in enumerate(quad_ranges.keys()):
    # Distribuir K dispositivos aleatoriamente no quadrante
    x_coords, y_coords = distribui_disp(K, quad)
    
    # Formar a entrada e normalizá-la
    X = np.column_stack((x_coords, y_coords)).reshape(1, 10, 2)
    X_expanded = np.tile(X, (1, 1, 10))  # Ajuste de dimensão para (1, 10, 20)
    X_expanded_normalizado = scaler_X.transform(X_expanded.reshape(-1, X_expanded.shape[-1])).reshape(X_expanded.shape)
    
    # Passar a entrada normalizada ao modelo
    previsao_normalizada = model.predict(X_expanded_normalizado)
    
    # Desnormalizar a previsão do PB
    previsao_desnormalizada = scaler_y.inverse_transform(previsao_normalizada)
    pb_x_desnorm, pb_y_desnorm = previsao_desnormalizada[0, 0], previsao_desnormalizada[0, 1]
    
    # Desnormalizar X_expanded para plotagem
    X_expanded_desnormalizado = scaler_X.inverse_transform(X_expanded_normalizado.reshape(-1, X_expanded.shape[-1])).reshape(X_expanded.shape)
    
    # Plotar os dispositivos desnormalizados 
    if i == 0:
        plt.scatter(X_expanded_desnormalizado[0, :, 0], X_expanded_desnormalizado[0, :, 1], color=cores_dispositivos[quad], label='Dispositivos', alpha=0.6)
    else:
        plt.scatter(X_expanded_desnormalizado[0, :, 0], X_expanded_desnormalizado[0, :, 1], color=cores_dispositivos[quad], alpha=0.6)

    # Plotar a previsão do PB desnormalizada
    plt.scatter(pb_x_desnorm, pb_y_desnorm, color=cores_pbs[quad], label=f'PB previsto {quad[-1]}', marker='X', s=100)

plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.title('Distribuição de Dispositivos e Predição de PBs Desnormalizados')
plt.legend()
plt.show()
