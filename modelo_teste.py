import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping

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

# Configuração da validação cruzada
n_splits = 5  # Número de folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Variável para armazenar a perda de cada fold
fold_losses = []

# Loop para validação cruzada
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"\nTreinando fold {fold + 1}/{n_splits}...")
    
    # Dividir os dados de treino e validação para o fold atual
    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    
    # Construir e compilar o modelo para cada fold
    model = tuner.hypermodel.build(best_hps)

    # Treinar o modelo no fold atual
    history = model.fit(
        X_fold_train, y_fold_train,
        epochs=100,
        validation_data=(X_fold_val, y_fold_val),
        verbose=1
    )
    
    # Avaliar a perda no conjunto de validação para o fold atual
    val_loss = model.evaluate(X_fold_val, y_fold_val, verbose=0)
    print(f"Perda de validação para fold {fold + 1}: {val_loss}")
    fold_losses.append(val_loss)

# Calcular a média da perda entre os folds
mean_loss = np.mean(fold_losses)
print(f"\nPerda média de validação cruzada: {mean_loss}")

# Avaliar o modelo final no conjunto de teste
test_loss = model.evaluate(X_test, y_test)
print(f'\nPerda no teste: {test_loss}')

# Plotar a perda de treino e validação para o último fold
plt.figure(figsize=(10, 6))  # Aumentar o tamanho da figura
plt.plot(history.history['loss'], label='Perda no treinamento')
plt.plot(history.history['val_loss'], label='Perda na validação')

# Ajuste dos tamanhos das fontes
plt.xlabel('Épocas', fontsize=16)
plt.ylabel('Perda', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

y_pred = model.predict(X_test)# mandar x teste e y_pred e salvar pra cada valor de K (desnormalizados)
#PLOTAGEM DE DISPOSITIVOS E PREVISOES
# Desnormalizar as previsões e o conjunto de teste para todos os dispositivos
pred_desnormalizado = scaler_y.inverse_transform(y_pred)
X_test_desnormalizado = scaler_X.inverse_transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Separar as coordenadas x e y dos dispositivos
X_test_x = X_test_desnormalizado[:, :, 0:10]
X_test_y = X_test_desnormalizado[:, :, 10:20]

plt.figure(figsize=(10, 8))

# Loop para plotar apenas as duas primeiras amostras com 10 dispositivos cada
for i in range(2):  # Limita a iteração às duas primeiras amostras
    # Plotar os 10 dispositivos para a amostra i
    plt.scatter(X_test_x[i, 0, :], X_test_y[i, 0, :], marker='o', label=f'Dispositivos Amostra {i+1}')
    
    # Verificar se realmente estamos plotando 10 dispositivos
    print(f"Coordenadas dos 10 dispositivos para a amostra {i+1} (X):", X_test_x[i, 0, :])
    print(f"Coordenadas dos 10 dispositivos para a amostra {i+1} (Y):", X_test_y[i, 0, :])
    
    # Plotar o PB previsto para a amostra i
    plt.scatter(pred_desnormalizado[i, 0], pred_desnormalizado[i, 1], marker='x', color='red', label=f'PB Previsto Amostra {i+1}')

plt.xlabel("Coordenada X", fontsize=14)
plt.ylabel("Coordenada Y", fontsize=14)
plt.title("Distribuição de Dispositivos e PBs Previstos (Amostras 1 e 2)", fontsize=16)
plt.legend(fontsize=12)
plt.show()

#PREDICOES E GERACAO DE ARQUIVOS TXT
y_pred = model.predict(X_test)# mandar x teste e y_pred e salvar pra cada valor de K (desnormalizados)

# Desnormalizar os valores de X_test e y_pred
X_test_desnormalizado = scaler_X.inverse_transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
pred_desnormalizado = scaler_y.inverse_transform(y_pred)

# Salvar X_test_desnormalizado em um arquivo txt (para cada valor de K)
with open("X_test_desnormalizado_K10.txt", "w") as file_X:
    for i in range(len(X_test_desnormalizado)):
        for j in range(X_test_desnormalizado.shape[1]):  # para cada timestep K
            # Escrever as coordenadas X e Y separadas por espaço
            linha = " ".join(map(str, X_test_desnormalizado[i, j])) + "\n"
            file_X.write(linha)

# Salvar pred_desnormalizado em um arquivo txt
with open("pred_desnormalizado_K10.txt", "w") as file_y:
    for i in range(len(pred_desnormalizado)):
        # Escrever as coordenadas desnormalizadas do PB separadas por espaço
        linha = " ".join(map(str, pred_desnormalizado[i])) + "\n"
        file_y.write(linha)

# Desnormalizar as previsões e o conjunto de teste para todos os dispositivos
pred_desnormalizado = scaler_y.inverse_transform(y_pred)
X_test_desnormalizado = scaler_X.inverse_transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Extrair as coordenadas x e y dos dispositivos, assumindo que as primeiras 10 colunas são X e as últimas 10 são Y
X_test_x = X_test_desnormalizado[:, :, :10]  # Primeiras 10 colunas como coordenadas X
X_test_y = X_test_desnormalizado[:, :, 10:20]  # Próximas 10 colunas como coordenadas Y

#PLOTAGEM POR QUADRANTES
# Parâmetros dos quadrantes
esc = 100
quad_ranges = {
    'quad1': {'x': (0, 1*esc), 'y': (0, 3*esc)},
    'quad2': {'x': ((1*esc)+1, 2*esc), 'y': (0, 3*esc)},
    'quad3': {'x': ((2*esc)+1, 3*esc), 'y': (0, 3*esc)},
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

    min = i* esc
    max = esc *( i + 1 )  
    
    # Formar a entrada e normalizá-la
    X = np.column_stack((x_coords, y_coords)).reshape(1, 10, 2)
    X_expanded = np.tile(X, (1, 1, 10))  # Ajuste de dimensão para (1, 10, 20)
    X_expanded_normalizado = scaler_X.fit_transform(X_expanded.reshape(-1, X_expanded.shape[-1])).reshape(X_expanded.shape)
    
    # Passar a entrada normalizada ao modelo
    previsao_normalizada = model.predict(X_expanded_normalizado)
    print(previsao_normalizada)

    prev_desnorm_x = previsao_normalizada[0, 0] * ( max - min ) + min
    prev_desnorm_y = previsao_normalizada[0, 1] * 3 * esc
    
    # Desnormalizar a previsão do PB
    # Desnormalizar com base no quadrante
    previsao_desnormalizada = scaler_y.inverse_transform(previsao_normalizada)
    pb_x_desnorm, pb_y_desnorm = previsao_desnormalizada[0, 0], previsao_desnormalizada[0, 1]

    desnorm_x = X_expanded_normalizado[0, :, 0] * ( max - min ) + min
    desnorm_y = X_expanded_normalizado[0, :, 1] * 3 * esc
    
    # Desnormalizar X_expanded para plotagem
    X_expanded_desnormalizado = scaler_X.inverse_transform(X_expanded_normalizado.reshape(-1, X_expanded.shape[-1])).reshape(X_expanded.shape)
    
    # Plotar os dispositivos desnormalizados 
    if i == 0:
        plt.scatter(desnorm_x, desnorm_y, color=cores_dispositivos[quad], label='Dispositivos', alpha=0.6)
    else:
        plt.scatter(desnorm_x, desnorm_y, color=cores_dispositivos[quad], alpha=0.6)

    # Plotar a previsão do PB desnormalizada
    plt.scatter(prev_desnorm_x, prev_desnorm_y, color=cores_pbs[quad], label=f'PB', marker='X', s=100)

plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.title('Distribuição de Dispositivos e Predição de PBs Desnormalizados')
plt.legend()
plt.show()
