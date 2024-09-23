import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
from keras_tuner import RandomSearch

# Load the dataset
df = pd.read_csv('dataset_limpo.csv')

# Normalize the data
scaler = MinMaxScaler()
df[['Local_X', 'Local_Y']] = scaler.fit_transform(df[['Local_X', 'Local_Y']])

# Prepare the data
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Group by Vehicle_ID and create sequences
n_steps = 10
sequences = []

for vehicle_id in df['Vehicle_ID'].unique():
    vehicle_data = df[df['Vehicle_ID'] == vehicle_id][['Local_X', 'Local_Y']].values
    X, y = create_sequences(vehicle_data, n_steps)
    sequences.append((X, y))

# Concatenate all sequences
X = np.concatenate([seq[0] for seq in sequences], axis=0)
y = np.concatenate([seq[1] for seq in sequences], axis=0)

# Reshape X for LSTM input (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 2)

# Split the data into 80% training and 20% testing
train_size = int(X.shape[0] * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define a function to build the model for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    # Tuning the number of LSTM units
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), 
                   activation='relu', input_shape=(n_steps, 2)))
    # Adding dropout layer
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(2))  # 2 output units for Local_X and Local_Y
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

# Use Keras Tuner to find the best hyperparameters
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=2,  # Número de modelos a serem testados
    executions_per_trial=1,  # Número de execuções para cada conjunto de hiperparâmetros
    directory='C:/Users/jkspa/Desktop/arquivosTCC/TCC/tuner_results',  # Usando 'Desktop' em vez de 'Área de Trabalho'
    project_name='lstm_vehicle_prediction'
)


# Run hyperparameter search
tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model again for more epochs
history = best_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Making predictions on the test set
predicted = best_model.predict(X_test)

# Inverse scaling to original values
predicted = scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predicted)
print(f'Test MSE: {mse}')

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
