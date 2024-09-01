import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import math

# Load the dataset
df = pd.read_csv('dataset_limpo.csv')

# Normalize the data
scaler = MinMaxScaler()
df[['Local_X', 'Local_Y']] = scaler.fit_transform(df[['Local_X', 'Local_Y']])

# Prepare the data
# Assuming you want to predict the next position based on the last 10 positions
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

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 2)))
model.add(Dense(2))  # 2 output units for Local_X and Local_Y
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Making predictions on the test set
predicted = model.predict(X_test)

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
