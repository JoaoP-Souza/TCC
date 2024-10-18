import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout  # Import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load your CSV file
data = pd.read_csv('df_final.csv')

# Extract input features and target values
X = data.drop(['Best_X', 'Best_Y'], axis=1).values
y = data[['Best_X', 'Best_Y']].values

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape data for LSTM (samples, time steps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)

# Function to create and train LSTM model
def create_and_train_model(X_train, y_train):
    # Create a new LSTM model instance
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))  # Added return_sequences=True
    model.add(Dropout(0.2))  # Add Dropout layer (20% dropout rate)
    model.add(LSTM(50, activation='relu'))  # Another LSTM layer
    model.add(Dropout(0.2))  # Another Dropout layer
    model.add(Dense(2))  # 2 outputs for Best_X and Best_Y
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, verbose=1)

    return model

# Training the model
model = create_and_train_model(X_train, y_train)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')