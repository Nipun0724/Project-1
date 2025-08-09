import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

NUM_SERVERS = 3
SIM_TIME = 500  # total time to simulate (not # of rows)
TASK_INTERVAL = 1.0  # average interval between tasks
CPU_CAPACITY = 100
NET_CAPACITY = 100

records = []
timestamp = 0.0  # start time

while timestamp < SIM_TIME:
    # Exponentially distributed inter-arrival time
    timestamp += random.expovariate(1.0 / TASK_INTERVAL)
    for s in range(NUM_SERVERS):
        # CPU/net patterns
        cpu_base = 50 + 40 * np.sin(2 * np.pi * timestamp / 100)
        net_base = 30 + 25 * np.sin(2 * np.pi * (timestamp + s*10) / 100)

        # Add noise
        cpu = int(np.clip(cpu_base + random.gauss(0, 10), 0, CPU_CAPACITY))
        net_in = int(np.clip(net_base + random.gauss(0, 5), 0, NET_CAPACITY))
        net_out = int(np.clip(net_base + random.gauss(0, 5), 0, NET_CAPACITY))
        q_len = int(np.clip(cpu / 10 + random.gauss(0, 2), 0, 10))

        # Inject synthetic bottleneck
        if 250 <= timestamp <= 260 and s == 1:
            cpu = 95
            net_in = 90
            net_out = 90

        bottleneck = int(
            cpu >= 0.8 * CPU_CAPACITY or
            net_in >= 0.8 * NET_CAPACITY or
            net_out >= 0.8 * NET_CAPACITY
        )

        records.append({
            'timestamp': round(timestamp, 2),
            'server_id': f'Server {s}',
            'cpu_used': cpu,
            'q_len': q_len,
            'network_in': net_in,
            'network_out': net_out,
            'bottleneck': bottleneck
        })

# Save dataset
df = pd.DataFrame(records)
df.to_csv('synthetic_realistic_poisson.csv', index=False)

# --- Corrected Preprocessing and Training Flow ---

df = pd.read_csv('synthetic_realistic_poisson.csv')

# Features & Target
FEATURES = ['cpu_used', 'q_len', 'network_in', 'network_out']
TARGET = 'bottleneck'
WINDOW_SIZE = 20

# 1. Create sequences from the original, unscaled data first
X_raw, Y = [], []
for server_id in df['server_id'].unique():
    server_df = df[df['server_id'] == server_id].reset_index(drop=True)
    for i in range(len(server_df) - WINDOW_SIZE):
        window = server_df.loc[i:i + WINDOW_SIZE - 1, FEATURES].values
        label = server_df.loc[i + WINDOW_SIZE, TARGET]
        X_raw.append(window)
        Y.append(label)

X_raw = np.array(X_raw)
Y = np.array(Y)

# 2. Split the raw data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=42
)

# 3. Fit the scaler ONLY on the training data
scaler = MinMaxScaler()
# Reshape data to 2D for scaler, fit, then reshape back to 3D
nsamples, nsteps, nfeatures = X_train_raw.shape
X_train_reshaped = X_train_raw.reshape((nsamples * nsteps, nfeatures))
scaler.fit(X_train_reshaped)

# 4. Save the correctly fitted scaler
joblib.dump(scaler, "minmax_scaler.save")

# 5. Transform both train and test sets using the fitted scaler
X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
X_train = X_train_scaled_reshaped.reshape(nsamples, nsteps, nfeatures)

nsamples_test, nsteps_test, nfeatures_test = X_test_raw.shape
X_test_reshaped = X_test_raw.reshape((nsamples_test * nsteps_test, nfeatures_test))
X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
X_test = X_test_scaled_reshaped.reshape(nsamples_test, nsteps_test, nfeatures_test)

# Define model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model.save("bottleneck_predictor.keras")