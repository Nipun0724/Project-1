import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# --- Simulation Parameters for Data Generation ---
NUM_SERVERS = 3
SIM_TIME = 500
TASK_INTERVAL = 1.0
CPU_CAPACITY = 100
NET_CAPACITY = 100
# --- IMPORTANT: This MUST match the MAX_QUEUE_LEN in your simulation ---
MAX_QUEUE_LEN = 5 

records = []
timestamp = 0.0

print("Generating synthetic data...")
while timestamp < SIM_TIME:
    timestamp += random.expovariate(1.0 / TASK_INTERVAL)
    for s in range(NUM_SERVERS):
        cpu_base = 50 + 40 * np.sin(2 * np.pi * timestamp / 100)
        net_base = 30 + 25 * np.sin(2 * np.pi * (timestamp + s*10) / 100)
        
        cpu = int(np.clip(cpu_base + random.gauss(0, 10), 0, CPU_CAPACITY))
        net_in = int(np.clip(net_base + random.gauss(0, 5), 0, NET_CAPACITY))
        net_out = int(np.clip(net_base + random.gauss(0, 5), 0, NET_CAPACITY))
        
        # Make q_len more strongly correlated with CPU for realism
        q_len = int(np.clip((cpu / 10) * (MAX_QUEUE_LEN / 10) + random.gauss(0, 1.5), 0, MAX_QUEUE_LEN))

        # --- ### THE CRITICAL CHANGE ### ---
        # The new definition of a bottleneck.
        # A bottleneck is now defined as a server whose queue is ALMOST full.
        # This predicts the state that causes the dispatcher to stop.
        bottleneck = int(
            cpu >= 0.8 * CPU_CAPACITY or
            net_in >= 0.8 * NET_CAPACITY or
            net_out >= 0.8 * NET_CAPACITY
        )

        records.append({
            'timestamp': round(timestamp, 2), 'server_id': f'Server {s}',
            'cpu_used': cpu, 'q_len': q_len, 'network_in': net_in,
            'network_out': net_out, 'bottleneck': bottleneck
        })

df = pd.DataFrame(records)
df.to_csv('synthetic_data_for_queue_prediction.csv', index=False)
print("Data generation complete.")

# --- Preprocessing and Training (No changes needed here) ---
FEATURES = ['cpu_used', 'q_len', 'network_in', 'network_out']
TARGET = 'bottleneck'
WINDOW_SIZE = 20

# Create sequences
X_raw, Y = [], []
for server_id in df['server_id'].unique():
    server_df = df[df['server_id'] == server_id].reset_index(drop=True)
    for i in range(len(server_df) - WINDOW_SIZE):
        window = server_df.loc[i:i + WINDOW_SIZE - 1, FEATURES].values
        label = server_df.loc[i + WINDOW_SIZE, TARGET]
        X_raw.append(window); Y.append(label)

X_raw, Y = np.array(X_raw), np.array(Y)

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=42
)

# Fit scaler ONLY on training data
scaler = MinMaxScaler()
ns, steps, feats = X_train_raw.shape
X_train_reshaped = X_train_raw.reshape((ns * steps, feats))
scaler.fit(X_train_reshaped)
joblib.dump(scaler, "minmax_scaler.save")

# Transform both sets
X_train = scaler.transform(X_train_reshaped).reshape(ns, steps, feats)
ns_test, steps_test, feats_test = X_test_raw.shape
X_test_reshaped = X_test_raw.reshape((ns_test * steps_test, feats_test))
X_test = scaler.transform(X_test_reshaped).reshape(ns_test, steps_test, feats_test)

# Define and train the model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining new model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
model.save("bottleneck_predictor.keras")
print("Model training complete and saved as 'bottleneck_predictor.keras'.")
