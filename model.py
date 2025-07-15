import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

CPU_CAPACITY = 100
WINDOW_SIZE = 20

df = pd.read_csv('resource_usage_log.csv')

df['bottleneck'] = (df['cpu_used'] / CPU_CAPACITY >= 0.8).astype(int)

features = ['cpu_used', 'q_len', 'network_in', 'network_out']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X = []
Y = []

for server_id in df['server_id'].unique():
    server_df = df[df['server_id'] == server_id].reset_index(drop = True)
    for i in range(len(server_df) - WINDOW_SIZE):
        window = server_df.loc[i:i+WINDOW_SIZE-1, features].values
        label = server_df.loc[i+WINDOW_SIZE, 'bottleneck']
        X.append(window)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(X)
print(Y)