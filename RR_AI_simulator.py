import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import joblib
from tensorflow import keras
from keras.models import load_model

# --- Global Configuration ---
TARGET_COMPLETED_TASKS = 500
SIM_TIME = 500 # Simulation duration in time units
TASK_INTERVAL = 1.0 # Average time between task arrivals (exponential distribution)
WINDOW_SIZE = 20 # <<< AI ADDITION: Needed for the prediction model

# Configuration Profiles
CONFIG = "LIGHT_TASKS" # Options: "BALANCED", "HIGH_CAPACITY", "LIGHT_TASKS"

if CONFIG == "BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 6
    MAX_QUEUE_LEN = 8
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10

elif CONFIG == "HIGH_CAPACITY":
    RANDOM_SEED = 42
    CPU_CAPACITY = 150
    NET_CAPACITY = 150
    NUM_SERVERS = 8
    MAX_QUEUE_LEN = 5
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10

elif CONFIG == "LIGHT_TASKS":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 4
    MAX_QUEUE_LEN = 6
    USE_LIGHTER_TASKS = True
    TASK_PRIORITY_RANGE = 10

# --- Global Simulation Metrics ---
SLA_VIOLATIONS = 0
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
TASK_ID = 0
TASK_TIMES = {}
SLA_VIOLATIONS_PER_SERVER = defaultdict(int)

random.seed(RANDOM_SEED)

# --- Server Class ---
class Server:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.cpu_capacity = CPU_CAPACITY
        self.net_capacity = NET_CAPACITY
        self.cpu_used = 0
        self.net_in = 0
        self.net_out = 0
        self.current_tasks = 0
        self.queue_items = []
        
        # <<< AI ADDITION: Buffer to store recent history for predictions
        self.history_buffer = deque(maxlen=WINDOW_SIZE)
        
        self.env.process(self.run())

    @property
    def q_len(self):
        return len(self.queue_items)

    def run(self):
        global COMPLETED_TASKS, TASK_TIMES
        while True:
            if not self.queue_items:
                yield self.env.timeout(0.5)
                continue
            
            self.queue_items.sort(key=lambda x: (x[5], TASK_TIMES[x[4]]['arrival']))
            
            task_to_process_index = -1
            for i, task in enumerate(self.queue_items):
                cpu_demand, net_in_demand, net_out_demand, _, _, _ = task
                if (cpu_demand + self.cpu_used <= self.cpu_capacity and
                    net_in_demand + self.net_in <= self.net_capacity and
                    net_out_demand + self.net_out <= self.net_capacity):
                    task_to_process_index = i
                    break
            
            if task_to_process_index != -1:
                task = self.queue_items.pop(task_to_process_index)
                cpu_demand, net_in_demand, net_out_demand, duration, task_id, _ = task
                self.cpu_used += cpu_demand
                self.net_in += net_in_demand
                self.net_out += net_out_demand
                self.current_tasks += 1
                self.env.process(self.process_task(cpu_demand, net_in_demand, net_out_demand, duration, task_id))
            else:
                yield self.env.timeout(0.5)

    def process_task(self, cpu_demand, net_in_demand, net_out_demand, duration, task_id):
        global COMPLETED_TASKS, TASK_TIMES
        yield self.env.timeout(duration)
        self.cpu_used -= cpu_demand
        self.net_in -= net_in_demand
        self.net_out -= net_out_demand
        self.current_tasks -= 1
        COMPLETED_TASKS += 1
        TASK_TIMES[task_id]['completion'] = self.env.now

# --- AI ADDITION: Server Selection Logic ---
def choose_server(servers, model, scaler):
    """
    Chooses the best server based on ML predictions.
    Falls back to simple load balancing if not enough data is available.
    """
    input_data = []
    candidate_servers = []

    for server in servers:
        if len(server.history_buffer) == WINDOW_SIZE:
            raw_data = np.array(server.history_buffer).astype(np.float32)
            normalized = scaler.transform(raw_data)
            input_data.append(normalized)
            candidate_servers.append(server)

    if not input_data:
        # Fallback: if no server has enough history, choose based on shortest queue
        return min(servers, key=lambda s: s.q_len)

    # Predict bottleneck scores
    input_tensor = np.array(input_data)
    preds = model.predict(input_tensor, verbose=0).flatten()

    # Pair predictions with servers and sort by lowest predicted bottleneck
    sorted_candidates = sorted(zip(preds, candidate_servers), key=lambda x: x[0])
    
    # Choose the best-predicted server that has queue space
    for pred_score, server in sorted_candidates:
        if server.q_len < MAX_QUEUE_LEN:
            return server

    # If all candidate servers have full queues, return the one with the best score (and shortest queue as tie-breaker)
    return min(candidate_servers, key=lambda s: s.q_len)

# --- MODIFIED: Task Processing now uses AI ---
def process_incoming_task(env, servers, model, scaler):
    """Generates a task and dispatches it using the AI model."""
    global TOTAL_TASKS, SLA_VIOLATIONS, TASK_ID, TASK_TIMES

    if USE_LIGHTER_TASKS:
        cpu_demand = random.randint(8, 25)
        net_in_demand = random.randint(1, 8)
        net_out_demand = random.randint(1, 8)
        duration = random.randint(2, 8)
    else:
        cpu_demand = random.randint(40, 90)
        net_in_demand = random.randint(5, 15)
        net_out_demand = random.randint(5, 15)
        duration = random.randint(5, 15)

    TASK_ID += 1
    task_id = TASK_ID
    arrival_time = env.now
    priority = random.randint(1, TASK_PRIORITY_RANGE)
    TASK_TIMES[task_id] = {'arrival': arrival_time, 'priority': priority}

    # <<< CORE CHANGE: Use AI to choose the server, replacing Round Robin
    selected_server = choose_server(servers, model, scaler)
    
    # Check if the chosen server can accept the task
    if selected_server.q_len >= MAX_QUEUE_LEN:
        SLA_VIOLATIONS += 1
        SLA_VIOLATIONS_PER_SERVER[selected_server.name] += 1
        if task_id % 50 == 0 or TOTAL_TASKS < 10:
            print(f"[{env.now:.2f}] Task {task_id} rejected - AI-chosen server {selected_server.name} queue full.")
    else:
        selected_server.queue_items.append((cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority))
        
    TOTAL_TASKS += 1

# --- MODIFIED: Task Generator now passes AI models ---
def generate_task(env, servers, model, scaler):
    """Continuously generates tasks."""
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL))
        process_incoming_task(env, servers, model, scaler)

# --- MODIFIED: Logger now populates AI history buffer ---
def periodic_logger(env, servers, log_data, interval=1):
    """Logs metrics and populates the history buffer for the AI."""
    while True:
        yield env.timeout(interval)
        for server in servers:
            # <<< AI ADDITION: Populate history buffer for each server
            server.history_buffer.append([
                server.q_len,
                server.cpu_used,
                server.net_in,
                server.net_out
            ])
            
            # Log data for post-simulation analysis
            log_data.append({
                'timestamp'   : env.now,
                'server_id'   : server.name,
                'cpu_used'    : server.cpu_used,
                'q_len'       : server.q_len,
                'network_in'  : server.net_in,
                'network_out' : server.net_out,
                'current_tasks': server.current_tasks
            })

# --- Main Simulation Function ---
def simulation():
    global SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID, TASK_TIMES, SLA_VIOLATIONS_PER_SERVER

    # <<< AI ADDITION: Load the trained model and scaler
    try:
        model = load_model("bottleneck_predictor.keras")
        scaler = joblib.load('minmax_scaler.save')
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Please ensure 'bottleneck_predictor.keras' and 'minmax_scaler.save' are in the directory.")
        return

    print("Starting simulation with AI-Driven Scheduler...")
    print(f"Configuration: {CONFIG}")
    print(f"Number of Servers: {NUM_SERVERS}, Task Interval: {TASK_INTERVAL}, Max Queue: {MAX_QUEUE_LEN}")
    print("="*60)

    # Reset metrics
    SLA_VIOLATIONS = 0
    TOTAL_TASKS = 0
    COMPLETED_TASKS = 0
    TASK_ID = 0
    TASK_TIMES = {}
    SLA_VIOLATIONS_PER_SERVER.clear()

    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    servers = [Server(env, f"Server {i+1}") for i in range(NUM_SERVERS)]
    log_data = []
    
    # <<< AI MODIFICATION: Pass model and scaler to processes
    env.process(generate_task(env, servers, model, scaler))
    env.process(periodic_logger(env, servers, log_data))

    env.run(until=SIM_TIME)

    # --- Post-simulation Analysis (remains the same) ---
    df_server_logs = pd.DataFrame(log_data)
    df_server_logs.to_csv("resource_usage_log_ai_scheduler.csv", index=False)

    sla_rate = (SLA_VIOLATIONS / TOTAL_TASKS) * 100 if TOTAL_TASKS else 0
    completion_rate_of_target = (COMPLETED_TASKS / TARGET_COMPLETED_TASKS) * 100
    
    if not df_server_logs.empty:
        avg_q_len = df_server_logs['q_len'].mean()
        avg_cpu_util = (df_server_logs['cpu_used'] / CPU_CAPACITY).mean() * 100
        avg_net_util = ((df_server_logs['network_in'] + df_server_logs['network_out']) / (2 * NET_CAPACITY)).mean() * 100
    else:
        avg_q_len = avg_cpu_util = avg_net_util = 0

    turnaround_times = [t['completion'] - t['arrival'] for t in TASK_TIMES.values() if 'completion' in t]
    avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0

    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Configuration Used: {CONFIG}")
    print(f"Target Completed Tasks: {TARGET_COMPLETED_TASKS}")
    print(f"Actual Completed Tasks: {COMPLETED_TASKS} ({completion_rate_of_target:.1f}% of target)")
    print(f"Total Tasks Generated: {TOTAL_TASKS}")
    print(f"Tasks Rejected (SLA Violations): {SLA_VIOLATIONS}")
    print(f"SLA Violation Rate: {sla_rate:.2f}%")
    print(f"Average Queue Length: {avg_q_len:.2f}")
    print(f"Average CPU Utilization: {avg_cpu_util:.2f}%")
    print(f"Average Network Utilization: {avg_net_util:.2f}%")
    print(f"Average Turnaround Time: {avg_turnaround:.2f}")
    print(f"System Task Completion Rate: {(COMPLETED_TASKS/TOTAL_TASKS)*100:.1f}% (Completed vs. Generated)")

    if completion_rate_of_target >= 95:
        print("✅ EXCELLENT: Target achieved!")
    elif completion_rate_of_target >= 80:
        print("✅ GOOD: Close to target")
    elif completion_rate_of_target >= 60:
        print("⚠️  MODERATE: Needs tuning")
    else:
        print("❌ POOR: Significant underperformance")

    print(f"SLA Violations per Server: {dict(SLA_VIOLATIONS_PER_SERVER)}")

    # --- Tuning Recommendations ---
    if completion_rate_of_target < 90 or sla_rate > 5:
        print(f"\n=== TUNING RECOMMENDATIONS ===")
        if sla_rate > 5:
            print("- High SLA violations detected (tasks rejected)")
            print("  → Increase NUM_SERVERS or MAX_QUEUE_LEN.")
            print("  → Alternatively, reduce TASK_INTERVAL (slower task arrival).")
        
        if avg_q_len < 0.5:
            print("- Low average queue length detected.")
            print("  → Consider reducing NUM_SERVERS or increasing TASK_INTERVAL.")

        if avg_turnaround > (SIM_TIME / 10): # Arbitrary threshold for high turnaround
            print("- High average turnaround time detected (tasks take long to complete).")
            print("  → Reduce task complexity (duration, CPU/Net demands) or increase server capacities (CPU_CAPACITY, NET_CAPACITY).")
            print("  → Increase NUM_SERVERS.")
        
    # --- Plotting Results (remains the same) ---
    if not df_server_logs.empty:
        plt.figure(figsize=(18, 8))

        # Plot 1: Queue Length Over Time
        plt.subplot(2, 2, 1)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            plt.plot(server_data['timestamp'], server_data['q_len'], label=f'{server_id}', alpha=0.7)
        plt.title('Queue Length Over Time per Server')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        # Plot 2: CPU Usage Over Time
        plt.subplot(2, 2, 2)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            plt.plot(server_data['timestamp'], server_data['cpu_used'], label=f'{server_id}', alpha=0.7)
        plt.title('CPU Usage Over Time per Server')
        plt.xlabel('Time')
        plt.ylabel('CPU Used')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        # Plot 3: Network Usage Over Time
        plt.subplot(2, 2, 3)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            plt.plot(server_data['timestamp'], server_data['network_in'], label=f'{server_id} In', linestyle='-', alpha=0.7)
            plt.plot(server_data['timestamp'], server_data['network_out'], label=f'{server_id} Out', linestyle='--', alpha=0.7)
        plt.title('Network Usage Over Time per Server')
        plt.xlabel('Time')
        plt.ylabel('Network Usage')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        # Plot 4: Task Completion Time Distribution
        plt.subplot(2, 2, 4)
        if turnaround_times:
            plt.hist(turnaround_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Task Turnaround Time Distribution')
            plt.xlabel('Turnaround Time')
            plt.ylabel('Frequency')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No tasks completed to plot.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
        plt.suptitle(f'Simulation Results for {CONFIG} Configuration (Round Robin with AI, No Scaling)', fontsize=16, y=0.98)
        plt.show()

if __name__ == '__main__':
    simulation()
