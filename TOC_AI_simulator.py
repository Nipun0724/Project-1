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

# Configuration Profiles
CONFIG = "SCALABLE_BALANCED" # Options: "BALANCED", "HIGH_CAPACITY", "LIGHT_TASKS", "SCALABLE_BALANCED"

if CONFIG == "BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 6
    WINDOW_SIZE = 20
    MAX_QUEUE_LEN = 8
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10 # Tasks will have priorities from 1 to this value (lower is higher priority)
    LOAD_BALANCING_WEIGHTS = {'q_len': 0.4, 'cpu': 0.4, 'net': 0.2} # Weights for server fitness calculation
    # Dynamic Scaling (disabled for BALANCED profile)
    MIN_NUM_SERVERS = NUM_SERVERS
    MAX_NUM_SERVERS = NUM_SERVERS
    SCALE_UP_THRESHOLD = 0.7 # Average queue utilization threshold to scale up (0.0 to 1.0)
    SCALE_DOWN_THRESHOLD = 0.3 # Average queue utilization threshold to scale down (0.0 to 1.0)
    SCALING_CHECK_INTERVAL = 10 # How often to check for scaling opportunities

elif CONFIG == "HIGH_CAPACITY":
    RANDOM_SEED = 42
    CPU_CAPACITY = 150
    NET_CAPACITY = 150
    NUM_SERVERS = 8
    WINDOW_SIZE = 20
    MAX_QUEUE_LEN = 5
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10
    LOAD_BALANCING_WEIGHTS = {'q_len': 0.4, 'cpu': 0.4, 'net': 0.2}
    # Dynamic Scaling (disabled for HIGH_CAPACITY profile)
    MIN_NUM_SERVERS = NUM_SERVERS
    MAX_NUM_SERVERS = NUM_SERVERS
    SCALE_UP_THRESHOLD = 0.7
    SCALE_DOWN_THRESHOLD = 0.3
    SCALING_CHECK_INTERVAL = 10

elif CONFIG == "LIGHT_TASKS":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 4
    WINDOW_SIZE = 20
    MAX_QUEUE_LEN = 6
    USE_LIGHTER_TASKS = True
    TASK_PRIORITY_RANGE = 10
    LOAD_BALANCING_WEIGHTS = {'q_len': 0.4, 'cpu': 0.4, 'net': 0.2}
    # Dynamic Scaling (disabled for LIGHT_TASKS profile)
    MIN_NUM_SERVERS = NUM_SERVERS
    MAX_NUM_SERVERS = NUM_SERVERS
    SCALE_UP_THRESHOLD = 0.7
    SCALE_DOWN_THRESHOLD = 0.3
    SCALING_CHECK_INTERVAL = 10

elif CONFIG == "SCALABLE_BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 3 # Starting number of servers
    WINDOW_SIZE = 20
    MAX_QUEUE_LEN = 8
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10
    LOAD_BALANCING_WEIGHTS = {'q_len': 0.4, 'cpu': 0.4, 'net': 0.2}
    # Dynamic Scaling (enabled for SCALABLE_BALANCED profile)
    MIN_NUM_SERVERS = 2
    MAX_NUM_SERVERS = 10
    SCALE_UP_THRESHOLD = 0.6 # If average queue utilization > 60%, scale up
    SCALE_DOWN_THRESHOLD = 0.2 # If average queue utilization < 20%, scale down
    SCALING_CHECK_INTERVAL = 10


# --- Global Simulation Metrics (will be reset for each simulation run) ---
SLA_VIOLATIONS = 0
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
TASK_ID = 0 # Global task counter
TASK_TIMES = {} # Stores arrival and completion times for each task
SLA_VIOLATIONS_PER_SERVER = defaultdict(int)
ACTIVE_SERVERS_HISTORY = [] # To log the number of active servers over time

# Set initial random seed for reproducibility
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
        self.current_tasks = 0 # Number of tasks currently being processed
        self.is_active = True # Flag to indicate if server is active (for scaling)

        # Using a list as a custom priority queue (lower priority number = higher priority)
        # Task format in queue_items: (cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority)
        self.queue_items = []

        # For ML prediction
        self.history_buffer = deque(maxlen=WINDOW_SIZE)
        self.env.process(self.run())

    @property
    def q_len(self):
        """Return current queue length."""
        return len(self.queue_items)

    def run(self):
        """Server's main process: continuously fetches and processes tasks."""
        global COMPLETED_TASKS, TASK_TIMES, SLA_VIOLATIONS, SLA_VIOLATIONS_PER_SERVER

        while True:
            # If the server is not active, it should not process tasks
            if not self.is_active:
                yield self.env.timeout(1) # Wait if inactive
                continue

            # If queue is empty, wait for tasks
            if not self.queue_items:
                yield self.env.timeout(0.5)
                continue

            # Implement priority queue logic: sort by priority (lower is higher)
            # and then by arrival time (to ensure FIFO among same priority tasks)
            # Task format: (cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority)
            self.queue_items.sort(key=lambda x: (x[5], TASK_TIMES[x[4]]['arrival']))

            # Try to find a task that can be processed with current resources
            task_to_process_index = -1
            for i, task in enumerate(self.queue_items):
                cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority = task
                if (cpu_demand + self.cpu_used <= self.cpu_capacity and
                    net_in_demand + self.net_in <= self.net_capacity and
                    net_out_demand + self.net_out <= self.net_capacity):
                    task_to_process_index = i
                    break # Found a task that fits

            if task_to_process_index != -1:
                # Resources are available for the selected task, accept and start processing
                task = self.queue_items.pop(task_to_process_index)
                cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority = task

                self.cpu_used += cpu_demand
                self.net_in += net_in_demand
                self.net_out += net_out_demand
                self.current_tasks += 1

                # Start processing task as a separate process
                self.env.process(self.process_task(cpu_demand, net_in_demand, net_out_demand, duration, task_id))
            else:
                # No task in the queue can be processed right now due to resource constraints, wait
                yield self.env.timeout(0.5)

    def process_task(self, cpu_demand, net_in_demand, net_out_demand, duration, task_id):
        """Simulates the actual processing of a task."""
        global COMPLETED_TASKS, TASK_TIMES

        yield self.env.timeout(duration) # Task consumes resources for 'duration' time

        # Release resources after task completion
        self.cpu_used -= cpu_demand
        self.net_in -= net_in_demand
        self.net_out -= net_out_demand
        self.current_tasks -= 1

        COMPLETED_TASKS += 1
        TASK_TIMES[task_id]['completion'] = self.env.now

# --- Load Balancer / Server Selection ---
# Load the pre-trained scaler
try:
    scaler = joblib.load('minmax_scaler.save')
except FileNotFoundError:
    print("Error: 'minmax_scaler.save' not found. Please ensure the scaler file is in the correct directory.")
    exit()


def choose_server(servers, model):
    """
    Chooses the best server based on ML predictions.
    Falls back to simple load balancing if not enough data is available.
    """
    input_data = []
    candidate_servers = []

    # Filter for servers that have a full history buffer for prediction
    for server in servers:
        if len(server.history_buffer) == WINDOW_SIZE:
            raw_data = np.array(server.history_buffer).astype(np.float32)
            normalized = scaler.transform(raw_data)
            input_data.append(normalized)
            candidate_servers.append(server)

    # If no servers have enough history, use simple load balancing as a fallback
    if not input_data:
        # Find server with shortest queue that's not full
        available_servers = [s for s in servers if s.q_len < MAX_QUEUE_LEN]
        if available_servers:
            return min(available_servers, key=lambda s: (s.q_len, s.cpu_used))
        else:
            # All queues are full, return the one with the shortest queue anyway
            return min(servers, key=lambda s: s.q_len)

    # Use the model to predict bottleneck scores
    input_tensor = np.array(input_data)
    preds = model.predict(input_tensor, verbose=0).flatten()

    # Pair predictions with servers and sort by the lowest predicted bottleneck score
    sorted_candidates = sorted(zip(preds, candidate_servers), key=lambda x: x[0])

    # Choose the best-predicted server that has queue space
    for pred_score, server in sorted_candidates:
        if server.q_len < MAX_QUEUE_LEN:
            return server

    # If all candidate servers have full queues, return the one with the best score (and shortest queue as tie-breaker)
    return min(candidate_servers, key=lambda s: s.q_len)


# --- Task Processing and Generation Functions ---
def process_incoming_task(env, servers, model):
    """
    Generates a new task and dispatches it to the most suitable server using AI prediction.
    This is the UPDATED function.
    """
    global TOTAL_TASKS, SLA_VIOLATIONS, TASK_ID, TASK_TIMES

    # Determine task demands based on configuration
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

    # --- CORRECTED LOGIC ---
    # Always use the AI model to choose the best server from the list of active ones.
    active_servers = [s for s in servers if s.is_active]
    if not active_servers:
        # No active servers, task is rejected immediately
        print(f"Task {task_id} rejected - No active servers available at {env.now:.2f}")
        SLA_VIOLATIONS += 1
        TOTAL_TASKS += 1
        return

    selected_server = choose_server(active_servers, model)
    # --- END OF CORRECTION ---

    if selected_server.q_len >= MAX_QUEUE_LEN:
        SLA_VIOLATIONS += 1
        SLA_VIOLATIONS_PER_SERVER[selected_server.name] += 1
        if task_id % 50 == 0 or TOTAL_TASKS < 10:
            print(f"[{env.now:.2f}] Task {task_id} (P:{priority}) rejected - Server {selected_server.name} queue full ({selected_server.q_len}/{MAX_QUEUE_LEN})")
    else:
        selected_server.queue_items.append((cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority))

    TOTAL_TASKS += 1


def generate_task(env, servers, model):
    """Process that continuously generates tasks."""
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL))
        process_incoming_task(env, servers, model)


# --- Monitoring and Logging Functions ---
def periodic_logger(env, servers, log_data, interval=1):
    """Logs server and system metrics at regular intervals."""
    while True:
        yield env.timeout(interval)
        # Log data for history buffer used by the ML model
        for server in servers:
            if server.is_active:
                 server.history_buffer.append([
                    server.q_len,
                    server.cpu_used,
                    server.net_in,
                    server.net_out
                ])

        # Log detailed data for post-simulation analysis
        for server in servers:
            log_data.append({
                'timestamp'   : env.now,
                'server_id'   : server.name,
                'is_active'   : server.is_active,
                'cpu_used'    : server.cpu_used,
                'q_len'       : server.q_len,
                'network_in'  : server.net_in,
                'network_out' : server.net_out,
                'current_tasks': server.current_tasks
            })
        
        # Log active servers count for scaling analysis
        ACTIVE_SERVERS_HISTORY.append({
            'timestamp': env.now,
            'active_servers_count': sum(1 for s in servers if s.is_active)
        })


def monitor_system(env, servers):
    """Monitors system load and scales servers up or down."""
    global NUM_SERVERS

    while True:
        yield env.timeout(SCALING_CHECK_INTERVAL)

        active_servers = [s for s in servers if s.is_active]
        if not active_servers:
            if NUM_SERVERS < MIN_NUM_SERVERS and len(servers) > NUM_SERVERS:
                servers[NUM_SERVERS].is_active = True
                NUM_SERVERS += 1
                print(f"[{env.now:.2f}] Scaling up: Activated {servers[NUM_SERVERS-1].name}. Total active: {NUM_SERVERS}")
            continue

        total_q_len = sum(s.q_len for s in active_servers)
        avg_q_utilization = (total_q_len / (len(active_servers) * MAX_QUEUE_LEN)) if (len(active_servers) * MAX_QUEUE_LEN) > 0 else 0

        # Scale Up Logic
        if avg_q_utilization > SCALE_UP_THRESHOLD and NUM_SERVERS < MAX_NUM_SERVERS:
            inactive_servers = [s for s in servers if not s.is_active]
            if inactive_servers:
                server_to_activate = inactive_servers[0]
                server_to_activate.is_active = True
                NUM_SERVERS += 1
                print(f"[{env.now:.2f}] Scaling up: Activated {server_to_activate.name}. Total active: {NUM_SERVERS}")

        # Scale Down Logic
        elif avg_q_utilization < SCALE_DOWN_THRESHOLD and NUM_SERVERS > MIN_NUM_SERVERS:
            eligible_for_shutdown = [s for s in active_servers if s.current_tasks == 0 and s.q_len == 0]
            if eligible_for_shutdown:
                server_to_deactivate = eligible_for_shutdown[0]
                server_to_deactivate.is_active = False
                NUM_SERVERS -= 1
                print(f"[{env.now:.2f}] Scaling down: Deactivated {server_to_deactivate.name}. Total active: {NUM_SERVERS}")


# --- Main Simulation Function ---
def simulation():
    global SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID, TASK_TIMES, SLA_VIOLATIONS_PER_SERVER, ACTIVE_SERVERS_HISTORY, NUM_SERVERS

    try:
        model = load_model("bottleneck_predictor.keras")
    except (IOError, ImportError) as e:
        print(f"Error loading Keras model 'bottleneck_predictor.keras': {e}")
        print("Please ensure the model file is present and you have TensorFlow/Keras installed.")
        return

    print("Starting simulation...")
    print(f"Configuration: {CONFIG}")
    print(f"Initial Servers: {NUM_SERVERS}, Task Interval: {TASK_INTERVAL}, Max Queue: {MAX_QUEUE_LEN}")
    print(f"Server Capacity: CPU={CPU_CAPACITY}, NET={NET_CAPACITY}")
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
        print(f"Dynamic Scaling: Min Servers={MIN_NUM_SERVERS}, Max Servers={MAX_NUM_SERVERS}")
    else:
        print("Dynamic Scaling: Disabled (Fixed number of servers)")
    print("="*60)

    # Reset global metrics for a new simulation run
    SLA_VIOLATIONS = 0
    TOTAL_TASKS = 0
    COMPLETED_TASKS = 0
    TASK_ID = 0
    TASK_TIMES = {}
    SLA_VIOLATIONS_PER_SERVER.clear()
    ACTIVE_SERVERS_HISTORY = []
    
    # Reset NUM_SERVERS to its initial value for the chosen CONFIG
    initial_num_servers = {
        "BALANCED": 6,
        "HIGH_CAPACITY": 8,
        "LIGHT_TASKS": 4,
        "SCALABLE_BALANCED": 3
    }.get(CONFIG, 3)
    NUM_SERVERS = initial_num_servers

    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    all_servers = [Server(env, f"Server {i+1}") for i in range(MAX_NUM_SERVERS)]
    for i in range(NUM_SERVERS):
        all_servers[i].is_active = True
    for i in range(NUM_SERVERS, MAX_NUM_SERVERS):
        all_servers[i].is_active = False

    log_data = []
    
    env.process(generate_task(env, all_servers, model))
    env.process(periodic_logger(env, all_servers, log_data))
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
        env.process(monitor_system(env, all_servers))

    env.run(until=SIM_TIME)

    # --- Post-simulation Analysis and Results (Expanded Version) ---
    df_server_logs = pd.DataFrame(log_data)
    df_server_logs.to_csv("resource_usage_log_ai_toc.csv", index=False) # Changed filename for clarity
    df_active_servers = pd.DataFrame(ACTIVE_SERVERS_HISTORY)

    sla_rate = (SLA_VIOLATIONS / TOTAL_TASKS) * 100 if TOTAL_TASKS else 0
    completion_rate_of_target = (COMPLETED_TASKS / TARGET_COMPLETED_TASKS) * 100

    # Calculate average metrics only for active servers
    if not df_server_logs.empty:
        active_server_logs = df_server_logs[df_server_logs['is_active'] == True]
        avg_q_len = active_server_logs['q_len'].mean() if not active_server_logs.empty else 0
        avg_cpu_util = (active_server_logs['cpu_used'] / CPU_CAPACITY).mean() * 100 if not active_server_logs.empty else 0
        # Add the missing network utilization calculation
        avg_net_util = ((active_server_logs['network_in'] + active_server_logs['network_out']) / (2 * NET_CAPACITY)).mean() * 100 if not active_server_logs.empty else 0
    else:
        avg_q_len = 0
        avg_cpu_util = 0
        avg_net_util = 0

    turnaround_times = [
        t['completion'] - t['arrival'] for t in TASK_TIMES.values() if 'completion' in t
    ]
    avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0

    print(f"\n=== SIMULATION RESULTS (AI + TOC) ===")
    print(f"Configuration Used: {CONFIG}")
    print(f"Target Completed Tasks: {TARGET_COMPLETED_TASKS}")
    print(f"Actual Completed Tasks: {COMPLETED_TASKS} ({completion_rate_of_target:.1f}% of target)")
    print(f"Total Tasks Generated: {TOTAL_TASKS}")
    print(f"Tasks Rejected (SLA Violations): {SLA_VIOLATIONS}")
    print(f"SLA Violation Rate: {sla_rate:.2f}%")
    print(f"Average Queue Length (Active Servers): {avg_q_len:.2f}")
    print(f"Average CPU Utilization (Active Servers): {avg_cpu_util:.2f}%")
    print(f"Average Network Utilization (Active Servers): {avg_net_util:.2f}%")
    print(f"Average Turnaround Time: {avg_turnaround:.2f}")
    print(f"System Task Completion Rate: {(COMPLETED_TASKS/TOTAL_TASKS)*100:.1f}% (Completed vs. Generated)")

    # Add performance grade
    if completion_rate_of_target >= 95:
        print("✅ EXCELLENT: Target achieved!")
    elif completion_rate_of_target >= 80:
        print("✅ GOOD: Close to target")
    elif completion_rate_of_target >= 60:
        print("⚠️  MODERATE: Needs tuning")
    else:
        print("❌ POOR: Significant underperformance")

    # Add violations per server
    print(f"SLA Violations per Server: {dict(SLA_VIOLATIONS_PER_SERVER)}")

    # Add tuning recommendations
    if completion_rate_of_target < 90 or sla_rate > 5:
        print(f"\n=== TUNING RECOMMENDATIONS ===")
        if sla_rate > 5:
            print("- High SLA violations detected (tasks rejected)")
            if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
                print("  → Consider adjusting SCALE_UP_THRESHOLD lower or increasing MAX_NUM_SERVERS.")
            else:
                print("  → Increase NUM_SERVERS or MAX_QUEUE_LEN.")
            print("  → Alternatively, reduce TASK_INTERVAL (slower task arrival rate).")
        
        if avg_q_len < 0.5 and (MAX_NUM_SERVERS > MIN_NUM_SERVERS and NUM_SERVERS > MIN_NUM_SERVERS):
            print("- Low average queue length detected, and scaling is enabled.")
            print("  → Consider adjusting SCALE_DOWN_THRESHOLD higher or decreasing MIN_NUM_SERVERS.")
        elif avg_q_len < 0.5 and (MAX_NUM_SERVERS == MIN_NUM_SERVERS):
            print("- Low average queue length detected, scaling is disabled.")
            print("  → Consider reducing NUM_SERVERS or increasing TASK_INTERVAL.")

        if avg_turnaround > (SIM_TIME / 10): # Arbitrary threshold for high turnaround
            print("- High average turnaround time detected (tasks take long to complete).")
            print("  → Reduce task complexity (duration, CPU/Net demands) or increase server capacities (CPU_CAPACITY, NET_CAPACITY).")
            if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
                print("  → Ensure SCALE_UP_THRESHOLD is not too high, preventing timely scaling.")
            else:
                print("  → Increase NUM_SERVERS.")
    
    # --- Plotting Results ---
    if not df_server_logs.empty:
        plt.figure(figsize=(18, 12))

        # Plot 1: Queue Length Over Time
        plt.subplot(3, 2, 1)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            plt.plot(server_data['timestamp'], server_data['q_len'], label=f'{server_id}', alpha=0.7)
        plt.title('Queue Length Over Time per Server')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        # Plot 2: CPU Usage Over Time
        plt.subplot(3, 2, 2)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            plt.plot(server_data['timestamp'], server_data['cpu_used'], label=f'{server_id}', alpha=0.7)
        plt.title('CPU Usage Over Time per Server')
        plt.xlabel('Time')
        plt.ylabel('CPU Used')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        # Plot 3: Active Servers Over Time
        plt.subplot(3, 2, 3)
        if not df_active_servers.empty:
            plt.plot(df_active_servers['timestamp'], df_active_servers['active_servers_count'], marker='o', linestyle='-', color='purple')
            plt.title('Number of Active Servers Over Time')
            plt.xlabel('Time')
            plt.ylabel('Active Servers')
            plt.grid(True)
            if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
                plt.yticks(range(MIN_NUM_SERVERS, MAX_NUM_SERVERS + 2))

        # Plot 4: Task Completion Time Distribution
        plt.subplot(3, 2, 4)
        if turnaround_times:
            plt.hist(turnaround_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Task Turnaround Time Distribution')
            plt.xlabel('Turnaround Time')
            plt.ylabel('Frequency')
            plt.grid(True)
        
        # Plot 5: Overall System CPU Utilization
        plt.subplot(3, 2, 5)
        if not df_active_servers.empty:
            df_merged = df_server_logs.merge(df_active_servers, on='timestamp', how='left').ffill()
            df_merged['total_cpu_capacity'] = df_merged['active_servers_count'] * CPU_CAPACITY
            df_summed_usage = df_server_logs[df_server_logs['is_active']].groupby('timestamp').agg(total_cpu_used=('cpu_used', 'sum')).reset_index()
            df_util = df_summed_usage.merge(df_merged[['timestamp', 'total_cpu_capacity']].drop_duplicates(), on='timestamp', how='left')
            df_util['overall_cpu_util'] = (df_util['total_cpu_used'] / df_util['total_cpu_capacity']) * 100
            plt.plot(df_util['timestamp'], df_util['overall_cpu_util'], label='Overall CPU Utilization', color='red')
            plt.title('Overall System CPU Utilization')
            plt.xlabel('Time')
            plt.ylabel('Utilization (%)')
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f'Simulation Results for {CONFIG} Configuration', fontsize=16, y=0.98)
        plt.show()

if __name__ == '__main__':
    simulation()
