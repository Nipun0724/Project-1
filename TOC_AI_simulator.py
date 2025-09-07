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
SIM_TIME = 500
TASK_INTERVAL = 1.0
WINDOW_SIZE = 20

CONFIG = "SCALABLE_BALANCED"

if CONFIG == "SCALABLE_BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 3
    MAX_QUEUE_LEN = 5
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10
    
    MIN_NUM_SERVERS = 2
    MAX_NUM_SERVERS = 10
    REACTIVE_SCALE_UP_THRESHOLD = 0.75
    PREDICTIVE_SCALE_UP_THRESHOLD = 0.60
    SCALE_DOWN_THRESHOLD = 0.40
    SCALING_CHECK_INTERVAL = 15

# --- Global Metrics ---
SLA_VIOLATIONS=0; TOTAL_TASKS=0; COMPLETED_TASKS=0; TASK_ID=0
TASK_TIMES={}; ACTIVE_SERVERS_HISTORY=[]
random.seed(RANDOM_SEED)

# --- ### CRITICAL REFACTOR: On-Demand Scheduling Manager ---

class SchedulingManager:
    """
    The new central brain. It is NOT a simpy process. It provides on-demand
    constraint detection by switching between reactive and predictive modes.
    """
    def __init__(self, env, all_servers, model, scaler):
        self.env = env
        self.all_servers = all_servers
        self.model = model
        self.scaler = scaler
        self.warmup_period = WINDOW_SIZE
        self.history_buffers = {s.name: deque(maxlen=WINDOW_SIZE) for s in all_servers}

    def update_history(self, server):
        """Called by the logger to feed the latest data."""
        self.history_buffers[server.name].append([
            server.cpu_used, server.q_len, server.net_in, server.net_out
        ])

    def get_current_constraint(self):
        """
        The core on-demand method. It is called by other processes to get
        a real-time assessment of the system's constraint.
        """
        if self.env.now < self.warmup_period:
            # --- REACTIVE MODE (During Warm-up) ---
            max_util, constraint_name = -1.0, 'None (Reactive)'
            for server in [s for s in self.all_servers if s.is_active]:
                cpu_util = server.cpu_used / server.cpu_capacity
                if cpu_util > max_util:
                    max_util, constraint_name = cpu_util, f"{server.name} (Reactive)"
            return {'name': constraint_name, 'score': 0.0, 'util': max_util}
        else:
            # --- PREDICTIVE MODE (AI-Driven) ---
            input_data, candidate_servers = [], []
            for server in [s for s in self.all_servers if s.is_active]:
                if len(self.history_buffers[server.name]) == WINDOW_SIZE:
                    raw_data = np.array(self.history_buffers[server.name]).astype(np.float32)
                    normalized = self.scaler.transform(raw_data)
                    input_data.append(normalized); candidate_servers.append(server)
            
            if not candidate_servers: 
                return {'name': 'Awaiting Data (Predicted)', 'score': 0.0, 'util': 0.0}

            pred_scores = self.model.predict(np.array(input_data), verbose=0).flatten()
            max_idx = np.argmax(pred_scores)
            
            return {
                'name': f"{candidate_servers[max_idx].name} (Predicted)",
                'score': float(pred_scores[max_idx]),
                'util': 0.0 # Util is not relevant in predictive mode
            }

# --- Dispatcher (Rope) ---
class Dispatcher:
    def __init__(self, env, all_servers, manager):
        self.env, self.all_servers, self.manager = env, all_servers, manager
        self.task_buffer = []; self.env.process(self.run())
    def add_task(self, task):
        self.task_buffer.append(task); self.task_buffer.sort(key=lambda x: x[5])
    def run(self):
        while True:
            if not self.task_buffer:
                yield self.env.timeout(0.5); continue
            
            # --- REFACTOR: Get a fresh, on-demand constraint check ---
            constraint = self.manager.get_current_constraint()
            constraint_name = " ".join(constraint['name'].split(' ')[0:2])
            constraint_server = next((s for s in self.all_servers if s.name == constraint_name), None)
            
            if constraint_server and constraint_server.q_len < MAX_QUEUE_LEN:
                available = [s for s in self.all_servers if s.is_active and s.q_len < MAX_QUEUE_LEN]
                if available:
                    min(available, key=lambda s: s.q_len).add_to_buffer(self.task_buffer.pop(0))
            yield self.env.timeout(0.2)

# --- Server Class (Unchanged) ---
class Server:
    def __init__(self, env, name):
        self.env, self.name = env, name
        self.cpu_capacity, self.net_capacity = CPU_CAPACITY, NET_CAPACITY
        self.cpu_used, self.net_in, self.net_out, self.current_tasks = 0, 0, 0, 0
        self.is_active = True; self.queue_items = []; self.env.process(self.run())
    def add_to_buffer(self, task): self.queue_items.append(task)
    @property
    def q_len(self): return len(self.queue_items)
    def run(self):
        while True:
            if not self.is_active or not self.queue_items:
                yield self.env.timeout(0.5); continue
            task_idx = -1
            for i, task in enumerate(self.queue_items):
                cpu, net_in, net_out, _, _, _ = task
                if (cpu + self.cpu_used <= self.cpu_capacity and net_in + self.net_in <= self.net_capacity and net_out + self.net_out <= self.net_capacity):
                    task_idx = i; break
            if task_idx != -1:
                cpu, net_in, net_out, dur, task_id, _ = self.queue_items.pop(task_idx)
                self.cpu_used += cpu; self.net_in += net_in; self.net_out += net_out; self.current_tasks += 1
                self.env.process(self.process_task(task_id, dur, cpu, net_in, net_out))
            else: yield self.env.timeout(0.5)
    def process_task(self, task_id, duration, cpu, net_in, net_out):
        global COMPLETED_TASKS, TASK_TIMES
        yield self.env.timeout(duration)
        self.cpu_used -= cpu; self.net_in -= net_in; self.net_out -= net_out
        self.current_tasks -= 1; COMPLETED_TASKS += 1
        TASK_TIMES[task_id]['completion'] = self.env.now

# --- Task Generation (Unchanged) ---
def process_incoming_task(env, dispatcher):
    global TOTAL_TASKS, TASK_ID, TASK_TIMES, SLA_VIOLATIONS
    if len(dispatcher.task_buffer) >= MAX_CENTRAL_BUFFER_LEN:
        SLA_VIOLATIONS += 1; TOTAL_TASKS += 1; return
    cpu, net_in, net_out, dur = (random.randint(8, 25), random.randint(1, 8), random.randint(1, 8), random.randint(2, 8)) if USE_LIGHTER_TASKS else (random.randint(40, 90), random.randint(5, 15), random.randint(5, 15), random.randint(5, 15))
    TASK_ID += 1; task_id = TASK_ID
    TASK_TIMES[task_id] = {'arrival': env.now, 'priority': random.randint(1, TASK_PRIORITY_RANGE)}
    task = (cpu, net_in, net_out, dur, task_id, TASK_TIMES[task_id]['priority'])
    dispatcher.add_task(task); TOTAL_TASKS += 1
def generate_task(env, dispatcher):
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL)); process_incoming_task(env, dispatcher)

# --- ELEVATE Step (Refactored to use the manager) ---
def monitor_system(env, servers, manager):
    global NUM_SERVERS
    while True:
        yield env.timeout(SCALING_CHECK_INTERVAL)
        
        # --- REFACTOR: Get a fresh, on-demand constraint check ---
        constraint = manager.get_current_constraint()
        
        if env.now < manager.warmup_period:
            if constraint['util'] > REACTIVE_SCALE_UP_THRESHOLD and NUM_SERVERS < MAX_NUM_SERVERS:
                inactive = [s for s in servers if not s.is_active];
                if inactive: inactive[0].is_active = True; NUM_SERVERS += 1; print(f"[{env.now:.2f}] ELEVATE (Reactive): Util {(constraint['util']*100):.1f}%. Scaling up. Active: {NUM_SERVERS}")
        else:
            if constraint['score'] > PREDICTIVE_SCALE_UP_THRESHOLD and NUM_SERVERS < MAX_NUM_SERVERS:
                inactive = [s for s in servers if not s.is_active];
                if inactive: inactive[0].is_active = True; NUM_SERVERS += 1; print(f"[{env.now:.2f}] ELEVATE (Predictive): Score {(constraint['score']*100):.1f}%. Scaling up. Active: {NUM_SERVERS}")

        if env.now > manager.warmup_period:
            active = [s for s in servers if s.is_active]
            avg_util = sum(s.cpu_used for s in active) / (len(active) * CPU_CAPACITY) if active else 0
            if avg_util < SCALE_DOWN_THRESHOLD and NUM_SERVERS > MIN_NUM_SERVERS:
                eligible = [s for s in active if s.current_tasks == 0 and s.q_len == 0]
                if eligible:
                    eligible[-1].is_active = False; NUM_SERVERS -= 1; print(f"[{env.now:.2f}] DE-ELEVATE: Low util ({(avg_util*100):.1f}%). Scaling down. Active: {NUM_SERVERS}")

# --- Logger ---
def periodic_logger(env, servers, log_data, manager):
    while True:
        yield env.timeout(1)
        # --- REFACTOR: Log the on-demand constraint info ---
        constraint = manager.get_current_constraint()
        ACTIVE_SERVERS_HISTORY.append({'timestamp': env.now, 'active_servers_count': sum(1 for s in servers if s.is_active),
                                     'constraint': constraint['name'], 'constraint_score': constraint['score']})
        for server in servers:
            manager.update_history(server)
            log_data.append({'timestamp': env.now, 'server_id': server.name, 'is_active': server.is_active,
                             'cpu_used': server.cpu_used, 'q_len': server.q_len, 'network_in': server.net_in, 'network_out': server.net_out})

# --- Main Simulation ---
def simulation():
    global NUM_SERVERS, SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID, TASK_TIMES, ACTIVE_SERVERS_HISTORY
    try: model = load_model("bottleneck_predictor.keras"); scaler = joblib.load('minmax_scaler.save')
    except Exception as e: print(f"ERROR: Could not load AI model. {e}"); return

    SLA_VIOLATIONS=0; TOTAL_TASKS=0; COMPLETED_TASKS=0; TASK_ID=0
    TASK_TIMES={}; ACTIVE_SERVERS_HISTORY=[]
    if CONFIG == "SCALABLE_BALANCED": NUM_SERVERS = 3
    
    random.seed(RANDOM_SEED); env = simpy.Environment()
    all_servers = [Server(env, f"Server {i+1}") for i in range(MAX_NUM_SERVERS)]
    for i in range(NUM_SERVERS): all_servers[i].is_active = True
    for i in range(NUM_SERVERS, MAX_NUM_SERVERS): all_servers[i].is_active = False

    log_data = []
    # --- REFACTOR: Instantiate the new manager ---
    manager = SchedulingManager(env, all_servers, model, scaler)
    dispatcher = Dispatcher(env, all_servers, manager)
    
    env.process(generate_task(env, dispatcher))
    env.process(periodic_logger(env, all_servers, log_data, manager))
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
        env.process(monitor_system(env, all_servers, manager))

    print("--- Starting Final Hybrid AI+TOC Simulation ---")
    print(f"Reactive Scale Up (first {WINDOW_SIZE}s): {REACTIVE_SCALE_UP_THRESHOLD*100:.0f}%")
    print(f"Predictive Scale Up (AI-driven): {PREDICTIVE_SCALE_UP_THRESHOLD*100:.0f}%")
    
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
    plt.figure(figsize=(18, 12))
    plt.suptitle(f'AI+TOC Simulation Results for {CONFIG}', fontsize=16)

    # Plot 1: Active Servers and Predicted Bottleneck Score
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df_active_servers['timestamp'], df_active_servers['active_servers_count'], marker='.', linestyle='-', color='purple', label='Active Servers')
    ax1.set_ylabel('Active Servers', color='purple'); ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_yticks(range(MIN_NUM_SERVERS, MAX_NUM_SERVERS + 2)); ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(df_active_servers['timestamp'], df_active_servers['constraint_score'], color='red', linestyle='--', label='Constraint Score')
    ax2.set_ylabel('Predicted Score', color='red'); ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.1); ax1.set_title('Scaling vs. Predicted Score')

    # Plot 2: DBR Buffer Sizes
    plt.subplot(3, 2, 2)
    for server_id in df_server_logs['server_id'].unique():
        server_data = df_server_logs[df_server_logs['server_id'] == server_id]
        if server_data['is_active'].any(): plt.plot(server_data['timestamp'], server_data['q_len'], label=f'{server_id}', alpha=0.7)
    plt.title('DBR Buffer Sizes (Queue Length)'); plt.ylabel('Buffer Length'); plt.legend(loc='upper right'); plt.grid(True)
    
    # Plot 3: CPU Usage
    plt.subplot(3, 2, 3)
    for server_id in df_server_logs['server_id'].unique():
        server_data = df_server_logs[df_server_logs['server_id'] == server_id]
        if server_data['is_active'].any(): plt.plot(server_data['timestamp'], server_data['cpu_used'], label=f'{server_id}', alpha=0.7)
    plt.title('CPU Usage per Server'); plt.ylabel('CPU Units'); plt.legend(loc='upper right'); plt.grid(True)

    # Plot 4: Identified Constraint
    plt.subplot(3, 2, 4)
    unique_constraints = df_active_servers.dropna(subset=['constraint'])['constraint'].unique()
    for constraint in unique_constraints:
        subset = df_active_servers[df_active_servers['constraint'] == constraint]
        plt.scatter(subset['timestamp'], subset['constraint'], label=constraint, s=10)
    plt.title('Predicted System Constraint'); plt.ylabel('Constraint'); plt.xticks(rotation=15); plt.legend(loc='upper right'); plt.grid(True)

    # Plot 5: Turnaround Time Distribution
    plt.subplot(3, 2, 5)
    if turnaround_times:
        plt.hist(turnaround_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Task Turnaround Time Distribution'); plt.xlabel('Time'); plt.ylabel('Frequency'); plt.grid(True)
    
    # Plot 6: Overall System Utilization
    plt.subplot(3, 2, 6)
    if not df_active_servers.empty:
        df_merged = df_server_logs.merge(df_active_servers, on='timestamp', how='left').ffill()
        df_merged['total_cpu_capacity'] = df_merged['active_servers_count'] * CPU_CAPACITY
        df_summed = df_server_logs[df_server_logs['is_active']].groupby('timestamp').agg(total_cpu=('cpu_used', 'sum')).reset_index()
        df_util = df_summed.merge(df_merged[['timestamp', 'total_cpu_capacity']].drop_duplicates(), on='timestamp', how='left')
        df_util['overall_cpu_util'] = (df_util['total_cpu'] / df_util['total_cpu_capacity']) * 100
        plt.plot(df_util['timestamp'], df_util['overall_cpu_util'], label='Overall CPU Usage', color='red')
        plt.title('Overall System CPU Utilization'); plt.xlabel('Time'); plt.ylabel('Utilization (%)'); plt.ylim(0, 100); plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    simulation()
