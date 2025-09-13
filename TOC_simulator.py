import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Global Configuration ---
TARGET_COMPLETED_TASKS = 500
SIM_TIME = 500 # Simulation duration in time units
TASK_INTERVAL = 1.0 # Average time between task arrivals (exponential distribution)

# --- TOC: Configuration for the Constraint Detector ---
# EWMA (Exponentially Weighted Moving Average) alpha for smoothing utilization metrics
EWMA_ALPHA = 0.2 
CONSTRAINT_CHECK_INTERVAL = 5 # How often to re-evaluate the system constraint
USE_CYCLICAL_LOAD = True

# Configuration Profiles
CONFIG = "SCALABLE_BALANCED" # Options: "BALANCED", "HIGH_CAPACITY", "LIGHT_TASKS", "SCALABLE_BALANCED"

if CONFIG == "BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 6
    MAX_QUEUE_LEN = 8
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10 # Tasks will have priorities from 1 to this value (lower is higher priority)
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
    MAX_QUEUE_LEN = 5
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10
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
    MAX_QUEUE_LEN = 6
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = True
    TASK_PRIORITY_RANGE = 10
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
    MAX_QUEUE_LEN = 5
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10
    # Dynamic Scaling (enabled for SCALABLE_BALANCED profile)
    MIN_NUM_SERVERS = 2
    MAX_NUM_SERVERS = 10
    SCALE_UP_THRESHOLD = 0.8 # If average queue utilization > 70%, scale up
    SCALE_DOWN_THRESHOLD = 0.4 # If average queue utilization < 20%, scale down
    SCALING_CHECK_INTERVAL = 15


# --- Global Simulation Metrics (will be reset for each simulation run) ---
SLA_VIOLATIONS = 0
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
TASK_ID = 0 # Global task counter
TASK_TIMES = {} # Stores arrival and completion times for each task
ACTIVE_SERVERS_HISTORY = [] # To log the number of active servers over time

# Set initial random seed for reproducibility
random.seed(RANDOM_SEED)

# --- TOC STEP 1: IDENTIFY ---
class ConstraintDetector:
    """Monitors all resources to identify the single system constraint."""
    def __init__(self, env, all_servers):
        self.env = env
        self.all_servers = all_servers
        self.current_constraint = {'name': 'None', 'util': 0.0}
        # Initialize smoothed utilization for all potential resources
        self.smoothed_utils = {
            f"{s.name}_{res}": 0.0 
            for s in all_servers 
            for res in ['cpu', 'net']
        }
        self.env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(CONSTRAINT_CHECK_INTERVAL)
            
            max_util = -1.0
            constraint_name = 'None'

            active_servers = [s for s in self.all_servers if s.is_active]
            for server in active_servers:
                # Calculate current utilization
                cpu_util = server.cpu_used / server.cpu_capacity
                net_util = (server.net_in + server.net_out) / (2 * server.net_capacity)

                # Update smoothed utilization using EWMA
                self.smoothed_utils[f"{server.name}_cpu"] = (EWMA_ALPHA * cpu_util) + \
                    (1 - EWMA_ALPHA) * self.smoothed_utils[f"{server.name}_cpu"]
                
                self.smoothed_utils[f"{server.name}_net"] = (EWMA_ALPHA * net_util) + \
                    (1 - EWMA_ALPHA) * self.smoothed_utils[f"{server.name}_net"]

                # Check if this resource is the new constraint
                if self.smoothed_utils[f"{server.name}_cpu"] > max_util:
                    max_util = self.smoothed_utils[f"{server.name}_cpu"]
                    constraint_name = f"{server.name} CPU"
                
                if self.smoothed_utils[f"{server.name}_net"] > max_util:
                    max_util = self.smoothed_utils[f"{server.name}_net"]
                    constraint_name = f"{server.name} Network"

            self.current_constraint['name'] = constraint_name
            self.current_constraint['util'] = max_util

# --- TOC STEPS 2 & 3: EXPLOIT & SUBORDINATE ---
class Dispatcher:
    """Acts as the gatekeeper (Rope) managing the central task buffer."""
    def __init__(self, env, all_servers, constraint_detector):
        self.env = env
        self.all_servers = all_servers
        self.constraint_detector = constraint_detector
        self.task_buffer = [] # Central buffer for all incoming tasks
        self.env.process(self.run())

    def add_task(self, task):
        self.task_buffer.append(task)
        # Sort by priority
        self.task_buffer.sort(key=lambda x: x[5])

    def run(self):
        while True:
            if not self.task_buffer:
                yield self.env.timeout(0.5) # Wait if no tasks
                continue

            # Find the server object that is the current constraint
            constraint_name_parts = self.constraint_detector.current_constraint['name'].split(' ')
            constraint_server_name = " ".join(constraint_name_parts[0:2]) if len(constraint_name_parts) > 1 else None
            constraint_server = next((s for s in self.all_servers if s.name == constraint_server_name), None)
            
            # DBR Logic: The "Rope"
            # Only release work if the constraint's buffer has space.
            if constraint_server and constraint_server.q_len < MAX_QUEUE_LEN:
                 # Find a server for the highest priority task
                task_to_dispatch = self.task_buffer.pop(0)
                
                # Simple dispatch: find first available server
                # A more advanced TOC model would subordinate even this choice
                # to better protect the constraint.
                available_servers = [s for s in self.all_servers if s.is_active and s.q_len < MAX_QUEUE_LEN]
                if available_servers:
                    # Dispatch to the least loaded available server
                    best_server = min(available_servers, key=lambda s: s.q_len)
                    best_server.add_to_buffer(task_to_dispatch)

            yield self.env.timeout(0.2) # Check frequently

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
        self.env.process(self.run())
    
    def add_to_buffer(self, task):
        self.queue_items.append(task)
    
    @property
    def q_len(self):
        """Return current queue length."""
        return len(self.queue_items)

    def run(self):
        """Server's main process: continuously fetches and processes tasks."""
        global COMPLETED_TASKS, TASK_TIMES

        while True:
            # If the server is not active, it should not process tasks
            if not self.is_active or not self.queue_items:
                yield self.env.timeout(0.5) # Wait if inactive
                continue

            # Try to find a task that can be processed with current resources
            task_to_process_index = -1
            for i, task in enumerate(self.queue_items):
                cpu_demand, net_in_demand, net_out_demand, _, _, _ = task
                if (cpu_demand + self.cpu_used <= self.cpu_capacity and
                    net_in_demand + self.net_in <= self.net_capacity and
                    net_out_demand + self.net_out <= self.net_capacity):
                    task_to_process_index = i
                    break # Found a task that fits

            if task_to_process_index != -1:
                # Resources are available for the selected task, accept and start processing
                task = self.queue_items.pop(task_to_process_index)
                cpu_demand, net_in_demand, net_out_demand, duration, task_id, _ = task

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

# --- Task Processing and Generation Functions ---
def process_incoming_task(env, dispatcher):
    """Generates a new task and passes it to the central dispatcher."""
    global TOTAL_TASKS, TASK_ID, TASK_TIMES, SLA_VIOLATIONS
    
    # --- Check for SLA violation before creating the task ---
    if len(dispatcher.task_buffer) >= MAX_CENTRAL_BUFFER_LEN:
        SLA_VIOLATIONS += 1
        TOTAL_TASKS += 1 # Still count it as an attempt
        return

    # Task property generation
    if USE_CYCLICAL_LOAD:
        # This logic mimics the sine-wave pattern from the training data script.
        time_in_cycle = env.now % 100  # 100-second cycle
        cpu_base = 70 + 40 * np.sin(2 * np.pi * time_in_cycle / 100)
        net_base = 50 + 25 * np.sin(2 * np.pi * time_in_cycle / 100)
        cpu_demand = int(np.clip(cpu_base + random.gauss(0, 10), 10, CPU_CAPACITY))
        net_in_demand = int(np.clip(net_base + random.gauss(0, 5), 5, NET_CAPACITY))
        net_out_demand = int(np.clip(net_base + random.gauss(0, 5), 5, NET_CAPACITY))
        duration = random.randint(5, 15)
    elif USE_LIGHTER_TASKS:
        cpu_demand, net_in_demand, net_out_demand, duration = (
            random.randint(8, 25),
            random.randint(1, 8),
            random.randint(1, 8),
            random.randint(2, 8),
        )
    else:  # Original random load
        cpu_demand, net_in_demand, net_out_demand, duration = (
            random.randint(40, 90),
            random.randint(5, 15),
            random.randint(5, 15),
            random.randint(5, 15),
        )
    TASK_ID += 1
    task_id = TASK_ID
    arrival_time = env.now
    priority = random.randint(1, TASK_PRIORITY_RANGE)
    TASK_TIMES[task_id] = {'arrival': arrival_time, 'priority': priority}
    
    # Task is added to the dispatcher instead of a server
    task = (cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority)
    dispatcher.add_task(task)
    
    TOTAL_TASKS += 1

def generate_task(env, dispatcher):
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL))
        process_incoming_task(env, dispatcher)

# --- TOC STEP 4: ELEVATE ---
def monitor_system(env, servers, constraint_detector):
    """Monitors the constraint and scales servers up or down."""
    global NUM_SERVERS

    while True:
        yield env.timeout(SCALING_CHECK_INTERVAL)
        
        # Scale Up Logic: Based on the identified constraint's utilization
        if constraint_detector.current_constraint['util'] > SCALE_UP_THRESHOLD and NUM_SERVERS < MAX_NUM_SERVERS:
            inactive_servers = [s for s in servers if not s.is_active]
            if inactive_servers:
                server_to_activate = inactive_servers[0]
                server_to_activate.is_active = True
                NUM_SERVERS += 1
                print(f"[{env.now:.2f}] ELEVATE: Constraint {constraint_detector.current_constraint['name']} at {(constraint_detector.current_constraint['util']*100):.1f}%. Scaling up. Active servers: {NUM_SERVERS}")
        
        # Scale Down Logic: Based on overall system idleness
        else:
            active_servers = [s for s in servers if s.is_active]
            total_cpu = sum(s.cpu_used for s in active_servers)
            total_cpu_capacity = len(active_servers) * CPU_CAPACITY
            avg_sys_util = total_cpu / total_cpu_capacity if total_cpu_capacity > 0 else 0

            if avg_sys_util < SCALE_DOWN_THRESHOLD and NUM_SERVERS > MIN_NUM_SERVERS:
                # Find an idle server to shut down
                eligible_for_shutdown = [s for s in active_servers if s.current_tasks == 0 and s.q_len == 0]
                if eligible_for_shutdown:
                    server_to_deactivate = eligible_for_shutdown[-1] # Pick last one
                    server_to_deactivate.is_active = False
                    NUM_SERVERS -= 1
                    print(f"[{env.now:.2f}] DE-ELEVATE: Low system utilization ({(avg_sys_util*100):.1f}%). Scaling down. Active servers: {NUM_SERVERS}")

# --- Logger and Main Simulation ---
def periodic_logger(env, servers, log_data, constraint_detector):
    while True:
        yield env.timeout(1)
        # Log active servers count
        ACTIVE_SERVERS_HISTORY.append({
            'timestamp': env.now,
            'active_servers_count': sum(1 for s in servers if s.is_active),
            'constraint': constraint_detector.current_constraint['name'],
            'constraint_util': constraint_detector.current_constraint['util']
        })
        for server in servers:
            log_data.append({
                'timestamp': env.now, 'server_id': server.name, 'is_active': server.is_active,
                'cpu_used': server.cpu_used, 'q_len': server.q_len,
                'network_in': server.net_in, 'network_out': server.net_out,
            })

# --- Main Simulation Function ---
def simulation():
    global SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID, TASK_TIMES, ACTIVE_SERVERS_HISTORY, NUM_SERVERS

    print("Starting simulation...")
    print(f"Configuration: {CONFIG}")
    print(f"Initial Servers: {NUM_SERVERS}, Task Interval: {TASK_INTERVAL}, Max Queue: {MAX_QUEUE_LEN}")
    print(f"Server Capacity: CPU={CPU_CAPACITY}, NET={NET_CAPACITY}")
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
        print(f"Dynamic Scaling: Min Servers={MIN_NUM_SERVERS}, Max Servers={MAX_NUM_SERVERS}")
        print(f"Scale Up Threshold (Avg Q Util): {SCALE_UP_THRESHOLD*100:.0f}%")
        print(f"Scale Down Threshold (Avg Q Util): {SCALE_DOWN_THRESHOLD*100:.0f}%")
        print(f"Scaling Check Interval: {SCALING_CHECK_INTERVAL} time units")
    else:
        print("Dynamic Scaling: Disabled (Fixed number of servers)")

    print(f"Target: ~{TARGET_COMPLETED_TASKS} completed tasks in {SIM_TIME} time units")
    print("="*60)

    # Reset global metrics for a new simulation run
    SLA_VIOLATIONS = 0
    TOTAL_TASKS = 0
    COMPLETED_TASKS = 0
    TASK_ID = 0
    TASK_TIMES = {}
    ACTIVE_SERVERS_HISTORY = []
    # Reset NUM_SERVERS to its initial value for the chosen CONFIG
    if CONFIG == "SCALABLE_BALANCED":
        NUM_SERVERS = 3 # Starting value for scalable config
    else:
        NUM_SERVERS = 6 if CONFIG == "BALANCED" else (8 if CONFIG == "HIGH_CAPACITY" else 4)

    random.seed(RANDOM_SEED) # Ensure reproducibility for each run
    env = simpy.Environment()

    # Create all possible servers up to MAX_NUM_SERVERS, but only activate initial NUM_SERVERS
    all_servers = [Server(env, f"Server {i+1}") for i in range(MAX_NUM_SERVERS)]
    for i in range(NUM_SERVERS): # Activate initial set of servers
        all_servers[i].is_active = True
    for i in range(NUM_SERVERS, MAX_NUM_SERVERS): # Deactivate the rest initially
        all_servers[i].is_active = False

    log_data = [] # Data for server metrics logging
    
    # --- TOC: Instantiate the new components ---
    constraint_detector = ConstraintDetector(env, all_servers)
    dispatcher = Dispatcher(env, all_servers, constraint_detector)
    
    # Start simulation processes
    env.process(generate_task(env, dispatcher))
    env.process(periodic_logger(env, all_servers, log_data, constraint_detector))
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS: # Only start monitor if scaling is enabled
        env.process(monitor_system(env, all_servers, constraint_detector))

    # Run the simulation
    env.run(until=SIM_TIME)

    # --- Post-simulation Analysis and Results ---
    df_server_logs = pd.DataFrame(log_data)
    df_server_logs.to_csv("resource_usage_log_enhanced.csv", index=False)

    df_active_servers = pd.DataFrame(ACTIVE_SERVERS_HISTORY)

    sla_rate = (SLA_VIOLATIONS / TOTAL_TASKS) * 100 if TOTAL_TASKS else 0
    completion_rate_of_target = (COMPLETED_TASKS / TARGET_COMPLETED_TASKS) * 100
    
    # Calculate average queue length only for active servers
    if not df_server_logs.empty:
        active_server_logs = df_server_logs[df_server_logs['is_active'] == True]
        avg_q_len = active_server_logs['q_len'].mean() if not active_server_logs.empty else 0
        avg_cpu_util = (active_server_logs['cpu_used'] / CPU_CAPACITY).mean() * 100 if not active_server_logs.empty else 0
        avg_net_util = ((active_server_logs['network_in'] + active_server_logs['network_out']) / (2 * NET_CAPACITY)).mean() * 100 if not active_server_logs.empty else 0
    else:
        avg_q_len = 0
        avg_cpu_util = 0
        avg_net_util = 0

    turnaround_times = [
        t['completion'] - t['arrival'] for t in TASK_TIMES.values() if 'completion' in t
    ]
    avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0

    print(f"\n=== SIMULATION RESULTS ===")
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

    if completion_rate_of_target >= 95:
        print("✅ EXCELLENT: Target achieved!")
    elif completion_rate_of_target >= 80:
        print("✅ GOOD: Close to target")
    elif completion_rate_of_target >= 60:
        print("⚠️  MODERATE: Needs tuning")
    else:
        print("❌ POOR: Significant underperformance")


    # --- Tuning Recommendations ---
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

    # --- Plotting Results (Corrected and Expanded) ---
    if not df_server_logs.empty:
        plt.figure(figsize=(18, 12)) # Increased size for a 3x2 grid
        plt.suptitle(f'TOC Simulation Results for {CONFIG}', fontsize=16)

        # Plot 1: Active Servers and Constraint Utilization
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df_active_servers['timestamp'], df_active_servers['active_servers_count'], marker='.', linestyle='-', color='purple', label='Active Servers')
        ax1.set_ylabel('Active Servers', color='purple'); ax1.tick_params(axis='y', labelcolor='purple')
        ax1.set_yticks(range(MIN_NUM_SERVERS, MAX_NUM_SERVERS + 2)); ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(df_active_servers['timestamp'], df_active_servers['constraint_util'], color='red', linestyle='--', label='Constraint Util')
        ax2.set_ylabel('Constraint Utilization', color='red'); ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1.1)
        ax1.set_title('Scaling vs. Constraint Utilization')
        ax1.legend(loc='upper left'); ax2.legend(loc='lower right')

        # Plot 2: DBR Buffer Sizes (Queue Length)
        plt.subplot(3, 2, 2)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            if server_data['is_active'].any():
                plt.plot(server_data['timestamp'], server_data['q_len'], label=f'{server_id}', alpha=0.7)
        plt.title('DBR Buffer Sizes (Queue Length)'); plt.ylabel('Buffer Length')
        plt.legend(loc='best'); plt.grid(True)
        
        # Plot 3: Identified System Constraint
        plt.subplot(3, 2, 3)
        unique_constraints = df_active_servers.dropna(subset=['constraint'])['constraint'].unique()
        for constraint in unique_constraints:
            subset = df_active_servers[df_active_servers['constraint'] == constraint]
            plt.scatter(subset['timestamp'], subset['constraint'], label=constraint, s=10)
        plt.title('System Constraint Over Time'); plt.ylabel('Constraint')
        plt.xticks(rotation=15); plt.legend(loc='best'); plt.grid(True)

        # Plot 4: Task Turnaround Time Distribution
        plt.subplot(3, 2, 4)
        if turnaround_times:
            plt.hist(turnaround_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Task Turnaround Time Distribution'); plt.xlabel('Time')
            plt.ylabel('Frequency'); plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No tasks completed.', ha='center', va='center')

        # Plot 5: CPU Usage Per Server
        plt.subplot(3, 2, 5)
        for server_id in df_server_logs['server_id'].unique():
            server_data = df_server_logs[df_server_logs['server_id'] == server_id]
            if server_data['is_active'].any():
                plt.plot(server_data['timestamp'], server_data['cpu_used'], label=f'{server_id}', alpha=0.7)
        plt.title('CPU Usage per Server'); plt.ylabel('CPU Units'); plt.xlabel('Time')
        plt.legend(loc='best'); plt.grid(True)

        # Plot 6: Overall System Utilization (CPU & Network)
        plt.subplot(3, 2, 6)
        if not df_active_servers.empty:
            df_merged = df_server_logs.merge(df_active_servers, on='timestamp', how='left').ffill()
            df_merged['total_cpu_capacity'] = df_merged['active_servers_count'] * CPU_CAPACITY
            df_merged['total_net_capacity'] = df_merged['active_servers_count'] * NET_CAPACITY * 2
            
            df_summed = df_server_logs[df_server_logs['is_active']].groupby('timestamp').agg(
                total_cpu=('cpu_used', 'sum'),
                total_net=('network_in', lambda x: x.sum() + df_server_logs.loc[x.index, 'network_out'].sum())
            ).reset_index()
            
            df_util = df_summed.merge(df_merged[['timestamp', 'total_cpu_capacity', 'total_net_capacity']].drop_duplicates(), on='timestamp', how='left')
            
            df_util['overall_cpu_util'] = (df_util['total_cpu'] / df_util['total_cpu_capacity']) * 100
            df_util['overall_net_util'] = (df_util['total_net'] / df_util['total_net_capacity']) * 100
            
            plt.plot(df_util['timestamp'], df_util['overall_cpu_util'], label='Overall CPU Usage', color='red')
            plt.plot(df_util['timestamp'], df_util['overall_net_util'], label='Overall Network Usage', color='blue', linestyle='--')
            
            plt.title('Overall System Utilization'); plt.xlabel('Time'); plt.ylabel('Utilization (%)')
            plt.ylim(0, 100); plt.legend(); plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    simulation()
