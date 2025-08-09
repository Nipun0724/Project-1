import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Global Configuration ---
TARGET_COMPLETED_TASKS = 500
SIM_TIME = 500 # Simulation duration in time units
TASK_INTERVAL = 1.0 # Average time between task arrivals (exponential distribution)

# Configuration Profiles
CONFIG = "BALANCED" # Options: "BALANCED", "HIGH_CAPACITY", "LIGHT_TASKS"

if CONFIG == "BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 6
    MAX_QUEUE_LEN = 8
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10 # Tasks will have priorities from 1 to this value (lower is higher priority)

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


# --- Global Simulation Metrics (will be reset for each simulation run) ---
SLA_VIOLATIONS = 0
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
TASK_ID = 0 # Global task counter
TASK_TIMES = {} # Stores arrival and completion times for each task
SLA_VIOLATIONS_PER_SERVER = defaultdict(int)
LAST_SERVER_INDEX = -1 # Keeps track of the last server assigned in round-robin

# Set initial random seed for reproducibility
random.seed(RANDOM_SEED)

# --- Server Class ---
class Server:
    """
    Represents a server in the simulation. It has a fixed CPU and network capacity,
    and processes tasks from its queue.
    """
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.cpu_capacity = CPU_CAPACITY
        self.net_capacity = NET_CAPACITY
        self.cpu_used = 0
        self.net_in = 0
        self.net_out = 0
        self.current_tasks = 0 # Number of tasks currently being processed

        # Using a list as a custom priority queue (lower priority number = higher priority)
        # Task format in queue_items: (cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority)
        self.queue_items = []
        self.env.process(self.run())

    @property
    def q_len(self):
        """Return current queue length."""
        return len(self.queue_items)

    def run(self):
        """Server's main process: continuously fetches and processes tasks."""
        global COMPLETED_TASKS, TASK_TIMES, SLA_VIOLATIONS, SLA_VIOLATIONS_PER_SERVER

        while True:
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
def process_incoming_task(env, servers):
    """Generates a new task and dispatches it to a server using round-robin."""
    global TOTAL_TASKS, SLA_VIOLATIONS, TASK_ID, TASK_TIMES, LAST_SERVER_INDEX

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
    priority = random.randint(1, TASK_PRIORITY_RANGE) # Assign a random priority

    TASK_TIMES[task_id] = {'arrival': arrival_time, 'priority': priority}

    selected_server = None
    # Implement Round Robin logic
    start_index = (LAST_SERVER_INDEX + 1) % len(servers)
    
    # Iterate through servers starting from start_index in a round-robin fashion
    for i in range(len(servers)):
        current_server_index = (start_index + i) % len(servers)
        server_candidate = servers[current_server_index]

        if server_candidate.q_len < MAX_QUEUE_LEN:
            # Check if resources are available for this task on this server
            if (cpu_demand + server_candidate.cpu_used <= server_candidate.cpu_capacity and
                net_in_demand + server_candidate.net_in <= server_candidate.net_capacity and
                net_out_demand + server_candidate.net_out <= server_candidate.net_capacity):
                selected_server = server_candidate
                LAST_SERVER_INDEX = current_server_index # Update last selected server
                break # Found a suitable server, exit loop

    if selected_server:
        # Add task to the selected server's queue
        selected_server.queue_items.append((cpu_demand, net_in_demand, net_out_demand, duration, task_id, priority))
    else:
        # If no server was found that could accept the task (either queue full or no resources)
        server_for_logging = servers[start_index] # The server that was attempted first
        SLA_VIOLATIONS += 1
        SLA_VIOLATIONS_PER_SERVER[server_for_logging.name] += 1
        if task_id % 50 == 0 or TOTAL_TASKS < 10:
            print(f"[{env.now:.2f}] Task {task_id} (P:{priority}) rejected - No suitable server found (attempted {server_for_logging.name})")

    TOTAL_TASKS += 1

def generate_task(env, servers):
    """Process that continuously generates tasks."""
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL))
        process_incoming_task(env, servers)

# --- Monitoring and Logging Functions ---
def periodic_logger(env, servers, log_data, interval=1):
    """Logs server and system metrics at regular intervals."""
    while True:
        yield env.timeout(interval)
        for server in servers:
            log_data.append({
                'timestamp'    : env.now,
                'server_id'    : server.name,
                'cpu_used'     : server.cpu_used,
                'q_len'        : server.q_len,
                'network_in'   : server.net_in,
                'network_out'  : server.net_out,
                'current_tasks': server.current_tasks
            })

# --- Main Simulation Function ---
def simulation():
    """Sets up and runs the simulation, then analyzes and plots the results."""
    global SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID, TASK_TIMES, SLA_VIOLATIONS_PER_SERVER, LAST_SERVER_INDEX

    print("Starting simulation...")
    print(f"Configuration: {CONFIG}")
    print(f"Number of Servers: {NUM_SERVERS}, Task Interval: {TASK_INTERVAL}, Max Queue: {MAX_QUEUE_LEN}")
    print(f"Server Capacity: CPU={CPU_CAPACITY}, NET={NET_CAPACITY}")
    print(f"Lighter Tasks: {USE_LIGHTER_TASKS}")
    print(f"Task Priority Range: 1 (highest) to {TASK_PRIORITY_RANGE} (lowest)")
    print("Dynamic Scaling: Disabled (Fixed number of servers)")
    print(f"Target: ~{TARGET_COMPLETED_TASKS} completed tasks in {SIM_TIME} time units")
    print("="*60)

    # Reset global metrics for a new simulation run
    SLA_VIOLATIONS = 0
    TOTAL_TASKS = 0
    COMPLETED_TASKS = 0
    TASK_ID = 0
    TASK_TIMES = {}
    SLA_VIOLATIONS_PER_SERVER.clear()
    LAST_SERVER_INDEX = -1 # Reset for new simulation run

    random.seed(RANDOM_SEED) # Ensure reproducibility for each run
    env = simpy.Environment()

    # Create the fixed number of servers
    servers = [Server(env, f"Server {i+1}") for i in range(NUM_SERVERS)]
    
    log_data = [] # Data for server metrics logging
    
    # Start simulation processes
    env.process(generate_task(env, servers))
    env.process(periodic_logger(env, servers, log_data))

    # Run the simulation
    env.run(until=SIM_TIME)

    # --- Post-simulation Analysis and Results ---
    df_server_logs = pd.DataFrame(log_data)
    df_server_logs.to_csv("resource_usage_log_round_robin_no_scaling.csv", index=False)

    sla_rate = (SLA_VIOLATIONS / TOTAL_TASKS) * 100 if TOTAL_TASKS else 0
    completion_rate_of_target = (COMPLETED_TASKS / TARGET_COMPLETED_TASKS) * 100
    
    if not df_server_logs.empty:
        avg_q_len = df_server_logs['q_len'].mean()
        avg_cpu_util = (df_server_logs['cpu_used'] / CPU_CAPACITY).mean() * 100
        avg_net_util = ((df_server_logs['network_in'] + df_server_logs['network_out']) / (2 * NET_CAPACITY)).mean() * 100
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

    # --- Plotting Results ---
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
        plt.suptitle(f'Simulation Results for {CONFIG} Configuration (Round Robin, No Scaling)', fontsize=16, y=0.98)
        plt.show()

if __name__ == '__main__':
    simulation()
