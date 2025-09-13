import random
from collections import deque

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy

# --- tensorflow and keras can be heavy dependencies, so we handle the import gracefully ---
try:
    import tensorflow as tf
    from keras.models import load_model
except ImportError:
    print("Warning: TensorFlow/Keras not found. AI features will not be available.")
    print("Please install with: pip install tensorflow")
    load_model = None


# --- Global Configuration ---
TARGET_COMPLETED_TASKS = 500
SIM_TIME = 500  # Simulation duration in time units
TASK_INTERVAL = 1.0  # Average time between task arrivals (exponential distribution)
WINDOW_SIZE = 20
# --- NEW: Configuration to select load pattern ---
# Set to True to use a cyclical load similar to the training data. This is crucial for model accuracy.
# Set to False to use a more unpredictable, purely random load pattern.
USE_CYCLICAL_LOAD = True

# Configuration Profiles
CONFIG = "SCALABLE_BALANCED"  # Options: "BALANCED", "HIGH_CAPACITY", "LIGHT_TASKS", "SCALABLE_BALANCED"

if CONFIG == "BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 6
    MAX_QUEUE_LEN = 8
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = (
        10  # Tasks will have priorities from 1 to this value (lower is higher priority)
    )
    # Dynamic Scaling (disabled for BALANCED profile)
    MIN_NUM_SERVERS = NUM_SERVERS
    MAX_NUM_SERVERS = NUM_SERVERS
    SCALE_UP_PROBABILITY_THRESHOLD = 0.7
    SCALE_DOWN_THRESHOLD = 0.3
    SCALING_CHECK_INTERVAL = 10

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
    SCALE_UP_PROBABILITY_THRESHOLD = 0.7
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
    SCALE_UP_PROBABILITY_THRESHOLD = 0.7
    SCALE_DOWN_THRESHOLD = 0.3
    SCALING_CHECK_INTERVAL = 10

elif CONFIG == "SCALABLE_BALANCED":
    RANDOM_SEED = 42
    CPU_CAPACITY = 100
    NET_CAPACITY = 100
    NUM_SERVERS = 3  # Starting number of servers
    MAX_QUEUE_LEN = 5
    MAX_CENTRAL_BUFFER_LEN = 50
    USE_LIGHTER_TASKS = False
    TASK_PRIORITY_RANGE = 10
    # Dynamic Scaling (enabled for SCALABLE_BALANCED profile)
    MIN_NUM_SERVERS = 2
    MAX_NUM_SERVERS = 10
    # If predicted bottleneck score > 60%, scale up
    SCALE_UP_PROBABILITY_THRESHOLD = 0.65
    # If average system CPU util < 40%, scale down
    SCALE_DOWN_THRESHOLD = 0.4
    SCALING_CHECK_INTERVAL = 15


# --- Global Simulation Metrics (will be reset for each simulation run) ---
SLA_VIOLATIONS = 0
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
TASK_ID = 0  # Global task counter
TASK_TIMES = {}  # Stores arrival and completion times for each task
ACTIVE_SERVERS_HISTORY = []  # To log the number of active servers over time

# Set initial random seed for reproducibility
random.seed(RANDOM_SEED)


# --- TOC STEP 1: IDENTIFY (AI INTEGRATION: AI-Powered Constraint Detector) ---
class AiConstraintDetector:
    """Uses a predictive model to identify the most likely future constraint."""

    def __init__(self, env, all_servers, model, scaler):
        self.env = env
        self.all_servers = all_servers
        self.model = model
        self.scaler = scaler
        self.current_constraint = {"server": None, "score": 0.0}
        self.history_buffers = {s.name: deque(maxlen=WINDOW_SIZE) for s in all_servers}
        self.env.process(self.run())

    @property
    def constraint_name(self):
        """Returns the name of the current constraint server for logging."""
        if self.current_constraint["server"]:
            return f"{self.current_constraint['server'].name} (Predicted)"
        return "None"

    def update_history(self, server):
        """Called by the logger to add the latest data."""
        self.history_buffers[server.name].append(
            [server.cpu_used, server.q_len, server.net_in, server.net_out]
        )

    def run(self):
        # Wait for the initial window to be filled before starting predictions
        yield self.env.timeout(WINDOW_SIZE)

        while True:
            # Prediction interval
            yield self.env.timeout(5)

            # Do not predict if the model is not loaded
            if not self.model:
                continue

            input_data, candidate_servers = [], []
            active_servers = [s for s in self.all_servers if s.is_active]

            for server in active_servers:
                if len(self.history_buffers[server.name]) == WINDOW_SIZE:
                    raw_data = np.array(self.history_buffers[server.name]).astype(
                        np.float32
                    )
                    # Use the scaler to transform the window of data
                    normalized = self.scaler.transform(raw_data)
                    input_data.append(normalized)
                    candidate_servers.append(server)

            if not candidate_servers:
                continue

            # Make batch prediction for all candidate servers
            input_tensor = np.array(input_data)
            pred_scores = self.model.predict(input_tensor, verbose=0).flatten()

            # Identify the server with the highest bottleneck probability
            max_score_index = np.argmax(pred_scores)
            self.current_constraint["server"] = candidate_servers[max_score_index]
            self.current_constraint["score"] = float(pred_scores[max_score_index])


# --- TOC STEPS 2 & 3: EXPLOIT & SUBORDINATE ---
class Dispatcher:
    """
    Intelligently dispatches tasks based on the predicted constraint.
    This class now correctly implements the TOC principle of "subordination"
    by avoiding the predicted bottleneck.
    """

    def __init__(self, env, all_servers, constraint_detector):
        self.env = env
        self.all_servers = all_servers
        self.constraint_detector = constraint_detector
        self.task_buffer = []
        self.env.process(self.run())

    def add_task(self, task):
        """Adds a task to the central buffer and sorts by priority."""
        self.task_buffer.append(task)
        # Sort by priority (lower is better)
        self.task_buffer.sort(key=lambda x: x[5])

    def run(self):
        while True:
            if not self.task_buffer:
                yield self.env.timeout(0.5)
                continue

            # Get the live, updated constraint from the detector
            constraint_server = self.constraint_detector.current_constraint["server"]
            task_to_dispatch = self.task_buffer[0]

            # --- CRITICAL FIX: Inverted scheduling logic ---
            # The goal is to AVOID the constraint, not send work to it.

            # 1. Identify all eligible servers that are NOT the predicted constraint.
            eligible_servers = [
                s
                for s in self.all_servers
                if s.is_active and s != constraint_server and s.q_len < MAX_QUEUE_LEN
            ]

            best_server = None
            if eligible_servers:
                # 2. From the pool of healthy servers, choose the one with the shortest queue.
                best_server = min(eligible_servers, key=lambda s: s.q_len)
            elif (
                constraint_server
                and constraint_server.is_active
                and constraint_server.q_len < MAX_QUEUE_LEN
            ):
                # 3. ONLY if no other server is available, use the constraint as a last resort.
                # This ensures the system doesn't grind to a halt.
                best_server = constraint_server

            if best_server:
                # 4. Dispatch the highest-priority task to the selected server.
                self.task_buffer.pop(0)
                best_server.add_to_buffer(task_to_dispatch)
            else:
                # If no server has capacity, wait before re-evaluating.
                yield self.env.timeout(0.5)
                continue

            # Short delay between dispatches to prevent frantic loops
            yield self.env.timeout(0.2)


# --- Server Class ---
class Server:
    """
    Represents a single server that processes tasks from its local queue.
    Now corrected to prevent head-of-line blocking.
    """

    def __init__(self, env, name):
        self.env, self.name = env, name
        self.cpu_capacity, self.net_capacity = CPU_CAPACITY, NET_CAPACITY
        self.cpu_used, self.net_in, self.net_out, self.current_tasks = 0, 0, 0, 0
        self.is_active = True
        self.queue_items = []
        self.env.process(self.run())

    def add_to_buffer(self, task):
        self.queue_items.append(task)

    @property
    def q_len(self):
        return len(self.queue_items)

    def run(self):
        while True:
            if not self.is_active or not self.queue_items:
                yield self.env.timeout(0.5)
                continue

            # --- FIX: Prevent Head-of-Line Blocking ---
            # Sort by priority to ensure we check the most important tasks first.
            self.queue_items.sort(key=lambda x: x[5])

            task_to_process = None
            task_idx_to_pop = -1

            # Iterate through the entire sorted queue to find the first task that fits.
            for i, task in enumerate(self.queue_items):
                cpu, net_in, net_out, _, _, _ = task
                if (
                    cpu + self.cpu_used <= self.cpu_capacity
                    and net_in + self.net_in <= self.net_capacity
                    and net_out + self.net_out <= self.net_capacity
                ):
                    # Found the highest-priority task that can be processed *now*.
                    task_to_process = task
                    task_idx_to_pop = i
                    break  # Break because we found our candidate

            if task_to_process:
                self.queue_items.pop(task_idx_to_pop)
                cpu, net_in, net_out, duration, task_id, _ = task_to_process

                # Allocate resources and start processing
                self.cpu_used += cpu
                self.net_in += net_in
                self.net_out += net_out
                self.current_tasks += 1
                self.env.process(
                    self.process_task(task_id, duration, cpu, net_in, net_out)
                )
            else:
                # If NO task in the queue could run, the server must wait.
                yield self.env.timeout(0.5)

    def process_task(self, task_id, duration, cpu, net_in, net_out):
        """Simulates the actual work being done on a task."""
        global COMPLETED_TASKS, TASK_TIMES
        yield self.env.timeout(duration)

        # Release resources
        self.cpu_used -= cpu
        self.net_in -= net_in
        self.net_out -= net_out
        self.current_tasks -= 1

        # Log completion
        COMPLETED_TASKS += 1
        TASK_TIMES[task_id]["completion"] = self.env.now


# --- Task Generation ---
def process_incoming_task(env, dispatcher):
    """Creates a single task and adds it to the dispatcher's central buffer."""
    global TOTAL_TASKS, TASK_ID, TASK_TIMES, SLA_VIOLATIONS

    # Reject task if the central buffer is full (SLA violation)
    if len(dispatcher.task_buffer) >= MAX_CENTRAL_BUFFER_LEN:
        SLA_VIOLATIONS += 1
        TOTAL_TASKS += 1
        return

    # Task resource requirements can be cyclical to match training data
    if USE_CYCLICAL_LOAD:
        # This logic mimics the sine-wave pattern from the training data script.
        time_in_cycle = env.now % 100  # 100-second cycle
        cpu_base = 70 + 40 * np.sin(2 * np.pi * time_in_cycle / 100)
        net_base = 50 + 25 * np.sin(2 * np.pi * time_in_cycle / 100)
        cpu = int(np.clip(cpu_base + random.gauss(0, 10), 10, CPU_CAPACITY))
        net_in = int(np.clip(net_base + random.gauss(0, 5), 5, NET_CAPACITY))
        net_out = int(np.clip(net_base + random.gauss(0, 5), 5, NET_CAPACITY))
        dur = random.randint(5, 15)
    elif USE_LIGHTER_TASKS:
        cpu, net_in, net_out, dur = (
            random.randint(8, 25),
            random.randint(1, 8),
            random.randint(1, 8),
            random.randint(2, 8),
        )
    else:  # Original random load
        cpu, net_in, net_out, dur = (
            random.randint(40, 90),
            random.randint(5, 15),
            random.randint(5, 15),
            random.randint(5, 15),
        )

    TASK_ID += 1
    task_id = TASK_ID
    TASK_TIMES[task_id] = {
        "arrival": env.now,
        "priority": random.randint(1, TASK_PRIORITY_RANGE),
    }
    task = (cpu, net_in, net_out, dur, task_id, TASK_TIMES[task_id]["priority"])
    dispatcher.add_task(task)
    TOTAL_TASKS += 1


def generate_task(env, dispatcher):
    """A SimPy process that generates tasks at a given interval."""
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL))
        process_incoming_task(env, dispatcher)


# --- TOC STEP 4: ELEVATE ---
def monitor_system(env, servers, constraint_detector):
    """
    Autoscaler with hybrid logic: uses AI predictions for proactive scaling
    and current utilization as a reactive safety net.
    """
    global NUM_SERVERS
    while True:
        yield env.timeout(SCALING_CHECK_INTERVAL)

        constraint_server = constraint_detector.current_constraint.get("server")
        prediction_score = constraint_detector.current_constraint.get("score", 0.0)

        # --- REVISED HYBRID SCALING LOGIC ---
        # Condition 1: Proactive scale-up based on high AI prediction score.
        proactive_scale_up = prediction_score > SCALE_UP_PROBABILITY_THRESHOLD

        # Condition 2: Reactive scale-up if the predicted constraint server is already overloaded.
        # This acts as a safety net if the model is not confident but the system is struggling.
        reactive_scale_up = False
        if constraint_server and constraint_server.is_active:
            # Use 85% as a critical "danger" threshold for immediate reaction.

            current_constraint_util = max(
                constraint_server.cpu_used / constraint_server.cpu_capacity,
                (constraint_server.net_in + constraint_server.net_out) / (2 * constraint_server.net_capacity)
            )
            if current_constraint_util > 0.85:
                reactive_scale_up = True

        # ELEVATE if either the proactive or reactive trigger is met.
        if (proactive_scale_up or reactive_scale_up) and NUM_SERVERS < MAX_NUM_SERVERS:
            inactive = [s for s in servers if not s.is_active]
            if inactive:
                inactive[0].is_active = True
                NUM_SERVERS += 1
                # Log the reason for scaling to see which trigger is more frequent.
                reason = (
                    f"AI score {(prediction_score*100):.1f}%"
                    if proactive_scale_up
                    else f"High load on {constraint_server.name}"
                )
                print(
                    f"[{env.now:.2f}] ELEVATE: {reason}. Scaling up. Active: {NUM_SERVERS}"
                )
        # Scale-down logic remains the same, based on low overall utilization.
        elif env.now > WINDOW_SIZE:  # Don't scale down too early
            active = [s for s in servers if s.is_active]
            if not active:
                continue # No active servers to scale down

            # --- FIX: Correct calculation for average network utilization ---
            total_cpu_used = sum(s.cpu_used for s in active)
            total_net_used = sum(s.net_in + s.net_out for s in active)

            total_cpu_capacity = len(active) * CPU_CAPACITY
            total_net_capacity = len(active) * NET_CAPACITY * 2 # In + Out

            avg_cpu_util = total_cpu_used / total_cpu_capacity
            avg_net_util = total_net_used / total_net_capacity
            
            # Using the max of the two is a good heuristic to decide if the system is idle
            avg_util = max(avg_cpu_util, avg_net_util)

            if avg_util < SCALE_DOWN_THRESHOLD and NUM_SERVERS > MIN_NUM_SERVERS:
                # Find an idle server to shut down
                eligible = [s for s in active if s.current_tasks == 0 and s.q_len == 0]
                if eligible:
                    eligible[-1].is_active = False  # Deactivate the last eligible server
                    NUM_SERVERS -= 1
                    print(
                        f"[{env.now:.2f}] DE-ELEVATE: Low util ({(avg_util*100):.1f}%). Scaling down. Active: {NUM_SERVERS}"
                    )


# --- Logger ---
def periodic_logger(env, servers, log_data, constraint_detector):
    """Logs system state at regular intervals for analysis and plotting."""
    while True:
        yield env.timeout(1)
        ACTIVE_SERVERS_HISTORY.append(
            {
                "timestamp": env.now,
                "active_servers_count": sum(1 for s in servers if s.is_active),
                "constraint": constraint_detector.constraint_name,
                "constraint_score": constraint_detector.current_constraint["score"],
            }
        )
        for server in servers:
            constraint_detector.update_history(server)
            log_data.append(
                {
                    "timestamp": env.now,
                    "server_id": server.name,
                    "is_active": server.is_active,
                    "cpu_used": server.cpu_used,
                    "q_len": server.q_len,
                    "network_in": server.net_in,
                    "network_out": server.net_out,
                }
            )


# --- Main Simulation Function ---
def simulation():
    """Sets up and runs the entire simulation, then analyzes and plots the results."""
    global SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID, TASK_TIMES, ACTIVE_SERVERS_HISTORY, NUM_SERVERS

    try:
        # Gracefully handle missing model files
        if load_model is None:
            raise FileNotFoundError("Keras/TensorFlow is not installed.")
        model = load_model("bottleneck_predictor.keras")
        scaler = joblib.load("minmax_scaler.save")
    except Exception as e:
        print(f"ERROR: Could not load AI model or scaler. {e}")
        print(
            "Please ensure 'bottleneck_predictor.keras' and 'minmax_scaler.save' are in the same directory."
        )
        print("Simulation will run without AI-based scheduling and scaling.")
        model, scaler = None, None

    # Reset metrics for the new run
    SLA_VIOLATIONS, TOTAL_TASKS, COMPLETED_TASKS, TASK_ID = 0, 0, 0, 0
    TASK_TIMES, ACTIVE_SERVERS_HISTORY = {}, []

    initial_num_servers = {
        "BALANCED": 6,
        "HIGH_CAPACITY": 8,
        "LIGHT_TASKS": 4,
        "SCALABLE_BALANCED": 3,
    }.get(CONFIG, 3)
    NUM_SERVERS = initial_num_servers

    print("Starting simulation...")
    print(f"Configuration: {CONFIG}")
    print(f"Using Cyclical Load (matching training data): {USE_CYCLICAL_LOAD}")

    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    # Create all possible servers, but only activate the initial set
    all_servers = [Server(env, f"Server-{i+1}") for i in range(MAX_NUM_SERVERS)]
    for i in range(NUM_SERVERS):
        all_servers[i].is_active = True
    for i in range(NUM_SERVERS, MAX_NUM_SERVERS):
        all_servers[i].is_active = False

    log_data = []
    constraint_detector = AiConstraintDetector(env, all_servers, model, scaler)
    dispatcher = Dispatcher(env, all_servers, constraint_detector)

    # Start SimPy processes
    env.process(generate_task(env, dispatcher))
    env.process(periodic_logger(env, all_servers, log_data, constraint_detector))
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS and model:  # Scaling requires the AI model
        env.process(monitor_system(env, all_servers, constraint_detector))

    env.run(until=SIM_TIME)

    # --- Post-simulation Analysis ---
    df_server_logs = pd.DataFrame(log_data)
    df_active_servers = pd.DataFrame(ACTIVE_SERVERS_HISTORY)

    sla_rate = (SLA_VIOLATIONS / TOTAL_TASKS) * 100 if TOTAL_TASKS else 0
    completion_rate_of_target = (COMPLETED_TASKS / TARGET_COMPLETED_TASKS) * 100

    if not df_server_logs.empty:
        active_logs = df_server_logs[df_server_logs["is_active"] == True]
        avg_q_len = active_logs["q_len"].mean() if not active_logs.empty else 0
        avg_cpu_util = (
            (active_logs["cpu_used"].sum() / (len(active_logs) * CPU_CAPACITY)) * 100
            if len(active_logs) > 0
            else 0
        )
        avg_net_util = ((active_logs['network_in'] + active_logs['network_out']) / (2 * NET_CAPACITY)).mean() * 100 if not active_logs.empty else 0
    else:
        avg_q_len, avg_cpu_util = 0, 0

    turnaround_times = [
        t["completion"] - t["arrival"] for t in TASK_TIMES.values() if "completion" in t
    ]
    avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0

    print(f"\n=== SIMULATION RESULTS (AI + TOC) ===")
    print(f"Configuration Used: {CONFIG}")
    print(f"Target Completed Tasks: {TARGET_COMPLETED_TASKS}")
    print(
        f"Actual Completed Tasks: {COMPLETED_TASKS} ({completion_rate_of_target:.1f}% of target)"
    )
    print(f"Total Tasks Generated: {TOTAL_TASKS}")
    print(f"System Task Completion Rate: {(COMPLETED_TASKS/TOTAL_TASKS)*100:.1f}% (Completed vs. Generated)")
    print(f"SLA Violation Rate: {sla_rate:.2f}%")
    print(f"Average Queue Length (Active Servers): {avg_q_len:.2f}")
    print(f"Average CPU Utilization (Active Servers): {avg_cpu_util:.2f}%")
    print(f"Average Network Utilization (Active Servers): {avg_net_util:.2f}%")
    print(f"Average Turnaround Time: {avg_turnaround:.2f}")

    if completion_rate_of_target >= 95:
        print("✅ EXCELLENT: Target achieved!")
    elif completion_rate_of_target >= 80:
        print("✅ GOOD: Close to target")
    elif completion_rate_of_target >= 60:
        print("⚠️  MODERATE: Needs tuning")
    else:
        print("❌ POOR: Significant underperformance")

    # --- Plotting Results ---
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(18, 12))
    plt.suptitle(
        f"AI+TOC Simulation Results for {CONFIG} (Cyclical Load: {USE_CYCLICAL_LOAD})",
        fontsize=16,
    )

    # Plot 1: Active Servers and Predicted Bottleneck Score
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(
        df_active_servers["timestamp"],
        df_active_servers["active_servers_count"],
        marker=".",
        linestyle="-",
        color="purple",
        label="Active Servers",
    )
    ax1.set_ylabel("Active Servers", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")
    if MAX_NUM_SERVERS > MIN_NUM_SERVERS:
        ax1.set_yticks(range(MIN_NUM_SERVERS, MAX_NUM_SERVERS + 2))
    ax2 = ax1.twinx()
    ax2.plot(
        df_active_servers["timestamp"],
        df_active_servers["constraint_score"],
        color="red",
        linestyle="--",
        label="Constraint Score",
    )
    ax2.set_ylabel("Predicted Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1.1)
    ax1.set_title("Scaling vs. Predicted Score")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Plot 2: DBR Buffer Sizes (Queue Length)
    plt.subplot(3, 2, 2)
    for server_id in df_server_logs["server_id"].unique():
        server_data = df_server_logs[df_server_logs["server_id"] == server_id]
        if server_data["is_active"].any():
            plt.plot(
                server_data["timestamp"],
                server_data["q_len"],
                label=f"{server_id}",
                alpha=0.7,
            )
    plt.title("DBR Buffer Sizes (Queue Length)")
    plt.ylabel("Buffer Length")
    plt.legend(loc="upper right", fontsize="small")

    # Plot 3: CPU Usage per Server
    plt.subplot(3, 2, 3)
    for server_id in df_server_logs["server_id"].unique():
        server_data = df_server_logs[df_server_logs["server_id"] == server_id]
        if server_data["is_active"].any():
            plt.plot(
                server_data["timestamp"],
                server_data["cpu_used"],
                label=f"{server_id}",
                alpha=0.7,
            )
    plt.title("CPU Usage per Server")
    plt.ylabel("CPU Units")
    plt.legend(loc="upper right", fontsize="small")

    # Plot 4: Identified Constraint
    plt.subplot(3, 2, 4)
    if not df_active_servers.dropna(subset=["constraint"]).empty:
        unique_constraints = df_active_servers.dropna(subset=["constraint"])[
            "constraint"
        ].unique()
        for constraint in unique_constraints:
            subset = df_active_servers[df_active_servers["constraint"] == constraint]
            plt.scatter(
                subset["timestamp"], subset["constraint"], label=constraint, s=10
            )
    plt.title("Predicted System Constraint")
    plt.ylabel("Constraint")
    plt.xticks(rotation=15)
    plt.legend(loc="upper right", fontsize="small")

    # Plot 5: Turnaround Time Distribution
    plt.subplot(3, 2, 5)
    if turnaround_times:
        plt.hist(
            turnaround_times, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        plt.title("Task Turnaround Time Distribution")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

    # Plot 6: Overall System Utilization
    plt.subplot(3, 2, 6)
    if not df_server_logs.empty and not df_active_servers.empty:
        df_summed = (
            df_server_logs[df_server_logs["is_active"]]
            .groupby("timestamp")
            .agg(total_cpu=("cpu_used", "sum"))
            .reset_index()
        )
        df_merged = pd.merge(
            df_summed,
            df_active_servers[["timestamp", "active_servers_count"]],
            on="timestamp",
            how="left",
        ).ffill()
        df_merged["total_cpu_capacity"] = (
            df_merged["active_servers_count"] * CPU_CAPACITY
        )
        df_merged["overall_cpu_util"] = (
            df_merged["total_cpu"] / df_merged["total_cpu_capacity"]
        ) * 100
        plt.plot(
            df_merged["timestamp"],
            df_merged["overall_cpu_util"],
            label="Overall CPU Usage",
            color="crimson",
        )
        plt.title("Overall System CPU Utilization")
        plt.xlabel("Time")
        plt.ylabel("Utilization (%)")
        plt.ylim(0, 105)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    simulation()
