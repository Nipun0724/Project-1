import simpy
import random
import pandas as pd

RANDOM_SEED = 42
CPU_CAPACITY = 100
NET_CAPACITY = 100
TASK_INTERVAL = 1
NUM_SERVERS = 3
MAX_QUEUE_LEN = 5
SIM_TIME = 500
SLA_VIOLATIONS = 0
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
TASK_ID = 0
TASK_TIMES = {}

class Server:
    def __init__( self, env, name ):
        self.env = env
        self.name = name
        self.cpu_capacity = CPU_CAPACITY
        self.net_capacity = NET_CAPACITY
        self.cpu_used = 0
        self.net_in = 0
        self.net_out = 0
        self.q_len = 0

        self.queue = simpy.Store(env)

        self.env.process(self.run())

    def run( self ):
        global COMPLETED_TASKS, TASK_TIMES

        while True:
            task = yield self.queue.get()
            cpu_demand, net_in_demand, net_out_demand, duration, task_id = task

            if ( 
                cpu_demand + self.cpu_used > self.cpu_capacity or
                net_in_demand + self.net_in > self.net_capacity or
                net_out_demand + self.net_out > self.net_capacity
            ):
                yield self.queue.put(task)
                yield self.env.timeout(1)
                continue
            
            self.cpu_used += cpu_demand
            self.net_in += net_in_demand
            self.net_out += net_out_demand
            self.q_len += 1

            yield self.env.timeout(duration)

            self.cpu_used -= cpu_demand
            self.net_in -= net_in_demand
            self.net_out -= net_out_demand
            self.q_len -= 1

            COMPLETED_TASKS += 1
            TASK_TIMES[task_id]['completion'] = self.env.now

def process_task(servers, rr_index):
    global TOTAL_TASKS, SLA_VIOLATIONS, TASK_ID, TASK_TIMES

    cpu_demand = random.randint(40, 90)
    net_in_demand = random.randint(5, 15)
    net_out_demand = random.randint(5, 15)
    duration = random.randint(5, 15)

    TASK_ID += 1
    task_id = TASK_ID
    arrival_time = servers[0].env.now
    TASK_TIMES[task_id] = {'arrival': arrival_time}

    server = servers[rr_index[0] % len(servers)]
    rr_index[0] += 1

    if len(server.queue.items) >= MAX_QUEUE_LEN:
        SLA_VIOLATIONS += 1
        print(f"Task {task_id} dropped at time {arrival_time} — queue full at {server.name}")
    else:
        server.queue.put((cpu_demand, net_in_demand, net_out_demand, duration, task_id))

    TOTAL_TASKS += 1

def generate_task( env, servers ):
    rr_index = [0]
    while True:
        yield env.timeout(random.expovariate(1.0 / TASK_INTERVAL))
        process_task(servers, rr_index)

def periodic_logger( env, servers, log_data, interval = 1 ):
    while True:
        yield env.timeout(interval)
        for server in servers:
            log_data.append({
                'timestamp'  : env.now,
                'server_id'  : server.name,
                'cpu_used'   : server.cpu_used,
                'q_len' : server.q_len,
                'network_in' : server.net_in,
                'network_out': server.net_out
            })

def simulation():
    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    servers = [Server(env, f"Server {i}") for i in range(NUM_SERVERS)]

    log_data = []

    env.process(generate_task(env, servers))
    env.process(periodic_logger(env, servers, log_data, 1))

    env.run(until=SIM_TIME)

    df = pd.DataFrame(log_data)
    df.to_csv("resource_usage_log_no_toc.csv", index=False)
    print(df.head())

    sla_violation_rate = (SLA_VIOLATIONS / TOTAL_TASKS) * 100 if TOTAL_TASKS else 0

    avg_q_len = df['q_len'].mean()

    for sid in df['server_id'].unique():
        cpu_util = df[df['server_id'] == sid]['cpu_used'].mean() / CPU_CAPACITY * 100
        net_in_util = df[df['server_id'] == sid]['network_in'].mean() / NET_CAPACITY * 100
        net_out_util = df[df['server_id'] == sid]['network_out'].mean() / NET_CAPACITY * 100
        print(f"{sid} — Avg CPU: {cpu_util:.2f}%, Net In: {net_in_util:.2f}%, Net Out: {net_out_util:.2f}%")

    print(f"Throughput: {COMPLETED_TASKS} tasks completed.")

    turnaround_times = []
    for t in TASK_TIMES.values():
        if 'completion' in t:
            turnaround_times.append(t['completion'] - t['arrival'])

    avg_turnaround = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0

    print("\n===== Final Metrics =====")
    print(f"SLA Violation Rate: {sla_violation_rate:.2f}%")
    print(f"Average Queue Length: {avg_q_len:.2f}")
    print(f"Throughput: {COMPLETED_TASKS} / {TOTAL_TASKS}")
    print(f"Average Turnaround Time: {avg_turnaround:.2f}")

if __name__ == '__main__':
    simulation()