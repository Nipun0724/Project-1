import simpy
import random
import pandas as pd

RANDOM_SEED = 42
CPU_CAPACITY = 100
NET_CAPACITY = 100
TASK_INTERVAL = 1
NUM_SERVERS = 3
SIM_TIME = 500

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
        while True:
            task = yield self.queue.get()
            cpu_demand, net_in_demand, net_out_demand, duration = task

            if cpu_demand + self.cpu_used > CPU_CAPACITY:
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

def process_task( servers, rr_index ):
    cpu_demand = random.randint(40, 90)
    net_in_demand = random.randint(5, 15)
    net_out_demand = random.randint(5, 15)
    duration = random.randint(5, 15)

    server = servers[rr_index[0] % len(servers)]
    rr_index[0] += 1

    server.queue.put((cpu_demand, net_in_demand, net_out_demand, duration))

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
    df.to_csv("resource_usage_log.csv", index=False)
    print(df.head())

if __name__ == '__main__':
    simulation()