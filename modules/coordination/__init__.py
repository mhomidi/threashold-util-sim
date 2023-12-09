import numpy as np


class Worker:

    def __init__(self, agents, w2c_queue, c2w_queue):
        self.id = id
        self.agents = agents
        self.agents_len = len(self.agents)
        self.w2c_queue = w2c_queue
        self.c2w_queue = c2w_queue

    def run(self, path):
        demands = np.zeros(self.agents_len)

        while True:
            self.w2c_queue.put(demands)

            info = self.c2w_queue.get()
            if info == 'stop':
                for agent in self.agents:
                    agent.stop(path)

            assignments, extra, iteration = info
            for i, agent in enumerate(self.agents):
                agent.set_extra(extra)
                demands[i] = agent.run_agent(iteration, assignments[i])


class Coordinator:

    def __init__(self, scheduler, num_iterations, num_agents, num_workers, w2c_queues, c2w_queues):
        self.scheduler = scheduler
        self.w2c_queues = w2c_queues
        self.c2w_queues = c2w_queues
        self.num_workers = num_workers
        self.num_agents = num_agents
        self.num_iterations = num_iterations

    def run(self):
        demands_array = np.zeros(self.num_agents)
        agent_ids = np.arange(0, self.num_agents)
        workers_agent_ids = np.array_split(agent_ids, self.num_workers)

        for iteration in range(self.num_iterations):

            for q, ids in zip(self.w2c_queues, workers_agent_ids):
                demands = q.get()
                demands_array[ids] = demands

            assignments, extra = self.scheduler.run_scheduler(iteration, demands_array)

            if extra is None:
                extra = np.zeros(self.num_agents)

            workers_assignments = np.array_split(assignments, self.num_workers)
            workers_extra = np.array_split(extra, self.num_workers)

            for q, w_s, w_t in zip(self.c2w_queues, workers_assignments, workers_extra):
                q.put((w_s, w_t, iteration))

        for q in self.c2w_queues:
            q.put('stop')
