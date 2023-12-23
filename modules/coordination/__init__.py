from multiprocessing import Queue
import time
import numpy as np

ITER_PRINT_STEP = 100


class Worker:

    def __init__(self, agents, num_clusters, w2c_queue: Queue, c2w_queue: Queue):
        self.agents = agents
        self.agents_len = len(self.agents)
        self.w2c_queue = w2c_queue
        self.c2w_queue = c2w_queue
        self.num_clusters = num_clusters

    def run(self, path):
        demands = np.zeros((self.agents_len, self.num_clusters))
        for i, agent in enumerate(self.agents):
            demands[i] = agent.init_demand

        more_data = np.ones((self.agents_len, self.num_clusters))
        while True:
            self.w2c_queue.put((demands, more_data))

            info = self.c2w_queue.get()
            if info == 'stop':
                print('Start worker stoping ...')
                for agent in self.agents:
                    agent.stop(path)
                break

            assignments, extra, iteration = info
            for i, agent in enumerate(self.agents):
                agent.set_extra(extra)
                demands[i] = agent.run_agent(iteration, assignments[i])
                more_data[i] = agent.get_more_data()
        print('Worker done')
        return


class Coordinator:

    def __init__(self, scheduler, num_iterations, num_agents, num_clusters, num_workers, w2c_queues, c2w_queues):
        self.scheduler = scheduler
        self.w2c_queues: list[Queue] = w2c_queues
        self.c2w_queues: list[Queue] = c2w_queues
        self.num_workers = num_workers
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.num_clusters = num_clusters

    def run(self):
        agent_ids = np.arange(0, self.num_agents)
        workers_agent_ids = np.array_split(agent_ids, self.num_workers)
        time_duration = time.time()

        for iteration in range(self.num_iterations):
            demands_array = np.zeros((self.num_agents, self.num_clusters))
            more = np.zeros((self.num_agents, self.num_clusters))
            for q, ids in zip(self.w2c_queues, workers_agent_ids):
                demands, more_data = q.get()
                demands_array[ids] = demands
                more[ids] = more_data

            self.scheduler.set_more_data(more)

            assignments, extra = self.scheduler.run_scheduler(
                iteration, demands_array)
            # print(assignments)
            if extra is None:
                extra = np.zeros(self.num_agents)

            workers_assignments = np.array_split(assignments, self.num_workers)

            for q, w_s in zip(self.c2w_queues, workers_assignments):
                q.put((w_s, extra, iteration))

            if (iteration + 1) % ITER_PRINT_STEP == 0:
                print('iteration {iter} done in {t:.2f}s'.format(
                    iter=iteration + 1, t=time.time() - time_duration))
                time_duration = time.time()
        print('sending stop ...')
        for q in self.c2w_queues:
            q.put('stop')
        print('Coordinator done')
        return
