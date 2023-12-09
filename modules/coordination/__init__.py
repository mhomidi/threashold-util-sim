from modules.agents import Agent
from utils.pipe import *
import config
from modules.scheduler import Scheduler
import time
import numpy as np


class Worker:

    def __init__(self, id: int) -> None:
        self.id = id
        self.agents = list()
        self.w2c_pipe = Pipe(self.id)
        self.c2w_pipe = Pipe(self.id)

    def add_agent(self, agent):
        self.agents.append(agent)

    def run(self):
        data['demands'] = np.zeros(
            (len(self.agents), config.get(config.CLUSTER_NUM)))
        for iter in range(config.get(config.TOTAL_ITERATION)):
            self.w2c_pipe.put(data)
            while self.c2w_pipe.is_empty():
                # TODO: wait lock
                time.sleep(0.0001)
            data = self.c2w_pipe.get()
            assignments = data['assignments']
            demands = [None for _ in range(len(self.agents))]
            for index, agent in enumerate(self.agents):
                agent: Agent
                agent_assignment = assignments[index]
                demands[index] = agent.run_agent(iter, agent_assignment)
            data['demands'] = np.array(demands)


class Coordinator:

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.update_demands = list()
        self.w2c_queues = list()
        self.c2w_queues = list()

    def add_pipes(self, w2c_queue: Pipe, c2w_queue: Pipe):
        self.update_demands.append(False)
        self.w2c_queues.append(w2c_queue)
        self.c2w_queues.append(c2w_queue)

    def is_demands_new(self):
        ud = np.array(self.update_demands, dtype=np.int8)
        return ud.sum() == len(self.update_demands)

    def recieve_demands(self):
        demands = [None for i in range(len(self.w2c_queues))]
        while not self.is_demands_new():
            # TODO: wait lock
            time.sleep(0.0001)
            for id, pipe in enumerate(self.w2c_queues):
                pipe: Pipe
                if not self.demands_update[id]:
                    if pipe.is_empty():
                        continue
                    data = pipe.get()
                    if data:
                        demands[id] = data['demands']
                        self.demands_update[id] = True

        self.demands_update = [False for _ in range(len(self.w2c_queues))]
        return np.array(demands).reshape((config.get(config.AGENT_NUM), config.get(config.CLUSTER_NUM)))

    def send_assignments(self, assignments):
        start_point = 0
        w_num_agent = config.get(config.WORKER_AGENT_NUM)
        for queue in self.c2w_queues:
            queue: Pipe
            queue.put(
                data={'assignments': assignments[start_point: start_point + w_num_agent]})
            start_point += w_num_agent

    def run(self):
        for iteration in range(config.get(config.TOTAL_ITERATION)):
            demands = self.recieve_demands()
            assignments = self.scheduler.run_scheduler(iteration, demands)
            self.send_assignments(assignments)
            if iteration % 500 == 499:
                print("episode {e} done".format(e=iteration + 1))
