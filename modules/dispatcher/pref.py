
from modules.dispatcher import Dispatcher
from utils.pipe import Pipe
import time


class PrefDispatcher(Dispatcher):

    def set_budgets(self, new_budgets: list) -> None:
        self.budgets = new_budgets

    def set_dist_token(self, dist: list) -> None:
        self.token_dist = dist

    def set_bid(self, agent_id: int, budget: int, bid: list) -> None:
        self.update_report(agent_id, data=[budget, bid])

    def send_data(self) -> None:
        for id, queue in enumerate(self.out_pipes):
            queue: Pipe
            data = dict()
            data['budget'] = self.budgets[id]
            data["assignments"] = self.cluster_assignments
            data["token_dist"] = self.token_dist
            queue.put(data=data)

    def recieve_data(self) -> None:
        while not self.is_report_new():
            time.sleep(0.0001)
            for id, queue in enumerate(self.incoming_pipes):
                queue: Pipe
                if queue.is_empty():
                    continue
                data = queue.get()
                if data:
                    id = data['id']
                    budget = data['budget']
                    pref = data['pref']
                    self.set_bid(id, budget, pref)

        self.report_update = [False for _ in range(len(self.incoming_pipes))]
