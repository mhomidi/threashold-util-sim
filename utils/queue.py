
import threading


class BaseQueue:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.buffer = list()

    def enqueue(self, data: dict) -> None:
        self.lock.acquire()
        self.buffer.append(data)
        self.lock.release()

    def dequeue(self) -> dict:
        data = None
        self.lock.acquire()
        if len(self.buffer) != 0:
            data = self.buffer.pop(0)
        self.lock.release()
        return data
    
    def put(self):
        raise NotImplementedError()
    
    def get(self):
        raise NotImplementedError()
    

class AgentQueue(BaseQueue):

    def __init__(self, id: int) -> None:
        super().__init__()
        self.id = id
    
    def set_id(self, id: int) -> int:
        self.id = id

    def get_id(self) -> int:
        return self.id
    

class AgentToDispatcherQueue(AgentQueue):

    def put(self, budget: int, pref: list) -> None:
        data = dict()
        data["id"] = self.id
        data["budget"] = budget
        data["pref"] = pref
        self.enqueue(data=data)

    def get(self) -> list:
        data = self.dequeue()
        if data is None:
            return None
        return [data["id"], data["budget"], data["pref"]]

    
class DispatcherToAgentQueue(AgentQueue):

    def put(self, budget: int, assignments: list) -> None:
        data = dict()
        data["id"] = self.id
        data["budget"] = budget
        data["assignments"] = assignments
        self.enqueue(data=data)

    def get(self) -> list:
        data = self.dequeue()
        if data is None:
            return None
        return [data["id"], data["budget"], data["assignments"]]


class SchedulerToDispatcherQueue(BaseQueue):

    def put(self, budgets: list, assignments: list):
        data = dict()
        data["budgets"] = budgets
        data["assignments"] = assignments
        self.enqueue(data=data)
    
    def get(self):
        data = self.dequeue()
        if data is None:
            return None
        return [data["budgets"], data["assignments"]]


class DispatcherToSchedulerQueue(BaseQueue):
    
    def put(self, report: list) -> None:
        data = dict()
        data["report"] = report
        self.enqueue(data=data)

    def get(self):
        data = self.dequeue()
        if data is None:
            return None
        return data["report"]
    