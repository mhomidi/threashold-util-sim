
import threading


class Pipe:

    def __init__(self, id: int) -> None:
        self.lock = threading.Lock()
        self.buffer = list()
        self.id = id

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

    def is_empty(self) -> bool:
        self.lock.acquire()
        l = len(self.buffer)
        self.lock.release()
        return l == 0

    def set_id(self, id: int) -> int:
        self.id = id

    def get_id(self) -> int:
        return self.id

    def put(self, data):
        data["id"] = self.id
        self.enqueue(data=data)

    def get(self):
        return self.dequeue()
