from time import time
from datetime import timedelta


class Timer:
    def __init__(self, msg: str):
        self.msg = msg
        self.start_time = None

    def start(self):
        self.start_time = time()

    def stop(self):
        if self.start_time is None:
            raise Exception('Must set start time.')
        end = time()
        delta = timedelta(seconds=end - self.start_time)
        print(self.msg, str(delta))
