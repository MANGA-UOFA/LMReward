import time
import math


class AvgStat:
    def __init__(self):
        self.c = 0
        self.v = 0
    
    def take(self, v):
        if math.isnan(v):
            return
        self.v += v
        self.c += 1
    
    def pop(self):
        if self.c == 0:
            return 0
        v = self.v / self.c
        self.v = 0
        self.c = 0
        return v


class Logger:
    def log(self, *args, **kwargs):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), end=' | ')
        print(*args, **kwargs, flush=True)