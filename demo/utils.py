import random
from collections import deque
import math
import torch


class MovingAverager:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be positive")
        self.data = deque(maxlen=n)
        self.total = 0

    def __call__(self, x):
        if len(self.data) == self.data.maxlen:
            self.total -= self.data[0]
        self.data.append(x)
        self.total += x
        return self.total / len(self.data)


@torch.no_grad()
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return -10 * torch.log10(mse)


class IndexRandomChooser:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.indices = []

    def __call__(self):
        if not self.indices:
            self.indices = list(range(self.n))
            random.shuffle(self.indices)
        return self.indices.pop()