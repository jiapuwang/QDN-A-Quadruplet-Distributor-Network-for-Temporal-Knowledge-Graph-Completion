from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]

class La3(Regularizer):
    def __init__(self, weight: float):
        super(La3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        dif = torch.mm(factor[1:], factor[:-1].transpose(0,1))
        return - self.weight* dif.trace() / (factor.shape[0] - 1)
