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


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)
       
class La3(Regularizer):
    def __init__(self, weight: float):
        super(La3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        dif = torch.mm(factor[1:], factor[:-1].transpose(0,1))
        return - self.weight* dif.trace() / (factor.shape[0] - 1)

class Linear3(Regularizer):
    def __init__(self, weight: float):
        super(Linear3, self).__init__()
        self.weight = weight

    def forward(self, factor, W):
        rank = int(factor.shape[1] / 2)
        #print(rank.size())
        ddiff = factor[1:] - factor[:-1] - W.weight[:rank*2].t()
        print(ddiff.size())
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        print(diff.size())
        y = ddiff[:, :rank]**2
        print(y.size())
        x = self.weight * torch.sum(diff) / (factor.shape[0] - 1)
        print(x.size())
        return x
        
'''        
class La3(Regularizer):
    def __init__(self, weight: float):
        super(La3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        #print(factor[1:].size())
        #print(factor[:-1].size())
        ddiff = torch.mm(factor[1:], factor[:-1].transpose(0,1))
        return -self.weight * ddiff.trace() / (factor.shape[0] - 1)
''' 
