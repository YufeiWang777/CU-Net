
from abc import ABC

import torch.nn as nn

loss_names = ['l1', 'l2', 'ls']

class L2Loss(nn.Module, ABC):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = None

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]

        # tmp = (diff**2).sum()
        self.loss = (diff**2).mean()

        return self.loss

