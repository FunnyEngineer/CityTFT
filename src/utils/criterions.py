import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('q', torch.tensor(quantiles))

    def forward(self, predictions, targets):
        diff = predictions - targets
        ql = (1-self.q)*F.relu(diff) + self.q*F.relu(-diff)
        losses = ql.view(-1, ql.shape[-1]).mean(0)
        return losses

def qrisk(pred, tgt, quantiles):
    diff = pred - tgt
    ql = (1-quantiles)*np.clip(diff,0, float('inf')) + quantiles*np.clip(-diff,0, float('inf'))
    losses = ql.reshape(-1, ql.shape[-1])
    normalizer = np.abs(tgt).mean()
    risk = 2 * losses / normalizer
    return risk.mean(0)