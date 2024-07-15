import torch
import torch.nn as nn
from .bert import SBERT


class SBERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """
    def __init__(self, sbert: SBERT, num_classes):
        super().__init__()
        self.sbert = sbert
        self.linear = nn.Linear(self.sbert.d_model, num_classes)

    def forward(self, x, time, mask):
        x = self.sbert(x, time, mask)
        x = torch.mean(x.permute(0, 2, 1), dim=-1)  # average across time dimension
        x = self.linear(x)
        return x
