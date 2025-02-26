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

    def forward(self, x, time, attention_mask, averaging_mask=None):
        # x has shape (batch_size, sequence length, features)
        # time has shape (batch_size, sequence length)
        # attention_mask has shape (batch_size, sequence length)
        # attention_mask is 0 where there is data and 1 else
        # averaging_mask has shape (batch_size, sequence length)

        x = self.sbert(x, time, attention_mask)
        x = x.permute(0, 2, 1)  # shape: batchsize, hidden dim, sequence length

        if averaging_mask is not None:
            x = x * averaging_mask.unsqueeze(1)
            x = torch.sum(x, dim=-1) / torch.sum(averaging_mask, dim=-1).unsqueeze(-1)
        else:
            x = torch.mean(x, dim=-1)  # average across time dimension, shape: batchsize, hidden dim

        x = self.linear(x)
        return x
