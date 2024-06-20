import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=10 * 366):
        """Positional / temporal encoding for time series of maximum 'max_len' days. embedding_dim is the embedding
        dimension of the network input. Later, the positional embedding is added to the embedded input.

        The embedding slightly differs from the original BERT embedding because we make use of the fact that the year
        has 365.25 days. That's the base period, which is divided by powers of two from 2^-3 to 2^6 (roughly 8 years to
        about a week).
        """
        super().__init__()
        year = 365.2422
        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(max_len).unsqueeze(1)         # [max_len, 1]
        # div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).exp()  # [d_model/2,]
        periods = year / (2 ** torch.arange(6, -3, -9/embedding_dim * 2))

        pe[:, 0::2] = torch.sin(2 * math.pi * position / periods)   # broadcasting to [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(2 * math.pi * position / periods)   # broadcasting to [max_len, d_model/2]

        self.register_buffer('pos_embed', pe)

    def forward(self, dayssinceepoch):
        return self.pos_embed[dayssinceepoch, :]
