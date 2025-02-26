import torch
import torch.nn as nn
from .position import PositionalEncoding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a fully connected layer
        2. PositionalEncoding : adding positional information using sin, cos

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, embedding_dim, max_pos_embed_val=10*366, dropout=0.1):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_embed = nn.Linear(in_features=num_features, out_features=embedding_dim)
        self.pos_embed = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_pos_embed_val)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_sequence, doy_sequence):
        embedded_input = self.input_embed(input_sequence)  # [batch_size, seq_length, embedding_dim]
        embedded_pos = self.pos_embed(doy_sequence)
        embedding = embedded_input + embedded_pos
        return embedding


class ConcatEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a fully connected layer
        2. PositionalEncoding : adding positional information using sin, cos

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, embedding_dim, max_pos_embed_val=10*366, dropout=0.1):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_embed = nn.Linear(in_features=num_features, out_features=embedding_dim)
        self.pos_embed = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_pos_embed_val)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_sequence, doy_sequence):
        embedded_input = self.input_embed(input_sequence)  # [batch_size, seq_length, embedding_dim]
        embedded_pos = self.pos_embed(doy_sequence)
        embedding = torch.concat((embedded_input, embedded_pos), dim=-1)
        return embedding