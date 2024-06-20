import torch
import torch.nn as nn
from .embedding import BERTEmbedding
from .transformer import TransformerBlock
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class SBERT(nn.Module):

    def __init__(self, num_features, d_model, n_layers, attn_heads, max_embedding_size=366, dropout=0.1):
        """
        :param num_features: number of input features
        :param d_model: hidden size of the SITS-BERT model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = BERTEmbedding(num_features, d_model, max_pos_embed_val=max_embedding_size)
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=attn_heads,
                                                dim_feedforward=4*d_model,
                                                batch_first=True,
                                                activation="gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x, time, mask=None):
        """
        Args:
            x: (N,L,C) input tensor where L is the sequence length
            time: (N,L) tensor containing the timestamps in days since some t0
            mask: (N, L) tensor containing a validity mask for the input sequence. 0 means valid, 1 means invalid.
        """
        x = self.embedding(input_sequence=x, doy_sequence=time)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x
