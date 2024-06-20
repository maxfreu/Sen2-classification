import torch.nn as nn

from torch.nn import MultiheadAttention
# from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, hum_heads, feed_forward_hidden, dropout):
        """
        :param embed_dim: hidden size of transformer
        :param hum_heads: head count of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiheadAttention(embed_dim=embed_dim, hum_heads=hum_heads)
        self.feed_forward = PositionwiseFeedForward(d_model=embed_dim, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=embed_dim, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
