import torch
from torch.nn import Linear, BatchNorm1d, LayerNorm
from ..utils import sparse2dense_timeseries_batched_torch
from .sitsbert.model.embedding.position import PositionalEncoding
from ..satellite_classifier import SatelliteClassifier


class LinearModel(SatelliteClassifier):
    def __init__(self,
                 max_time,
                 num_classes,
                 lr=3e-3,
                 embed_time=False,
                 embedding_dim=64,
                 use_weighted_loss=False,
                 loss_weights=None,
                 satellite_input_channels=10,
                 classes: "list[str]" = None  # only needed for the test step
                 ):
        super().__init__(num_classes, lr, loss_weights, use_weighted_loss, classes)
        # self.save_hyperparameters({"max_time": max_time,
        #                            "num_classes":num_classes,
        #                            "lr":lr,
        #                            "embed_time":embed_time,
        #                            "embedding_dim":embedding_dim,
        #                            "use_weighted_loss": use_weighted_loss
        #                            },
        #                           ignore=["loss_weights", "classes"])
        self.max_time = max_time
        if embed_time:
            num_features = satellite_input_channels * embedding_dim
            self.norm = LayerNorm([satellite_input_channels, embedding_dim])
        else:
            num_features = satellite_input_channels * max_time
            self.norm = LayerNorm([max_time, satellite_input_channels])
        self.linear = Linear(num_features, num_classes)
        self.embed_time = embed_time

        if self.embed_time:
            self.pos_encoding = PositionalEncoding(embedding_dim=embedding_dim, max_len=self.max_time)
            self.register_buffer("pos_embed", self.pos_encoding.pos_embed)

    def forward(self, boa_batch, time_batch, unused):  # unused is only there for API consistency
        batch_size = boa_batch.shape[0]
        with torch.no_grad():
            boa_batch[boa_batch<0] = 0

        x = sparse2dense_timeseries_batched_torch(boa_batch,
                                                  time_batch,
                                                  self.max_time,
                                                  subsampling=1)

        if self.embed_time:
            x = torch.matmul(torch.permute(x, (0,2,1)), self.pos_embed)

        x = self.norm(x)
        return self.linear(x.view(batch_size, -1))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0005)
