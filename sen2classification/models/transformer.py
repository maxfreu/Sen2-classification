import torch.nn
from torch import optim
from .sitsbert.model.classification_model import SBERT, SBERTClassification
from ..satellite_classifier import SatelliteClassifier
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, SequentialLR


#%%
class SBERTClassifier(SatelliteClassifier):
    def __init__(self,
                 num_classes: int,
                 lr: float,
                 loss_weights: list = None,
                 use_weighted_loss: bool = False,
                 satellite_input_channels: int = 10,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_attention_heads: int = 8,
                 max_time: int = 366,
                 cosine_init_period: int = 900,
                 classes: "list[str]" = None,  # only needed for the test step
                 pretrained_model_path = "",
                 dropout=0.1,
                 layernorm_on_input=False,
                 embedding_type="bert",
                 **kwargs
                 ):
        super().__init__(num_classes, lr, loss_weights, use_weighted_loss, classes, **kwargs)
        # self.save_hyperparameters("num_classes", "lr", "loss_weights", "satellite_input_channels", "hidden_dim",
        #                           "transformer_layercount", "num_attention_heads")
        self.cosine_init_period = cosine_init_period

        if pretrained_model_path:
            model = SBERTPretrain.load_from_checkpoint(pretrained_model_path,
                                                       satellite_input_channels=satellite_input_channels,
                                                       hidden_dim=hidden_dim,
                                                       num_layers=num_layers,
                                                       num_attention_heads=num_attention_heads,
                                                       max_time=max_time,
                                                       layernorm_on_input=layernorm_on_input,
                                                       embedding_type=embedding_type
                                                       )
            self.sbert = model.sbert
        else:
            self.sbert = SBERT(num_features=satellite_input_channels,
                               d_model=hidden_dim,
                               num_layers=num_layers,
                               attn_heads=num_attention_heads,
                               max_embedding_size=max_time,
                               dropout=dropout,
                               layernorm_on_input=layernorm_on_input,
                               embedding_type=embedding_type)
        self.classifier = SBERTClassification(self.sbert, self.num_classes)

    def forward(self, boa, time, mask, averaging_mask=None):
        return self.classifier(boa, time, mask, averaging_mask=None)


class SBERTPretrain(LightningModule):
    def __init__(self,
                 lr: float = 5e-4,
                 satellite_input_channels: int = 10,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_attention_heads: int = 8,
                 max_time: int = 366,
                 cosine_init_period: int = 900,
                 dropout: float = 0.1,
                 warmup_steps: int = 4000,
                 layernorm_on_input: bool = False,
                 embedding_type: str = "bert"
                 ):
        super().__init__()
        self.lr = lr
        self.cosine_init_period = cosine_init_period
        self.sbert = SBERT(num_features=satellite_input_channels,
                           d_model=hidden_dim,
                           num_layers=num_layers,
                           attn_heads=num_attention_heads,
                           max_embedding_size=max_time,
                           dropout=dropout,
                           embedding_type=embedding_type,
                           layernorm_on_input=layernorm_on_input)
        self.linear = torch.nn.Linear(hidden_dim, satellite_input_channels)
        self.loss = torch.nn.MSELoss(reduction="none")
        self.warmup_steps = warmup_steps

    def forward(self, boa, time, mask):
        x = self.sbert(boa, time, mask)  # shape: batchsize, sequence length, hidden dim
        x = self.linear(x)
        return x

    def shared_step(self, batch, train_or_val):
        boa, time, transformer_mask, data_mask = batch
        # filename = os.path.basename(filename).split('.')[0]
        masked_boa = boa.clone()
        masked_boa[data_mask] = 0
        pred = self(masked_boa, time, transformer_mask)
        loss = self.loss(pred, boa)
        masked_loss = (loss * data_mask.unsqueeze(-1)).sum() / data_mask.sum()
        neg_tm = ~transformer_mask.unsqueeze(-1)
        unmasked_loss = (loss * neg_tm).sum() / neg_tm.sum()
        total_loss = masked_loss + unmasked_loss * 1e-2

        self.log(f"{train_or_val}/loss", total_loss)
        return total_loss

    def training_step(self, batch, batch_index):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_index):
        return self.shared_step(batch, "val")

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.sbert, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.lr)
        # optimizer = optim.RAdam(params=self.parameters(), lr=self.lr, weight_decay=0)
        train_scheduler = CosineAnnealingWarmRestarts(optimizer, self.cosine_init_period, T_mult=2, eta_min=1e-6, last_epoch=-1)

        def warmup(current_step: int):
            return current_step / self.warmup_steps

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [self.warmup_steps])

        lr_config = {"scheduler": scheduler,
                     "interval": "step",
                     "frequency": 1,
                     }
        return {"optimizer": optimizer, "lr_scheduler": lr_config}

    # def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
