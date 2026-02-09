import torch
import pytorch_lightning as L
from torch import nn, optim
from torch.nn.functional import relu
from torchmetrics.classification import Accuracy
from ..satellite_classifier import SatelliteClassifier
from .sitsbert.model.embedding.bert import BERTEmbedding, ConcatEmbedding


class GRU(SatelliteClassifier):
    def __init__(self,
                 satellite_input_channels=10,
                 hidden_dim=256,
                 num_layers=3,
                 num_classes=5,
                 loss_weights=None,
                 use_weighted_loss=False,
                 dropout=0.2,
                 fc_size=512,
                 bidirectional=False,
                 lr=1e-3,
                 cosine_init_period=3000,
                 classes=None,
                 embedding_dim=None,
                 embedding_type="bert",
                 max_time=8*366,
                 layernorm_on_input=False,
                 **kwargs):
        """ Instantiates a GRU model for time series classification.

        Args:
            satellite_input_channels (int): Number of input channels of the satellite data.
            hidden_dim (int): Number of hidden units in the GRU.
            num_layers (int): Number of layers in the GRU.
            num_classes (int): Number of classes to classify.
            loss_weights (list): Weights for the loss function.
            use_weighted_loss (bool): Whether to use a weighted loss function.
            dropout (float): Dropout rate.
            fc_size (int): Size of the fully connected layer.
            bidirectional (bool): Whether the GRU should be bidirectional.
            lr (float): Learning rate.
            cosine_init_period (int): Period of the cosine annealing learning rate schedule.
            classes (list): List of class names.
            embedding_dim (int): Dimension of the embedding. If None, no embedding is used.
            embedding_type (str): Type of the embedding.
            max_time (int): Maximum time value for the embedding.
            layernorm_on_input (bool): Whether to use layer normalization on the input.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_classes, lr, loss_weights, use_weighted_loss, classes, **kwargs)
        self.cosine_init_period = cosine_init_period
        self.embed_time = True if embedding_dim else False
        gru_input_channels = satellite_input_channels
        self.norm = nn.LayerNorm(satellite_input_channels) if layernorm_on_input else torch.nn.Identity()
        if embedding_dim:
            gru_input_channels = embedding_dim
            if embedding_type == "bert":
                self.pos_embed = BERTEmbedding(num_features=satellite_input_channels, embedding_dim=embedding_dim, max_pos_embed_val=max_time)
            elif embedding_type == "concat":
                assert embedding_dim % 2 == 0, "Embedding dim must be divisible by two if embedding type concat is chosen."
                self.pos_embed = ConcatEmbedding(num_features=satellite_input_channels,
                                                 embedding_dim=embedding_dim//2,
                                                 max_pos_embed_val=max_time)

        self.gru = nn.GRU(gru_input_channels, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout, bidirectional=bidirectional)
        fc_in = hidden_dim * (1 + bidirectional)
        self.fc1 = nn.Linear(in_features=fc_in, out_features=fc_size)
        self.clf = nn.Linear(fc_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, boa, times, mask, validation_mask=None):
        z = self.norm(boa)

        if self.embed_time:
            z = self.pos_embed(z, times)

        z = nn.utils.rnn.pack_padded_sequence(z, torch.sum(torch.logical_not(mask), 1).cpu(), batch_first=True, enforce_sorted=False)
        z, h = self.gru(z)
        z, mask_lens = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=mask.shape[1])
        z = z[0:len(mask_lens), mask_lens-1]
        z = self.dropout(z)
        z = self.fc1(z)
        z = relu(z)
        z = self.clf(z)
        return z


class TimeSeriesModel(L.LightningModule):
    def __init__(self, time_layer, num_features, sequence_length=32,
                 hidden_size=256, num_layers=3, n_classes=5,
                 class_weights=None, dropout=0.0, fc_size=512,
                 bidirectional=False, lr=1e-3, scheduler='cosine_warm',
                 use_time_feature=False, hyperparams=None):
        super().__init__()
        self.sequence_length = sequence_length
        if time_layer.lower() == 'lstm':
            self.lstm1 = nn.LSTM(num_features, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True,
                                 dropout=dropout, bidirectional=bidirectional)
        elif time_layer.lower() == 'gru':
            self.lstm1 = nn.GRU(num_features, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True,
                                dropout=dropout, bidirectional=bidirectional)
        elif time_layer.lower() == 'cnn':
            inputs = [num_features] + [hidden_size] * (num_layers - 1)
            self.cnns = [nn.Conv1d(input_, hidden_size, kernel_size=7, bias=False, padding='same', device='cuda') for input_ in inputs]
            # todo: BatchNorm besser?
            self.bns = [nn.LayerNorm((hidden_size, sequence_length), device='cuda') for _ in range(num_layers)]
        elif time_layer.lower() == 'both':
            num_lstm = num_layers//2
            num_cnn = num_layers - num_lstm
            inputs = [num_features] + [hidden_size//2] * (num_cnn - 1)
            self.cnns = [nn.Conv1d(input_, hidden_size//2, kernel_size=7, bias=False, padding='same', device='cuda') for input_ in inputs]
            self.bns = [nn.BatchNorm1d(hidden_size//2, device='cuda') for _ in range(num_cnn)]
            self.lstm1 = nn.GRU(hidden_size//2, hidden_size=hidden_size, num_layers=num_lstm, batch_first=True,
                                dropout=dropout, bidirectional=bidirectional)
        else:
            print(f'Unknown time layer {time_layer}! Valid options are [lstm, gru, cnn].')
        fc_in = hidden_size*sequence_length if time_layer.lower() == 'cnn' else hidden_size*(1+bidirectional)
        self.fc1 = nn.Linear(in_features=fc_in, out_features=fc_size)
        # self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.clf = nn.Linear(fc_size, n_classes)
        self.relu = nn.ReLU()
        self.dr = nn.Dropout(dropout)
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        class_weights = torch.Tensor(class_weights) if class_weights is not None else None
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.time_layer = time_layer.lower()

        self.lr = lr
        self.scheduler = scheduler
        self.use_time_feature = use_time_feature
        self.hyperparams = hyperparams
        self.save_hyperparameters(ignore='hyperparams')

    def forward(self, x):
        id_, x, times, mask, y = x
        if self.time_layer in ['lstm', 'gru']:
            if self.use_time_feature:
                timespans = torch.concat((torch.zeros(times.size(0), 1, dtype=times.dtype, device=times.device),
                                        ((times[:, 1:]-times[:, :-1])-2.0)),
                                        dim=-1)/366.
                timespans = timespans.unsqueeze(-1)
                x = torch.concat((x, timespans), dim=-1)
            z = nn.utils.rnn.pack_padded_sequence(x, torch.sum(mask, axis=1).cpu(), batch_first=True, enforce_sorted=False)
            z = z.to(self.device)  # needed for a notebook ,don't know why
            z, h = self.lstm1(z)
            z, mask_lens = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=mask.shape[1])
            # z = torch.stack([z_[i, :] for z_,i in zip(z, mask_lens-1)])
            z = z[torch.arange(len(mask_lens)), mask_lens-1]
        elif self.time_layer == 'cnn':
            z = torch.transpose(x, 1, 2)
            for c, b in zip(self.cnns, self.bns):
                z = self.relu(b(c(z)))
            z = nn.Flatten()(z)
        elif self.time_layer == 'both':
            z = torch.transpose(x, 1, 2)
            for c, b in zip(self.cnns, self.bns):
                z = self.relu(b(c(z)))
            z = torch.transpose(z, 1, 2)
            z, h = self.lstm1(z)
            mask_lens = torch.sum(torch.logical_not(mask), 1)
            z = z[torch.arange(len(mask_lens)), mask_lens-1]
        z = self.dr(z)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.clf(z)
        return z

    def training_step(self, batch, batch_idx):
        _, _, _, _, y = batch
        z = self.forward(batch)
        loss = self.loss(z, y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.accuracy(z, y)
        self.log('acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, _, _, y = batch
        z = self.forward(batch)
        loss = self.loss(z, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.accuracy(z, y)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # def prediction_step(self, batch, batch_idx):
    #     return self.forward(batch)

    def on_train_epoch_end(self) -> None:
        self.log("step", self.current_epoch + 1)
        print('')

    def on_validation_epoch_end(self) -> None:
        self.log("step", self.current_epoch + 1)
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer
        # optimizer = optim.Adam(self.parameters(), lr=5e-4)
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.lr, weight_decay=1e-2)
        # optimizer = optim.Adamax(self.parameters(), lr=1e-3)
        if self.scheduler == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.25, patience=8,
                mode='max', verbose=True)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_acc",
                    "interval": "epoch",
                    }
                }

