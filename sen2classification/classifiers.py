import os
import numpy as np
import torch
import pandas as pd
from . import utils
from typing import Any
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from pytorch_lightning import LightningModule
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import MLP
from torchmetrics.functional import accuracy, precision, recall
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection
from .sitsbert.model.classification_model import SBERT, SBERTClassification
from plotting import plot_confusion_matrix
from .datasets import InMemoryTimeSeriesDataset
import datetime
from .utils import GeneratorDataset, read_img, array_to_tif
import rioxarray


#%%
class TreeClassifier(LightningModule):
    def __init__(self, num_classes: int, lr: float, loss_weights: list, use_weighted_loss: bool = False):
        super().__init__()
        self.save_hyperparameters("num_classes", "lr", "loss_weights")
        self.lr = lr
        self.num_classes = num_classes
        self.task = "binary" if self.num_classes == 2 else "multiclass"
        if use_weighted_loss and loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights)
            # self.register_buffer("loss_weights", loss_weights)
            self.loss_weights.requires_grad = False
        else:
            self.loss_weights = None

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        utils.change_resnet_classes(self.model, num_classes=self.num_classes)
        utils.change_resnet_input(self.model, 4)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    # the loss weight is not moved automatically for some reason,
    # move it manually here
    def on_fit_start(self) -> None:
        if self.loss_weights is not None:
            self.loss_weights = self.loss_weights.to(dtype=self.dtype, device=self.device)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = cross_entropy(y_pred, y_true, weight=self.loss_weights)

        y_pred_labels = torch.argmax(y_pred, dim=1)

        acc = accuracy(y_pred_labels, y_true,  task=self.task, num_classes=self.num_classes)
        pre = precision(y_pred_labels, y_true, task=self.task, num_classes=self.num_classes)
        rec = recall(y_pred_labels, y_true,    task=self.task, num_classes=self.num_classes)

        metrics_dict = {"train_loss": loss, "train_acc": acc, "train_pre": pre, "train_rec": rec}
        self.log_dict(metrics_dict, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = cross_entropy(y_pred, y_true)

        y_pred_labels = torch.argmax(y_pred, dim=1)

        acc = accuracy(y_pred_labels, y_true,  task=self.task, num_classes=self.num_classes)
        pre = precision(y_pred_labels, y_true, task=self.task, num_classes=self.num_classes)
        rec = recall(y_pred_labels, y_true,    task=self.task, num_classes=self.num_classes)

        metrics_dict = {"val_loss": loss, "val_acc": acc, "val_pre": pre, "val_rec": rec}
        self.log_dict(metrics_dict, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(params=self.parameters(), lr=self.lr)


class SBERTClassifier(LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float,
                 loss_weights: list = None,
                 use_weighted_loss: bool = False,
                 satellite_input_channels: int = 10,
                 hidden_dim: int = 256,
                 transformer_layercount: int = 3,
                 num_attention_heads: int = 8,
                 max_embedding_size: int = 366,
                 cosine_init_period: int = 900,
                 classes: "list[str]" = None  # only needed for the test step
                ):
        super().__init__()
        self.save_hyperparameters("num_classes", "lr", "loss_weights", "satellite_input_channels", "hidden_dim",
                                  "transformer_layercount", "num_attention_heads")
        self.num_classes = num_classes
        self.classes = classes
        self.lr = lr
        self.loss_weights = loss_weights
        self.use_weighted_loss = use_weighted_loss
        self.satellite_input_channels = satellite_input_channels
        self.hidden_dim = hidden_dim
        self.transformer_layercount = transformer_layercount
        self.num_attention_heads = num_attention_heads
        self.cosine_init_period = cosine_init_period

        self.task = "binary" if self.num_classes == 2 else "multiclass"  # required for torchmetrics
        if use_weighted_loss and loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights)
            # self.register_buffer("loss_weights", loss_weights)
            self.loss_weights.requires_grad = False

        self.sbert = SBERT(num_features=satellite_input_channels,
                           hidden=hidden_dim,
                           n_layers=transformer_layercount,
                           attn_heads=num_attention_heads,
                           max_embedding_size=max_embedding_size)
        self.transformer = SBERTClassification(self.sbert, self.num_classes)

        metric = MetricCollection({'acc': Accuracy(task=self.task, num_classes=self.num_classes)})
        self.train_metric = metric.clone(prefix='train_')
        self.valid_metric = metric.clone(prefix='val_')

        self.test_acc = Accuracy(task=self.task, num_classes=self.num_classes)
        self.test_cm = ConfusionMatrix(task=self.task, num_classes=self.num_classes)

    def forward(self, x):
        return self.transformer(*x)
    
    def on_fit_start(self) -> None:
        if self.loss_weights is not None:
            self.loss_weights = self.loss_weights.to(dtype=self.dtype, device=self.device)

    def on_train_start(self) -> None:
        # log hyperparams
        self.logger.log_hyperparams(self.hparams, {'train_acc_step': 0, 'val_acc': 0})
        return super().on_train_start()
    
    def shared_step(self, batch, batch_idx, metric):
        _, boa, doy, mask, y_true = batch
        y_pred = self((boa, doy, mask))

        loss = cross_entropy(y_pred, y_true, weight=self.loss_weights)

        with torch.no_grad():
            y_pred_labels = torch.argmax(y_pred, dim=1)

        # torchmetrics are synced implicitly, no sync_dist needed
        # https://github.com/Lightning-AI/lightning/discussions/6501
        # but I can't find any syncs in the code, so keep it
        self.log_dict(metric(y_pred_labels, y_true), on_epoch=True, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.train_metric)
        self.log("train_loss", loss.item(), on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.valid_metric)
        self.log("val_loss", loss.item(), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, boa, doy, mask, y_true = batch
        y_pred = self((boa, doy, mask))
        y_pred_labels = torch.argmax(y_pred, dim=1)

        self.test_acc.update(y_pred_labels, y_true)
        self.test_cm.update(y_pred_labels, y_true)

    def on_test_end(self):
        acc = self.test_acc.compute()
        cm = self.test_cm.compute().cpu().numpy()

        outpath = self.logger.log_dir
        version = self.logger.version

        with open(os.path.join(outpath, f"report_{version}.txt"), "w") as f:
            f.write(str(self.classes) + "\n")
            f.write(f"acc: {acc * 100:.1f}\n")

        np.savetxt(os.path.join(outpath, f"cm_{version}.txt"), cm)

        plot_confusion_matrix(cm, classes=self.classes, fmt=".0f",
                              outfile=os.path.join(outpath, f"confmat_unnormalized_{version}.png"), fontsize=4)

        plot_confusion_matrix(cm, classes=self.classes, fmt=".2f", normalize="precision",
                              outfile=os.path.join(outpath, f"confmat_precision_{version}.png"), fontsize="xx-small")

        plot_confusion_matrix(cm, classes=self.classes, fmt=".2f", normalize="recall",
                              outfile=os.path.join(outpath, f"confmat_recall_{version}.png"), fontsize="xx-small")

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.cosine_init_period, T_mult=2, eta_min=1e-6, last_epoch=-1)
        lr_config = {"scheduler": scheduler,
                     "interval": "step",
                     "frequency": 1,
                     }
        return {"optimizer": optimizer, "lr_scheduler": lr_config}
    
    def predict_dataset(self, ds: InMemoryTimeSeriesDataset, batch_size=128, num_workers=4):
        self.eval()
        ids = []
        trues = []
        preds = []
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(ds, batch_size=batch_size, num_workers=num_workers)):
                tree_ids, boa, doy, mask, y_true = batch
                boa = boa.to(self.device)
                doy = doy.to(self.device)
                mask = mask.to(self.device)
                y_pred = self((boa, doy, mask))
                y_pred_labels = torch.argmax(y_pred, dim=1)
                ids.extend(tree_ids.numpy())
                preds.extend([self.classes[x] for x in y_pred_labels.cpu().numpy()])
                trues.extend([self.classes[x] for x in y_true.cpu().numpy()])
        return pd.DataFrame({"tree_id": ids, "y_true": trues, "y_pred": preds, "correct": np.array(trues) == np.array(preds)})

    def predict_timeseries(self, input_folder, qai, seq_len, output_filepath=None, save=True, t0=datetime.date(2015, 1, 1), batch_size=16, num_workers=0):
        self.eval()
        files = os.listdir(input_folder)
        boa_filenames = list(sorted(filter(lambda x: 'SEN2' in x and 'BOA' in x, files)))[-seq_len:]

        if qai > 0:
            qais = list(sorted(filter(lambda x: 'SEN2' in x and 'QAI' in x, files)))[-seq_len:]

        seq_len = min(seq_len, len(boa_filenames))

        dates = [datetime.datetime.strptime(s[:8], '%Y%m%d').date() for s in boa_filenames]
        days_since_t0 = np.array([(date - t0).days for date in dates])

        print("Loading images")
        sample_boa = read_img(os.path.join(input_folder, boa_filenames[0]), dim_ordering="HWC", dtype=np.int16)
        h,w,c = sample_boa.shape
        all_boas = np.empty((h,w,seq_len,c), dtype=np.int16)

        # read all the files
        for (i,f) in enumerate(boa_filenames):
            fname = os.path.join(input_folder, f)
            img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
            all_boas[:,:,i,:] = img

        all_boas = all_boas.reshape((-1, seq_len, c))

        if qai > 0:
            validity_mask = np.empty((h, w, seq_len, 1), dtype=bool)
            for (i,f) in enumerate(qais):
                fname = os.path.join(input_folder, f)
                img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
                validity_mask[:,:,i,:] = (img & qai) == 0

            validity_mask = validity_mask.reshape((-1, seq_len))
            n_obs = np.sum(validity_mask, axis=1)

            def pixel_generator():
                i = 0
                while i < all_boas.shape[0]:
                    n = n_obs[i]
                    mask = validity_mask[i]
                    transformer_mask = np.zeros(seq_len, dtype=int)
                    transformer_mask[:n] = 1
                    this_pixels_times = np.zeros(seq_len, dtype=int)
                    this_pixels_times[:n] = days_since_t0[mask]
                    boa = np.zeros((seq_len, c), dtype=np.float32)
                    boa[:n, :] = all_boas[i][mask]
                    yield torch.from_numpy(boa), torch.from_numpy(this_pixels_times), torch.from_numpy(transformer_mask)
                    i += 1
        else:
            def pixel_generator():
                i = 0
                while i < all_boas.shape[0]:
                    transformer_mask = np.ones(seq_len, dtype=int)
                    yield torch.from_numpy(all_boas[i]), torch.from_numpy(days_since_t0), torch.from_numpy(transformer_mask)
                    i += 1

        gen = GeneratorDataset(pixel_generator())
        dl = DataLoader(gen, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        output = np.zeros(w*h, dtype=np.uint8)

        print("Starting prediction")
        print(f"0 / {w*h//batch_size}")
        with torch.no_grad():
            for (i,batch) in enumerate(dl):
                if i%100 == 0:
                    print(f"{i} / {w * h // batch_size}")
                try:
                    boa, doy, mask = batch
                    boa = boa.to(self.device)
                    doy = doy.to(self.device)
                    mask = mask.to(self.device)
                    pred = torch.argmax(self((boa, doy, mask)), dim=1)
                    start = i*batch_size
                    stop = start + pred.shape[0]
                    output[start:stop] = pred.cpu().numpy()
                except Exception as err:
                    print(err)
                    break

        output = output.reshape((h, w))

        if save:
            if output_filepath is None:
                if os.path.isdir(input_folder) and input_folder[-1] != os.sep:
                    input_folder += os.sep

                output_filename = input_folder.split(os.sep)[-2]
                assert output_filename != "", "Calculated empty output file name, check input directory."
                output_filepath = os.path.join(os.path.abspath(input_folder), f"{output_filename}.tif")

            sample_filepath = os.path.join(input_folder, boa_filenames[0])
            print(f"Writing file {output_filepath}")
            array_to_tif(output, output_filepath, src_raster=sample_filepath)
        return output


class MultiModalTreeClassifier(LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float,
                 loss_weights: list,
                 use_weighted_loss: bool = False,
                 satellite_input_channels: int = 10,
                 hidden_dim: int = 256,
                 transformer_layercount: int = 3,
                 num_attention_heads: int = 8
                 ):
        super().__init__()
        self.save_hyperparameters("num_classes", "lr", "loss_weights", "satellite_input_channels", "hidden_dim",
                                  "transformer_layercount", "num_attention_heads")
        self.num_classes = num_classes
        self.lr = lr
        self.loss_weights = loss_weights
        self.use_weighted_loss = use_weighted_loss
        self.satellite_input_channels = satellite_input_channels
        self.hidden_dim = hidden_dim
        self.transformer_layercount = transformer_layercount
        self.num_attention_heads = num_attention_heads

        self.task = "binary" if self.num_classes == 2 else "multiclass"  # required for torchmetrics
        if use_weighted_loss and loss_weights is not None:
            self.loss_weights = torch.tensor(loss_weights)
            # self.register_buffer("loss_weights", loss_weights)
            self.loss_weights.requires_grad = False
        else:
            self.loss_weights = None

        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        utils.change_resnet_classes(self.cnn, num_classes=self.num_classes)
        utils.change_resnet_input(self.cnn, 4)

        self.sbert = SBERT(num_features=satellite_input_channels,
                           hidden=hidden_dim,
                           n_layers=transformer_layercount,
                           attn_heads=num_attention_heads)
        self.transformer = SBERTClassification(self.sbert, self.num_classes)
        self.head = MLP(2*num_classes, [64,64,64, num_classes])
        # self.head_1 = torch.nn.Linear(2 * self.num_classes, self.num_classes)  # really simple combination
        # self.head_2 = torch.nn.Linear(2*self.num_classes, num_classes)

    def forward(self, image, transformer_input):
        y_cnn = self.cnn(image)
        y_transformer = self.transformer(*transformer_input)
        combined = torch.cat((y_cnn, y_transformer), dim=-1)
        out = self.head(torch.nn.functional.relu(combined))
        # out = self.head_2(combined)
        # softmax happens in the loss function
        return y_cnn, y_transformer, out
    
    def on_fit_start(self) -> None:
        if self.loss_weights is not None:
            self.loss_weights = self.loss_weights.to(dtype=self.dtype, device=self.device)

    def training_step(self, batch, batch_idx):
        x_cnn, x_transformer, y_true = batch
        y_cnn, y_transformer, y_pred = self(x_cnn, x_transformer)

        loss_transformer = cross_entropy(y_transformer, y_true, weight=self.loss_weights)
        loss_cnn = cross_entropy(y_cnn, y_true, weight=self.loss_weights)
        loss_combined = cross_entropy(y_pred, y_true, weight=self.loss_weights)

        loss = loss_combined + loss_cnn + loss_transformer

        with torch.no_grad():
            y_pred_labels = torch.argmax(y_pred, dim=1)
            y_transformer_labels = torch.argmax(y_transformer, dim=1)
            y_cnn_labels = torch.argmax(y_cnn, dim=1)

        # blabla for logging
        acc             = accuracy(y_pred_labels,        y_true, task=self.task, num_classes=self.num_classes)
        acc_cnn         = accuracy(y_cnn_labels,         y_true, task=self.task, num_classes=self.num_classes)
        acc_transformer = accuracy(y_transformer_labels, y_true, task=self.task, num_classes=self.num_classes)
        pre = precision(y_pred_labels, y_true, task=self.task, num_classes=self.num_classes)
        rec = recall(y_pred_labels, y_true, task=self.task, num_classes=self.num_classes)

        metrics_dict = {"train_loss_total": loss.item(), "train_loss_cnn": loss_cnn.item(),
                        "train_loss_transformer": loss_transformer.item(),
                        "train_acc": acc, "train_pre": pre, "train_rec": rec,
                        "train_acc_transformer": acc_transformer, "train_acc_cnn": acc_cnn}
        self.log_dict(metrics_dict, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_cnn, x_trans, y_true = batch
        y_cnn, y_transformer, y_pred = self(x_cnn, x_trans)

        loss_transformer = cross_entropy(y_transformer, y_true, weight=self.loss_weights)
        loss_cnn = cross_entropy(y_cnn, y_true, weight=self.loss_weights)
        loss_combined = cross_entropy(y_pred, y_true, weight=self.loss_weights)

        loss = loss_combined + loss_cnn + loss_transformer

        y_pred_labels = torch.argmax(y_pred, dim=1)
        y_transformer_labels = torch.argmax(y_transformer, dim=1)
        y_cnn_labels = torch.argmax(y_cnn, dim=1)

        # blabla for logging
        acc             = accuracy(y_pred_labels,        y_true, task=self.task, num_classes=self.num_classes)
        acc_cnn         = accuracy(y_cnn_labels,         y_true, task=self.task, num_classes=self.num_classes)
        acc_transformer = accuracy(y_transformer_labels, y_true, task=self.task, num_classes=self.num_classes)
        pre = precision(y_pred_labels, y_true, task=self.task, num_classes=self.num_classes)
        rec = recall(y_pred_labels, y_true, task=self.task, num_classes=self.num_classes)

        metrics_dict = {"val_loss": loss, "val_loss_cnn": loss_cnn, "val_loss_transformer": loss_transformer,
                        "val_acc": acc, "val_pre": pre, "val_rec": rec,
                        "val_acc_transformer": acc_transformer, "val_acc_cnn": acc_cnn}
        self.log_dict(metrics_dict, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.lr)


class MockSBERTClassifier(LightningModule):
    """Only exists to test CLI setup quickly."""
    def __init__(self,
                 num_classes: int,
                 lr: float,
                 loss_weights: list,
                 use_weighted_loss: bool = False,
                 satellite_input_channels: int = 10,
                 hidden_dim: int = 256,
                 transformer_layercount: int = 3,
                 num_attention_heads: int = 8,
                 max_embedding_size: int = 366,
                 cosine_init_period: int = 900,
                 classes: "list[str]" = None  # only needed for the test step
                ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.classes = classes

    def forward(self, x):
        pass