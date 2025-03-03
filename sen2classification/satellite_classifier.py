import os
from typing import Dict, Any, Optional

import torch
import datetime
import numpy as np
import pandas as pd

from glob import glob
from uuid import uuid4

from torch.optim import AdamW
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection
from sen2classification.utils import GeneratorDataset, read_img, array_to_tif, predict_on_batches
from .utils import load_and_prepare_timeseries_folder, assemble_batch_cpu
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, SequentialLR
from schedulefree import AdamWScheduleFree
from sen2classification.focalloss import FocalLoss
from filelock import FileLock


class SatelliteClassifier(LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float,
                 loss_weights: list = None,
                 use_weighted_loss: bool = False,
                 classes: "list[str]" = None,  # only needed for the test step
                 cosine_init_period: int = 0,
                 warmup_steps: int = 15000,
                 weight_decay: float = 1e-3,
                 focalloss_gamma=0
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.classes = classes
        self.lr = lr
        self.loss_weights = loss_weights
        self.use_weighted_loss = use_weighted_loss
        self.cosine_init_period = cosine_init_period
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        if focalloss_gamma == 0:
            if use_weighted_loss:
                assert loss_weights is not None
                self.loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(loss_weights)).to(torch.float32))
            else:
                self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = FocalLoss(focalloss_gamma)

        self.task = "binary" if self.num_classes == 2 else "multiclass"  # required for torchmetrics
        metric = MetricCollection({'acc': Accuracy(task=self.task, num_classes=self.num_classes)})
        self.train_metric = metric.clone(prefix='train_')
        self.valid_metric = metric.clone(prefix='val_')

    def shared_step(self, batch, metric):
        _, boa, doy, mask, y_true = batch
        y_pred = self(boa, doy, mask)
        loss = self.loss(y_pred, y_true)

        with torch.no_grad():
            y_pred_labels = torch.argmax(y_pred.detach(), dim=1)

        # torchmetrics are synced implicitly, no sync_dist needed
        # https://github.com/Lightning-AI/lightning/discussions/6501
        # but I can't find any syncs in the code, so keep it
        self.log_dict(metric(y_pred_labels, y_true), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, self.train_metric)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, self.valid_metric)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def set_optimizer_state(self, state: str):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        for opt in opts:
            if isinstance(opt, AdamWScheduleFree):
                if state == "train":
                    opt.train()
                elif state == "eval":
                    opt.eval()
                else:
                    raise ValueError(f"Unknown train state {state}")

    def on_train_epoch_start(self) -> None:
        self.set_optimizer_state("train")

    def on_validation_epoch_start(self) -> None:
        self.set_optimizer_state("eval")

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.parameters(), lr=self.lr, warmup_steps=self.warmup_steps)
        return optimizer

        # optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # if self.cosine_init_period > 0:
        #     scheduler = CosineAnnealingWarmRestarts(optimizer, self.cosine_init_period, T_mult=2, eta_min=1e-6, last_epoch=-1)
        #
        #     def warmup(current_step: int):
        #         return current_step / self.warmup_steps
        #
        #     warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
        #     scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], [self.warmup_steps])
        #
        #     return {"optimizer": optimizer,
        #             "lr_scheduler": {
        #                 "scheduler": scheduler,
        #                 "interval": "step",
        #                 }
        #             }
        # else:
        #     return optimizer

    @rank_zero_only
    def test_on_dataloader(self, dataloader):
        """Takes a dataloader and makes predictions on all samples.

        Returns:
            accuracy, numpy confusion matrix, pandas dataframe with predictions.
        """
        self.eval()
        tp = 0
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        ids = []
        trues = []
        preds = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if (i % 100) == 0:
                    print(f"{i + 1}/{len(dataloader)}")
                tree_ids, boa, doy, mask, y_true = batch
                boa = boa.to(self.device)
                doy = doy.to(self.device)
                mask = mask.to(self.device)
                y_pred = self(boa, doy, mask)
                y_pred_labels = torch.argmax(y_pred, dim=1)

                # compute metrics on cpu
                y_pred_labels = y_pred_labels.cpu()
                y_true = y_true.cpu()

                ids.extend(tree_ids.numpy())
                tp += (y_pred_labels == y_true).sum()
                cm += confusion_matrix(y_true, y_pred_labels, labels=range(self.num_classes))
                preds.extend([self.classes[x] for x in y_pred_labels])
                trues.extend([self.classes[x] for x in y_true])

        acc = tp / len(trues)

        return (acc,
                cm,
                pd.DataFrame({"tree_id": ids,
                              "y_true": trues,
                              "y_pred": preds,
                              "correct": np.array(trues) == np.array(preds)})
                )

    def predict_force_folder(self,
                             input_folder,
                             qai,
                             seq_len,
                             output_filepath=None,
                             save=True,
                             batch_size=128,
                             mean=np.zeros(10, dtype=np.float32),
                             stddev=np.ones(10, dtype=np.float32) * 10000,
                             fname2date=lambda s: datetime.datetime.strptime(s[:8], '%Y%m%d').date(),
                             time_encoding="absolute",
                             apply_argmax=True,
                             num_classes=0,
                             band_reordering=None,
                             t0=datetime.date(2015, 1, 1),
                             tmin_data=datetime.date(2015, 1, 1),
                             tmax_data=datetime.date(2024, 1, 1),
                             tmin_inference=None,
                             tmax_inference=None,
                             verbose=False,
                             ):
        """
        Predicts on all images in a folder, saves the result as a tif file and returns the prediction as a numpy array.

        Args:
            input_folder: Input folder with images
            qai: Quality Assurance Information bit flags
            seq_len: Sequence length
            output_filepath: Output file path
            save: Whether to save the prediction as a tif file
            batch_size: Batch size
            mean: Mean of the training data
            stddev: Standard deviation of the training data
            fname2date: Function to convert filenames to dates
            time_encoding: Whether the time encoding the network expects is absolute (calculated from t0) or relative (day of year)
            apply_argmax (bool): Whether to apply argmax to the output, i.e. whether to convert soft classifications into hard decisions for one class
            num_classes (int): Only applicable if apply_argmax=False. Number of classes the network outputs.
            band_reordering (list): An optional list for reordering the output bands. The list should contain all indices of the old array in the new one, e.g. [2,0,1] to move the last band to first position.
            t0: Time origin
            tmin_data: Starting point in time for loading the data
            tmax_data: End point in time for loading the data
            tmin_inference: Starting point in time for inference. If None, tmin_data is used.
            tmax_inference: End point in time for inference (exclusive). If None, tmax_data is used.
            verbose: Whether to print progress information

        Returns:
            Prediction as numpy array
        """
        if verbose:
            print(f"Predicting on images in folder {input_folder}")
        self.eval()

        if tmin_inference is None:
            tmin_inference = tmin_data
        if tmax_inference is None:
            tmax_inference = tmax_data

        if     tmin_inference < tmin_data \
            or tmax_inference < tmin_data \
            or tmin_inference > tmax_data \
            or tmax_inference > tmax_data \
            or tmin_inference > tmax_inference \
            or tmin_data > tmax_data:
            raise ValueError("Invalid time range for inference or data loading.")

        boa_filenames = glob(os.path.join(input_folder, "*.tif"))

        # with FileLock("/home/max/copy_lock.lock") as lock:
        # all_boas has shape (h*w, loaded images (sequence length), c)
        (h, w, c), all_boas, times, inference_date_mask, validity_mask, n_obs = load_and_prepare_timeseries_folder(
            input_folder,
            qai,
            seq_len,
            time_encoding,
            t0,
            fname2date=fname2date,
            tmin_data=tmin_data,
            tmax_data=tmax_data,
            tmin_inference=tmin_inference,
            tmax_inference=tmax_inference,
        )

        # (model, c, all_boas, times, validity_mask, n_obs, mean, stddev, batch_size,
        #  inference_date_mask, verbose)
        output = predict_on_batches(self, all_boas, times, validity_mask, n_obs, mean, stddev, batch_size,
                                    inference_date_mask, verbose, apply_argmax, num_classes)

        if apply_argmax:
            # TODO: Reorder before argmax! Right now it's a trap!
            output = output.reshape(h, w).cpu().numpy()
        else:
            # output is softmaxed
            output = output.reshape(h, w, num_classes).cpu().numpy()

            # TODO: move this logic into predict_on_batches!
            if band_reordering:
                print("reordering bands")
                output = output[:, :, list(band_reordering)]

        if save:
            if output_filepath is None:
                if os.path.isdir(input_folder) and input_folder[-1] != os.sep:
                    input_folder += os.sep

                output_filename = input_folder.split(os.sep)[-2] + "_pred_" + str(uuid4())[:4]
                assert output_filename != "", "Calculated empty output file name, check input directory."
                output_filepath = os.path.join(os.path.abspath(input_folder), f"{output_filename}.tif")

            sample_filepath = os.path.join(input_folder, boa_filenames[0])
            if verbose:
                print(f"Writing file {output_filepath}")
            array_to_tif(output, output_filepath, src_raster=sample_filepath)

        return output

#%%
