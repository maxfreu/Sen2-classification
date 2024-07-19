import os
import torch
import datetime
import numpy as np
import pandas as pd

from time import time
from glob import glob
from uuid import uuid4
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection
from sen2classification.plotting import plot_confusion_matrix
from sen2classification.utils import GeneratorDataset, read_img, array_to_tif
from .utils import load_and_prepare_timeseries_folder, assemble_batch_cpu
from sklearn.metrics import confusion_matrix


class SatelliteClassifier(LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float,
                 loss_weights: list = None,
                 use_weighted_loss: bool = False,
                 classes: "list[str]" = None  # only needed for the test step
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.classes = classes
        self.lr = lr
        self.loss_weights = loss_weights
        self.use_weighted_loss = use_weighted_loss

        if use_weighted_loss:
            assert loss_weights is not None
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(loss_weights)).to(torch.float32))
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        self.task = "binary" if self.num_classes == 2 else "multiclass"  # required for torchmetrics
        metric = MetricCollection({'acc': Accuracy(task=self.task, num_classes=self.num_classes)})
        self.train_metric = metric.clone(prefix='train/')
        self.valid_metric = metric.clone(prefix='val/')

    def shared_step(self, batch, metric):
        _, boa, doy, mask, y_true = batch
        y_pred = self(boa, doy, mask)
        loss = self.loss(y_pred, y_true)

        with torch.no_grad():
            y_pred_labels = torch.argmax(y_pred, dim=1)

        # torchmetrics are synced implicitly, no sync_dist needed
        # https://github.com/Lightning-AI/lightning/discussions/6501
        # but I can't find any syncs in the code, so keep it
        self.log_dict(metric(y_pred_labels, y_true), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, self.train_metric)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, self.valid_metric)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    @rank_zero_only
    def test_on_dataloader(self, dataloader, outpath, version, seq_len, return_mode):
        """Takes a dataloader, makes predictions on all samples, computes acc and confusion matrix, writes out result
        text files and confusion matrix plots into the log dir. Returns pandas dataframe with predictions."""
        self.eval()
        tp = 0
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        ids = []
        trues = []
        preds = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
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

        with open(os.path.join(outpath, f"report_v={version}_seq_len={seq_len}_ret_mode={return_mode}.txt"), "w") as f:
            f.write(f"seq_len={seq_len}")
            f.write(f"ret_mode={return_mode}\n")
            f.write(f"acc: {acc * 100:.1f}\n")

        with open(os.path.join(outpath, "classes.txt"), "w") as f:
            for (i,c) in zip(range(len(self.classes)), self.classes):
                f.write(f"{i},{c}\n")

        np.savetxt(os.path.join(outpath, f"cm_v={version}_seq_len={seq_len}_ret_mode={return_mode}.txt"), cm)

        plot_confusion_matrix(cm, classes=self.classes, fmt=".0f",
                              outfile=os.path.join(outpath,
                                                   f"confmat_unnormalized_v={version}_seq_len={seq_len}_ret_mode={return_mode}.png"),
                              fontsize=4)

        plot_confusion_matrix(cm, classes=self.classes, fmt=".2f", normalize="precision", title="Precision",
                              outfile=os.path.join(outpath,
                                                   f"confmat_precision_v={version}_seq_len={seq_len}_ret_mode={return_mode}.png"),
                              fontsize="xx-small")

        plot_confusion_matrix(cm, classes=self.classes, fmt=".2f", normalize="recall", title="Recall",
                              outfile=os.path.join(outpath,
                                                   f"confmat_recall_v={version}_seq_len={seq_len}_ret_mode={return_mode}.png"),
                              fontsize="xx-small")

        return pd.DataFrame(
            {"tree_id": ids, "y_true": trues, "y_pred": preds, "correct": np.array(trues) == np.array(preds)})

    def predict_force_folder(self,
                             input_folder,
                             qai,
                             seq_len,
                             time_encoding="absolute",
                             output_filepath=None,
                             save=True, t0=datetime.date(2015, 1, 1),
                             batch_size=128,
                             div_by=10000,
                             fname2date=lambda s: datetime.datetime.strptime(s[:8], '%Y%m%d').date(),
                             verbose=False,
                             tmin=datetime.date(2015, 1, 1),
                             tmax=datetime.date(2024, 1, 1),
                             ):
        if verbose:
            print(f"Predicting on images in folder {input_folder}")
        self.eval()
        boa_filenames = glob(os.path.join(input_folder, "*.tif"))
        # all_boas has shape (h*w, seq_len, c)
        (h, w, c), all_boas, times, validity_mask, n_obs = load_and_prepare_timeseries_folder(input_folder,
                                                                                              qai,
                                                                                              seq_len,
                                                                                              time_encoding,
                                                                                              t0,
                                                                                              fname2date=fname2date,
                                                                                              tmin=tmin,
                                                                                              tmax=tmax
                                                                                              )
        all_boas[all_boas < 0] = 0
        # defer division to later as raw data is in uint16 and we dont want to waste space

        if validity_mask is not None:
            all_boas[np.logical_not(validity_mask)] = 0

        batch_size = min(batch_size, h*w)

        boa_batch  = np.zeros((batch_size, seq_len, c), dtype=np.float32)
        time_batch = np.zeros((batch_size, seq_len), dtype=np.int32)
        mask_batch = np.zeros((batch_size, seq_len), dtype=bool)

        # pre-allocate data on GPU to recycle memory, boosts perf by ca 4%
        boa_torch  = torch.zeros((batch_size, seq_len, c), device=self.device, dtype=torch.float32)
        time_torch = torch.zeros((batch_size, seq_len), device=self.device, dtype=torch.int32)
        mask_torch = torch.zeros((batch_size, seq_len), device=self.device, dtype=bool)

        output = torch.zeros(h * w, dtype=torch.uint8, device=self.device)

        # predict
        t0 = time()
        with torch.no_grad():
            i = 0  # batch counter
            start = 0
            while start < h * w:
                if i % 100 == 0 and verbose:
                    print(f"{i} / {w * h // batch_size}")
                stop = min(start + batch_size, h * w)
                bs = stop - start
                # adjust the size of the last batch
                # lot of repetition, but needed to avoid allocations
                if start + batch_size > h * w:
                    boa_batch = np.zeros((bs, seq_len, c), dtype=np.float32)
                    time_batch = np.zeros((bs, seq_len), dtype=np.int32)
                    mask_batch = np.zeros((bs, seq_len), dtype=bool)
                    boa_torch  = torch.zeros((bs, seq_len, c), device=self.device, dtype=torch.float32)
                    time_torch = torch.zeros((bs, seq_len), device=self.device, dtype=torch.int32)
                    mask_torch = torch.zeros((bs, seq_len), device=self.device, dtype=bool)

                assemble_batch_cpu(boa_batch, time_batch, mask_batch, start, all_boas, n_obs, validity_mask, times)

                # copy data over to GPU without allocations
                boa_torch[:]  = torch.from_numpy(boa_batch)
                time_torch[:] = torch.from_numpy(time_batch)
                mask_torch[:] = torch.from_numpy(mask_batch)
                boa_torch /= div_by  # here we can safely divide, as data is float32
                # mask_torch = torch.from_numpy(mask_batch).to(self.device)
                pred = self(boa_torch, time_torch, mask_torch).argmax(dim=-1).to(torch.uint8)
                output[start:stop] = pred
                start += batch_size
                i += 1

        output = output.reshape(h, w).cpu().numpy()
        if verbose:
            print("Prediction time: ", time() - t0)

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
