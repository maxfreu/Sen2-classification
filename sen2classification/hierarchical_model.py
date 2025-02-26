import os
import torch
import datetime
import numpy as np
from glob import glob
from uuid import uuid4
from torch.nn.functional import one_hot
from .utils import array_to_tif, assemble_batch_cpu, load_and_prepare_timeseries_folder, predict_on_batches
from pytorch_lightning import LightningModule
from sen2classification.satellite_classifier import SatelliteClassifier


class HierarchicalModel(SatelliteClassifier):
    def __init__(self, parent_model, child_models, num_classes: list, classes: list, reorder_dict: dict = None):
        """
        Takes a list of models. Input is passed through the parent model first to determine which child model to use.
        The parent model's output is argmax'ed and based on that the respective child model is called with the input.
        The returned class indices will be in the same order as the child models, just offset by their cumulated sum. If
        there is a background class (i.e. None is passed as child model in some position), 0 will be the background class
        and all other output will be in ascending order from 1 up.

        In forward call it dispatches to child models via for loop, probably super slow.

        This model is not differentiable and therefore not trainable!

        Args:
            parent_model: The parent model used to decide which child model to use. The background class should be 0 after argmaxing the output.
            child_models: The child models that output the final classification. The child model order has to match the
                order of classes of the parent model. E.g. argmaxed parent output 1 will call the child model with index 1.
            num_classes (list): Number of output classes per child model, in the same order.
            reorder_dict (dict): A dict that maps hierarchical model output class indices to new values. If given,
                these substitutions are always applied to all model outputs.
            classes (list): A list of strings containing the output class names.
        """
        super().__init__(num_classes=sum(num_classes)+1, classes=classes, lr=0)
        self.parent_model = parent_model
        self.child_models = child_models
        self.class_offset = torch.tensor(np.cumsum([1] + num_classes), requires_grad=False, device=parent_model.device)
        self.lookup_table = torch.arange(256, device=parent_model.device)
        if reorder_dict is not None:
            for key, value in reorder_dict.items():
                assert key < 256, "The class reordering is invalid and maps to a value greater than 255."
                self.lookup_table[key] = value

        print(f"class offset {self.class_offset}")

    def forward(self, boa, times, mask):
        # x is a tuple of (boa, doy, mask) values
        # each of them containing items of a batch
        with torch.no_grad():
            # make prediction with parent model and argmax the values to obtain the class indices
            # background must have value 0! then followed by the other classes
            parent_classes = torch.argmax(self.parent_model(boa, times, mask), dim=1)
            output = torch.zeros_like(parent_classes)

            for cls, model in enumerate(self.child_models, start=1):
                # select all items from batch equal to cls
                class_mask = parent_classes == cls

                # do nothing if there are no values to make predictions for
                if class_mask.sum() == 0:
                    continue

                # set background values to 0
                # if model is None:
                #     output[class_mask] = 0
                #     continue

                # make prediction for the other classes
                pred = model(boa[class_mask], times[class_mask], mask[class_mask])
                y = torch.argmax(pred, dim=1)
                output[class_mask] = y + self.class_offset[cls-1]  # add one, because 0 is already background

            output[:] = self.lookup_table[output]

            return one_hot(output, num_classes=self.num_classes)

    @classmethod
    def from_jitted_weights(cls, parent_model_path, child_model_paths, num_classes):
        """For stand-alone runnable / jitted pytorch models."""
        parent_model = torch.jit.load(parent_model_path)
        child_models = [torch.jit.load(p) if p is not None else None for p in child_model_paths]
        return cls(parent_model, child_models, num_classes)

    @classmethod
    def from_checkpoints(cls, parent_class, parent_model_path, child_class, child_model_paths, num_classes,
                         device="cpu", **kwargs):
        """For pytorch lighning modules and checkpoints."""
        parent_model = parent_class.load_from_checkpoint(parent_model_path, map_location=device, **kwargs)
        child_models = [child_class.load_from_checkpoint(p, map_location=device, **kwargs) if p is not None else None for p in child_model_paths]
        return cls(parent_model, child_models, num_classes)

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

        if tmin_inference < tmin_data \
                or tmax_inference < tmin_data \
                or tmin_inference > tmax_data \
                or tmax_inference > tmax_data \
                or tmin_inference > tmax_inference \
                or tmin_data > tmax_data:
            raise ValueError("Invalid time range for inference or data loading.")

        boa_filenames = glob(os.path.join(input_folder, "*.tif"))

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

        output = torch.zeros(h * w, dtype=torch.uint8, device=self.parent_model.device)

        parent_output = predict_on_batches(self.parent_model, all_boas, times, validity_mask, n_obs, mean, stddev,
                                           batch_size, inference_date_mask, verbose)

        output[:] = parent_output

        # make prediction for the other classes
        for (cls, model) in enumerate(self.child_models, start=1):
            class_mask = parent_output == cls
            class_mask_cpu = class_mask.cpu().numpy()

            # do nothing if there are no values to make predictions for
            if class_mask.sum() == 0:
                continue

            child_output = predict_on_batches(model, all_boas[class_mask_cpu], times, validity_mask[class_mask_cpu],
                                              n_obs[class_mask_cpu], mean, stddev, batch_size,
                                              inference_date_mask, verbose)

            output[class_mask] = child_output + self.class_offset[cls-1]  # add one, because 0 is already background

        output[:] = self.lookup_table[output.to(torch.int32)]

        output = output.reshape(h, w).cpu().numpy()

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
