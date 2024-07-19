import os
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import GeneratorDataset, read_img, array_to_tif
from pytorch_lightning import LightningModule


class HierarchicalModel(LightningModule):
    def __init__(self, parent_model, child_models, num_classes: list):
        """
        Takes a list of models. Input is passed through the parent model first to determine which child model to use.
        The parent model's output is argmax'ed and based on that the respective child model is called with the input.
        The returned class indices will be in the same order as the child models, just offset by their cumulated sum. If
        there is a background class (i.e. None is passed as child model in some position), 0 will be the background class
        and all other output will be in ascending order from 1 up.

        In forward call it dispatches to child models via for loop, probably super slow.

        This model is not differentiable and therefore not trainable!

        Args:
            parent_model: The parent model used to decide which child model to use
            child_models: The child models that output the final classification. The child model order has to match the
                order of classes of the parent model. E.g. argmaxed parent output 1 will call the child model with index 1.
            num_classes (list): Number of output classes per child model, in the same order. Background class counts as one.
        """
        super().__init__()
        self.parent_model = parent_model
        self.child_models = child_models
        self.class_offset = torch.tensor(np.cumsum(num_classes), requires_grad=False)

        print(f"class offset {self.class_offset}")

    def forward(self, x):
        # x is a tuple of (boa, doy, mask) values
        # each of them containing items of a batch
        with torch.no_grad():
            # make prediction with parent model and argmax the values to obtain the class indices
            parent_classes = torch.argmax(self.parent_model(x), dim=1)
            output = torch.zeros_like(parent_classes)

            # select all items from batch equal to cls where cls ranges from 0 to the length of child models
            # including the None which indicates the background class position in the parent model
            for cls in range(len(self.child_models)):
                mask = parent_classes == cls
                model = self.child_models[cls]

                # do nothing if there are no values to make predictions for
                if mask.sum() == 0:
                    continue

                # set background values to 0
                if model is None:
                    output[mask] = 0
                    continue

                # make prediction for the other classes
                y = torch.argmax(model((x[0][mask], x[1][mask], x[2][mask])), dim=1)
                output[mask] = y + self.class_offset[cls-1]

            return output

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

    # ok duplicate code bomb is coming
    def predict_timeseries_slow(self, input_folder, qai, seq_len, output_filepath=None, save=True,
                           t0=datetime.date(2015, 1, 1), batch_size=16, num_workers=0):
        self.parent_model.eval()
        for m in self.child_models:
            if m is not None:
                m.eval()

        files = os.listdir(input_folder)
        boa_filenames = list(sorted(filter(lambda x: 'SEN2' in x and 'BOA' in x, files)))[-seq_len:]

        dates = [datetime.datetime.strptime(s[:8], '%Y%m%d').date() for s in boa_filenames]
        days_since_t0 = np.array([(date - t0).days for date in dates])

        print("Loading images")
        sample_boa = read_img(os.path.join(input_folder, boa_filenames[0]), dim_ordering="HWC", dtype=np.int16)
        h, w, c = sample_boa.shape
        all_boas = np.empty((h, w, seq_len, c), dtype=np.int16)

        # read all the files
        for (i, f) in enumerate(boa_filenames):
            fname = os.path.join(input_folder, f)
            img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
            all_boas[:, :, i, :] = img

        all_boas = all_boas.reshape((-1, seq_len, c))

        if qai > 0:
            qais = list(sorted(filter(lambda x: 'SEN2' in x and 'QAI' in x, files)))[-seq_len:]
            validity_mask = np.empty((h, w, seq_len, 1), dtype=bool)
            for (i, f) in enumerate(qais):
                fname = os.path.join(input_folder, f)
                img = read_img(fname, dim_ordering="HWC", dtype=np.uint32)
                validity_mask[:, :, i, :] = (img & qai) == 0

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
                    yield torch.from_numpy(all_boas[i]), torch.from_numpy(days_since_t0), torch.from_numpy(
                        transformer_mask)
                    i += 1

        gen = GeneratorDataset(pixel_generator())
        dl = DataLoader(gen, batch_size=batch_size, num_workers=num_workers)
        output = np.zeros(w * h, dtype=np.uint8)

        print("Starting prediction")
        print(f"0 / {w * h // batch_size}")
        with torch.no_grad():
            for (i, batch) in enumerate(dl):
                if i % 100 == 0:
                    print(f"{i} / {w * h // batch_size}")
                boa, doy, mask = batch
                boa = boa.to(self.device)
                doy = doy.to(self.device)
                mask = mask.to(self.device)
                pred = self((boa, doy, mask))
                start = i * batch_size
                stop = start + pred.shape[0]
                output[start:stop] = pred.cpu().numpy()

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

    def predict_timeseries(self, input_folder, qai, seq_len, output_filepath=None, save=True, t0=datetime.date(2015, 1, 1), batch_size=16, num_workers=0):
        self.parent_model.eval()
        for m in self.child_models:
            if m is not None:
                m.eval()

        files = os.listdir(input_folder)
        boa_filenames = list(sorted(filter(lambda x: 'SEN2' in x and 'BOA' in x, files)))[-seq_len:]

        dates = [datetime.datetime.strptime(s[:8], '%Y%m%d').date() for s in boa_filenames]
        days_since_t0 = np.array([(date - t0).days for date in dates])

        print("Loading images")
        sample_boa = read_img(os.path.join(input_folder, boa_filenames[0]), dim_ordering="HWC", dtype=np.int16)
        h, w, c = sample_boa.shape
        all_boas = np.empty((h, w, seq_len, c), dtype=np.int16)

        # eat ALL the RAM :)))
        for (i, f) in enumerate(boa_filenames):
            fname = os.path.join(input_folder, f)
            img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
            all_boas[:, :, i, :] = img

        all_boas = all_boas.reshape((-1, seq_len, c))

        if qai > 0:
            qais = list(sorted(filter(lambda x: 'SEN2' in x and 'QAI' in x, files)))[-seq_len:]
            validity_mask_raw = np.empty((h, w, seq_len, 1), dtype=bool)
            for (i,f) in enumerate(qais):
                fname = os.path.join(input_folder, f)
                img = read_img(fname, dim_ordering="HWC", dtype=np.uint32)
                validity_mask_raw[:,:,i,:] = (img & qai) == 0

            validity_mask = validity_mask_raw.reshape((-1, seq_len))
            n_obs = np.sum(validity_mask, axis=1)

            def pixel_generator(validity_mask):
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

            gen = GeneratorDataset(pixel_generator(validity_mask))

        else:
            def pixel_generator():
                i = 0
                while i < all_boas.shape[0]:
                    transformer_mask = np.ones(seq_len, dtype=int)
                    yield torch.from_numpy(all_boas[i]), torch.from_numpy(days_since_t0), torch.from_numpy(transformer_mask)
                    i += 1

            gen = GeneratorDataset(pixel_generator())

        dl = DataLoader(gen, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        parent_prediction = np.zeros(w*h, dtype=np.uint8)
        final_prediction = np.zeros(w*h, dtype=np.uint8)

        print("Starting prediction")
        print(f"0 / {w*h//batch_size}")
        with torch.no_grad():
            for (i,batch) in enumerate(dl):
                print(f"{i} / {w * h // batch_size}")
                boa, doy, mask = batch
                boa = boa.to(self.device)
                doy = doy.to(self.device)
                mask = mask.to(self.device)
                pred = torch.argmax(self.parent_model((boa, doy, mask)), dim=1)
                start = i*batch_size
                stop = start + pred.shape[0]
                parent_prediction[start:stop] = pred.cpu().numpy()

        parent_prediction = parent_prediction.reshape((h, w))

        for i, m in enumerate(self.child_models):
            if m is None:
                continue

            mask = parent_prediction == i
            this_class_validity_mask

