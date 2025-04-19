# due to some bug in torch this has to be at the top
from .datasets import InMemoryTimeSeriesDataset, CLOUD_OR_NODATA, JLDS

import time
import torch
import random
import numpy as np
import pytorch_lightning as L
from .augmentations import augment_boa_and_time


def filepath_to_classname(path):
    return path.split('/')[-2]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32  # the initial seed differs for each worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TimeSeriesClassificationDataModule(L.LightningDataModule):
    def __init__(self,
                 input_file: str,
                 dbname: str,
                 batch_size: int,
                 val_file: str = "",
                 sequence_length: int = 32,
                 satellite_input_channels: int = 10,
                 quality_mask: int = CLOUD_OR_NODATA,
                 class_mapping: dict[str, str] = None,
                 return_mode: str = "random",
                 time_encoding: str = "doy",
                 num_workers: int = 8,
                 train_split: float = 0.7,
                 seed: int = 42,
                 where: str = "",
                 val_where: str = "none",
                 append_ndvi: bool = False,
                 eliminate_nodata: bool = False,
                 pickle_path: str = "/tmp",
                 mean=np.zeros(10),
                 stddev=np.ones(10) * 10000,
                 train_ids=None,
                 val_ids=None,
                 augmentation_kwargs=None
                 ):
        super().__init__()
        self.input_filepath = input_file
        self.val_file = val_file or input_file
        self.dbname = dbname
        self.sequence_length = sequence_length
        self.satellite_input_channels = satellite_input_channels
        self.quality_mask = quality_mask
        self.batch_size = batch_size
        self.class_mapping = class_mapping
        self.return_mode = return_mode
        self.time_encoding = time_encoding
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self.where = where
        self.val_where = val_where if val_where != "none" else where
        self.append_ndvi = append_ndvi
        self.eliminate_nodata = eliminate_nodata
        self.pickle_path = pickle_path
        self.mean = np.array(mean)
        self.stddev = np.array(stddev)
        self.training_data = None
        self.val_data = None
        self.is_setup = False
        self.prepare_data_per_node = True
        self.classes_ = None
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.augmentation_kwargs = augmentation_kwargs or {}
        self.rng = np.random.default_rng(42)

    def train_augmentation(self, boa, time, doy, mean, stddev):
        return augment_boa_and_time(boa, time, doy, mean, stddev, rng=self.rng, **self.augmentation_kwargs)

    def setup(self, stage=None) -> None:
        if self.is_setup:
            return

        if self.train_ids is None:
            # the brackets around 'where' are fuckin important
            # because of operator precedence!!!
            train_where = f"({self.where}) AND is_train = TRUE" if self.where else "is_train = TRUE"
        else:
            train_where = self.where

        if self.val_ids is None:
            val_where = f"({self.val_where}) AND is_train = FALSE" if self.val_where else "is_train = FALSE"
        else:
            val_where = self.val_where

        print(f"Where clause for loading training data: {train_where}")
        print(f"Where clause for loading validation data: {val_where}")

        print(f"Loading training dataset.")
        t0 = time.time()
        self.training_data = InMemoryTimeSeriesDataset(self.input_filepath,
                                                       self.dbname,
                                                       self.train_augmentation,
                                                       self.sequence_length,
                                                       self.satellite_input_channels,
                                                       self.quality_mask,
                                                       class_mapping=self.class_mapping,
                                                       return_mode=self.return_mode,
                                                       time_encoding=self.time_encoding,
                                                       where=train_where,
                                                       append_ndvi=self.append_ndvi,
                                                       eliminate_nodata=self.eliminate_nodata,
                                                       plot_ids=self.train_ids,
                                                       mean=self.mean,
                                                       stddev=self.stddev
                                                       )
        print(f"Loading training ds took {time.time() - t0}s.")

        print(f"Loading val dataset.")
        t0 = time.time()
        self.val_data = InMemoryTimeSeriesDataset(self.val_file,
                                                  self.dbname,
                                                  sequence_length=self.sequence_length,
                                                  satellite_input_channels=self.satellite_input_channels,
                                                  quality_mask=self.quality_mask,
                                                  class_mapping=self.class_mapping,
                                                  return_mode=self.return_mode,
                                                  time_encoding=self.time_encoding,
                                                  where=val_where,
                                                  append_ndvi=self.append_ndvi,
                                                  eliminate_nodata=self.eliminate_nodata,
                                                  plot_ids=self.val_ids,
                                                  mean=self.mean,
                                                  stddev=self.stddev
                                                  )
        print("Classes in val / test set: ", np.unique(self.val_data.df["species"]))
        print(f"Loading val ds took {time.time() - t0}s.")

        self.is_setup = True

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(self.training_data,
                                           pin_memory=True,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           persistent_workers=False,
                                           generator=g,
                                           worker_init_fn=seed_worker)

    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(self.val_data,
                                           pin_memory=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           persistent_workers=False,
                                           generator=g,
                                           worker_init_fn=seed_worker)

    def test_dataloader(self):
        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(self.val_data,
                                           pin_memory=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           persistent_workers=False,
                                           generator=g,
                                           worker_init_fn=seed_worker)

    @property
    def num_classes(self):
        if self.class_mapping is not None:
            self.classes_ = list(sorted(set(self.class_mapping.values())))
            return len(self.classes_)
        else:
            # in this case we have to get the train ids and look up the classes in via sql query
            raise NotImplementedError("class mapping must be given")
        
    @property
    def classes(self):
        if self.class_mapping is not None:
            self.classes_ = list(sorted(set(self.class_mapping.values())))
            return self.classes_
        else:
            raise NotImplementedError("class mapping must be given")

    @property
    def loss_weights(self):
        if not self.is_setup:
            self.setup("fit")
            self.is_setup = True
        return self.training_data.compute_class_weights()


class TimeSeriesClassificationDataModuleJL(L.LightningDataModule):
    def __init__(self,
                 input_file: str,
                 dbname: str,
                 batch_size: int,
                 val_file: str = "",
                 sequence_length: int = 32,
                 satellite_input_channels: int = 10,
                 quality_mask: int = CLOUD_OR_NODATA,
                 class_mapping: dict[str, str] = None,
                 return_mode: str = "random",
                 time_encoding: str = "doy",
                 num_workers: int = 8,
                 train_split: float = 0.7,
                 seed: int = 42,
                 where: str = "",
                 val_where: str = "none",
                 append_ndvi: bool = False,
                 eliminate_nodata: bool = False,
                 pickle_path: str = "/tmp",
                 mean=np.zeros(10),
                 stddev=np.ones(10) * 10000,
                 train_ids=None,
                 val_ids=None,
                 augmentation_kwargs=None
                 ):
        super().__init__()
        self.input_filepath = input_file
        self.val_file = val_file or input_file
        self.dbname = dbname
        self.sequence_length = sequence_length
        self.satellite_input_channels = satellite_input_channels
        self.quality_mask = quality_mask
        self.batch_size = batch_size
        self.class_mapping = class_mapping
        self.return_mode = return_mode
        self.time_encoding = time_encoding
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self.where = where
        self.val_where = val_where if val_where != "none" else where
        self.append_ndvi = append_ndvi
        self.eliminate_nodata = eliminate_nodata
        self.pickle_path = pickle_path
        self.mean = np.array(mean)
        self.stddev = np.array(stddev)
        self.training_data = None
        self.val_data = None
        self.is_setup = False
        self.prepare_data_per_node = True
        self.classes_ = None
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.augmentation_kwargs = augmentation_kwargs or {}
        self.rng = np.random.default_rng(42)

    def setup(self, stage=None) -> None:
        if self.train_ids is None:
            if self.where:
                # the brackets around 'where' are fuckin important
                # because of operator precedence!!!
                train_where = f"({self.where}) AND is_train = TRUE AND (qai & {self.quality_mask}) = 0"
            else:
                train_where = f"is_train = TRUE AND (qai & {self.quality_mask}) = 0"
        else:
            train_where = f"({self.where}) AND (qai & {self.quality_mask}) = 0"

        if self.val_ids is None:
            if self.val_where:
                val_where = f"({self.val_where}) AND is_train = FALSE AND (qai & {self.quality_mask}) = 0"
            else:
                val_where = f"is_train = FALSE AND (qai & {self.quality_mask}) = 0"
        else:
            val_where = f"({self.val_where}) AND (qai & {self.quality_mask}) = 0"

        print(f"Where clause for loading training data: {train_where}")
        print(f"Where clause for loading validation data: {val_where}")

        print(f"Loading training dataset.")
        t0 = time.time()
        self.training_data = JLDS(self.input_filepath,
                                  train_where,
                                  self.augmentation_kwargs,
                                  self.class_mapping,
                                  sequence_length=self.sequence_length,
                                  satellite_input_channels=self.satellite_input_channels,
                                  time_encoding=self.time_encoding,
                                  plot_ids=self.train_ids,
                                  mean=self.mean,
                                  stddev=self.stddev,
                                  seed=self.seed,
                                  batch_size=self.batch_size,
                                  )
        print(f"Loading training ds took {time.time() - t0}s.")

        print(f"Loading val dataset.")
        t0 = time.time()
        self.val_data = JLDS(self.input_filepath,
                             val_where,
                             self.augmentation_kwargs,
                             self.class_mapping,
                             sequence_length=self.sequence_length,
                             satellite_input_channels=self.satellite_input_channels,
                             time_encoding=self.time_encoding,
                             plot_ids=self.val_ids,
                             mean=self.mean,
                             stddev=self.stddev,
                             seed=self.seed,
                             batch_size=self.batch_size,
                             )
        print("Classes in val / test set: ", np.unique(self.val_data.ds.classes))
        print(f"Loading val ds took {time.time() - t0}s.")

        self.is_setup = True

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data,
                                           pin_memory=True,
                                           batch_size=None,
                                           num_workers=0,
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           pin_memory=True,
                                           batch_size=None,
                                           num_workers=0,
                                           )

    def test_dataloader(self):
        return self.val_dataloader()

    @property
    def num_classes(self):
        if self.class_mapping is not None:
            self.classes_ = list(sorted(set(self.class_mapping.values())))
            return len(self.classes_)
        else:
            # in this case we have to get the train ids and look up the classes in via sql query
            raise NotImplementedError("class mapping must be given")

    @property
    def classes(self):
        if self.class_mapping is not None:
            self.classes_ = list(sorted(set(self.class_mapping.values())))
            return self.classes_
        else:
            raise NotImplementedError("class mapping must be given")
