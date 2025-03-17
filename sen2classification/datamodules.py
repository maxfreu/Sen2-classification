import time
import torch
import random
import numpy as np
from . import utils
import pytorch_lightning as L
from torch.utils.data import Dataset
from .datasets import InMemoryImageClassificationDataset, InMemoryTimeSeriesDataset, CLOUD_OR_NODATA
from .datasets import MultiModalDataset
import torchvision.transforms as T
from .augmentations import augment_boa_and_time


def filepath_to_classname(path):
    return path.split('/')[-2]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32  # the initial seed differs for each worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ClassificationDataModule(L.LightningDataModule):
    def __init__(self, input_folder: str, seed: int, batch_size: int, class_mapping: dict[str, str] = None, num_workers: int = 8):
        super().__init__()
        self.input_folder = input_folder
        self.seed = seed
        self.batch_size = batch_size
        self.class_mapping = class_mapping
        self.num_workers = num_workers
        self.training_data = None
        self.val_data = None
        self.is_setup = False

    def setup(self, stage: str) -> None:
        files = utils.get_all_image_paths(self.input_folder)
        gen = torch.Generator().manual_seed(42)
        train_files, val_files = torch.utils.data.random_split(files, lengths=[0.7, 0.3], generator=gen)

        train_augmentations = torch.nn.Sequential(T.RandomVerticalFlip(),
                                                  T.RandomRotation((0, 270)),
                                                  T.ColorJitter(),
                                                  )

        self.training_data = InMemoryImageClassificationDataset(train_files,
                                                                class_mapping=self.class_mapping,
                                                                filepath_to_classname=filepath_to_classname,
                                                                augmentation=train_augmentations,
                                                                nprocs=10)

        self.val_data = InMemoryImageClassificationDataset(val_files,
                                                           class_mapping=self.class_mapping,
                                                           filepath_to_classname=filepath_to_classname,
                                                           augmentation=lambda x: x,
                                                           nprocs=10)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, shuffle=True, batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def num_classes(self):
        if not self.is_setup:
            self.setup("fit")
            self.is_setup = True
        return self.training_data.num_classes

    @property
    def class_weights(self):
        if not self.is_setup:
            self.setup("fit")
            self.is_setup = True
        return self.training_data.compute_class_weights()


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
                 time_shift: int = 4,
                 pickle_path: str = "/tmp",
                 mean=np.zeros(10),
                 stddev=np.ones(10) * 10000,
                 train_ids=None,
                 val_ids=None
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
        self.time_shift = time_shift
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
                                                       augment_boa_and_time,
                                                       self.sequence_length,
                                                       self.satellite_input_channels,
                                                       self.quality_mask,
                                                       class_mapping=self.class_mapping,
                                                       return_mode=self.return_mode,
                                                       time_encoding=self.time_encoding,
                                                       where=train_where,
                                                       append_ndvi=self.append_ndvi,
                                                       time_shift=self.time_shift,
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


class MockTimeSeriesClassificationDataModule(L.LightningDataModule):
    """This only exists to test the CLI setup."""
    def __init__(self,
                 sqlite_path: str,
                 dbname: str,
                 batch_size: int,
                 sequence_length: int = 32,
                 satellite_input_channels: int = 10,
                 quality_mask: int = CLOUD_OR_NODATA,
                 class_mapping: dict[str, str] = None,
                 return_mode: str = "random",
                 num_workers: int = 8,
                 train_split: float = 0.7,
                 seed: int = 42,
                 ):
        self.num_classes = 0
        self.class_weights = [0]
        self.classes = []


class MultiModalClassificationDataModule(L.LightningDataModule):
    def __init__(self,
                 input_folder: str,
                 sqlite_path: str,
                 dbname: str,
                 seed: int,
                 batch_size: int,
                 sequence_length: int,
                 satellite_input_channels: int,
                 class_mapping: dict[str, str] = None,
                 num_workers: int = 8,
                 ):
        super().__init__()
        self.input_folder = input_folder
        self.sqlite_path = sqlite_path
        self.dbname = dbname
        self.seed = seed
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.satellite_input_channels = satellite_input_channels
        self.class_mapping = class_mapping
        self.num_workers = num_workers
        self.training_data = None
        self.val_data = None
        self.is_setup = False

    def setup(self, stage: str) -> None:
        files = utils.get_all_image_paths(self.input_folder)
        gen = torch.Generator().manual_seed(42)
        train_files, val_files = torch.utils.data.random_split(files, lengths=[0.7, 0.3], generator=gen)

        train_augmentations = torch.nn.Sequential(T.RandomVerticalFlip(),
                                                  T.RandomRotation((0, 270)),
                                                  T.ColorJitter(),
                                                  )

        self.training_data = MultiModalDataset(train_files,
                                               class_mapping=self.class_mapping,
                                               filepath_to_classname=filepath_to_classname,
                                               image_augmentation=train_augmentations,
                                               nprocs=10,
                                               sqlite_path=self.sqlite_path,
                                               dbname=self.dbname,
                                               )

        self.val_data = MultiModalDataset(val_files,
                                          class_mapping=self.class_mapping,
                                          filepath_to_classname=filepath_to_classname,
                                          image_augmentation=lambda x: x,
                                          nprocs=10,
                                          sqlite_path=self.sqlite_path,
                                          dbname=self.dbname,
                                          )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, shuffle=False, batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def num_classes(self):
        if not self.is_setup:
            self.setup("fit")
            self.is_setup = True
        return self.training_data.image_dataset.num_classes

    @property
    def class_weights(self):
        if not self.is_setup:
            self.setup("fit")
            self.is_setup = True
        return self.training_data.image_dataset.compute_class_weights()
