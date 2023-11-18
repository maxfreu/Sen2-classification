import os.path
import time
import torch
import shutil
import numpy as np
from . import utils
import pytorch_lightning as L
from torch.utils.data import Dataset
from .datasets import InMemoryImageClassificationDataset, InMemoryTimeSeriesDataset, CLOUD_OR_NODATA
from .datasets import MultiModalDataset
import torchvision.transforms as T
import pandas as pd
import sqlite3
from torch.utils.data import random_split


def filepath_to_classname(path):
    return path.split('/')[-2]


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
        super().__init__()
        self.sqlite_path = sqlite_path
        self.dbname = dbname
        self.sequence_length = sequence_length
        self.satellite_input_channels = satellite_input_channels
        self.quality_mask = quality_mask
        self.batch_size = batch_size
        self.class_mapping = class_mapping
        self.return_mode = return_mode
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self.training_data = None
        self.val_data = None
        self.is_setup = False
        self.prepare_data_per_node = True
        self.classes_ = None

    def prepare_data(self) -> None:
        tmppath = f"/tmp/{self.dbname}.sqlite"
        if not os.path.isfile(tmppath) or os.path.getsize(tmppath) == 0:
            shutil.copy2(self.sqlite_path, tmppath)

    def get_train_val_ids(self):
        tmppath = f"/tmp/{self.dbname}.sqlite"
        rstate = np.random.default_rng(self.seed)
        conn = sqlite3.connect(tmppath)
        tnrs = list(pd.read_sql_query(f"SELECT DISTINCT tnr FROM {self.dbname}", conn).tnr)
        rstate.shuffle(tnrs)
        conn.close()

        train_ids, val_ids = random_split(tnrs, [self.train_split, 1-self.train_split],
                                          generator=torch.Generator().manual_seed(self.seed))
        return train_ids, val_ids
        
    def setup(self, stage: str) -> None:
        if self.is_setup:
            return
        
        tmppath = f"/tmp/{self.dbname}.sqlite"
        train_ids, val_ids = self.get_train_val_ids()

        print(f"Loading training dataset.")
        t0 = time.time()
        self.training_data = InMemoryTimeSeriesDataset(tmppath,
                                                       self.dbname,
                                                       self.sequence_length,
                                                       self.satellite_input_channels,
                                                       self.quality_mask,
                                                       class_mapping=self.class_mapping,
                                                       return_mode=self.return_mode,
                                                       plot_ids=train_ids)
        print(f"Loading training ds took {time.time() -t0}s.")

        print(f"Loading val dataset.")
        t0 = time.time()
        self.val_data = InMemoryTimeSeriesDataset(tmppath,
                                                  self.dbname,
                                                  self.sequence_length,
                                                  self.satellite_input_channels,
                                                  self.quality_mask,
                                                  class_mapping=self.class_mapping,
                                                  return_mode=self.return_mode,
                                                  plot_ids=val_ids)
        print(f"Loading val ds took {time.time() - t0}s.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, pin_memory=True, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def num_classes(self):
        if self.class_mapping is not None:
            self.classes_ = list(sorted(set(self.class_mapping.values())))
            return len(self.classes_)
        else:
            # in this case we have to get the train ids and look up the classes in via sql query
            raise NotImplementedError("class mapping must be given")
        
        # if not self.is_setup:
        #     self.prepare_data()
        #     self.setup("fit")
        #     self.is_setup = True
        # return self.training_data.compute_class_weights()

    @property
    def classes(self):
        if self.class_mapping is not None:
            self.classes_ = list(sorted(set(self.class_mapping.values())))
            return self.classes_
        else:
            raise NotImplementedError("class mapping must be given")

    @property
    def class_counts(self):
        counts = []
        if self.class_mapping is None:
            mapped_classes = np.array(list(self.grouped_df.ba.first()))
        else:
            mapped_classes = np.array([self.class_mapping[int(ba)] for ba in np.array(list(self.grouped_df.ba.first()))])
        for cls in self.classes:
            counts.append((mapped_classes == cls).sum())
        return counts
    
    @property
    def class_weights(self):
        if not self.is_setup:
            self.prepare_data()
            self.setup("fit")
            self.is_setup = True
        return self.training_data.compute_class_weights()

    # def teardown(self, stage: str):
    #     try:
    #         os.remove(f"/tmp/{self.dbname}.sqlite")
    #     except FileNotFoundError:
    #         pass


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
