import os
import torch
import sqlite3
import duckdb
import numpy as np
import multiprocessing as mp
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
from .utils import default_image_loader
from os.path import basename, splitext
import datetime

# Force processing framework quality bits
# quality filters can be created by chaining the different flags with bit-wise "or" (|)
VALID        = 0b000000000000000
NODATA       = 0b000000000000001
CLOUD_BUFFER = 0b000000000000010
CLOUD_OPAQUE = 0b000000000000100
CLOUD_CIRRUS = 0b000000000000110
CLOUD_SHADOW = 0b000000000001000
SNOW         = 0b000000000010000
WATER        = 0b000000000100000
AOD_INT      = 0b000000001000000
AOD_HIGH     = 0b000000010000000
AOD_FILL     = 0b000000011000000
SUBZERO      = 0b000000100000000
SATURATION   = 0b000001000000000
SUN_LOW      = 0b000010000000000
ILLUMIN_LOW  = 0b000100000000000
ILLUMIN_POOR = 0b001000000000000
ILLUMIN_NONE = 0b001100000000000
SLOPED       = 0b010000000000000
WVP_NONE     = 0b100000000000000

CLOUD_OR_NODATA = NODATA | CLOUD_BUFFER | CLOUD_CIRRUS | CLOUD_OPAQUE | CLOUD_SHADOW


class InMemoryImageClassificationDataset(Dataset):
    def __init__(self,
                 file_paths: list[str],
                 filepath_to_classname,
                 augmentation,
                 classes: list[str] = None,
                 image_loader=default_image_loader,
                 class_mapping: dict[str, str] = None,
                 device="cpu",
                 nprocs=1,
                 seed=1337,
                 ):
        """
        file_paths: list[str],
        classes: list[str],
        filepath_to_classname,
        augmentation,
        image_loader = ,
        class_mapping: dict[str, str] = None,
        device="cpu",
        nprocs=1,
        seed=1337,
        """
        self.files = np.array(file_paths)
        self.device = device
        self.augmentation = augmentation
        self.class_mapping = class_mapping
        self.filepath_to_classname = filepath_to_classname
        self.seed = seed
        if class_mapping is not None and classes is None:
            self.classes = list(sorted(set(class_mapping.values())))
        elif class_mapping is None and classes is None:
            self.classes = list(sorted(np.unique([filepath_to_classname(p) for p in self.files])))
        else:
            self.classes = classes

        self.num_classes = len(self.classes)

        rstate = np.random.RandomState(seed=seed)
        rstate.shuffle(self.files)

        # decompress images in parallel and put them onto the desired device
        # this is still slow due to pickling and unpickling
        with mp.Pool(nprocs) as p:
            self.images = p.map(image_loader, self.files)

        self.images = [torch.from_numpy(img).to(device=self.device) for img in self.images]
        self.class_indices = np.array([self.filepath_to_classindex(path) for path in self.files])

    def class_index(self, classname):
        """Returns the numerical ID of a class."""
        return self.classes.index(classname)

    def class_name(self, class_index):
        """Returns class name at index."""
        return self.classes[class_index]

    def filepath_to_classindex(self, path):
        name = self.filepath_to_classname(path)
        if self.class_mapping is None:
            return self.class_index(name)
        else:
            mapped_class = self.class_mapping[name]
            return self.class_index(mapped_class)

    def class_counts(self):
        counts = []
        for cls in self.classes:
            counts.append((self.class_indices == self.class_index(cls)).sum())
        return counts

    def compute_class_weights(self):
        return len(self) / np.array(self.class_counts())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        class_index = self.class_indices[index]
        img = self.images[index]
        return self.augmentation(img), class_index


class InMemoryTimeSeriesDataset(Dataset):
    def __init__(self,
                 input_filepath,
                 dbname,
                 augmentation=None,
                 sequence_length=32,
                 satellite_input_channels=10,
                 quality_mask=CLOUD_OR_NODATA,
                 min_obs: int = 12,
                 class_mapping: dict = None,
                 return_mode: str = "random",
                 pos_encode: str = "doy",
                 plot_ids: tuple = None,
                 num_workers: int = 0,
                 where: str = ""
                 ):
        """ In-memory dataset for time series classification.

        Args:
            input_filepath: Path to the sqlite or parquet dataset file
            dbname: Name of the contained table if input file is sqlite
            augmentation: Optional augmentation function that acts on a single BOA observation
            sequence_length: The maximum sequence length
            satellite_input_channels: Number of satellite image channels
            quality_mask: Bit-encoded quality mask (see webpage of FORCE satellite processing toolbox)
            min_obs: Minimum number of observations / time points to return
            class_mapping: Dictionary optionally mapping tree species ids to classes
            return_mode: Can be 'random' (default) to return a sequence with at maximum 'sequence_length' 
                samples drawn from a random starting point in time onwards, 'single' to return data from a 
                random single year, 'last' to return at most the latest 'sequence_length' points or 'all' to return
                all available data for a given tree.
                WARNING: If you choose 'all', you have to ensure that the sequence length is long enough to store the
                longest time series!
            pos_encode: How to encode the temporal information. Can be either 'doy' (default) to give all time stamps as
                day of year or 'absolute' to encode the number of days passed since 2015-01-01.
            plot_ids: Plot ids (from tnr field) to load; can be used for training / val selection
            num_workers: Dataloader worker count
            where: SQL Where clause to filter data while loading, e.g. `species > 100 AND is_pure = TRUE` to select only
                deciduous trees from pure stands.
        """
        if return_mode not in ("random", "single", "last", "all"):
            raise RuntimeError(f"Please provide the correct return mode; either random, single, last or all. Received {return_mode}")
        if pos_encode not in ("doy", "absolute"):
            raise RuntimeError(f"Wrong position embedding. Choose doy or absolute. Received {pos_encode}")
        self.sequence_length = sequence_length
        self.augmentation = augmentation
        self.satellite_input_channels = satellite_input_channels
        self.min_obs = min_obs
        self.return_mode = return_mode
        self.pos_encode = pos_encode

        input_filetype = os.path.splitext(os.path.basename(input_filepath))[1]

        columns = "tree_id, species, boa, qai, time, doy"
        if input_filetype == ".sqlite":
            conn = sqlite3.connect(input_filepath)
            conn.text_factory = bytes  # this makes sqlite return strings as bytes that we can parse via numpy

            # load all data or only some plots
            if plot_ids is None:
                if where:
                    self.df = pd.read_sql_query(f"SELECT {columns} FROM {dbname} WHERE {where}", conn)
                else:
                    self.df = pd.read_sql_query(f"SELECT {columns} FROM {dbname}", conn)
            else:
                if where:
                    self.df = pd.read_sql_query(f"SELECT {columns} FROM {dbname} WHERE tnr IN {tuple(plot_ids)} AND {where}", conn)
                else:
                    self.df = pd.read_sql_query(f"SELECT {columns} FROM {dbname} WHERE tnr IN {tuple(plot_ids)}", conn)

            conn.close()

        elif input_filetype == ".parquet" or input_filetype == ".parq":
            if plot_ids is None:
                if where:
                    self.df = duckdb.query(f"select {columns} from '{input_filepath}' where {where}").df()
                else:
                    self.df = duckdb.query(f"select {columns} from '{input_filepath}'").df()
            else:
                if where:
                    self.df = duckdb.query(f"select {columns} from '{input_filepath}' WHERE tnr IN {tuple(plot_ids)} AND {where}").df()
                else:
                    self.df = duckdb.query(f"select {columns} from '{input_filepath}' WHERE tnr IN {tuple(plot_ids)}").df()

        self.df = self.df[(self.df.qai & quality_mask) == 0]

        # convert the bytes to a numpy array
        # 16 bit is a storage format - we convert it to 32 bit for faster calculations at the cost of RAM
        self.df.boa = [np.frombuffer(x, dtype=np.int16).astype(np.int32) for x in self.df.boa]
        self.df.time = [datetime.date.fromtimestamp(t) for t in self.df.time]
        self.df["dayssinceepoch"] = [(t - datetime.date(2015,1,1)).days for t in self.df.time]
        self.df["year"] = [t.year for t in self.df.time]
        self.df.sort_values("time", inplace=True)
        self.grouped_df = self.df.groupby("tree_id")
        self.tree_ids = list(self.grouped_df.groups.keys())
        np.random.shuffle(self.tree_ids)

        self.num_workers = num_workers if num_workers > 0 else 1
        self.class_mapping = class_mapping
        if class_mapping is not None:
            self.classes = list(sorted(set(class_mapping.values())))
        else:
            self.classes = list(sorted(self.df.species.unique()))

        self.num_classes = len(self.classes)

    def class_index(self, classname):
        """Returns the numerical ID of a class."""
        return self.classes.index(classname)

    def class_name(self, class_index):
        """Returns class name at index."""
        return self.classes[class_index]

    def class_counts(self):
        counts = []
        if self.class_mapping is None:
            mapped_classes = np.array(list(self.grouped_df.species.first()))
        else:
            mapped_classes = np.array([self.class_mapping[int(sp)] for sp in np.array(list(self.grouped_df.species.first()))])
        for cls in self.classes:
            counts.append((mapped_classes == cls).sum())
        return counts

    def compute_class_weights(self):
        return len(self) / np.array(self.class_counts())

    def get_tree_data(self, tree_id):
        subgroup = self.grouped_df.get_group(tree_id)
        # subgroup = subgroup.sort_values("time")

        # return single year or all available data
        if self.return_mode == "single":
            year_group = subgroup.groupby("year")
            available_years = np.array(list(year_group.groups.keys()))
            # select years with enough observations, so that the transformer has
            # the chance to learn something
            years_with_enough_obs = available_years[year_group.count().tnr >= self.min_obs]

            if len(years_with_enough_obs) > 0:
                random_year = np.random.choice(years_with_enough_obs)  # select random year as augmentation
            else:
                random_year = np.random.choice(available_years)

            selection = year_group.get_group(random_year)[:self.sequence_length]
        elif self.return_mode == "random":
            n = subgroup.shape[0]
            if n > self.sequence_length:
                start = np.random.randint(0, high=n-self.sequence_length)
                selection = subgroup[start:start+self.sequence_length]
            else:
                selection = subgroup
        elif self.return_mode == "last":
            n = subgroup.shape[0]
            if n > self.sequence_length:
                selection = subgroup[-self.sequence_length:]
            else:
                selection = subgroup
        elif self.return_mode == "all":
            selection = subgroup
        else:
            raise RuntimeError(f"Return mode must be one of random, single, last or all, but is {self.return_mode}")

        n_obs = min(selection.shape[0], self.sequence_length)

        boa = np.zeros((self.sequence_length, self.satellite_input_channels), dtype=np.float32)
        
        if self.augmentation is not None:
            augmented_boas = [self.augmentation(obs) for obs in selection.boa[:n_obs]]
            boa[:n_obs, :] = np.stack(augmented_boas)
        else:
            boa[:n_obs, :] = np.stack(selection.boa[:n_obs])

        times = np.zeros(self.sequence_length, dtype=int)

        if self.pos_encode == "doy":
            times[:n_obs] = selection.doy[:n_obs]
        else:
            # if we work with the entire time series, we take the
            # days since beginning of launch year of sentinel 2
            times[:n_obs] = selection.dayssinceepoch[:n_obs]

        mask = np.zeros(self.sequence_length, dtype=bool)
        mask[n_obs:] = True

        cls = selection.species.iloc[0]
        if self.class_mapping is not None:
            cls = self.class_mapping[int(cls)]

        return tree_id, boa / 10000, times, mask, self.class_index(cls)

    def __getitem__(self, item):
        tree_id = self.tree_ids[item]  # indirection to be able to index into the dataset with 0..len-1
        return self.get_tree_data(tree_id)

    def __len__(self):
        """Returns the number of samples per worker!!!"""
        return len(self.tree_ids) // self.num_workers
    
    @staticmethod
    def worker_init_fn(worker_id):
        """Here we highly reduce the memory footprint by splitting the dataset across workers."""
        worker_info = torch.utils.data.get_worker_info()
        print(f"setting up worker {worker_id}")
        dataset = worker_info.dataset
        all_ids = dataset.tree_ids
        subset = np.array_split(all_ids, worker_info.num_workers)[worker_id]
        print(f"worker {worker_id} subset size: {len(subset)} / {len(all_ids)}")

        dataset.tree_ids = subset
        dataset.df = dataset.df[dataset.df["tree_id"].isin(subset)]
        dataset.grouped_df = dataset.df.groupby("tree_id")
    
    # @staticmethod
    # def get_random_365_days(startdate, enddate):
    #     dates_bet = enddate - startdate
    #     total_days = dates_bet.days
    #     random_day = np.random.choice(total_days)
    #     res = startdate + datetime.timedelta(days=int(random_day))
    #     return res, res + datetime.timedelta(days=365)


class MultiModalDataset(IterableDataset):
    """Train and test split is dictated by the image dataset."""
    def __init__(self,
                 file_paths: list[str],
                 filepath_to_classname,
                 image_augmentation,
                 sqlite_path,
                 dbname,
                 classes: list[str] = None,
                 image_loader=default_image_loader,
                 class_mapping: dict[str, str] = None,
                 device="cpu",
                 nprocs=1,
                 seed=1337,
                 sequence_length=32,
                 satellite_input_channels=10,
                 quality_mask=NODATA | CLOUD_BUFFER | CLOUD_CIRRUS | CLOUD_OPAQUE | CLOUD_SHADOW
                 ):
        self.image_dataset = InMemoryImageClassificationDataset(file_paths,
                                                                filepath_to_classname,
                                                                image_augmentation,
                                                                classes,
                                                                image_loader,
                                                                class_mapping,
                                                                device,
                                                                nprocs,
                                                                seed)
        self.timeseries_ds = InMemoryTimeSeriesDataset(sqlite_path,
                                                       dbname,
                                                       sequence_length,
                                                       satellite_input_channels,
                                                       quality_mask)

    def __iter__(self):
        i = -1
        while i < (len(self.image_dataset) - 1):
            i += 1
            try:
                tree_id = int( splitext(basename(self.image_dataset.files[i]))[0] )
                x_transformer = self.timeseries_ds.get_tree_data(tree_id)[:3]

                class_index = self.image_dataset.class_indices[i]
                img = self.image_dataset.images[i]
                x_cnn = self.image_dataset.augmentation(img)

                yield x_cnn, x_transformer, class_index

            except KeyError:
                continue
