# due to some bug in torch this has to be at the top
from juliacall import Main as jl
from juliacall import Pkg as jlpkg

import os
import torch
import sqlite3
import duckdb
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset

jlpkg.activate("../S2Dataset")
jl.seval("using S2Dataset")
jl.seval("using Random")


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


def ndvi(v):
    """Computes the NDVI for v, where v contains the 10 S2 bands. Uses B8 and B4."""
    red = 2
    nir = 6
    return (v[nir] - v[red]) / (v[nir] + v[red] + 1e-7)


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
                 return_year=None,
                 time_encoding: str = "doy",
                 plot_ids: tuple = None,
                 num_workers: int = 0,
                 where: str = "",
                 append_ndvi: bool = False,
                 eliminate_nodata: bool = False,
                 mean=np.zeros(10),
                 stddev=np.ones(10) * 10000,
                 ):
        """ In-memory dataset for time series classification.

        Args:
            input_filepath: Path to the sqlite or parquet dataset file
            dbname: Name of the contained table if input file is sqlite
            augmentation: Optional augmentation function that acts on a single BOA observation.
                Function signature must be augmentation(boas, times, doy_or_not, mean, stddev, **kwargs).
            sequence_length: The maximum sequence length
            satellite_input_channels: Number of satellite image channels
            quality_mask: Bit-encoded quality mask (see webpage of FORCE satellite processing toolbox)
            min_obs: Minimum number of observations / time points to return
            class_mapping: Dictionary optionally mapping tree species ids to classes
            return_mode: Can be 'random' (default) to return a sequence with at maximum 'sequence_length' 
                samples drawn from a random starting point in time onwards, 'single' to return data from a 
                random single year, 'double' to return data from two consecutive years, 'last' to return at most the
                latest 'sequence_length' points or 'all' to return all available data for a given tree.
                WARNING: If you choose 'all', you have to ensure that the sequence length is long enough to store the
                longest time series!
            return_year (int or None): If return mode is single, `return_year` selects a specific year for which to
                return the data. Data will be returned, no matter how few it is. Default is None, which returns random
                years.
            time_encoding: How to encode the temporal information. Can be either 'doy' (default) to give all time stamps as
                day of year or 'absolute' to encode the number of days passed since 2015-01-01.
            plot_ids: Plot ids (from tnr field) to load; can be used for training / val selection
            num_workers: Dataloader worker count
            where: SQL Where clause to filter data while loading, e.g. `species > 100 AND is_pure = TRUE` to select only
                deciduous trees from pure stands.
            append_ndvi (bool): Whether to append the NDVI to the BOA values. If True, you have to increase the
                number of satellite channels by one.
            eliminate_nodata: Whether to remove all records where the first BOA band has a value smaller than -5000.
            mean: Numpy vector representing the band-wise mean of the data. Is used for normalization.
            stddev: Numpy vector representing the band-wise standard deviation of the data. Is used for normalization.
        """
        if return_mode not in ("random", "single", "double", "last", "all"):
            raise RuntimeError(f"Please provide the correct return mode; either random, single, last or all. Received {return_mode}")
        if time_encoding not in ("doy", "absolute"):
            raise RuntimeError(f"Wrong position embedding. Choose doy or absolute. Received {time_encoding}")
        self.sequence_length = sequence_length
        self.augmentation = augmentation
        self.satellite_input_channels = satellite_input_channels
        self.min_obs = min_obs
        self.return_mode = return_mode
        self.return_year = return_year
        self.time_encoding = time_encoding
        self.mean = mean
        self.stddev = stddev + 1e-7
        self.df = self.load_data(input_filepath, dbname, where, plot_ids)
        self.df = self.df[(self.df.qai & quality_mask) == 0]

        # 16 bit is a storage format - we convert it to 32 bit for faster calculations at the cost of RAM
        # mean = mean.astype(np.float32)
        # inv_stddev = 1 / (stddev.astype(np.float32) + 1e-7)
        # self.df.boa = [(np.frombuffer(x, dtype=np.int16).astype(np.float32) - mean) * inv_stddev for x in self.df.boa]
        # convert the bytes to a numpy array
        self.df.boa = self.convert_bytearrays_to_numpy(self.df.boa, append_ndvi)

        # throw out all values smaller -5000
        # would be faster to remove all this in the file itself...
        if eliminate_nodata:
            self.df = self.df[[x[0] > -5000 for x in self.df.boa]]

        self.df.time = [datetime.date.fromtimestamp(t) for t in self.df.time]
        self.df["dayssinceepoch"] = [(t - datetime.date(2015, 1, 1)).days for t in self.df.time]
        self.df["year"] = [t.year for t in self.df.time]

        # throw out all disturbance years before the NFI
        self.df.loc[self.df['disturbance_year'] < 2011, 'disturbance_year'] = 9999

        # filter out all records that either
        # come after a disturbance or
        # for which a disturbance has happened between 2011 and 2014 (before the image acquisition period)
        self.df = self.df[np.logical_and(self.df.year < self.df.disturbance_year,
                                         np.logical_not(
                                             (self.df.disturbance_year >= 2011) & (self.df.disturbance_year <= 2014)))]
        # or that are spruce and not continuously present until 2022
        self.df = self.df[np.logical_or(self.df.species != 10, self.df.present_2022)]
        self.df = self.df.drop(["disturbance_year", "present_2022", "qai"], axis=1)
        self.df.sort_values("time", inplace=True)
        self.df = self.df.drop(["time"], axis=1)
        self.grouped_df = self.df.groupby("tree_id")
        self.tree_ids = np.array(list(self.grouped_df.groups.keys()))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(self.tree_ids)

        self.num_workers = num_workers if num_workers > 0 else 1
        self.class_mapping = class_mapping
        if class_mapping is not None:
            self.classes = list(sorted(set(class_mapping.values())))
        else:
            self.classes = list(sorted(self.df.species.unique()))

        self.num_classes = len(self.classes)

    @staticmethod
    def convert_bytearrays_to_numpy(bytearray_series, append_ndvi):
        concatenated_bytes = b''.join(bytearray_series.to_list())
        boa = np.frombuffer(concatenated_bytes, dtype=np.int16).astype(np.float32).reshape(len(bytearray_series), -1)
        if append_ndvi:
            red = boa[:, 2]
            nir = boa[:, 6]
            ndvi = (nir - red) / (nir + red + 1e-5)
            ndvi = np.clip(ndvi, -1, 1)
            boa = np.column_stack((boa, ndvi))
        return list(boa)

    def load_data(self, input_filepath, dbname, where, plot_ids):
        input_filetype = os.path.splitext(os.path.basename(input_filepath))[1]

        columns = "tree_id, species, boa, qai, time, doy, disturbance_year, present_2022"
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
                    self.df = pd.read_sql_query(
                        f"SELECT {columns} FROM {dbname} WHERE tnr IN {tuple(plot_ids)} AND ({where})", conn)
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
                    self.df = duckdb.query(
                        f"select {columns} from '{input_filepath}' WHERE tnr IN {tuple(plot_ids)} AND ({where})").df()
                else:
                    self.df = duckdb.query(
                        f"select {columns} from '{input_filepath}' WHERE tnr IN {tuple(plot_ids)}").df()

        return self.df

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
        return len(self) / (np.array(self.class_counts()) + 1)

    # @profile
    def get_tree_data(self, tree_id):
        subgroup = self.grouped_df.get_group(tree_id)

        if self.return_mode == "single":
            available_years, counts = np.unique(subgroup.year, return_counts=True)

            if self.return_year is None:
                # select years with enough observations, so that the transformer has
                # the chance to learn something
                years_with_enough_obs = available_years[counts > self.min_obs]

                if len(years_with_enough_obs) > 0:
                    year = np.random.choice(years_with_enough_obs)  # select random year as augmentation
                else:
                    year = np.random.choice(available_years)
            else:
                year = self.return_year
            # selection = year_group.get_group(year)[:self.sequence_length]
            selection = subgroup[subgroup.year == year]
        elif self.return_mode == "double":
            available_years, counts = np.unique(subgroup.year, return_counts=True)

            if len(available_years) > 1:
                if self.return_year is None:
                    first_year_index = np.random.randint(low=0, high=len(available_years)-1)
                    first_year = available_years[first_year_index]
                    second_year = available_years[first_year_index+1]
                else:
                    first_year = self.return_year - 1
                    second_year = self.return_year

                first_year_data  = subgroup[subgroup.year == first_year]
                second_year_data = subgroup[subgroup.year == second_year]
                selection = pd.concat([first_year_data, second_year_data], axis=0)[:self.sequence_length]
            else:
                selection = subgroup[:self.sequence_length]
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
        times = np.zeros(self.sequence_length, dtype=np.int32)

        boa_selection = np.stack(selection.boa[:n_obs])

        if self.time_encoding == "doy":
            time_selection = selection.doy[:n_obs].to_numpy()
        else:
            time_selection = selection.dayssinceepoch[:n_obs].to_numpy()

        if self.augmentation is not None:
            augmented_boa, augmented_times = self.augmentation(boa_selection, time_selection, self.time_encoding == "doy", self.mean, self.stddev)
            n_obs = len(augmented_times)  # Update n_obs to match new augmented length
            boa[:n_obs] = augmented_boa
            times[:n_obs] = augmented_times
        else:
            boa[:n_obs] = (boa_selection - self.mean) / self.stddev
            times[:n_obs] = time_selection

        mask = np.zeros(self.sequence_length, dtype=bool)
        mask[n_obs:] = True

        cls = subgroup.species.iloc[0]
        if self.class_mapping is not None:
            cls = self.class_mapping[int(cls)]

        return tree_id, boa, times, mask, self.class_index(cls)

    def __getitem__(self, item):
        tree_id = self.tree_ids[item]  # indirection to be able to index into the dataset with 0..len-1
        return self.get_tree_data(tree_id)

    def __len__(self):
        """Returns the number of samples per worker if the worker init fn is used!!!"""
        return len(self.tree_ids)
    
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


class JLDS(IterableDataset):
    def __init__(self,
                 filepath,
                 where_stmt,
                 augmentation_kwargs,
                 class_mapping,
                 seed=42,
                 sequence_length=64,
                 satellite_input_channels=10,
                 plot_ids=None,
                 time_encoding="doy",
                 mean=np.zeros(10),
                 stddev=np.ones(10),
                 batch_size=128,
                 max_rows=0,
                 ):
        args = " ".join([f"{k}={v}," for (k,v) in augmentation_kwargs.items()])
        jl.seval(f"""
        function aug(boa, time, doy_encoding, data_mean, stddev) 
            augment_boa_and_time(boa,
                                 time,
                                 doy_encoding,
                                 data_mean,
                                 stddev;
                                 rng=Xoshiro({seed}),
                                 {args})
        end
        """)
        self.ds = jl.S2ParquetDataset(filepath,
                                      where_stmt,
                                      jl.aug,
                                      class_mapping,
                                      max_rows=max_rows,
                                      plot_ids=plot_ids,
                                      seed=seed,
                                      sequence_length=sequence_length,
                                      satellite_input_channels=satellite_input_channels,
                                      time_encoding=time_encoding,
                                      mean_val=mean,
                                      stddev_val=stddev
                                      )
        self.batch_size = batch_size
        self.state = None

    def __getitem__(self, i):
        return self.ds[i]

    def __len__(self):
        return len(self.ds)
    
    def __iter__(self):
        jl.shuffle(self.ds)
        self.state = None
        self.generator = jl.iterate_batched(self.ds, self.batch_size)
        return self
    
    def __next__(self):
        if self.state is None:
            res = jl.iterate(self.generator)
        else:
            res = jl.iterate(self.generator, self.state)

        if res:
            (tree_id, boa, times, mask, cls_), self.state = res
            tree_id = np.array(tree_id)
            boa     = np.array(boa).transpose((2,1,0))
            times   = np.array(times).transpose((1,0))
            mask    = np.array(mask).transpose((1,0))
            cls_    = np.array(cls_)
            return tree_id, boa, times, mask, cls_
        else:
            raise StopIteration
                
    def __repr__(self):
        return repr(self.ds)
