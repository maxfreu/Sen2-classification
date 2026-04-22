import os
import ast
import yaml
import torch
import shutil
import sqlite3
import datetime
import warnings
import importlib
import numpy as np
from time import time
from osgeo import gdal, osr
import osgeo.gdalnumeric as gdn
from itertools import islice
from torch.utils.data import IterableDataset
# import numba
gdal.UseExceptions()

from .models.sitsbert.model.embedding.position import PositionalEncoding


def save_pandas_as_sqlite(outfile, dataframes, table_names, overwrite=False):
    if os.path.exists(outfile) and not overwrite:
        raise RuntimeError(f"Output file {outfile} exists. Set overwrite=True to if needed.")
    else:
        if os.path.exists(outfile) and overwrite:
            os.remove(outfile)
        conn = sqlite3.connect(outfile)
        for (df, name) in zip(dataframes, table_names):
            df.to_sql(name=name, con=conn)
        conn.close()


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def maybe_compile(model):
    if torch.cuda.is_available():
        maj, min =  torch.cuda.get_device_capability()
        if maj >= 7:
            return torch.compile(model)
        else:
            warnings.warn("CUDA capability lower than 7.0 - not compiling.")
            return model
    else:
        return torch.compile(model)


def read_img(input_file, dim_ordering="CHW", dtype='float32', band_mapping=None):
    """Reads an image from disk and returns it as numpy array.

    Args:
        input_file: Path to the input file.
        dim_ordering: One of HWC or CHW, C=Channels, H=Height, W=Width
        dtype: Desired data type for loading, e.g. np.uint8, np.float32...
        band_mapping: Dictionary of which image band to load into which array band. E.g. {1:0, 3:1}

    Returns:
        Numpy array containing the image and optionally the extent.
    """
    if not os.path.isfile(input_file):
        raise RuntimeError("Input file does not exist. Given path: {}".format(input_file))

    ds = gdal.Open(input_file)

    if band_mapping is None:
        num_bands = ds.RasterCount
        band_mapping = {i+1: i for i in range(num_bands)}
    elif isinstance(band_mapping, dict):
        num_bands = len(band_mapping)
    else:
        raise TypeError("band_mapping must be a dict, not {}.".format(type(band_mapping)))

    arr = np.empty((num_bands, ds.RasterYSize, ds.RasterXSize), dtype=dtype)

    for source_layer, target_layer in band_mapping.items():
        arr[target_layer] = gdn.BandReadAsArray(ds.GetRasterBand(source_layer))

    if dim_ordering == "HWC":
        arr = np.transpose(arr, (1, 2, 0))  # Reorders dimensions, so that channels are last
    elif dim_ordering == "CHW":
        pass
    else:
        raise ValueError("Dim ordering {} not supported. Choose one of 'HWC' or 'CHW'.".format(dim_ordering))

    return arr


def xarray_trafo_to_gdal_trafo(xarray_trafo):
    xres, xskew, xmin, yskew, yres, ymax = xarray_trafo
    return (xmin, xres, xskew, ymax, yskew, yres)


def array_to_tif(array, dst_filename, num_bands='multi', save_background=True, src_raster: str = "", transform=None,
                 crs=None):
    """ Takes a numpy array with predictions and writes a tif. Uses ZSTD compression.

    Args:
        array: numpy array (HWC)
        dst_filename (str): Destination file name/path
        num_bands (str): 'single' or 'multi'. If 'single' is chosen, everything is saved into one layer. The values
            in each layer of the input array are multiplied with the layer index and summed up. This is suitable for
            mutually exclusive categorical labels or single layer arrays. 'multi' is for normal images.
        save_background (bool): Whether or not to save the last layer, which is often the background class.
            Set to `True` for normal images.
        src_raster (str): Raster file used to determine the corner coords.
        transform: A geotransform in the gdal format
        crs: A coordinate reference system as proj4 string
    """
    if src_raster != "":
        src_raster = gdal.Open(src_raster)
        x_pixels = src_raster.RasterXSize
        y_pixels = src_raster.RasterYSize
    elif transform is not None and crs is not None:
        y_pixels, x_pixels = array.shape[:2]
    else:
        raise RuntimeError("Please provide either a source raster file or geotransform and coordinate reference "
                           "system.")

    bands = min( array.shape ) if array.ndim==3 else 1
    if not save_background and array.ndim==3: bands -= 1

    driver = gdal.GetDriverByName('GTiff')

    datatype = str(array.dtype)
    datatype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16,
                        'uint32': gdal.GDT_UInt32, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16,
                        'int32': gdal.GDT_Int32, 'float16': gdal.GDT_Float32, 'float32': gdal.GDT_Float32}
    options = ["COMPRESS=ZSTD"]
    if datatype == "float16":
        options.append("NBITS=16")

    out = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1 if num_bands == 'single' else bands,
        datatype_mapping[datatype],
        options=options)

    if src_raster != "":
        out.SetGeoTransform(src_raster.GetGeoTransform())
        out.SetProjection(src_raster.GetProjection())
    else:
        out.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(crs)
        out.SetProjection(srs.ExportToWkt())

    if array.ndim == 2:
        out.GetRasterBand(1).WriteArray(array)
        out.GetRasterBand(1).SetNoDataValue(0)
    else:
        if num_bands == 'single':
            singleband = np.zeros(array.shape[:2], dtype=array.dtype)
            for i in range(bands):
                singleband += (i+1)*array[:,:,i]
            out.GetRasterBand(1).WriteArray( singleband )
            out.GetRasterBand(1).SetNoDataValue(0)

        elif num_bands == 'multi':
            for i in range(bands):
                out.GetRasterBand(i+1).WriteArray( array[:,:,i] )
                out.GetRasterBand(i+1).SetNoDataValue(0)

    out.FlushCache()  # Write to disk.


def sparse2dense_timeseries(sparse_boa_series, times, max_length=512, subsampling=1):
    """Converts a batch of observations and points in time from sparse to dense.

    Args:
        sparse_boa_series: Observations (TxC)
        times: Observation times (as integers), (T).
        max_length: Must be the highest possible value of time_batch + 1.
        subsampling: Times can be divided by some subsampling factor to reduce time resolution and memory consumption.
    """
    # device = sparse__boa_series_batch.device
    seq_len, feature_dim = sparse_boa_series.shape
    dense_series = np.zeros(max_length * feature_dim // subsampling, dtype=np.float32)

    # squash time series
    times_ = times // subsampling

    # Calculate the indices for each batch and time step
    # Shape of t_idx: (max_length, feature_dim)
    time_indices = np.expand_dims(times_, -1) * feature_dim + np.arange(feature_dim)

    # Shape of time_indices: (max_length * feature_dim)
    # time_indices = time_indices.view(-1)
    time_indices = time_indices.reshape(-1)

    # Flatten boa_batch to match the shape of indices for scatter
    # Shape of boa_batch_flat: (N, 5120)
    boa_flat = sparse_boa_series.reshape(-1)

    # Scatter the values from boa_batch_flat into the data tensor at the appropriate indices
    # dense_series.scatter_(0, time_indices, boa_flat)
    np.put_along_axis(dense_series, time_indices, boa_flat, 0)
    return dense_series


def sparse2dense_timeseries_batched(sparse_boa_series, times, max_length=512, subsampling=1):
    """Converts a batch of observations and points in time from sparse to dense.

    Args:
        sparse_boa_series: Observations (NxTxC)
        times: Observation times (as integers), (NxT).
        max_length: Must be the highest possible value of time_batch + 1.
        subsampling: Times can be divided by some subsampling factor to reduce time resolution and memory consumption.
    """
    # device = sparse__boa_series_batch.device
    batch_size, seq_len, feature_dim = sparse_boa_series.shape
    dense_series = np.zeros((batch_size, max_length * feature_dim // subsampling), dtype=np.float32)

    # squash time series
    times_ = times // subsampling

    # Calculate the indices for each batch and time step
    # Shape of t_idx: (max_length, feature_dim)
    time_indices = np.expand_dims(times_, -1) * feature_dim + np.arange(feature_dim)

    # Shape of time_indices: (max_length * feature_dim)
    # time_indices = time_indices.view(-1)
    time_indices = time_indices.reshape(batch_size, -1)

    # Flatten boa_batch to match the shape of indices for scatter
    # Shape of boa_batch_flat: (N, 5120)
    boa_flat = sparse_boa_series.reshape(batch_size, -1)

    # Scatter the values from boa_batch_flat into the data tensor at the appropriate indices
    # dense_series.scatter_(0, time_indices, boa_flat)
    np.put_along_axis(dense_series, time_indices, boa_flat, 1)
    return dense_series


def sparse2dense_timeseries_batched_torch(sparse_boa_series, times, max_length=512, subsampling=1):
    """Converts a batch of observations and points in time from sparse to dense.

    Args:
        sparse_boa_series: Observations (NxTxC)
        times: Observation times (as integers), (NxT).
        max_length: Must be the highest possible value of time_batch + 1.
        subsampling: Times can be divided by some subsampling factor to reduce time resolution and memory consumption.
    """
    device = sparse_boa_series.device
    batch_size, seq_len, feature_dim = sparse_boa_series.shape
    dense_series = torch.zeros((batch_size, max_length * feature_dim // subsampling), dtype=sparse_boa_series.dtype, device=device)

    # squash time series
    times_ = times // subsampling

    # Calculate the indices for each batch and time step
    # Shape of t_idx: (max_length, feature_dim)
    time_indices = times_.unsqueeze(-1) * feature_dim + torch.arange(feature_dim, device=device)

    # Shape of time_indices: (max_length * feature_dim)
    # time_indices = time_indices.view(-1)
    time_indices = time_indices.view(batch_size, -1)

    # Flatten boa_batch to match the shape of indices for scatter
    # Shape of boa_batch_flat: (N, 5120)
    boa_flat = sparse_boa_series.view(batch_size, -1)

    # Scatter the values from boa_batch_flat into the data tensor at the appropriate indices
    dense_series.scatter_(1, time_indices, boa_flat)
    return dense_series.view(batch_size, max_length // subsampling, feature_dim)


def load_and_prepare_timeseries_folder(input_folder: str,
                                       qai: int,
                                       seq_len: int,
                                       time_encoding: str = "doy",
                                       t0: datetime.date = datetime.date(2015, 1, 1),
                                       fname2date = lambda s: datetime.datetime.strptime(s[:8], '%Y%m%d').date(),
                                       tmin_data=datetime.date(1, 1, 1),
                                       tmax_data=datetime.date(9999, 1, 1),
                                       tmin_inference=None,
                                       tmax_inference=None,
                                       append_ndvi=False,
                                       ):
    """Loads all `seq_len` last BOA and QAI files from the given input folder. Depending on seq_len, \
    this needs lots of RAM.

    Args:
        input_folder (str): The input folder path
        qai (int): A FORCE QAI binary flag
        seq_len (int): Maximum sequence length to load. The found dates can be filtered by tmin and tmax
            (default: no filtering) and then the last `seq_len` dates are loaded.
        time_encoding (str): Either doy for day of year time encoding or 'absolute' for the days passed since t0.
        t0 (datetime.date): Initial date from which to calculate the absolute time encoding
        fname2date (callable): Function converting a file name string to a datetime.date object.
        tmin_data (datetime.date): Minimum date to include (default year 0). Times are filtered as tmin <= t < tmax.
        tmax_data (datetime.date): Maximum date to include (default year 9999). Times are filtered as tmin <= t < tmax.
        tmin_inference (datetime.date): Minimum date to include for inference (default: tmin_data).
        tmax_inference (datetime.date): Maximum date for inference (exclusive) (default: tmax_data).

    Returns:
        If qai == 0:
            Input image shape (h,w,c),
            loaded BOAs as numpy array in format (h*w, seq_len, c),
            encoded times (seq_len),
            times valid for inference as boolean array of shape (seq_len),
            None,
            Mone
        else:
            Input image shape (h,w,c),
            loaded BOAs in same format,
            encoded times (seq_len),
            times valid for inference as boolean array of shape (seq_len),
            pixel-wise boolean validity mask (h*w, seq_len),
            number of observations per pixel (h*w)
    """
    if tmin_inference is None:
        tmin_inference = tmin_data
    if tmax_inference is None:
        tmax_inference = tmax_data

    assert time_encoding in ("doy", "absolute"), f"Please choose a valid time encoding: 'doy' for day of year or 'absolute' for days passed since t0"
    files = os.listdir(input_folder)
    assert len(files) > 0, f"No files found in input folder {input_folder}"
    boa_filenames = np.array(sorted(filter(lambda x: 'BOA' in x, files)))
    dates = np.array([fname2date(s) for s in boa_filenames])
    valid_dates = [tmin_data <= d < tmax_data for d in dates]
    boa_filenames = boa_filenames[valid_dates][-seq_len:]
    dates = dates[valid_dates][-seq_len:]
    inference_date_mask = np.array([tmin_inference <= d < tmax_inference for d in dates])

    seq_len = len(boa_filenames)
    
    if qai > 0:
        qais = np.array(sorted(filter(lambda x: 'QAI' in x, files)))[valid_dates][-seq_len:]
        assert len(boa_filenames) == len(qais), f"Length of BOA and QAI time series differ ({len(boa_filenames)} vs {len(qais)})."


    # TODO: use a function for below stuff
    if time_encoding == "doy":
        times = np.array([d.timetuple().tm_yday for d in dates])
    elif time_encoding == "absolute":
        times = np.array([(date - t0).days for date in dates])
    else:
        # can't happen
        raise RuntimeError(f"Argument time_encoding must be either doy or absolute, not {time_encoding}.")

    # print("Loading images")
    sample_boa = read_img(os.path.join(input_folder, boa_filenames[0]), dim_ordering="HWC", dtype=np.int16)
    h, w, c = sample_boa.shape

    if append_ndvi:
        c += 1

    all_boas = np.empty((h, w, seq_len, c), dtype=np.float32 if append_ndvi else np.int16)

    # read all the files
    for (i, f) in enumerate(boa_filenames):
        fname = os.path.join(input_folder, f)
        img = read_img(fname, dim_ordering="HWC", dtype=np.float32)
        if append_ndvi:
            all_boas[:, :, i, :-1] = img
        else:
            all_boas[:, :, i, :] = img

    if append_ndvi:
        red = all_boas[:, :, :, 2]
        nir = all_boas[:, :, :, 6]
        all_boas[:, :, :, -1] = (nir - red) / (nir + red + 1e-7)

    all_boas = all_boas.reshape((-1, seq_len, c))

    if qai > 0:
        validity_mask = np.empty((h, w, seq_len, 1), dtype=bool)
        for (i, f) in enumerate(qais):
            fname = os.path.join(input_folder, f)
            img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
            validity_mask[:, :, i, :] = (img & qai) == 0

        validity_mask = validity_mask.reshape((-1, seq_len))
        n_obs = np.sum(validity_mask, axis=1)

        return (h,w,c), all_boas, times, inference_date_mask, validity_mask, n_obs
    else:
        return (h,w,c), all_boas, times, inference_date_mask, None, None


def load_and_prepare_timeseries_files(boa_filenames: list[str],
                                      qai_filenames: list[str],
                                      qai: int,
                                      time_encoding: str = "doy",
                                      t0: datetime.date = datetime.date(2015, 1, 1),
                                      fname2date = lambda s: datetime.datetime.strptime(s[:8], '%Y%m%d').date()
                                      ):
    """Loads all `seq_len` last BOA and QAI files from the given input folder. Depending on seq_len, \
    this needs lots of RAM.

    Args:
        boa_filenames (list): Names of the files to load, must have the desired sequence length
        qai_filenames (list): Names of the files to load, must have the desired sequence length
        qai (int): A FORCE QAI binary flag
        time_encoding (str): Either 'doy' for day of year time encoding or 'absolute' for the days passed since t0
        t0 (datetime.date): Initial date from which to calculate the absolute time encoding
        fname2date (callable): Function converting a file name string to a datetime.date object.

    Returns:
        In case that QAI == 0:
            Input image shape (h,w), loaded BOAs as numpy array in format (h*w, seq_len, c), encoded times (seq_len), None, None
        else: Input image shape (h,w), loaded BOAs in same format, encoded times (seq_len), pixel-wise boolean validity mask (h*w, seq_len), number of observations per pixel (h*w)
    """
    # assert time_encoding in ("doy", "absolute"), f"Please choose a valid time encoding: 'doy' for day of year or 'absolute' for days passed since t0"
    # files = os.listdir(input_folder)
    # assert len(files) > 0, f"No files found in input folder {input_folder}"
    # boa_filenames = list(sorted(filter(lambda x: 'BOA' in x, files)))[-seq_len:]

    seq_len = len(boa_filenames)

    dates = [fname2date(s) for s in boa_filenames]

    if time_encoding == "doy":
        times = np.array([d.timetuple().tm_yday for d in dates])
    elif time_encoding == "absolute":
        times = np.array([(date - t0).days for date in dates])
    else:
        # can't happen
        raise RuntimeError(f"Argument time_encoding must be either doy or absolute, not {time_encoding}.")

    # print("Loading images")
    sample_boa = read_img(boa_filenames[0], dim_ordering="HWC", dtype=np.int16)
    h, w, c = sample_boa.shape
    all_boas = np.empty((h, w, seq_len, c), dtype=np.int16)

    # read all the files
    for (i, f) in enumerate(boa_filenames):
        img = read_img(f, dim_ordering="HWC", dtype=np.int32)
        all_boas[:, :, i, :] = img

    all_boas = all_boas.reshape((-1, seq_len, c))

    if qai > 0:
        validity_mask = np.empty((h, w, seq_len, 1), dtype=bool)
        for (i, f) in enumerate(qai_filenames):
            img = read_img(f, dim_ordering="HWC", dtype=np.int32)
            validity_mask[:, :, i, :] = (img & qai) == 0

        validity_mask = validity_mask.reshape((-1, seq_len))
        n_obs = np.sum(validity_mask, axis=1)

        return (h,w,c), all_boas, times, validity_mask, n_obs
    else:
        return (h,w,c), all_boas, times, None, None


# @numba.jit(nopython=True, boundscheck=False, parallel=False)
def assemble_batch_cpu(boa_batch, time_batch, mask_batch, start_idx, all_boas, n_obs, validity_mask, times):
    size_of_this_batch, seq_len, c = boa_batch.shape  # size of last batch might differ from the ones before
    assert start_idx + size_of_this_batch <= all_boas.shape[0]
    boa_batch.fill(0)
    mask_batch.fill(1)
    time_batch.fill(0)
    j = 0
    for i in range(start_idx, start_idx + size_of_this_batch):  # index in the source data
        # j = i % default_batch_size
        n = n_obs[i]
        mask = validity_mask[i]
        mask_batch[j, :n] = 0
        time_batch[j, :n] = times[mask]
        boa_batch[j, :n] = all_boas[i][mask]
        j += 1

    return boa_batch, time_batch, mask_batch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def instantiate_model_from_config(config_path, **override_kwargs):
    def parse_value(value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        return value

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    class_path = config['model']['class_path']

    # class_path = "<class 'sen2classification.models.gru.GRU'>"
    if class_path[0] == "<":
        class_path = class_path[8:-2]

    module_name = '.'.join(class_path.split('.')[:-1])
    class_name = class_path.split('.')[-1]

    init_args = {k: parse_value(v) for k, v in config['model']['init_args'].items()}
    init_args.update({k: parse_value(v) for k, v in override_kwargs.items()})

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_(**init_args), init_args


def load_model_from_configs_and_checkpoint(model_config_path, data_config_path, checkpoint_path):
    with open(data_config_path, 'r') as f:
        dataconfig = yaml.safe_load(f)

    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)["model"]

    classes = list(sorted(set(dataconfig["data"]["class_mapping"].values())))
    num_classes = len(classes)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    # fix latest possible inference date by increasing the max_time....
    max_time_state_dict = state_dict["pos_embed.pos_embed.pos_embed"].shape[0]
    max_time_config = model_config["init_args"]["max_time"]

    model, init_args = instantiate_model_from_config(config_path=model_config_path,
                                                     num_classes=num_classes,
                                                     classes=classes,
                                                     max_time=max_time_state_dict)

    if max_time_state_dict == max_time_config:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        embedding_dim = model_config["init_args"]["embedding_dim"]
        if model_config["init_args"]["embedding_type"] == "concat":
            embedding_dim //= 2
        model.pos_embed.pos_embed = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_time_config)

    return model, init_args


def classname(obj):
    return obj.__class__.__name__


def copy_file_to_destination(destination_folder, file):
    current_file_path = os.path.abspath(file)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    shutil.copy(current_file_path, os.path.join(destination_folder, os.path.basename(current_file_path)))


def listify(d):
    """Recursively converts all numpy arrays in a dictionary to lists."""
    if isinstance(d, dict):
        return {key: listify(value) for key, value in d.items()}
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d


def save_dict_to_yaml(data, filename):
    """
    Save a dictionary containing numpy arrays to a YAML file.

    Args:
        data (dict): The dictionary to save.
        filename (str): The filename where the dictionary should be saved.
    """
    with open(filename, 'w') as file:
        yaml.dump(listify(data), file, default_flow_style=False)


def predict_on_batches(model, all_boas, times, validity_mask, n_obs, mean, stddev, batch_size,
                       inference_date_mask, verbose, apply_argmax=True, num_classes=0, band_reordering=None):
        num_output_pixels = all_boas.shape[0]
        seq_len = all_boas.shape[1]
        c = all_boas.shape[2]  # channels
        seq_len = 2 ** int(np.ceil(np.log(seq_len) / np.log(2)))  # pad to next power of 2
        inference_date_mask = np.pad(inference_date_mask, (0, seq_len - len(inference_date_mask)), constant_values=0)

        # get rid of pixels with 0 observations
        valid_pixel_mask = n_obs > 0
        all_boas = all_boas[valid_pixel_mask]
        validity_mask = validity_mask[valid_pixel_mask]
        n_obs = n_obs[valid_pixel_mask]
        num_pixels = valid_pixel_mask.sum()

        # pre-allocate memory for different network inputs
        boa_batch = np.zeros((batch_size, seq_len, c), dtype=np.float32)
        time_batch = np.zeros((batch_size, seq_len), dtype=np.int32)
        data_mask_batch = np.zeros((batch_size, seq_len), dtype=bool)

        # pre-allocate data on GPU to recycle memory, boosts perf by ca 4%
        boa_torch = torch.zeros((batch_size, seq_len, c), device=model.device, dtype=torch.float32)
        time_torch = torch.zeros((batch_size, seq_len), device=model.device, dtype=torch.int32)
        data_mask_torch = torch.zeros((batch_size, seq_len), device=model.device, dtype=bool)

        # The inference mask can be used for transformer models to average only over tokens where
        # the inference mask is true. The mask is constant for all batches.
        inference_mask_torch = torch.from_numpy(inference_date_mask).to(model.device)
        inference_mask_torch_batch = inference_mask_torch.expand(batch_size, seq_len)

        mean = torch.from_numpy(mean).to(model.device)
        stddev = torch.from_numpy(stddev).to(model.device)

        # instantiate output for prediction as well as for the final output that will have all zeros or 255s, where
        # no prediction was made due to missing observations
        if apply_argmax:
            output = torch.zeros(num_pixels, dtype=torch.uint8, device=model.device)
            # set output to 255 so that we can filter out invalid pixels,
            # as it's highly unlikely that the model ever outputs 256 classes
            final_output = torch.ones(num_output_pixels, dtype=torch.uint8, device=model.device) * 255
        else:
            output = torch.zeros((num_pixels, num_classes), dtype=torch.uint8, device=model.device)
            # here we keep the zeros; invalid pixels will have 0 in every band
            final_output = torch.zeros((num_output_pixels, num_classes), dtype=torch.uint8, device=model.device)

        # predict
        t0 = time()
        with ((torch.no_grad())):
            i = 0  # batch counter
            start = 0
            while start < num_pixels:
                if i % 100 == 0 and verbose:
                    print(f"{i} / {num_pixels // batch_size}")
                stop = min(start + batch_size, num_pixels)
                bs = stop - start
                # adjust the size of the last batch
                # lot of repetition, but needed to avoid allocations
                if start + batch_size > num_pixels:
                    boa_batch = np.zeros((bs, seq_len, c), dtype=np.float32)
                    time_batch = np.zeros((bs, seq_len), dtype=np.int32)
                    data_mask_batch = np.zeros((bs, seq_len), dtype=bool)
                    boa_torch = torch.zeros((bs, seq_len, c), device=model.device, dtype=torch.float32)
                    time_torch = torch.zeros((bs, seq_len), device=model.device, dtype=torch.int32)
                    data_mask_torch = torch.zeros((bs, seq_len), device=model.device, dtype=bool)
                    inference_mask_torch_batch = inference_mask_torch.expand(bs, seq_len)

                assemble_batch_cpu(boa_batch, time_batch, data_mask_batch, start, all_boas, n_obs, validity_mask, times)

                # copy data over to GPU without allocations (hopefully)
                boa_torch[:] = torch.from_numpy(boa_batch)
                time_torch[:] = torch.from_numpy(time_batch)
                data_mask_torch[:] = torch.from_numpy(data_mask_batch)
                boa_torch[~data_mask_torch] -= mean
                boa_torch[~data_mask_torch] /= stddev + 1e-7
                pred = model(boa_torch, time_torch, data_mask_torch, inference_mask_torch_batch)
                if apply_argmax:
                    pred = pred.argmax(dim=-1)
                    output[start:stop] = pred.to(torch.uint8)
                else:
                    pred = pred.softmax(dim=-1) * 255
                    output[start:stop, :] = pred.to(torch.uint8)
                start += batch_size
                i += 1

        # output = output.reshape(h, w).cpu().numpy()
        if verbose:
            print("Prediction time: ", time() - t0)

        final_output[valid_pixel_mask] = output

        if band_reordering:
            # that was a joke, the final output is not final yet
            final_output = final_output[:, list(band_reordering)]

        return final_output


def k_fold_generator(n: int, k: int, test_fraction: float = 0.3, seed: int = 1):
    """ Infinitely running K-Fold generator for list indices.

    Args:
        n: Number of items in collection to be iterated over (length of list)
        k: Number of folds - n // k is the step size between two folds
        test_fraction: Test fraction
        seed: Seed for the internal RNG

    Returns:
        A generator object which infinitely yields a new set of (training_indices, test_indices) on each call of next().
    """
    i = 0
    indices = list(np.arange(n))
    rstate = np.random.RandomState(seed=seed)
    rstate.shuffle(indices)
    m = int(n * test_fraction)
    step = n // k

    print("Training samples: %d" % (n - m))
    print("Test samples: %d" % m)

    while True:
        test_start = i % n
        test_end = (i + m) % n

        if test_start < test_end:
            test_set = indices[test_start:test_end]
            training_set = indices[:test_start] + indices[test_end:]
        else:
            test_set = indices[:test_end] + indices[test_start:]
            training_set = indices[test_end:test_start]

        yield training_set, test_set
        i += step


def k_fold_generator_list(items: list, k: int, test_fraction: float = 0.3, seed: int = 1):
    """ Infinitely running K-Fold generator for list indices.

    Args:
        items: List of items to be iterated over (length of list)
        k: Number of folds - n // k is the step size between two folds
        test_fraction: Test fraction
        seed: Seed for the internal RNG

    Returns:
        A generator object which infinitely yields a new set of (training_indices, test_indices) on each call of next().
    """
    i = 0
    items = list(items)
    n = len(items)
    rstate = np.random.RandomState(seed=seed)
    rstate.shuffle(items)
    m = int(n * test_fraction)
    step = n // k

    print("Training samples: %d" % (n - m))
    print("Test samples: %d" % m)

    while True:
        test_start = i % n
        test_end = (i + m) % n

        if test_start < test_end:
            test_set = items[test_start:test_end]
            training_set = items[:test_start] + items[test_end:]
        else:
            test_set = items[:test_end] + items[test_start:]
            training_set = items[test_end:test_start]

        yield training_set, test_set
        i += step
