import os
import torch
import datetime
import numpy as np
import warnings
from osgeo import gdal, osr
import osgeo.gdalnumeric as gdn
from itertools import islice
from torch.utils.data import IterableDataset
import sqlite3
import numba
gdal.UseExceptions()


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


def change_resnet_classes(model, num_classes):
    """ Replaces the final resnet layer with one that output the desired number of classes. """
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)


def change_resnet_input(model, input_channels):
    old_conv : torch.nn.Conv2d = model.conv1
    new_conv = torch.nn.Conv2d(in_channels  = input_channels,
                               out_channels = old_conv.out_channels,
                               kernel_size  = old_conv.kernel_size,
                               stride       = old_conv.stride,
                               padding      = old_conv.padding,
                               bias         = old_conv.bias)
    model.conv1 = new_conv
    model.conv1.weight.data[:,:3,...] = old_conv.weight.data


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


def default_image_loader(path):
    return read_img(path) / 255


def get_all_image_paths(root_folder, image_extensions=("jpg", "jpeg", "png", "bmp", "gif", "tif")):
    """ Returns a list of all complete file paths of all the images in the subfolders of the given root folder.

    This function was written by ChatGPT.
    """
    image_paths = []  # List to store all the image file paths
    for root, dirs, files in os.walk(os.path.abspath(root_folder)):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                image_paths.append(os.path.join(root, file))
    return image_paths


def count_files_in_subfolders(directory):
    result = {}
    for root, dirs, files in os.walk(directory):
        # Ignore the root directory itself
        if root != directory:
            result[root] = len(files)
    return result


class GeneratorDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


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
                                       tmin=datetime.date(0, 1, 1),
                                       tmax=datetime.date(9999, 1, 1),
                                       ):
    """Loads all `seq_len` last BOA and QAI files from the given input folder. Depending on seq_len, \
    this needs lots of RAM.

    Args:
        input_folder (str): The input folder path
        qai (int): A FORCE QAI binary flag
        seq_len (int): Maximum sequence length to load. The found dates are filtered by tmin and tmax (default: no filtering) and then the last seq_len dates are loaded.
        time_encoding (str): Either doy for day of year time encoding or 'absolute' for the days passed since t0.
        t0 (datetime.date): Initial date from which to calculate the absolute time encoding
        fname2date (callable): Function converting a file name string to a datetime.date object.
        tmin (datetime.date): Minimum date to include (default year 0). Times are filtered as tmin <= t < tmax.
        tmax (datetime.date): Maximum date to include (default year 9999). Times are filtered as tmin <= t < tmax.

    Returns:
        In case that QAI == 0:
            Input image shape (h,w), loaded BOAs as numpy array in format (h*w, seq_len, c), encoded times (seq_len), None, None
        else: Input image shape (h,w), loaded BOAs in same format, encoded times (seq_len), pixel-wise boolean validity mask (h*w, seq_len), number of observations per pixel (h*w)
    """
    assert time_encoding in ("doy", "absolute"), f"Please choose a valid time encoding: 'doy' for day of year or 'absolute' for days passed since t0"
    files = os.listdir(input_folder)
    assert len(files) > 0, f"No files found in input folder {input_folder}"
    boa_filenames = np.array(sorted(filter(lambda x: 'BOA' in x, files)))
    dates = np.array([fname2date(s) for s in boa_filenames])
    valid_dates = [tmin <= d < tmax for d in dates][-seq_len:]
    boa_filenames = boa_filenames[valid_dates]
    dates = dates[valid_dates]

    if qai > 0:
        qais = np.array(sorted(filter(lambda x: 'QAI' in x, files)))[valid_dates]

    seq_len = len(boa_filenames)
    assert len(boa_filenames) == len(qais), f"Length of BOA and QAI time series differ ({len(boa_filenames)} vs {len(qais)})."

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
    all_boas = np.empty((h, w, seq_len, c), dtype=np.int16)

    # read all the files
    for (i, f) in enumerate(boa_filenames):
        fname = os.path.join(input_folder, f)
        img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
        all_boas[:, :, i, :] = img

    all_boas = all_boas.reshape((-1, seq_len, c))

    if qai > 0:
        validity_mask = np.empty((h, w, seq_len, 1), dtype=bool)
        for (i, f) in enumerate(qais):
            fname = os.path.join(input_folder, f)
            img = read_img(fname, dim_ordering="HWC", dtype=np.int32)
            validity_mask[:, :, i, :] = (img & qai) == 0

        validity_mask = validity_mask.reshape((-1, seq_len))
        n_obs = np.sum(validity_mask, axis=1)

        return (h,w,c), all_boas, times, validity_mask, n_obs
    else:
        return (h,w,c), all_boas, times, None, None


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
    size_of_this_batch, seq_len, c = boa_batch.shape  # size of last batch differs from the ones before
    assert start_idx + size_of_this_batch <= all_boas.shape[0]
    boa_batch[:] = 0
    mask_batch[:] = 1
    time_batch[:] = 0
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