import os
import torch
import numpy as np
import warnings
from osgeo import gdal, osr
import osgeo.gdalnumeric as gdn
from itertools import islice
from torch.utils.data import IterableDataset
gdal.UseExceptions()


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
