"""Reading and writing of the GEE raster exports."""

import numpy as np
import rasterio


def read_stack(path, band_names, nodata):
    """Read a multi-band GeoTIFF into a dictionary of arrays.

    Parameters
    ----------
    path : str or Path
        GeoTIFF exported from Google Earth Engine.
    band_names : list of str
        Band names in file order. Used when the file has no band descriptions.
    nodata : float
        Value used for masked pixels in the export.

    Returns
    -------
    bands : dict of str to numpy.ndarray
        Float arrays with nodata replaced by NaN.
    profile : dict
        Rasterio profile of the file.
    """
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        if any(n is None for n in names) or set(names) != set(band_names):
            if src.count != len(band_names):
                raise ValueError(
                    f"{path} has {src.count} bands but {len(band_names)} band names were given"
                )
            names = band_names
        data = src.read().astype("float32")
        profile = src.profile

    data[data == nodata] = np.nan
    return dict(zip(names, data)), profile


def pixel_area_ha(profile):
    """Area of one pixel in hectares, for a projected CRS in metres.

    Parameters
    ----------
    profile : dict
        Rasterio profile.

    Returns
    -------
    float
        Pixel area in hectares.
    """
    transform = profile["transform"]
    return abs(transform.a * transform.e) / 10_000


def plot_extent(profile):
    """Extent (left, right, bottom, top) in kilometres for matplotlib imshow.

    Parameters
    ----------
    profile : dict
        Rasterio profile.

    Returns
    -------
    tuple of float
        Extent relative to the lower-left corner of the raster, in km.
    """
    transform = profile["transform"]
    width_km = profile["width"] * abs(transform.a) / 1000
    height_km = profile["height"] * abs(transform.e) / 1000
    return (0.0, width_km, 0.0, height_km)


def write_single_band(path, array, profile, nodata=-9999.0):
    """Write a float array as a single-band GeoTIFF.

    Parameters
    ----------
    path : str or Path
        Output file.
    array : numpy.ndarray
        2-D array; NaN values are written as nodata.
    profile : dict
        Rasterio profile of the source stack.
    nodata : float, optional
        Nodata value of the output.
    """
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(np.where(np.isnan(array), nodata, array).astype("float32"), 1)
