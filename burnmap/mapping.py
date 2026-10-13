"""Helpers to display rasters on web maps."""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.warp import transform as warp_transform

WEB_MERCATOR = "EPSG:3857"


def to_web_mercator(array, profile, max_size=1500):
    """Reproject a single-band array to Web Mercator for map overlays.

    Parameters
    ----------
    array : numpy.ndarray
        2-D array in the CRS of profile (NaN = no data).
    profile : dict
        Rasterio profile with crs, transform, width and height.
    max_size : int, optional
        Maximum width or height of the output; larger rasters are downsampled.

    Returns
    -------
    projected : numpy.ndarray
        Reprojected float32 array (nearest neighbour), NaN outside the data.
    bounds : list
        [[south, west], [north, east]] in degrees, as expected by folium.
    """
    width, height = profile["width"], profile["height"]
    left, bottom, right, top = array_bounds(height, width, profile["transform"])
    dst_transform, dst_width, dst_height = calculate_default_transform(
        profile["crs"], WEB_MERCATOR, width, height, left, bottom, right, top
    )
    scale = max(dst_width, dst_height) / max_size
    if scale > 1:
        dst_width, dst_height = int(dst_width / scale), int(dst_height / scale)
        dst_transform = dst_transform @ dst_transform.scale(scale)

    projected = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=array.astype("float32"), destination=projected,
        src_transform=profile["transform"], src_crs=profile["crs"],
        dst_transform=dst_transform, dst_crs=WEB_MERCATOR,
        src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
    )
    w, s, e, n = array_bounds(dst_height, dst_width, dst_transform)
    lons, lats = warp_transform(WEB_MERCATOR, "EPSG:4326", [w, e], [s, n])
    return projected, [[lats[0], lons[0]], [lats[1], lons[1]]]


def classes_to_rgba(classes, colors, alpha=0.85):
    """Colour an integer class array; negative or NaN values are transparent.

    Parameters
    ----------
    classes : numpy.ndarray
        Class values 0..len(colors)-1.
    colors : list of str
        Colour per class.
    alpha : float, optional
        Opacity of coloured pixels.

    Returns
    -------
    numpy.ndarray
        uint8 RGBA image.
    """
    rgba = np.zeros(classes.shape + (4,), dtype="uint8")
    values = np.nan_to_num(classes, nan=-1).astype(int)
    for k, color in enumerate(colors):
        r, g, b, _ = to_rgba(color)
        rgba[values == k] = [int(255 * r), int(255 * g), int(255 * b), int(255 * alpha)]
    return rgba


def mask_to_rgba(mask, color, alpha=0.85):
    """Colour the True pixels of a boolean (or 0/1 float) array.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean or float array (NaN and 0 are transparent).
    color : str
        Colour of the True pixels.
    alpha : float, optional
        Opacity.

    Returns
    -------
    numpy.ndarray
        uint8 RGBA image.
    """
    return classes_to_rgba(np.where(np.nan_to_num(mask) > 0, 0, -1), [color], alpha)


def continuous_to_rgba(values, cmap, vmin, vmax, alpha=0.85):
    """Colour a continuous array with a matplotlib colormap; NaN is transparent.

    Parameters
    ----------
    values : numpy.ndarray
        Float array.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float
        Colour scale limits.
    alpha : float, optional
        Opacity of valid pixels.

    Returns
    -------
    numpy.ndarray
        uint8 RGBA image.
    """
    scaled = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(scaled)) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(values), int(255 * alpha), 0)
    return rgba
