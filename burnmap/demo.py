"""Synthetic dataset with the same format as the GEE exports.

The scene mimics a mountain landscape with forest, shrubland, dry grassland,
bare ground and a lake, a fire with a severity gradient, and summer drying of
grasslands that also lowers NBR and NDVI (a typical source of false positives
for threshold rules). Values are synthetic and only meant to run the pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from scipy.ndimage import binary_dilation, gaussian_filter

PARAMS = {
    "out_dir": "data/demo",
    "shape": (360, 450),           # rows, cols
    "pixel_m": 20,
    "origin": (290_000, 5_360_000),  # upper-left corner, UTM zone 19S
    "crs": "EPSG:32719",
    "nodata": -9999.0,
    "polygon_px": 4,               # training polygons are squares of polygon_px x polygon_px pixels
    "seed": 42,
}

BAND_NAMES = ["NBR_pre", "NBR_post", "dNBR", "NDVI_pre", "NDVI_post", "dNDVI", "MNDWI_pre", "MNDWI_post"]

# Pre-fire NDVI, NBR and MNDWI by cover type
COVER = {
    "forest":    {"ndvi": 0.75, "nbr": 0.60, "mndwi": -0.45},
    "shrubland": {"ndvi": 0.50, "nbr": 0.35, "mndwi": -0.35},
    "grassland": {"ndvi": 0.32, "nbr": 0.18, "mndwi": -0.30},
    "bare":      {"ndvi": 0.10, "nbr": 0.00, "mndwi": -0.20},
}


def smooth_field(rng, shape, sigma):
    """Zero-mean, unit-variance smooth random field."""
    field = gaussian_filter(rng.standard_normal(shape), sigma)
    return (field - field.mean()) / field.std()


def make_scene(params):
    """Simulate the feature stack and the true burned mask.

    Parameters
    ----------
    params : dict
        Generator parameters (see PARAMS).

    Returns
    -------
    bands : dict of str to numpy.ndarray
        Feature arrays, NaN on water and shoreline.
    truth : dict of str to numpy.ndarray
        Boolean masks: burned, cover types and the valid land area.
    """
    rng = np.random.default_rng(params["seed"])
    rows, cols = params["shape"]
    yy, xx = np.mgrid[0:rows, 0:cols]

    # Land cover
    relief = smooth_field(rng, (rows, cols), 25)
    patches = smooth_field(rng, (rows, cols), 8)
    cover = np.where(relief > -0.2, "forest", "shrubland").astype(object)
    cover[(relief < -0.9)] = "grassland"
    cover[(patches > 1.6) & (relief < 0.3)] = "bare"
    lake = ((xx - 0.82 * cols) / (0.16 * cols)) ** 2 + ((yy - 0.80 * rows) / (0.13 * rows)) ** 2 < 1
    shoreline = binary_dilation(lake, iterations=2)

    noise = lambda sd: rng.normal(0, sd, (rows, cols))
    ndvi_pre = np.zeros((rows, cols))
    nbr_pre = np.zeros((rows, cols))
    mndwi_pre = np.zeros((rows, cols))
    for name, values in COVER.items():
        m = cover == name
        ndvi_pre[m] = values["ndvi"]
        nbr_pre[m] = values["nbr"]
        mndwi_pre[m] = values["mndwi"]
    ndvi_pre += noise(0.04)
    nbr_pre += noise(0.04)
    mndwi_pre += noise(0.03)

    # Fire: irregular perimeter with severity decreasing towards the edges
    radial = np.hypot((xx - 0.38 * cols) / (0.30 * cols), (yy - 0.42 * rows) / (0.32 * rows))
    shape_noise = smooth_field(rng, (rows, cols), 12)
    fire_index = 1 - radial + 0.25 * shape_noise
    burnable = (cover != "bare") & ~shoreline
    burned = (fire_index > 0.25) & burnable
    severity = np.clip((fire_index - 0.25) / 0.6 + 0.15 * smooth_field(rng, (rows, cols), 4), 0, 1)
    severity[~burned] = 0

    # Changes between pre- and post-fire composites
    d_nbr = noise(0.03)
    d_ndvi = noise(0.03)
    d_mndwi = noise(0.02)

    fire_dnbr = 0.09 + 0.80 * severity + noise(0.035)
    d_nbr[burned] = fire_dnbr[burned]
    d_ndvi[burned] = (0.65 * fire_dnbr * ndvi_pre / 0.75 + noise(0.04))[burned]
    d_mndwi[burned] = (-0.03 - 0.18 * severity + noise(0.03))[burned]

    # Summer drying of grasslands and part of the shrubland (unburned)
    drought = smooth_field(rng, (rows, cols), 20) > 0.3
    drying = ~burned & drought & np.isin(cover, ["grassland", "shrubland"])
    d_nbr[drying] = (0.13 + noise(0.04))[drying]
    d_ndvi[drying] = (0.13 + noise(0.04))[drying]
    d_mndwi[drying] = (-0.02 + noise(0.03))[drying]

    bands = {
        "NBR_pre": nbr_pre, "NBR_post": nbr_pre - d_nbr, "dNBR": d_nbr,
        "NDVI_pre": ndvi_pre, "NDVI_post": ndvi_pre - d_ndvi, "dNDVI": d_ndvi,
        "MNDWI_pre": mndwi_pre, "MNDWI_post": mndwi_pre + d_mndwi,
    }
    for name in bands:
        bands[name] = np.where(shoreline, np.nan, bands[name]).astype("float32")

    truth = {"burned": burned, "drying": drying, "cover": cover, "valid": ~shoreline}
    return bands, truth


def sample_polygons(bands, truth, params, transform, crs):
    """Draw square training polygons in homogeneous areas and extract their pixels.

    Parameters
    ----------
    bands : dict of str to numpy.ndarray
        Feature arrays.
    truth : dict of str to numpy.ndarray
        Output of make_scene.
    params : dict
        Generator parameters.
    transform : affine.Affine
        Raster transform.
    crs : str
        Raster CRS.

    Returns
    -------
    pandas.DataFrame
        Same columns as the GEE sampleRegions export.
    """
    rng = np.random.default_rng(params["seed"] + 1)
    rows, cols = params["shape"]
    size = params["polygon_px"]
    burned, drying, cover = truth["burned"], truth["drying"], truth["cover"]
    high = burned & (bands["dNBR"] > 0.4)
    low = burned & (bands["dNBR"] <= 0.25)

    strata = {
        # name: (mask, label, number of polygons)
        "burned_high": (high, 1, 15),
        "burned_low": (low, 1, 20),
        "forest": (~burned & (cover == "forest"), 0, 15),
        "shrubland": (~burned & ~drying & (cover == "shrubland"), 0, 10),
        "grassland": (~burned & ~drying & (cover == "grassland"), 0, 10),
        "dry_vegetation": (drying, 0, 20),
        "bare": (cover == "bare", 0, 5),
    }

    records = []
    for name, (mask, label, n_polygons) in strata.items():
        drawn, attempts = 0, 0
        while drawn < n_polygons and attempts < 5000:
            attempts += 1
            r, c = rng.integers(0, rows - size), rng.integers(0, cols - size)
            window = (slice(r, r + size), slice(c, c + size))
            if not (mask[window].all() and truth["valid"][window].all()):
                continue
            poly_id = f"{name}_{drawn:02d}"
            for i in range(r, r + size):
                for j in range(c, c + size):
                    x, y = transform @ (j + 0.5, i + 0.5)
                    records.append({"class": label, "poly_id": poly_id, "x": x, "y": y,
                                    **{b: float(bands[b][i, j]) for b in BAND_NAMES}})
            drawn += 1

    df = pd.DataFrame(records)
    lon, lat = warp_transform(crs, "EPSG:4326", df["x"].tolist(), df["y"].tolist())
    df[".geo"] = [f'{{"type":"Point","coordinates":[{lo:.6f},{la:.6f}]}}' for lo, la in zip(lon, lat)]
    df.insert(0, "system:index", [f"{i}_0" for i in range(len(df))])
    return df.drop(columns=["x", "y"])



def write_demo_files(params=PARAMS):
    """Write the synthetic feature stack and training samples to disk.

    Parameters
    ----------
    params : dict, optional
        Generator parameters (see PARAMS).

    Returns
    -------
    features_path, samples_path : Path
        Paths of the GeoTIFF and CSV files.
    """
    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = params["shape"]
    transform = from_origin(*params["origin"], params["pixel_m"], params["pixel_m"])

    bands, truth = make_scene(params)

    profile = {
        "driver": "GTiff", "height": rows, "width": cols, "count": len(BAND_NAMES),
        "dtype": "float32", "crs": params["crs"], "transform": transform,
        "nodata": params["nodata"], "compress": "deflate",
    }
    features_path = out_dir / "burn_features.tif"
    with rasterio.open(features_path, "w", **profile) as dst:
        for k, name in enumerate(BAND_NAMES, start=1):
            dst.write(np.nan_to_num(bands[name], nan=params["nodata"]), k)
            dst.set_band_description(k, name)

    samples = sample_polygons(bands, truth, params, transform, params["crs"])
    samples_path = out_dir / "burn_training_samples.csv"
    samples.to_csv(samples_path, index=False)
    return features_path, samples_path


def load_demo(params=PARAMS):
    """Synthetic feature stack and training samples, kept in memory.

    Parameters
    ----------
    params : dict, optional
        Generator parameters (see PARAMS).

    Returns
    -------
    bands : dict of str to numpy.ndarray
        Feature arrays with NaN on water and shoreline.
    profile : dict
        Rasterio-style profile (crs, transform, width, height).
    samples : pandas.DataFrame
        Training samples in the format of the GEE export.
    """
    rows, cols = params["shape"]
    transform = from_origin(*params["origin"], params["pixel_m"], params["pixel_m"])
    bands, truth = make_scene(params)
    samples = sample_polygons(bands, truth, params, transform, params["crs"])
    profile = {"crs": rasterio.crs.CRS.from_string(params["crs"]), "transform": transform,
               "width": cols, "height": rows}
    return bands, profile, samples
