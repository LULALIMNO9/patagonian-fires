"""Burn severity classification from dNBR."""

import numpy as np
import pandas as pd


def classify_dnbr(dnbr, thresholds, valid=None):
    """Assign burn severity classes from dNBR.

    Parameters
    ----------
    dnbr : numpy.ndarray
        Differenced Normalized Burn Ratio (pre minus post).
    thresholds : list of float
        Ascending class limits; values below the first limit are class 0.
    valid : numpy.ndarray of bool, optional
        Pixels to classify. Other pixels and NaN values get class -1.

    Returns
    -------
    numpy.ndarray
        Integer classes from 0 to len(thresholds), -1 for excluded pixels.
    """
    classes = np.digitize(np.nan_to_num(dnbr, nan=-np.inf), thresholds).astype("int8")
    excluded = np.isnan(dnbr)
    if valid is not None:
        excluded |= ~valid
    classes[excluded] = -1
    return classes


def class_areas(classes, labels, pixel_ha):
    """Area and share of each severity class.

    Parameters
    ----------
    classes : numpy.ndarray
        Output of classify_dnbr.
    labels : list of str
        Class names, one per class value.
    pixel_ha : float
        Area of one pixel in hectares.

    Returns
    -------
    pandas.DataFrame
        One row per class with pixel count, area (ha) and percentage of the
        analysed area, plus the total burned area (all classes above 0).
    """
    counts = np.bincount(classes[classes >= 0].ravel(), minlength=len(labels))
    table = pd.DataFrame({
        "class": range(len(labels)),
        "severity": labels,
        "pixels": counts,
        "area_ha": counts * pixel_ha,
    })
    table["percent"] = 100 * table["pixels"] / table["pixels"].sum()
    burned = table[table["class"] > 0][["pixels", "area_ha", "percent"]].sum()
    total = pd.DataFrame([{"class": None, "severity": "Burned (low to high)", **burned.to_dict()}])
    return pd.concat([table, total], ignore_index=True)


def analysis_mask(bands, ndvi_pre_min, mndwi_water):
    """Land pixels that were vegetated before the fire.

    Parameters
    ----------
    bands : dict of str to numpy.ndarray
        Feature arrays with NDVI_pre, MNDWI_pre and dNBR.
    ndvi_pre_min : float
        Minimum pre-fire NDVI.
    mndwi_water : float
        MNDWI above this value is treated as water.

    Returns
    -------
    numpy.ndarray of bool
        True for pixels included in the analysis.
    """
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(bands["dNBR"])
            & (bands["NDVI_pre"] > ndvi_pre_min)
            & (bands["MNDWI_pre"] < mndwi_water)
        )
