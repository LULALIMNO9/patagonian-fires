"""Default parameters of the analysis."""

PATHS = {
    "features": "data/raw/burn_features.tif",
    "samples": "data/raw/burn_training_samples.csv",
    "results": "results",
    "figures": "figures",
}

RASTER = {
    # Band order of the GEE features export, used when the GeoTIFF has no band names
    "band_names": [
        "NBR_pre", "NBR_post", "dNBR",
        "NDVI_pre", "NDVI_post", "dNDVI",
        "MNDWI_pre", "MNDWI_post",
    ],
    "nodata": -9999.0,
}

SEVERITY = {
    # dNBR class limits (Key & Benson, 2006)
    "thresholds": [0.10, 0.27, 0.44, 0.66],
    "labels": ["Unburned", "Low", "Moderate-low", "Moderate-high", "High"],
    "colors": ["#d9f0d3", "#fee08b", "#fdae61", "#f46d43", "#a50026"],
    # Analysis restricted to land pixels vegetated before the fire (same rule as GEE)
    "ndvi_pre_min": 0.20,
    "mndwi_water": 0.10,
}

BASELINE = {
    # Rule used in the GEE script: candidate burned if both differences exceed these values
    "dnbr_min": 0.10,
    "dndvi_min": 0.10,
}

CLASSIFIER = {
    "features": [
        "NBR_pre", "NBR_post", "dNBR",
        "NDVI_pre", "NDVI_post", "dNDVI",
        "MNDWI_pre", "MNDWI_post",
    ],
    "target": "class",
    "group": "poly_id",
    "n_splits": 5,
    "random_state": 42,
}
