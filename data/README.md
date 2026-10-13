# Data

Inputs are not tracked in this repository. Export them from `gee/burn_severity_s2.js` and place them in `data/raw/`, or run `scripts/make_demo_data.py` to create synthetic data in `data/demo/`.

## `burn_features.tif`

8-band float GeoTIFF at 20 m, masked pixels (water and shoreline) set to -9999.

| Band | Name | Description |
|---|---|---|
| 1 | `NBR_pre` | Normalized Burn Ratio, pre-fire |
| 2 | `NBR_post` | Normalized Burn Ratio, post-fire |
| 3 | `dNBR` | NBR_pre − NBR_post |
| 4 | `NDVI_pre` | NDVI, pre-fire |
| 5 | `NDVI_post` | NDVI, post-fire |
| 6 | `dNDVI` | NDVI_pre − NDVI_post |
| 7 | `MNDWI_pre` | Modified Normalized Difference Water Index, pre-fire |
| 8 | `MNDWI_post` | Modified Normalized Difference Water Index, post-fire |

## `burn_training_samples.csv`

One row per pixel inside the `TRAINING` polygons: the 8 features above, `class` (1 = burned, 0 = not burned), `poly_id` (source polygon) and `.geo` (pixel centre as GeoJSON).

## Source

Copernicus Sentinel-2 Level-2A surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`), accessed through Google Earth Engine.
