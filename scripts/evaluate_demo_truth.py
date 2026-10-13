"""Compare the threshold rule and the classifier map with the synthetic ground truth.

Only possible with the demo data, where the true burned area is known.
Run after make_demo_data.py and train_classifier.py --demo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from burnmap.config import BASELINE
from burnmap.demo import PARAMS, make_scene


def main():
    bands, truth = make_scene(PARAMS)
    with rasterio.open("demo_output/results/burned_probability.tif") as src:
        probability = src.read(1)
        probability[probability == src.nodata] = np.nan
        pixel_ha = abs(src.transform.a * src.transform.e) / 10_000

    valid = np.isfinite(probability)
    burned, drying = truth["burned"] & valid, truth["drying"] & valid
    other_unburned = valid & ~truth["burned"] & ~truth["drying"]

    with np.errstate(invalid="ignore"):
        maps = {
            "Threshold rule (dNBR, dNDVI)": valid & (bands["dNBR"] > BASELINE["dnbr_min"])
            & (bands["dNDVI"] > BASELINE["dndvi_min"]),
            "Classifier (p > 0.5)": probability > 0.5,
        }

    rows = [{
        "method": "Ground truth",
        "burned_area_ha": burned.sum() * pixel_ha,
    }]
    for name, pred in maps.items():
        rows.append({
            "method": name,
            "burned_area_ha": pred.sum() * pixel_ha,
            "burned_detected_pct": 100 * pred[burned].mean(),
            "dry_vegetation_flagged_pct": 100 * pred[drying].mean(),
            "other_unburned_flagged_pct": 100 * pred[other_unburned].mean(),
        })
    table = pd.DataFrame(rows)
    out = Path("demo_output/results/demo_truth_check.csv")
    table.to_csv(out, index=False)
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
