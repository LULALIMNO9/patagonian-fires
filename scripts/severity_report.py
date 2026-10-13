"""Burn severity classes, area per class and figures from the GEE features export.

Usage
-----
python scripts/severity_report.py            # data/raw/burn_features.tif
python scripts/severity_report.py --demo     # synthetic demo data
"""

import argparse
from pathlib import Path

from burnmap import plots
from burnmap.config import PATHS, RASTER, SEVERITY
from burnmap.raster import pixel_area_ha, plot_extent, read_stack
from burnmap.severity import analysis_mask, class_areas, classify_dnbr

DEMO_PATHS = {
    "features": "data/demo/burn_features.tif",
    "results": "demo_output/results",
    "figures": "demo_output/figures",
}


def main(paths, caption=None):
    results_dir, figures_dir = Path(paths["results"]), Path(paths["figures"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bands, profile = read_stack(paths["features"], RASTER["band_names"], RASTER["nodata"])
    valid = analysis_mask(bands, SEVERITY["ndvi_pre_min"], SEVERITY["mndwi_water"])
    classes = classify_dnbr(bands["dNBR"], SEVERITY["thresholds"], valid)

    areas = class_areas(classes, SEVERITY["labels"], pixel_area_ha(profile))
    areas.to_csv(results_dir / "severity_area_ha.csv", index=False)
    print(areas.round(1).to_string(index=False))

    plots.severity_map(classes, SEVERITY["labels"], SEVERITY["colors"], plot_extent(profile),
                       figures_dir / "severity_map.png", caption)
    plots.dnbr_histogram(bands["dNBR"][valid], SEVERITY["thresholds"], SEVERITY["labels"],
                         SEVERITY["colors"], figures_dir / "dnbr_histogram.png", caption)
    print(f"severity_report: tables in {results_dir}, figures in {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="use the synthetic demo data")
    args = parser.parse_args()
    if args.demo:
        main({**PATHS, **DEMO_PATHS}, caption="Synthetic data")
    else:
        main(PATHS)
