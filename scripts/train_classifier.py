"""Train and evaluate burned-area classifiers, and map the burned probability.

Steps
-----
1. Grouped cross-validation (folds split by training polygon, stratified by class) of the threshold
   rule used in GEE against logistic regression, random forest and gradient
   boosting.
2. Permutation importance of the best model on held-out polygons.
3. Best model refitted on all samples and applied to the full feature stack.

Usage
-----
python scripts/train_classifier.py            # data/raw/
python scripts/train_classifier.py --demo     # synthetic demo data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from burnmap import plots
from burnmap.classifier import (
    build_models,
    cross_validate,
    grouped_permutation_importance,
    load_samples,
    predict_raster,
    summarise_scores,
)
from burnmap.config import BASELINE, CLASSIFIER, PATHS, RASTER, SEVERITY
from burnmap.raster import pixel_area_ha, plot_extent, read_stack, write_single_band
from burnmap.severity import analysis_mask

DEMO_PATHS = {
    "features": "data/demo/burn_features.tif",
    "samples": "data/demo/burn_training_samples.csv",
    "results": "demo_output/results",
    "figures": "demo_output/figures",
}


def main(paths, caption=None):
    results_dir, figures_dir = Path(paths["results"]), Path(paths["figures"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    features, target, group = CLASSIFIER["features"], CLASSIFIER["target"], CLASSIFIER["group"]

    # 1. Cross-validation
    samples = load_samples(paths["samples"], features, target, group)
    print(f"{len(samples)} pixels from {samples[group].nunique()} polygons "
          f"({samples[target].mean():.0%} burned)")

    models = build_models(CLASSIFIER["random_state"])
    scores, confusion = cross_validate(samples, features, target, group, models, BASELINE,
                                       CLASSIFIER["n_splits"], CLASSIFIER["random_state"])
    summary = summarise_scores(scores)
    scores.to_csv(results_dir / "cv_scores_by_fold.csv", index=False)
    summary.to_csv(results_dir / "cv_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    plots.confusion_matrices(confusion, figures_dir / "confusion_matrices.png", caption)

    # 2. Feature importance of the best classifier
    best_name = summary.loc[summary["model"].isin(models), "model"].iloc[0]
    best_model = models[best_name]
    importance = grouped_permutation_importance(samples, features, target, group, best_model,
                                                CLASSIFIER["n_splits"],
                                                random_state=CLASSIFIER["random_state"])
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    plots.feature_importance(importance, figures_dir / "feature_importance.png", caption)

    # 3. Burned probability map
    bands, profile = read_stack(paths["features"], RASTER["band_names"], RASTER["nodata"])
    valid = analysis_mask(bands, SEVERITY["ndvi_pre_min"], SEVERITY["mndwi_water"])
    fitted = clone(best_model).fit(samples[features], samples[target])
    probability = predict_raster(fitted, bands, features, valid)
    write_single_band(results_dir / "burned_probability.tif", probability, profile)
    plots.probability_map(probability, plot_extent(profile),
                          figures_dir / "burned_probability_map.png", caption)

    with np.errstate(invalid="ignore"):
        rule = valid & (bands["dNBR"] > BASELINE["dnbr_min"]) & (bands["dNDVI"] > BASELINE["dndvi_min"])
        model_burned = probability > 0.5
    ha = pixel_area_ha(profile)
    areas = pd.DataFrame({
        "method": ["Threshold rule (dNBR, dNDVI)", f"{best_name} (p > 0.5)"],
        "burned_area_ha": [rule.sum() * ha, model_burned.sum() * ha],
    })
    areas.to_csv(results_dir / "burned_area_by_method.csv", index=False)
    print(f"\nBest model: {best_name}")
    print(areas.round(1).to_string(index=False))
    print(f"train_classifier: tables in {results_dir}, figures in {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="use the synthetic demo data")
    args = parser.parse_args()
    if args.demo:
        main({**PATHS, **DEMO_PATHS}, caption="Synthetic data")
    else:
        main(PATHS)
