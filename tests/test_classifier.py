import numpy as np
import pandas as pd

from burnmap.classifier import build_models, cross_validate, predict_raster, threshold_baseline

FEATURES = ["dNBR", "dNDVI", "NDVI_pre"]


def make_samples(n_polygons=20, pixels=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_polygons):
        label = p % 2
        for _ in range(pixels):
            rows.append({
                "poly_id": f"p{p}",
                "class": label,
                "dNBR": 0.4 * label + rng.normal(0, 0.05),
                "dNDVI": 0.3 * label + rng.normal(0, 0.05),
                "NDVI_pre": 0.6 + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_threshold_baseline():
    df = pd.DataFrame({"dNBR": [0.05, 0.2, 0.2], "dNDVI": [0.2, 0.05, 0.2]})
    assert threshold_baseline(df, 0.1, 0.1).tolist() == [0, 0, 1]


def test_cross_validate_shapes():
    df = make_samples()
    models = build_models(random_state=0)
    scores, confusion = cross_validate(df, FEATURES, "class", "poly_id", models,
                                       {"dnbr_min": 0.1, "dndvi_min": 0.1}, n_splits=4)
    assert len(scores) == 4 * (len(models) + 1)
    for cm in confusion.values():
        assert cm.sum() == len(df)
    assert scores.loc[scores["model"] == "Random forest", "f1"].mean() > 0.9


def test_predict_raster_masks_invalid():
    df = make_samples()
    model = build_models(random_state=0)["Random forest"].fit(df[FEATURES], df["class"])
    bands = {f: np.full((2, 2), df[f].mean(), dtype="float32") for f in FEATURES}
    bands["dNBR"][0, 0] = np.nan
    valid = np.array([[True, True], [True, False]])
    probability = predict_raster(model, bands, FEATURES, valid)
    assert np.isnan(probability[0, 0]) and np.isnan(probability[1, 1])
    assert np.isfinite(probability[0, 1]) and np.isfinite(probability[1, 0])
