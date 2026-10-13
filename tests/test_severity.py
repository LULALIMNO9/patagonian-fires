import numpy as np

from burnmap.severity import class_areas, classify_dnbr

THRESHOLDS = [0.10, 0.27, 0.44, 0.66]
LABELS = ["Unburned", "Low", "Moderate-low", "Moderate-high", "High"]


def test_classify_dnbr_limits():
    dnbr = np.array([-0.2, 0.0999, 0.10, 0.30, 0.50, 0.66, 1.2])
    assert classify_dnbr(dnbr, THRESHOLDS).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_classify_dnbr_excludes_nan_and_invalid():
    dnbr = np.array([0.5, np.nan, 0.5])
    valid = np.array([True, True, False])
    assert classify_dnbr(dnbr, THRESHOLDS, valid).tolist() == [3, -1, -1]


def test_class_areas_totals():
    classes = np.array([0, 0, 1, 2, 4, -1])
    table = class_areas(classes, LABELS, pixel_ha=0.04)
    per_class = table[table["class"].notna()]
    assert per_class["pixels"].sum() == 5
    assert np.isclose(per_class["percent"].sum(), 100)
    burned = table[table["severity"] == "Burned (low to high)"].iloc[0]
    assert burned["pixels"] == 3
    assert np.isclose(burned["area_ha"], 0.12)
