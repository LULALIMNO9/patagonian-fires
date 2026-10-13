"""Supervised classification of burned pixels with spatially grouped cross-validation."""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_samples(path, features, target, group):
    """Load the training samples exported from GEE.

    Parameters
    ----------
    path : str or Path
        CSV exported by the GEE script (sampleRegions with geometries).
    features : list of str
        Feature columns to keep.
    target : str
        Label column (1 = burned, 0 = not burned).
    group : str
        Column identifying the source polygon of each pixel.

    Returns
    -------
    pandas.DataFrame
        Samples without missing values, with lon and lat columns when the
        export includes geometries.
    """
    df = pd.read_csv(path)
    if ".geo" in df.columns:
        coords = df[".geo"].map(lambda g: json.loads(g)["coordinates"])
        df["lon"] = coords.str[0]
        df["lat"] = coords.str[1]
    df = df.dropna(subset=features + [target, group])
    df[target] = df[target].astype(int)
    return df.reset_index(drop=True)


def threshold_baseline(df, dnbr_min, dndvi_min):
    """Rule-based classification used as baseline.

    Parameters
    ----------
    df : pandas.DataFrame
        Samples with dNBR and dNDVI columns.
    dnbr_min, dndvi_min : float
        A pixel is burned when both differences exceed these values.

    Returns
    -------
    numpy.ndarray
        Predicted labels (1 = burned).
    """
    return ((df["dNBR"] > dnbr_min) & (df["dNDVI"] > dndvi_min)).astype(int).to_numpy()


def build_models(random_state):
    """Candidate classifiers.

    Parameters
    ----------
    random_state : int
        Seed for the stochastic models.

    Returns
    -------
    dict of str to estimator
        Unfitted scikit-learn estimators.
    """
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=random_state,
        ),
    }


def _scores(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def cross_validate(df, features, target, group, models, baseline, n_splits, random_state=0):
    """Compare the threshold rule and the classifiers with grouped cross-validation.

    Pixels from the same training polygon are spatially autocorrelated, so
    folds are split by polygon: no polygon contributes pixels to both the
    training and the test set of a fold. Folds are also stratified so that
    each one contains burned and unburned polygons.

    Parameters
    ----------
    df : pandas.DataFrame
        Training samples.
    features : list of str
        Feature columns.
    target, group : str
        Label and polygon id columns.
    models : dict of str to estimator
        Output of build_models.
    baseline : dict
        Parameters of threshold_baseline (dnbr_min, dndvi_min).
    n_splits : int
        Number of folds.
    random_state : int, optional
        Seed used to shuffle polygons between folds.

    Returns
    -------
    scores : pandas.DataFrame
        Metrics per model and fold.
    confusion : dict of str to numpy.ndarray
        Confusion matrix per model, summed over folds.
    """
    X, y, groups = df[features], df[target].to_numpy(), df[group].to_numpy()
    rows = []
    confusion = {}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train, test) in enumerate(cv.split(X, y, groups)):
        predictions = {"Threshold rule (dNBR, dNDVI)": threshold_baseline(df.iloc[test], **baseline)}
        for name, model in models.items():
            fitted = clone(model).fit(X.iloc[train], y[train])
            predictions[name] = fitted.predict(X.iloc[test])
        for name, y_pred in predictions.items():
            rows.append({"model": name, "fold": fold, **_scores(y[test], y_pred)})
            cm = confusion_matrix(y[test], y_pred, labels=[0, 1])
            confusion[name] = confusion.get(name, 0) + cm
    return pd.DataFrame(rows), confusion


def summarise_scores(scores):
    """Mean and standard deviation of each metric per model.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of cross_validate.

    Returns
    -------
    pandas.DataFrame
        One row per model, sorted by mean F1.
    """
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    summary = scores.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.sort_values("f1_mean", ascending=False).reset_index()


def predict_raster(model, bands, features, valid):
    """Burned probability for every valid pixel of the feature stack.

    Parameters
    ----------
    model : estimator
        Fitted classifier with predict_proba.
    bands : dict of str to numpy.ndarray
        Feature arrays (output of burnmap.raster.read_stack).
    features : list of str
        Feature names in the order used for training.
    valid : numpy.ndarray of bool
        Pixels to predict.

    Returns
    -------
    numpy.ndarray
        Probability of the burned class, NaN outside valid pixels.
    """
    valid = valid & np.all([np.isfinite(bands[f]) for f in features], axis=0)
    X = pd.DataFrame({f: bands[f][valid] for f in features})
    probability = np.full(valid.shape, np.nan, dtype="float32")
    probability[valid] = model.predict_proba(X)[:, 1]
    return probability


def grouped_permutation_importance(df, features, target, group, model, n_splits,
                                   n_repeats=10, random_state=0):
    """Permutation importance on held-out polygons.

    The model is refitted in each grouped fold and the drop in F1 is measured
    on the test polygons, so the importance reflects generalisation to new
    areas rather than memorisation of training pixels.

    Parameters
    ----------
    df : pandas.DataFrame
        Training samples.
    features : list of str
        Feature columns.
    target, group : str
        Label and polygon id columns.
    model : estimator
        Unfitted classifier.
    n_splits : int
        Number of folds.
    n_repeats : int, optional
        Permutations per feature and fold.
    random_state : int, optional
        Seed of the fold assignment and the permutations.

    Returns
    -------
    pandas.DataFrame
        Columns feature, mean and std, sorted by decreasing mean.
    """
    from sklearn.inspection import permutation_importance

    X, y, groups = df[features], df[target].to_numpy(), df[group].to_numpy()
    per_fold = []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train, test in cv.split(X, y, groups):
        fitted = clone(model).fit(X.iloc[train], y[train])
        result = permutation_importance(fitted, X.iloc[test], y[test], scoring="f1",
                                        n_repeats=n_repeats, random_state=random_state)
        per_fold.append(result.importances_mean)
    per_fold = np.array(per_fold)
    importance = pd.DataFrame({
        "feature": features,
        "mean": per_fold.mean(axis=0),
        "std": per_fold.std(axis=0),
    })
    return importance.sort_values("mean", ascending=False).reset_index(drop=True)
