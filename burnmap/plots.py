"""Figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def severity_map(classes, labels, colors, extent, path, caption=None):
    """Map of burn severity classes.

    Parameters
    ----------
    classes : numpy.ndarray
        Output of burnmap.severity.classify_dnbr (-1 = excluded).
    labels, colors : list of str
        Class names and colours.
    extent : tuple of float
        Plot extent in km (see burnmap.raster.plot_extent).
    path : str or Path
        Output PNG.
    caption : str, optional
        Text added at the bottom right (e.g. "Synthetic data").
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#c6dbef")
    masked = np.ma.masked_less(classes, 0)
    ax.imshow(masked, cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1,
              extent=extent, interpolation="nearest")
    handles = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    handles.append(Patch(color="#c6dbef", label="Water / excluded"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              title="Burn severity (dNBR)")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title("Burn severity")
    _caption(fig, caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dnbr_histogram(dnbr, thresholds, labels, colors, path, caption=None):
    """Histogram of dNBR with severity class limits.

    Parameters
    ----------
    dnbr : numpy.ndarray
        dNBR values of the analysed pixels.
    thresholds : list of float
        Class limits.
    labels, colors : list of str
        Class names and colours.
    path : str or Path
        Output PNG.
    caption : str, optional
        Text added at the bottom right.
    """
    values = dnbr[np.isfinite(dnbr)]
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = [values.min()] + list(thresholds) + [values.max()]
    for lo, hi, color in zip(edges[:-1], edges[1:], colors):
        ax.axvspan(lo, hi, color=color, alpha=0.5, lw=0)
    ax.hist(values, bins=120, color="#333333")
    ax.set_yscale("log")
    ax.set_xlabel("dNBR")
    ax.set_ylabel("Pixels (log scale)")
    ax.set_title("dNBR distribution and severity classes")
    ax.legend(handles=[Patch(color=c, alpha=0.5, label=l) for c, l in zip(colors, labels)],
              frameon=False, fontsize=8)
    _caption(fig, caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_matrices(confusion, path, caption=None):
    """Confusion matrices (summed over folds), one panel per model.

    Parameters
    ----------
    confusion : dict of str to numpy.ndarray
        2x2 matrices (rows = true class, columns = predicted class).
    path : str or Path
        Output PNG.
    caption : str, optional
        Text added at the bottom right.
    """
    n = len(confusion)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    for ax, (name, cm) in zip(np.atleast_1d(axes), confusion.items()):
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if share[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Not burned", "Burned"])
        ax.set_yticks([0, 1], ["Not burned", "Burned"])
        ax.set_xlabel("Predicted")
        ax.set_title(name, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("True")
    _caption(fig, caption)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def feature_importance(importance, path, caption=None):
    """Horizontal bar chart of permutation importance.

    Parameters
    ----------
    importance : pandas.DataFrame
        Columns feature, mean and std, sorted by mean.
    path : str or Path
        Output PNG.
    caption : str, optional
        Text added at the bottom right.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance["feature"], importance["mean"], xerr=importance["std"], color="#4a7ab5")
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in F1 when the feature is permuted")
    ax.set_title("Permutation feature importance")
    _caption(fig, caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def probability_map(probability, extent, path, caption=None):
    """Map of the predicted burned probability.

    Parameters
    ----------
    probability : numpy.ndarray
        Probability of the burned class, NaN where not predicted.
    extent : tuple of float
        Plot extent in km.
    path : str or Path
        Output PNG.
    caption : str, optional
        Text added at the bottom right.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor("#c6dbef")
    image = ax.imshow(np.ma.masked_invalid(probability), cmap="inferno", vmin=0, vmax=1,
                      extent=extent, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Burned probability", fraction=0.035, pad=0.03)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title("Classifier output")
    _caption(fig, caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _caption(fig, caption):
    if caption:
        fig.text(0.99, 0.005, caption, ha="right", va="bottom", fontsize=8, color="#555555")
