"""Burn severity explorer: interactive view of the Sentinel-2 burn analysis.

Run with
    streamlit run streamlit_app.py
"""

import tempfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.base import clone
from streamlit_folium import st_folium

from burnmap.classifier import (
    build_models,
    cross_validate,
    grouped_permutation_importance,
    load_samples,
    predict_raster,
    summarise_scores,
)
from burnmap.config import BASELINE, CLASSIFIER, PATHS, RASTER, SEVERITY
from burnmap.demo import load_demo
from burnmap.mapping import classes_to_rgba, continuous_to_rgba, mask_to_rgba, to_web_mercator
from burnmap.raster import pixel_area_ha, read_stack
from burnmap.severity import analysis_mask, class_areas, classify_dnbr

st.set_page_config(page_title="Burn severity explorer", page_icon=":fire:", layout="wide")

SOURCE_DEMO = "Demo (synthetic data)"
SOURCE_UPLOAD = "Upload GEE exports"
SOURCE_LOCAL = "Local files (data/raw)"
RULE_NAME = "Threshold rule (dNBR, dNDVI)"


# Data loading ---------------------------------------------------------------

@st.cache_data(show_spinner="Generating demo data...")
def get_demo():
    return load_demo()


@st.cache_data(show_spinner="Reading GeoTIFF...")
def get_stack_from_bytes(data):
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tmp.write(data)
        tmp.flush()
        return read_stack(tmp.name, RASTER["band_names"], RASTER["nodata"])


@st.cache_data(show_spinner="Reading GeoTIFF...")
def get_stack_from_path(path):
    return read_stack(path, RASTER["band_names"], RASTER["nodata"])


def get_samples(source):
    features, target, group = CLASSIFIER["features"], CLASSIFIER["target"], CLASSIFIER["group"]
    return load_samples(source, features, target, group)


def select_data():
    """Sidebar data selector. Returns bands, profile, samples (or None) and a source label."""
    local_available = Path(PATHS["features"]).exists()
    options = [SOURCE_DEMO, SOURCE_UPLOAD] + ([SOURCE_LOCAL] if local_available else [])
    source = st.sidebar.radio("Data", options, index=2 if local_available else 0)

    if source == SOURCE_DEMO:
        bands, profile, samples = get_demo()
        samples = samples.dropna(subset=CLASSIFIER["features"])
        return bands, profile, samples, source

    if source == SOURCE_LOCAL:
        bands, profile = get_stack_from_path(PATHS["features"])
        samples_path = Path(PATHS["samples"])
        samples = get_samples(samples_path) if samples_path.exists() else None
        return bands, profile, samples, source

    features_file = st.sidebar.file_uploader("Features GeoTIFF (burn_features.tif)", type=["tif", "tiff"])
    samples_file = st.sidebar.file_uploader("Training samples CSV (optional)", type=["csv"])
    if features_file is None:
        st.info("Upload the features GeoTIFF exported by `gee/burn_severity_s2.js` to start.")
        st.stop()
    bands, profile = get_stack_from_bytes(features_file.getvalue())
    samples = get_samples(samples_file) if samples_file is not None else None
    return bands, profile, samples, source


# Parameters -----------------------------------------------------------------

def select_parameters():
    with st.sidebar.expander("Analysis parameters", expanded=False):
        ndvi_min = st.slider("Min pre-fire NDVI (vegetated)", 0.0, 0.6, SEVERITY["ndvi_pre_min"], 0.01)
        mndwi_water = st.slider("MNDWI water threshold", -0.2, 0.5, SEVERITY["mndwi_water"], 0.01)
        st.markdown("**dNBR severity limits**")
        defaults = SEVERITY["thresholds"]
        cols = st.columns(2)
        thresholds = [
            cols[i % 2].number_input(f"{SEVERITY['labels'][i + 1]} from", value=float(defaults[i]),
                                     step=0.01, format="%.2f", key=f"thr_{i}")
            for i in range(len(defaults))
        ]
        st.markdown("**Threshold rule**")
        dnbr_min = st.slider("dNBR >", 0.0, 0.5, BASELINE["dnbr_min"], 0.01)
        dndvi_min = st.slider("dNDVI >", 0.0, 0.5, BASELINE["dndvi_min"], 0.01)
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        st.sidebar.error("Severity limits must be increasing.")
        st.stop()
    return {"ndvi_min": ndvi_min, "mndwi_water": mndwi_water, "thresholds": thresholds,
            "baseline": {"dnbr_min": dnbr_min, "dndvi_min": dndvi_min}}


# Classifier -----------------------------------------------------------------

def run_classifier(samples, bands, valid, baseline):
    features, target, group = CLASSIFIER["features"], CLASSIFIER["target"], CLASSIFIER["group"]
    models = build_models(CLASSIFIER["random_state"])
    scores, confusion = cross_validate(samples, features, target, group, models, baseline,
                                       CLASSIFIER["n_splits"], CLASSIFIER["random_state"])
    summary = summarise_scores(scores)
    best_name = summary.loc[summary["model"].isin(models), "model"].iloc[0]
    importance = grouped_permutation_importance(samples, features, target, group, models[best_name],
                                                CLASSIFIER["n_splits"], n_repeats=5,
                                                random_state=CLASSIFIER["random_state"])
    fitted = clone(models[best_name]).fit(samples[features], samples[target])
    probability = predict_raster(fitted, bands, features, valid)
    return {"summary": summary, "confusion": confusion, "importance": importance,
            "best": best_name, "probability": probability}


# Figures --------------------------------------------------------------------

def legend_html(items, title):
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:3px 0">'
        f'<span style="width:14px;height:14px;background:{c};display:inline-block;'
        f'margin-right:8px;border:1px solid #999"></span>{l}</div>'
        for c, l in items
    )
    return f'<div style="font-size:0.9rem"><b>{title}</b>{rows}</div>'


def build_map(layers, bounds, basemap):
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    fmap = folium.Map(location=center, tiles=None, control_scale=True, scroll_wheel_zoom=False)
    if basemap:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite",
        ).add_to(fmap)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    for name, rgba, show in layers:
        folium.raster_layers.ImageOverlay(image=rgba, bounds=bounds, name=name, show=show,
                                          mercator_project=False).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(bounds)
    return fmap


def area_bar(areas):
    per_class = areas[areas["class"].notna()]
    fig = px.bar(per_class, x="severity", y="area_ha", color="severity",
                 color_discrete_sequence=SEVERITY["colors"], text_auto=".0f",
                 labels={"severity": "", "area_ha": "Area (ha)"})
    fig.update_layout(showlegend=False, height=380, margin=dict(t=30, b=10))
    fig.update_traces(marker_line_color="#555", marker_line_width=0.5)
    return fig


def dnbr_histogram(dnbr, thresholds):
    fig = go.Figure(go.Histogram(x=dnbr, nbinsx=120, marker_color="#333"))
    edges = [float(np.nanmin(dnbr))] + list(thresholds) + [float(np.nanmax(dnbr))]
    for lo, hi, color in zip(edges[:-1], edges[1:], SEVERITY["colors"]):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.45, line_width=0)
    fig.update_yaxes(type="log", title="Pixels (log)")
    fig.update_xaxes(title="dNBR")
    fig.update_layout(height=380, margin=dict(t=30, b=10), bargap=0)
    fig.add_annotation(x=1, y=1.06, xref="paper", yref="paper", showarrow=False, xanchor="right",
                       text="Shading: severity classes (colours as in the bar chart)", font_size=11)
    return fig


def change_scatter(bands, valid, classes, baseline, n=15000, seed=0):
    idx = np.flatnonzero(valid)
    idx = np.random.default_rng(seed).choice(idx, size=min(n, idx.size), replace=False)
    df = pd.DataFrame({
        "dNBR": bands["dNBR"].ravel()[idx],
        "dNDVI": bands["dNDVI"].ravel()[idx],
        "Severity": np.array(SEVERITY["labels"])[classes.ravel()[idx]],
    })
    fig = px.scatter(df, x="dNBR", y="dNDVI", color="Severity", opacity=0.5,
                     category_orders={"Severity": SEVERITY["labels"]},
                     color_discrete_sequence=SEVERITY["colors"])
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(legend=dict(itemsizing="constant"))
    fig.add_vline(x=baseline["dnbr_min"], line_dash="dash", line_color="#444")
    fig.add_hline(y=baseline["dndvi_min"], line_dash="dash", line_color="#444")
    fig.add_annotation(x=1, y=1, xref="paper", yref="paper", showarrow=False, xanchor="right",
                       text="Threshold rule: upper-right quadrant", font_size=11)
    fig.update_layout(height=420, margin=dict(t=30, b=10))
    return fig


def confusion_figure(confusion):
    names = list(confusion)
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names, horizontal_spacing=0.06)
    labels = ["Not burned", "Burned"]
    for k, name in enumerate(names, start=1):
        cm = confusion[name]
        share = cm / cm.sum(axis=1, keepdims=True)
        text = [[f"{share[i, j]:.0%}<br>({cm[i, j]})" for j in range(2)] for i in range(2)]
        fig.add_trace(go.Heatmap(z=share, x=labels, y=labels, text=text, texttemplate="%{text}",
                                 colorscale="Blues", zmin=0, zmax=1, showscale=False), row=1, col=k)
        fig.update_yaxes(autorange="reversed", row=1, col=k, showticklabels=(k == 1))
        fig.update_xaxes(title_text="Predicted", row=1, col=k)
    fig.update_yaxes(title_text="True", row=1, col=1)
    fig.update_layout(height=330, margin=dict(t=40, b=10))
    return fig


# App ------------------------------------------------------------------------

st.title("Burn severity explorer")
st.caption("Sentinel-2 dNBR / dNDVI analysis of forest fires, with a supervised classifier "
           "validated on held-out polygons.")

bands, profile, samples, source = select_data()
params = select_parameters()
is_demo = source == SOURCE_DEMO
if is_demo:
    st.info("Synthetic demo data: the landscape, lakes and fire are simulated to show the workflow. "
            "Select **Upload GEE exports** to explore a real fire.")

valid = analysis_mask(bands, params["ndvi_min"], params["mndwi_water"])
classes = classify_dnbr(bands["dNBR"], params["thresholds"], valid)
areas = class_areas(classes, SEVERITY["labels"], pixel_area_ha(profile))
with np.errstate(invalid="ignore"):
    rule = valid & (bands["dNBR"] > params["baseline"]["dnbr_min"]) & \
        (bands["dNDVI"] > params["baseline"]["dndvi_min"])
ha = pixel_area_ha(profile)

# Classifier results are kept while data and rule parameters stay the same
state_key = (source, profile["width"], profile["height"], tuple(params["thresholds"]),
             params["ndvi_min"], params["mndwi_water"], tuple(params["baseline"].values()))
if st.session_state.get("clf_key") != state_key:
    st.session_state.pop("clf", None)
clf = st.session_state.get("clf")

burned_row = areas[areas["severity"] == "Burned (low to high)"].iloc[0]
high_ha = areas.loc[areas["class"].isin([3, 4]), "area_ha"].sum()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Analysed vegetated area", f"{valid.sum() * ha:,.0f} ha")
m2.metric("Burned (dNBR ≥ low)", f"{burned_row['area_ha']:,.0f} ha", f"{burned_row['percent']:.1f}% of area",
          delta_color="off", delta_arrow="off")
m3.metric("Moderate-high and high severity", f"{high_ha:,.0f} ha")
m4.metric("Threshold rule (dNBR & dNDVI)", f"{rule.sum() * ha:,.0f} ha")

# Map (kept outside tabs: Leaflet cannot size itself inside a hidden tab)
with st.container():
    col_map, col_legend = st.columns([4, 1])
    severity_3857, bounds = to_web_mercator(classes.astype("float32"), profile)
    dnbr_3857, _ = to_web_mercator(bands["dNBR"], profile)
    rule_3857, _ = to_web_mercator(np.where(valid, rule, np.nan).astype("float32"), profile)
    severity_3857[severity_3857 < 0] = np.nan
    layers = [
        ("Burn severity", classes_to_rgba(severity_3857, SEVERITY["colors"]), True),
        ("dNBR", continuous_to_rgba(dnbr_3857, "RdYlBu_r", -0.3, 0.8), False),
        ("Threshold rule", mask_to_rgba(rule_3857, "#e31a1c"), False),
    ]
    if clf is not None:
        prob_3857, _ = to_web_mercator(clf["probability"], profile)
        layers.append((f"Burned probability ({clf['best']})",
                       continuous_to_rgba(prob_3857, "inferno", 0, 1), False))
    with col_map:
        fmap = build_map(layers, bounds, basemap=not is_demo)
        st_folium(fmap, height=540, use_container_width=True, returned_objects=[], key="map")
    with col_legend:
        st.markdown(legend_html(list(zip(SEVERITY["colors"], SEVERITY["labels"])), "Burn severity (dNBR)"),
                    unsafe_allow_html=True)
        st.markdown(legend_html([("#e31a1c", "dNBR and dNDVI above rule thresholds")], "Threshold rule"),
                    unsafe_allow_html=True)
        st.caption("Toggle layers with the control on the map. dNBR colour scale: −0.3 (blue) to 0.8 (red).")
        if clf is None:
            st.caption("Run the classifier in the **Classifier** tab to add its probability layer.")
        else:
            st.caption(f"Burned probability ({clf['best']}): black (0) to yellow (1).")

tab_stats, tab_clf, tab_about = st.tabs(["Severity statistics", "Classifier", "Method"])

with tab_stats:
    c1, c2 = st.columns(2)
    c1.subheader("Area by severity class")
    c1.plotly_chart(area_bar(areas), width="stretch")
    c2.subheader("dNBR distribution")
    c2.plotly_chart(dnbr_histogram(bands["dNBR"][valid], params["thresholds"]), width="stretch")

    table = areas.copy()
    table["class"] = table["class"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    st.dataframe(table.style.format({"pixels": "{:,.0f}", "area_ha": "{:,.1f}", "percent": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.download_button("Download table (CSV)", areas.to_csv(index=False), "severity_area_ha.csv", "text/csv")

    st.subheader("Change space: dNBR vs dNDVI")
    st.caption("Each point is a pixel (random sample). The dashed lines are the threshold rule.")
    st.plotly_chart(change_scatter(bands, valid, classes, params["baseline"]), width="stretch")

with tab_clf:
    if samples is None or len(samples) == 0:
        st.info("No training samples loaded. Draw `TRAINING` polygons in GEE (property `class`: "
                "1 = burned, 0 = not burned), export the samples CSV and upload it in the sidebar.")
    else:
        n_poly = samples[CLASSIFIER["group"]].nunique()
        st.markdown(f"**{len(samples):,} training pixels** from **{n_poly} polygons** "
                    f"({samples[CLASSIFIER['target']].mean():.0%} burned).")
        st.caption("Models are evaluated with cross-validation grouped by polygon: every polygon is "
                   "entirely in the training or the test set, so the scores reflect performance on new areas.")
        if st.button("Run cross-validation and map", type="primary", key="run_clf"):
            with st.spinner("Training and evaluating models..."):
                clf = run_classifier(samples, bands, valid, params["baseline"])
            st.session_state["clf"] = clf
            st.session_state["clf_key"] = state_key
            st.rerun()  # redraw the map with the probability layer

        if clf is not None:
            summary = clf["summary"]
            shown = summary[["model", "precision_mean", "recall_mean", "f1_mean", "balanced_accuracy_mean"]]
            shown.columns = ["Model", "Precision", "Recall", "F1", "Balanced accuracy"]
            st.subheader("Cross-validation on held-out polygons")
            st.dataframe(shown.style.format({c: "{:.2f}" for c in shown.columns[1:]}),
                         hide_index=True, width="stretch")
            st.plotly_chart(confusion_figure(clf["confusion"]), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"Feature importance ({clf['best']})")
                imp = clf["importance"].sort_values("mean")
                fig = px.bar(imp, x="mean", y="feature", error_x="std", orientation="h",
                             labels={"mean": "Decrease in F1 when permuted", "feature": ""})
                fig.update_traces(marker_color="#4a7ab5")
                fig.update_layout(height=380, margin=dict(t=10, b=10))
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.subheader("Burned area by method")
                model_ha = np.nansum(clf["probability"] > 0.5) * ha
                comp = pd.DataFrame({"Method": [RULE_NAME, f"{clf['best']} (p > 0.5)"],
                                     "Burned area (ha)": [rule.sum() * ha, model_ha]})
                st.dataframe(comp.style.format({"Burned area (ha)": "{:,.0f}"}), hide_index=True,
                             width="stretch")
                st.caption("The burned probability is added as a layer on the map above.")

with tab_about:
    st.markdown(f"""
**Workflow**

1. **Google Earth Engine** (`gee/burn_severity_s2.js`): Sentinel-2 L2A median composites before
   and after the fire, cloud masking with the Scene Classification Layer, NBR, NDVI and MNDWI,
   water and shoreline exclusion, and export of the index stack and training samples.
2. **Severity**: dNBR classes after Key & Benson (2006), restricted to land pixels vegetated before
   the fire. Current limits: {", ".join(f"{t:.2f}" for t in params["thresholds"])}.
3. **Threshold rule**: candidate burned pixels where dNBR > {params["baseline"]["dnbr_min"]:.2f}
   and dNDVI > {params["baseline"]["dndvi_min"]:.2f}.
4. **Classifier**: logistic regression, random forest and gradient boosting trained on labelled
   polygons, compared with the threshold rule using cross-validation grouped by polygon. The best
   model maps the burned probability of every pixel.

**Why a classifier?** Summer drying of grasslands and shrublands also lowers NBR and NDVI, and
low-severity burns can fall below fixed thresholds. Labelled examples let the model use the
whole spectral signature instead of two cut-offs.
""")
