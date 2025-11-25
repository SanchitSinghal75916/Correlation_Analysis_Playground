# pages/1_Modeler.py
import re
import numpy as np
import pandas as pd
import streamlit as st

from utils.transforms import (
    coerce_numeric_cols,
    suggest_transform,
    apply_table_transforms_no_pipeline,
)
from utils.modeling import (
    run_models_no_pipeline_parallel,
    compute_feature_relevance,
)
from utils.plots import small_text_heatmap, barplot_metric
from utils.io_utils import excel_bytes_from_sheets


st.set_page_config(page_title="Scotia Deposits: Table-Driven Modeling", layout="wide")
st.title("Scotia Deposits: Table-Driven Modeling")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Speed / Controls")
    fast_mode = st.toggle(
        "Fast mode (quicker, slightly less accurate)",
        value=False,
        help="Cuts CV folds to 3 and reduces RF trees to 200.",
    )
    folds = 3 if fast_mode else 5

# ----------------------------
# 1) Upload
# ----------------------------
st.subheader("1) Upload CSV")
st.caption("Bring your dataset. We'll parse numerics like $, %, and commas automatically.")
up = st.file_uploader("Upload CSV", type=["csv"], key="uploader")
if up is None:
    st.info("Upload your CSV to continue.")
    st.stop()

@st.cache_data(show_spinner=False)
def _read_csv(file):
    return pd.read_csv(file)

df_raw = _read_csv(up)
coerced = coerce_numeric_cols(df_raw)

# ----------------------------
# 2) Select Target & Features
# ----------------------------
st.subheader("2) Select Target & Features")
st.caption("Pick what you're predicting (target) and which numeric columns to use as inputs.")

all_cols = list(coerced.columns)
default_target = "Scotia Deposits" if "Scotia Deposits" in all_cols else all_cols[0]
TARGET = st.selectbox("Select target", options=all_cols, index=all_cols.index(default_target))

id_like = [c for c in all_cols if re.search(r"(CMA/CSD|id|identifier)", c, flags=re.I)]
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(coerced[c])]
default_features = [c for c in numeric_cols if c not in id_like + [TARGET]]
FEATURES = st.multiselect("Feature columns", options=numeric_cols, default=default_features)

if not FEATURES:
    st.warning("Pick at least one feature.")
    st.stop()

# ----------------------------
# 3) Transformations table
# ----------------------------
st.subheader("3) Set Transformations")
st.caption("We suggest transforms from skew/scale. Edit before applying if needed.")

rows = []
# Target first
target_tfm, target_hint = suggest_transform(coerced[TARGET])
rows.append({"Feature": TARGET, "Transformation": target_tfm, "Hint": f"(Target) {target_hint}"})
# Then features
for c in FEATURES:
    tfm, hint = suggest_transform(coerced[c])
    rows.append({"Feature": c, "Transformation": tfm, "Hint": hint})

trans_df = pd.DataFrame(rows)
trans_df = st.data_editor(
    trans_df,
    num_rows="fixed",
    column_config={
        "Transformation": st.column_config.SelectboxColumn(
            "Transformation", options=["None", "Log1p", "Standardize", "YeoJohnson"]
        ),
        "Hint": st.column_config.TextColumn(disabled=True),
    },
    key="trans_table_editor",
)

# ----------------------------
# 4) Apply transforms
# ----------------------------
if st.button("Apply from Table", key="apply_btn"):
    # Separate target row vs feature rows
    target_row = trans_df[trans_df["Feature"] == TARGET]
    feature_rows = trans_df[trans_df["Feature"] != TARGET]

    # Apply to features
    Xsel = coerced[FEATURES].copy()
    X_tx, notes_features = apply_table_transforms_no_pipeline(Xsel, feature_rows)

    # Apply to target
    y_raw = coerced[TARGET].copy()
    y_tx_df, notes_target = apply_table_transforms_no_pipeline(
        y_raw.to_frame(name=TARGET), target_row
    )
    y_tx = y_tx_df[TARGET]

    notes = notes_features + notes_target

    # Persist for later sections
    st.session_state["X_tx"] = X_tx
    st.session_state["y"] = y_tx
    st.session_state["y_raw"] = y_raw
    st.session_state["trans_table_final"] = trans_df

    st.success("Transforms applied to both features and target.")
    with st.expander("What we applied"):
        for n in notes:
            st.write("- ", n)

# Require transforms before proceeding
if "X_tx" not in st.session_state:
    st.stop()

X_tx = st.session_state["X_tx"].copy()
y = st.session_state["y"].copy()
y_raw = st.session_state["y_raw"].copy()

# ----------------------------
# 5) Row handling summary
# ----------------------------
mask_all = X_tx.notna().all(axis=1) & y.notna()
X_all = X_tx[mask_all]
y_all = y[mask_all]

mask_nz = mask_all & (y_raw != 0)
X_nz = X_tx[mask_nz]
y_nz = y[mask_nz]

st.info(
    f"Row handling: kept {len(y_all)} rows WITH zeros; {len(y_nz)} rows WITHOUT zeros. "
    f"Dropped {len(y) - len(y_all)} rows with NaNs."
)

# ----------------------------
# 6) Correlation Heatmap
# ----------------------------
st.subheader("4) Correlation (small text)")
st.caption("Quickly see linear relationships across inputs and the target (with zeros).")

try:
    corr = pd.concat([X_all, y_all.rename(TARGET)], axis=1).corr(numeric_only=True)
    small_text_heatmap(corr, "Correlation Heatmap (small text)")
except Exception as e:
    st.warning(f"Correlation failed: {e}")

# ----------------------------
# 7) Preview splits
# ----------------------------
st.subheader("5) Preview train/test splits")
st.caption("20% test split, fixed random seed, shown for both scenarios.")

if st.button("Preview splits", key="preview_btn"):
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    st.write("WITH zeros — train:", Xtr.shape, " test:", Xte.shape)
    st.dataframe(pd.concat([Xtr.head(), ytr.head()], axis=1))

    Xtr2, Xte2, ytr2, yte2 = train_test_split(X_nz, y_nz, test_size=0.2, random_state=42)
    st.write("WITHOUT zeros — train:", Xtr2.shape, " test:", Xte2.shape)
    st.dataframe(pd.concat([Xtr2.head(), ytr2.head()], axis=1))

# ----------------------------
# 8) Train models (parallel batch ONLY)
# ----------------------------
st.subheader("6) Train models (both scenarios)")
st.caption(
    f"We compare 7 models with cross-validation ({folds} folds). "
    f"Use the sidebar toggle if you want a quicker run."
)

start_batch = st.button("Run training (parallel batch)", key="train_batch_btn")

# Placeholders so results persist even if other buttons are clicked
ph_with = st.empty()
ph_without = st.empty()
ph_charts = st.empty()

if start_batch:
    res_with, res_without, combo = run_models_no_pipeline_parallel(
        X_all, y_all, X_nz, y_nz, folds=folds, fast=fast_mode
    )
    st.session_state["res_with"] = res_with
    st.session_state["res_without"] = res_without
    st.session_state["combo_results"] = combo

# Always render persisted results if present
if "res_with" in st.session_state and "res_without" in st.session_state:
    with ph_with.container():
        st.subheader("Results — WITH zeros in scotia deposits")
        st.dataframe(st.session_state["res_with"], use_container_width=True)
    with ph_without.container():
        st.subheader("Results — WITHOUT zeros in scotia deposits")
        st.dataframe(st.session_state["res_without"], use_container_width=True)
    with ph_charts.container():
        st.subheader("Final model comparison")
        combo = st.session_state["combo_results"]
        if combo is not None and not combo.empty:
            barplot_metric(combo, metric="Test_RMSE", title="Test_RMSE (lower is better)")
            barplot_metric(combo, metric="Test_R2",   title="Test_R² (higher is better)")
        else:
            st.info("No results to plot yet.")

# ----------------------------
# 9) Export
# ----------------------------
if (
    "res_with" in st.session_state
    and "res_without" in st.session_state
    and not st.session_state["res_with"].empty
    and not st.session_state["res_without"].empty
):
    xbytes = excel_bytes_from_sheets(
        {
            "transform_table": st.session_state.get("trans_table_final", pd.DataFrame()).reset_index(drop=True),
            "results_with_zeros": st.session_state["res_with"].reset_index(drop=True),
            "results_without_zeros": st.session_state["res_without"].reset_index(drop=True),
        }
    )
    st.download_button(
        "Download Excel (transforms + both results)",
        data=xbytes,
        file_name="scotia_deposits_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_btn",
    )

# ----------------------------
# 10) Feature relevance
# ----------------------------
st.subheader("7) Feature relevance")
st.caption("Multiple signals: Pearson r, Mutual Info, Tree importances, Ridge coefficients.")

if st.button("Compute feature relevance", key="relevance_btn"):
    rel_df = compute_feature_relevance(X_all, y_all)
    st.session_state["rel_df"] = rel_df

if "rel_df" in st.session_state:
    st.dataframe(st.session_state["rel_df"], use_container_width=True)
    st.download_button(
        "Download feature relevance (CSV)",
        data=st.session_state["rel_df"].to_csv(index=False).encode(),
        file_name="feature_relevance.csv",
        mime="text/csv",
        key="rel_dl",
    )

with st.expander("Model family relevance"):
    st.markdown(
        "- **Linear/Ridge/Lasso**: baselines; handle multicollinearity & shrinkage.\n"
        "- **RandomForest/DecisionTree**: nonlinear splits & interactions; robust to monotone transforms.\n"
        "- **GradientBoosting**: strong tabular baseline; captures complex patterns.\n"
        "- **SVR**: kernel method; benefits from scaling chosen in the table."
    )
