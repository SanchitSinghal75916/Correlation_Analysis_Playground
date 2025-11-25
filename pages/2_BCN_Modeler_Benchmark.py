# pages/2_BCN_modelr_State_1.py
# BCN_modelr (State 1): Fixed, read-only page for client review
# - Reads the exported Excel + correlation image from a local "files" directory (no uploads).
# - Shows RMSE & R² tables/charts, the exact transformations used, and the variables chosen.

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Fixed file locations (adjust names if yours differ)
# ----------------------------
FILES_DIR = r"files/"
RESULTS_XLSX = os.path.join(FILES_DIR, r"scotia_results.xlsx")
CORR_IMG = os.path.join(FILES_DIR, r"HeatMapCorr.png")  # saved image from Modeler

st.set_page_config(page_title="BCN_modelr (State 1)", layout="wide")
st.title("BCN_modelr (State 1) — Read-only Presentation View")

st.caption(
    "This page is fixed. "
    "shows exactly what was shared in the presentation: RMSE & R² comparisons, the chosen "
    "variables, the exact transformations, and the saved correlation heatmap."
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def _read_results_excel(path: str) -> dict:
    x = pd.ExcelFile(path)
    sheets = {}
    for name in ["transform_table", "results_with_zeros", "results_without_zeros"]:
        sheets[name] = pd.read_excel(x, sheet_name=name) if name in x.sheet_names else pd.DataFrame()
    return sheets

def _barplot_metric(df: pd.DataFrame, metric: str, title: str):
    if metric not in df.columns:
        st.info(f"Metric '{metric}' not found.")
        return
    data = df.dropna(subset=[metric]).copy()
    if data.empty:
        st.info(f"No data for '{metric}'.")
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(data=data, x="Model", y=metric, hue="Scenario", ax=ax)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(title="", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def _clean_perf(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["Model", "CV_RMSE", "CV_R2", "Test_RMSE", "Test_MAE", "Test_R2", "Scenario"]
    present = [c for c in keep if c in df.columns]
    out = df[present].copy()
    # Ensure numeric
    for c in ["CV_RMSE", "CV_R2", "Test_RMSE", "Test_MAE", "Test_R2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ----------------------------
# 1) Load from files/
# ----------------------------
st.subheader("1) Data source (fixed)")
st.caption("Reading from the local **files/** directory; no uploads needed.")

exists_xlsx = os.path.exists(RESULTS_XLSX)
exists_img = os.path.exists(CORR_IMG)

c1, c2 = st.columns(2)
with c1:
    st.write("**Results Excel:**", RESULTS_XLSX, "✅" if exists_xlsx else "❌")
with c2:
    st.write("**Correlation image:**", CORR_IMG, "✅" if exists_img else "❌")

if not exists_xlsx:
    st.error("The results Excel was not found. Please place the exported file in the files/ folder.")
    st.stop()

sheets = _read_results_excel(RESULTS_XLSX)

# ----------------------------
# 2) Transformations used (fixed list from presentation)
# ----------------------------
st.subheader("2) Transformations used (State 1)")
st.caption("Exactly as presented to the client.")

transform_map = {
    "Yeo-Johnson": [
        "Scotia Number of Branches",
        "Total Population",
        "Land_area_in_square_kilometres",
        "Population_density_per_square_kilometre",
        "Average age",
        "Total inequality measures",
        "Average value of dwellings",
        "Pupulation for education (15+)",
        "Bachelor´s degree or higher",
        "% with higher education",
        "Participation rate",
        "Unemployment rate",
        "BMO", "CIBC", "Desjardins", "NBC", "RBC", "TD", "RMS",
        "Scotia Deposits",
    ],
    "z-score": ["Median income after tax 2020"],
    "no transform": ["Average household size", "Gini index", "Employment rate"],
}

rows = []
for tfm, cols in transform_map.items():
    for c in cols:
        rows.append({"Transformation": tfm, "Variable": c})

tfm_df = pd.DataFrame(rows)
st.dataframe(tfm_df, use_container_width=True, height=min(420, 40 + 24 * len(rows)))

# ----------------------------
# 3) Variables chosen (from transform_table sheet)
# ----------------------------
st.subheader("3) Variables chosen")
st.caption("Parsed from the transform_table: all features except the target.")

trans_tbl = sheets.get("transform_table", pd.DataFrame()).copy()
chosen_vars = []
target_guess = None

if not trans_tbl.empty and "Feature" in trans_tbl.columns:
    # Heuristic: in our export, the target appears as the first row
    target_guess = str(trans_tbl.iloc[0]["Feature"])
    chosen_vars = [str(v) for v in trans_tbl["Feature"].tolist() if v != target_guess]

cleft, cright = st.columns([2, 3])
with cleft:
    st.write("**Target:**", target_guess or "Unknown")
    st.write("**# of chosen variables:**", len(chosen_vars))
with cright:
    if chosen_vars:
        st.write(", ".join(chosen_vars))
    else:
        st.info("No variables found in transform table.")

# ----------------------------
# 4) Model performance (R² & RMSE)
# ----------------------------
st.subheader("4) Model performance (R² & RMSE)")
st.caption("Side-by-side tables and compact comparison charts.")

res_with = sheets.get("results_with_zeros", pd.DataFrame()).copy()
res_without = sheets.get("results_without_zeros", pd.DataFrame()).copy()

# Ensure scenario labels if missing
if not res_with.empty and "Scenario" not in res_with.columns:
    res_with["Scenario"] = "With zeros"
if not res_without.empty and "Scenario" not in res_without.columns:
    res_without["Scenario"] = "Without zeros"

res_with = _clean_perf(res_with)
res_without = _clean_perf(res_without)
combo = _clean_perf(pd.concat([res_with, res_without], ignore_index=True))

t1, t2 = st.columns(2)
with t1:
    st.write("**With zeros**")
    st.dataframe(res_with.sort_values("Test_RMSE"), use_container_width=True)
with t2:
    st.write("**Without zeros**")
    st.dataframe(res_without.sort_values("Test_RMSE"), use_container_width=True)

st.markdown("**Charts**")
_barplot_metric(combo, "Test_RMSE", "Test RMSE (lower is better)")
_barplot_metric(combo, "Test_R2", "Test R² (higher is better)")

# Best models table
st.markdown("**Best models by scenario (lowest RMSE)**")
best_rows = []
for scen, g in combo.groupby("Scenario"):
    g2 = g.dropna(subset=["Test_RMSE"])
    if not g2.empty:
        best = g2.loc[g2["Test_RMSE"].idxmin()]
        best_rows.append(
            {
                "Scenario": scen,
                "Model": best["Model"],
                "Test_RMSE": float(best["Test_RMSE"]),
                "Test_R2": float(best["Test_R2"]) if "Test_R2" in best and not pd.isna(best["Test_R2"]) else np.nan,
            }
        )
if best_rows:
    st.dataframe(pd.DataFrame(best_rows), use_container_width=True)
else:
    st.info("No best-model summary available (empty results).")

# ----------------------------
# 5) Correlation heatmap (saved image)
# ----------------------------
st.subheader("5) Correlation heatmap (saved image)")
st.caption("Displays the correlation image saved during modeling.")
if exists_img:
    st.image(CORR_IMG, caption="Correlation Heatmap", use_column_width=True)
else:
    st.info("Correlation image was not found in files/. Place the image at the path shown above.")
