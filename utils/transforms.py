
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from typing import List, Tuple


def coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        out[c] = (
            out[c].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def suggest_transform(col: pd.Series) -> Tuple[str, str]:
    s = col.dropna()
    if len(s) == 0:
        return "None", "All NaN — no transform."
    skew = s.skew()
    pos_only = (s > 0).all()
    rng = s.max() - s.min()
    mean = s.mean()
    if pos_only and abs(skew) > 1.0:
        return "Log1p", f"Right-skewed (+ only), skew={skew:.2f} → log1p."
    if not pos_only and abs(skew) > 1.0:
        return "YeoJohnson", f"Skewed with ≤0 values, skew={skew:.2f} → Yeo-Johnson."
    if abs(mean) > 1e3 or rng > 1e4:
        return "Standardize", f"Large scale (mean≈{mean:.0f}, range≈{rng:.0f}) → z-score."
    return "None", f"Skew={skew:.2f}, range≈{rng:.0f} → none."


def apply_table_transforms_no_pipeline(
    X: pd.DataFrame, table_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    Xt = X.copy()
    notes: List[str] = []

    table_df = table_df[table_df["Feature"].isin(Xt.columns)].copy()
    if table_df.empty:
        return Xt, notes

    # Log1p
    log_cols = table_df.loc[table_df["Transformation"] == "Log1p", "Feature"].tolist()
    if log_cols:
        for c in log_cols:
            Xt[c] = np.log1p(Xt[c].clip(lower=0))
        notes.append("log1p → " + ", ".join(log_cols))

    # Yeo-Johnson
    yj_cols = table_df.loc[table_df["Transformation"] == "YeoJohnson", "Feature"].tolist()
    if yj_cols:
        pt = PowerTransformer(method="yeo-johnson")
        Xt[yj_cols] = pt.fit_transform(Xt[yj_cols])
        notes.append("Yeo-Johnson → " + ", ".join(yj_cols))

    # Standardize
    std_cols = table_df.loc[table_df["Transformation"] == "Standardize", "Feature"].tolist()
    if std_cols:
        scaler = StandardScaler()
        Xt[std_cols] = scaler.fit_transform(Xt[std_cols])
        notes.append("z-score → " + ", ".join(std_cols))

    none_cols = table_df.loc[table_df["Transformation"] == "None", "Feature"].tolist()
    if none_cols:
        notes.append("no transform → " + ", ".join(none_cols))

    return Xt, notes
