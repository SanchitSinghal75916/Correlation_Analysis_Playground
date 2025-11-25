# utils/modeling.py
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from joblib import Parallel, delayed


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cv_metrics(model, X_train, y_train, n_splits: int) -> Tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_vals, r2_vals = [], []
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = model.predict(X_train.iloc[val_idx])
        rmse_vals.append(_rmse(y_train.iloc[val_idx], pred))
        r2_vals.append(r2_score(y_train.iloc[val_idx], pred))
    return float(np.mean(rmse_vals)), float(np.mean(r2_vals))


def _fit_and_score(named_model, X, y, folds: int):
    name, model = named_model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.loc[X.index], test_size=0.2, random_state=42
    )
    n_splits = int(min(max(2, folds), 10, len(X_train)))

    cv_rmse, cv_r2 = _cv_metrics(model, X_train, y_train, n_splits)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return [
        name,
        cv_rmse,
        cv_r2,
        _rmse(y_test, pred),
        mean_absolute_error(y_test, pred),
        r2_score(y_test, pred),
    ]


def _model_list(fast: bool = False) -> List[Tuple[str, object]]:
    rf_estimators = 200 if fast else 400
    return [
        ("LinearRegression", LinearRegression()),
        ("RidgeCV", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ("LassoCV", LassoCV(alphas=np.logspace(-3, 3, 25), random_state=42, max_iter=10000)),
        ("RandomForest", RandomForestRegressor(n_estimators=rf_estimators, random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("DecisionTree", DecisionTreeRegressor(random_state=42)),
        ("SVR", SVR()),
    ]


def _results_df(results_list: list) -> pd.DataFrame:
    return pd.DataFrame(
        results_list,
        columns=["Model", "CV_RMSE", "CV_R2", "Test_RMSE", "Test_MAE", "Test_R2"],
    ).sort_values("Test_RMSE")


def run_models_no_pipeline_parallel(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_nz: pd.DataFrame,
    y_nz: pd.Series,
    folds: int = 5,
    fast: bool = False,
):
    """
    Run the 7 models in parallel (per scenario) and return:
    (res_with, res_without, combo)
    """
    models = _model_list(fast=fast)

    # Parallel across models for each scenario
    res_with_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(_fit_and_score)(m, X_all, y_all, folds) for m in models
    )
    res_without_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(_fit_and_score)(m, X_nz, y_nz, folds) for m in models
    )

    res_with = _results_df(res_with_list)
    res_without = _results_df(res_without_list)

    res_with["Scenario"] = "With zeros"
    res_without["Scenario"] = "Without zeros"
    combo = pd.concat([res_with, res_without], ignore_index=True)
    return res_with, res_without, combo


# ----------------------------
# Feature relevance
# ----------------------------
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV as _RidgeCV
from sklearn.ensemble import RandomForestRegressor as _RF, GradientBoostingRegressor as _GB


def compute_feature_relevance(Xr: pd.DataFrame, yr: pd.Series) -> pd.DataFrame:
    pearson = Xr.apply(lambda c: c.corr(yr))
    mi = mutual_info_regression(Xr, yr, random_state=42)

    rf = _RF(n_estimators=400, random_state=42).fit(Xr, yr)
    gb = _GB(random_state=42).fit(Xr, yr)

    x_mean = Xr.mean()
    x_std = Xr.std(ddof=0).replace(0, 1)
    Xz = (Xr - x_mean) / x_std
    y_std = yr.std(ddof=0) or 1.0
    yz = (yr - yr.mean()) / y_std
    ridge = _RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xz, yz)

    rel_df = pd.DataFrame({
        "Feature": Xr.columns,
        "Pearson_r": pearson.values,
        "|r|": np.abs(pearson.values),
        "Mutual_Info": mi,
        "RF_Importance": rf.feature_importances_,
        "GB_Importance": gb.feature_importances_,
        "Ridge_|coef|": np.abs(ridge.coef_),
    })

    rel_df = rel_df.sort_values(
        ["Mutual_Info", "RF_Importance", "GB_Importance", "|r|", "Ridge_|coef|"],
        ascending=False,
    ).reset_index(drop=True)
    return rel_df
