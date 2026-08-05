from __future__ import annotations

import argparse
import json
import random

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.common.paths import FEATURES_ROOT, MODELS_ROOT, ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def time_split(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["target_date"] = pd.to_datetime(df["target_month"] + "-01")
    df = df.sort_values("target_date").reset_index(drop=True)

    split_idx = max(1, int(len(df) * (1.0 - val_fraction)))
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def select_features(df: pd.DataFrame, mode: str) -> list[str]:
    numeric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "target_proxy",
            "target_month",
            "tile_id",
            "time_window",
            "imagery_reference",
            "sample_month",
            "target_date",
            "dataset_version",
            "city",
        }
    ]
    if mode == "baseline":
        return [c for c in numeric_cols if c.endswith("_lag0") or c.startswith("stress_accum_")]
    return numeric_cols


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def top_decile_lift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    pred_threshold = float(np.quantile(y_pred, 0.9))
    events = (y_true >= float(np.quantile(y_true, 0.75))).astype(float)
    top_mask = y_pred >= pred_threshold
    baseline = float(np.mean(events))
    if baseline <= 0:
        return 0.0
    return float(np.mean(events[top_mask]) / baseline)


def metrics_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman_rank_correlation(y_true, y_pred),
        "top_decile_lift": top_decile_lift(y_true, y_pred),
    }


def build_model(mode: str):
    if mode == "baseline":
        return Ridge(alpha=1.0)
    if mode == "temporal_gb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            random_state=42,
        )
    if mode == "temporal_rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    if mode == "temporal_et":
        return ExtraTreesRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        )
    if mode == "temporal_lgbm":
        return LGBMRegressor(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if mode == "temporal_xgb":
        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    if mode == "temporal_stack":
        return TimeStackRegressor(
            estimators=[
                ("lgbm", build_model("temporal_lgbm")),
                ("xgb", build_model("temporal_xgb")),
                ("rf", build_model("temporal_rf")),
                ("et", build_model("temporal_et")),
                ("gb", build_model("temporal_gb")),
            ],
            meta_fraction=0.25,
        )
    raise ValueError(f"unsupported mode: {mode}")


class TimeStackRegressor(BaseEstimator, RegressorMixin):
    """Stacked ensemble with a temporal holdout for the meta-learner.

    Rows must arrive in time order. Base models are fit on the earliest
    (1 - meta_fraction) of the training data, the Ridge meta-learner is fit on
    their predictions over the most recent meta_fraction, then base models are
    refit on the full training data for inference.
    """

    def __init__(self, estimators: list[tuple[str, object]], meta_fraction: float = 0.25):
        self.estimators = estimators
        self.meta_fraction = meta_fraction

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(X)
        split = max(1, min(n - 1, int(n * (1.0 - self.meta_fraction))))

        holdout_preds = []
        for _, est in self.estimators:
            m = clone(est)
            m.fit(X[:split], y[:split])
            holdout_preds.append(m.predict(X[split:]))

        self.meta_ = Ridge(alpha=1.0)
        self.meta_.fit(np.column_stack(holdout_preds), y[split:])

        self.fitted_ = [clone(est) for _, est in self.estimators]
        for m in self.fitted_:
            m.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        base = np.column_stack([m.predict(X) for m in self.fitted_])
        return self.meta_.predict(base)


def train_one(train_df: pd.DataFrame, val_df: pd.DataFrame, mode: str) -> dict[str, object]:
    feats = select_features(train_df, mode)
    x_train = train_df[feats].to_numpy(dtype=float)
    y_train = train_df["target_proxy"].to_numpy(dtype=float)

    x_val = val_df[feats].to_numpy(dtype=float)
    y_val = val_df["target_proxy"].to_numpy(dtype=float)

    model = build_model(mode)

    model.fit(x_train, y_train)
    pred = model.predict(x_val)

    metrics = {
        "model": mode,
        "feature_count": len(feats),
        **metrics_payload(y_val, pred),
        "validation_rows": int(len(val_df)),
    }

    return {
        "model": model,
        "features": feats,
        "metrics": metrics,
        "pred": pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and temporal regression models.")
    parser.add_argument("--dataset", default=str(FEATURES_ROOT / "dataset.parquet"))
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = pd.read_parquet(args.dataset)
    train_df, val_df = time_split(df, args.val_fraction)

    modes = [
        "baseline",
        "temporal_gb",
        "temporal_rf",
        "temporal_et",
        "temporal_lgbm",
        "temporal_xgb",
        "temporal_stack",
    ]
    trained: dict[str, dict[str, object]] = {}
    for mode in modes:
        trained[mode] = train_one(train_df, val_df, mode)
        print(json.dumps(trained[mode]["metrics"]))

    out_dir = ensure_dir(MODELS_ROOT)
    for mode in modes:
        joblib.dump(
            {"model": trained[mode]["model"], "features": trained[mode]["features"], "model_name": mode},
            out_dir / f"{'baseline_model' if mode == 'baseline' else mode + '_model'}.joblib",
        )

    best_name, best_obj = max(
        trained.items(),
        key=lambda x: (
            float(x[1]["metrics"]["top_decile_lift"]),
            float(x[1]["metrics"]["spearman"]),
            float(x[1]["metrics"]["r2"]),
        ),
    )
    joblib.dump({"model": best_obj["model"], "features": best_obj["features"], "model_name": best_name}, out_dir / "best_model.joblib")

    val_target = val_df["target_proxy"].to_numpy(dtype=float)
    yearly_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_obj in trained.items():
        preds = np.asarray(model_obj["pred"], dtype=float)
        by_year: dict[str, dict[str, float]] = {}
        for year in sorted(val_df["target_date"].dt.year.unique()):
            mask = val_df["target_date"].dt.year == year
            if int(mask.sum()) == 0:
                continue
            by_year[str(int(year))] = metrics_payload(val_target[mask.to_numpy()], preds[mask.to_numpy()])
        yearly_metrics[model_name] = by_year

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "seed": args.seed,
        **{mode: trained[mode]["metrics"] for mode in modes},
        "validation_metrics_by_year": yearly_metrics,
        "best_model": best_name,
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
