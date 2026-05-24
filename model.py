"""
model.py
--------
Trains and evaluates LightGBM regression models to predict
player kills, deaths, and assists.

One model per target (kills / deaths / assists).
Supports per-position models for better accuracy.

Evaluation uses time-based train/test split to prevent leakage.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from typing import Optional

warnings.filterwarnings("ignore")

TARGETS = ["kills", "deaths", "assists"]
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── LightGBM hyperparameters (sensible defaults; tune with tune_hyperparams) ──
DEFAULT_PARAMS = {
    "objective": "regression_l1",   # MAE loss — robust to outlier games
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ── Data splitting ────────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    test_cutoff: str = "2024-07-01",
    val_cutoff: str = "2024-01-01",
):
    """
    Chronological train/val/test split.
    - Train: everything before val_cutoff
    - Val:   val_cutoff → test_cutoff  (used for early stopping)
    - Test:  test_cutoff → present     (held out, never touched during training)
    """
    df = df.dropna(subset=[target] + feature_cols).copy()

    train = df[df["date"] < val_cutoff]
    val   = df[(df["date"] >= val_cutoff) & (df["date"] < test_cutoff)]
    test  = df[df["date"] >= test_cutoff]

    print(f"  [{target}] Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    X_train, y_train = train[feature_cols], train[target]
    X_val,   y_val   = val[feature_cols],   val[target]
    X_test,  y_test  = test[feature_cols],  test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train, y_train,
    X_val,   y_val,
    params: dict = None,
    target_name: str = "stat",
) -> lgb.LGBMRegressor:
    """Train a single LightGBM model with early stopping on val."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False),
                   lgb.log_evaluation(-1)],
    )
    best = model.best_iteration_
    print(f"  [{target_name}] Best iteration: {best}")
    return model


def evaluate(model, X_test, y_test, target_name: str) -> dict:
    """Compute MAE, RMSE, and within-0.5 / within-1.0 accuracy."""
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)   # KDA can't be negative

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    within_half = np.mean(np.abs(preds - y_test) <= 0.5)
    within_one  = np.mean(np.abs(preds - y_test) <= 1.0)

    metrics = {
        "target":       target_name,
        "n_test":       len(y_test),
        "mae":          round(mae, 4),
        "rmse":         round(rmse, 4),
        "within_0.5":   round(within_half, 4),
        "within_1.0":   round(within_one, 4),
        "mean_actual":  round(float(y_test.mean()), 4),
        "mean_pred":    round(float(preds.mean()), 4),
    }
    return metrics, preds


def print_metrics(metrics: dict):
    print(f"\n  ── {metrics['target'].upper()} ──")
    print(f"     MAE:         {metrics['mae']:.3f}  (avg error in {metrics['target']} per game)")
    print(f"     RMSE:        {metrics['rmse']:.3f}")
    print(f"     Within ±0.5: {metrics['within_0.5']*100:.1f}% of predictions")
    print(f"     Within ±1.0: {metrics['within_1.0']*100:.1f}% of predictions")
    print(f"     Mean actual: {metrics['mean_actual']:.2f}  |  Mean pred: {metrics['mean_pred']:.2f}")


# ── Feature importance ────────────────────────────────────────────────────────

def top_features(model, feature_cols: list[str], n: int = 20) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(n)
    return imp


def shap_summary(model, X_sample: pd.DataFrame, target_name: str):
    """Print top SHAP contributors for interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_|shap|": mean_abs
    }).sort_values("mean_|shap|", ascending=False).head(15)
    print(f"\n  SHAP top features for {target_name}:")
    print(top.to_string(index=False))


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, target: str, metrics: dict, feature_cols: list[str]):
    path = MODEL_DIR / f"{target}_model.pkl"
    payload = {
        "model":        model,
        "feature_cols": feature_cols,
        "metrics":      metrics,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  [save] Model saved to {path}")

    # Also save metrics as JSON for easy inspection
    mpath = MODEL_DIR / f"{target}_metrics.json"
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)


def load_model(target: str) -> tuple:
    path = MODEL_DIR / f"{target}_model.pkl"
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"], payload["metrics"]


# ── Full training run ─────────────────────────────────────────────────────────

def train_all(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_cutoff: str = None,
    val_cutoff:  str = None,
    shap_n_samples: int = 500,
) -> dict:
    """
    Train one model per target (kills/deaths/assists).
    Returns dict of {target: metrics}.
    """
    all_metrics = {}

    # Dynamically compute cutoffs based on actual data range
    # Use last 20% as test, previous 20% as val, rest as train
    dates = df["date"].dropna().sort_values()
    min_date = dates.iloc[0]
    max_date = dates.iloc[-1]
    total_days = (max_date - min_date).days

    if test_cutoff is None:
        test_cutoff  = (min_date + pd.Timedelta(days=int(total_days * 0.80))).strftime("%Y-%m-%d")
    if val_cutoff is None:
        val_cutoff   = (min_date + pd.Timedelta(days=int(total_days * 0.60))).strftime("%Y-%m-%d")

    print(f"  [split] val_cutoff={val_cutoff}  test_cutoff={test_cutoff}")

    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f" Training: {target.upper()}")
        print(f"{'='*50}")

        X_train, y_train, X_val, y_val, X_test, y_test = time_split(
            df, feature_cols, target, test_cutoff, val_cutoff
        )

        model = train_model(X_train, y_train, X_val, y_val, target_name=target)

        metrics, preds = evaluate(model, X_test, y_test, target)
        print_metrics(metrics)

        # Feature importance
        imp = top_features(model, feature_cols)
        print(f"\n  Top 10 features:")
        print(imp.head(10).to_string(index=False))

        # SHAP (on small sample for speed)
        sample = X_test.sample(min(shap_n_samples, len(X_test)), random_state=42)
        shap_summary(model, sample, target)

        save_model(model, target, metrics, feature_cols)
        all_metrics[target] = metrics

    return all_metrics


if __name__ == "__main__":
    from data_ingestion import load_raw, filter_major_leagues
    from feature_engineering import build_features, get_feature_columns

    print("=== Model Training Pipeline ===\n")
    raw   = load_raw(years=[2022, 2023, 2024])
    major = filter_major_leagues(raw)
    feat  = build_features(major)
    fcols = get_feature_columns(feat)

    results = train_all(feat, fcols)

    print("\n\n=== FINAL RESULTS ===")
    for t, m in results.items():
        print(f"  {t:8s} MAE={m['mae']:.3f}  within±1={m['within_1.0']*100:.1f}%")
