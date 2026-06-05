"""
model_kill_share.py
-------------------
Model 2: Predicts player kill share % (kills / team_kills).

Separate model per position — a support and mid laner have
completely different kill share distributions.

Target: kill_share (player kills / team kills, range 0-1)
Final kills = team_kills_prediction × kill_share_prediction
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

POSITIONS   = ["top", "jng", "mid", "bot", "sup"]
MODEL_DIR   = Path("models")

PARAMS = {
    "objective":            "regression_l1",
    "metric":               "mae",
    "learning_rate":        0.05,
    "num_leaves":           31,
    "min_child_samples":    15,
    "feature_fraction":     0.8,
    "bagging_fraction":     0.8,
    "bagging_freq":         5,
    "reg_alpha":            0.1,
    "reg_lambda":           0.1,
    "n_estimators":         500,
    "early_stopping_rounds":50,
    "verbose":              -1,
    "n_jobs":               -1,
    "random_state":         42,
}


def _model_path(position: str) -> Path:
    return MODEL_DIR / f"kill_share_{position}.pkl"


def _time_split(df, feature_cols, target):
    dates = df["date"].dropna().sort_values()
    n = len(dates)
    val_cut  = dates.iloc[int(n * 0.60)].date().isoformat()
    test_cut = dates.iloc[int(n * 0.80)].date().isoformat()

    valid = df.dropna(subset=[target] + feature_cols)
    train = valid[valid["date"] < val_cut]
    val   = valid[(valid["date"] >= val_cut) & (valid["date"] < test_cut)]
    test  = valid[valid["date"] >= test_cut]

    print(f"    train={len(train):,} val={len(val):,} test={len(test):,}")
    return (train[feature_cols], train[target],
            val[feature_cols],   val[target],
            test[feature_cols],  test[target])


def train_all(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train one kill share model per position."""
    MODEL_DIR.mkdir(exist_ok=True)
    results = {}

    for pos in POSITIONS:
        print(f"\n  [kill_share] Training position: {pos.upper()}")
        pos_df = df[df["position"] == pos].copy()

        if len(pos_df) < 100:
            print(f"    Skipping {pos} — only {len(pos_df)} rows")
            continue

        target = "kill_share"
        X_tr, y_tr, X_va, y_va, X_te, y_te = _time_split(pos_df, feature_cols, target)

        if len(X_tr) < 50:
            print(f"    Skipping {pos} — not enough training data")
            continue

        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        preds = model.predict(X_te).clip(0, 1)
        mae   = mean_absolute_error(y_te, preds)
        print(f"    MAE={mae:.4f} | mean_actual={y_te.mean():.3f} | mean_pred={preds.mean():.3f}")

        path = _model_path(pos)
        with open(path, "wb") as f:
            pickle.dump({
                "model":        model,
                "feature_cols": feature_cols,
                "mae":          mae,
                "pos_avg":      float(y_te.mean()),
            }, f)
        print(f"    Saved to {path}")
        results[pos] = {"mae": mae, "pos_avg": float(y_te.mean())}

    return results


def load(position: str) -> tuple:
    path = _model_path(position)
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["feature_cols"], d["mae"], d.get("pos_avg", 0.2)


def predict(model, X: pd.DataFrame, pos_avg: float) -> float:
    """Predict kill share for one player."""
    val = float(model.predict(X)[0])
    # Clamp to realistic range per position
    limits = {
        "top": (0.05, 0.55),
        "jng": (0.05, 0.55),
        "mid": (0.05, 0.60),
        "bot": (0.05, 0.65),
        "sup": (0.01, 0.30),
    }
    return val  # clamping applied in predict.py with position context
