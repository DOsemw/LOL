"""
model_team_kills.py
-------------------
Model 1: Predicts expected total team kills for a single map.

This is a regression model trained on team-level features.
One model for all positions (team kills is a team-level stat).

Target: team_kills (total kills scored by this team in this game)
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

MODEL_PATH = Path("models/team_kills_model.pkl")

PARAMS = {
    "objective":            "regression_l1",
    "metric":               "mae",
    "learning_rate":        0.05,
    "num_leaves":           31,
    "min_child_samples":    20,
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


def _time_split(df, feature_cols, target):
    """Chronological 60/20/20 split."""
    dates = df["date"].dropna().sort_values()
    n = len(dates)
    val_cut  = dates.iloc[int(n * 0.60)].date().isoformat()
    test_cut = dates.iloc[int(n * 0.80)].date().isoformat()

    train = df[df["date"] < val_cut]
    val   = df[(df["date"] >= val_cut) & (df["date"] < test_cut)]
    test  = df[df["date"] >= test_cut]

    valid = df.dropna(subset=[target] + feature_cols)
    train = valid[valid["date"] < val_cut]
    val   = valid[(valid["date"] >= val_cut) & (valid["date"] < test_cut)]
    test  = valid[valid["date"] >= test_cut]

    print(f"  [team_kills] train={len(train):,} val={len(val):,} test={len(test):,}")
    return (train[feature_cols], train[target],
            val[feature_cols],   val[target],
            test[feature_cols],  test[target])


def train(df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMRegressor:
    """Train team kills model on deduplicated team-game rows."""
    # One row per team per game (not per player)
    team_df = df.drop_duplicates(["gameid", "teamname"]).copy()
    target  = "team_kills"

    X_tr, y_tr, X_va, y_va, X_te, y_te = _time_split(team_df, feature_cols, target)

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X_te).clip(min=0)
    mae   = mean_absolute_error(y_te, preds)
    print(f"  [team_kills] MAE={mae:.3f} | mean_actual={y_te.mean():.2f} | mean_pred={preds.mean():.2f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "mae": mae}, f)
    print(f"  [team_kills] Saved to {MODEL_PATH}")
    return model


def load() -> tuple:
    with open(MODEL_PATH, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["feature_cols"], d["mae"]


def predict(model, X: pd.DataFrame) -> float:
    """Predict team kills for one team in one game."""
    val = float(model.predict(X)[0])
    return max(1.0, val)  # at least 1 kill
