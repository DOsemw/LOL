"""
calibration.py
--------------
Computes empirical scaling factors from the data:
  - How many more kills does a winner get vs a loser at each position?
  - These factors are used to scale predictions based on win probability

This replaces the hand-crafted moneyline adjustment tables with
data-driven scaling learned directly from the OE data.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

CACHE_PATH = Path("models/calibration.pkl")


def compute_win_loss_ratios(df: pd.DataFrame) -> dict:
    """
    For each position, compute:
      - avg kills when winning vs losing
      - avg deaths when winning vs losing  
      - avg assists when winning vs losing
      - the ratio (win_avg / loss_avg)
    
    These ratios tell us empirically how much win probability
    should scale the prediction.
    """
    if "result" not in df.columns:
        return {}

    ratios = {}
    for position in df["position"].unique():
        pos_df = df[df["position"] == position]
        
        win_df  = pos_df[pos_df["result"] == 1]
        loss_df = pos_df[pos_df["result"] == 0]
        
        pos_ratios = {}
        for stat in ["kills", "deaths", "assists"]:
            win_avg  = win_df[stat].mean()
            loss_avg = loss_df[stat].mean()
            
            if loss_avg > 0:
                ratio = win_avg / loss_avg
            else:
                ratio = 1.0
            
            pos_ratios[stat] = {
                "win_avg":  round(win_avg, 3),
                "loss_avg": round(loss_avg, 3),
                "ratio":    round(ratio, 3),
            }
        ratios[position] = pos_ratios

    return ratios


def scale_prediction(base_pred: float, stat: str, position: str,
                     win_prob: float, ratios: dict) -> float:
    """
    Scale a base prediction (trained on all games) based on win probability.
    
    At 50% win prob: no scaling (return base_pred)
    At 100% win prob: scale by full win/loss ratio
    At 0% win prob: scale down by full win/loss ratio
    
    Formula: base * (loss_avg + win_prob * (win_avg - loss_avg)) / overall_avg
    where overall_avg ≈ 0.5 * win_avg + 0.5 * loss_avg (approximate)
    """
    if position not in ratios or stat not in ratios[position]:
        return base_pred
    
    r = ratios[position][stat]
    win_avg  = r["win_avg"]
    loss_avg = r["loss_avg"]
    
    # Expected average at this win probability
    expected_at_win_prob = win_prob * win_avg + (1 - win_prob) * loss_avg
    
    # Overall average (what the model was trained on, approx 50/50)
    overall_avg = 0.5 * win_avg + 0.5 * loss_avg
    
    if overall_avg <= 0:
        return base_pred
    
    # Scale factor
    scale = expected_at_win_prob / overall_avg
    
    # Clamp to prevent extreme scaling
    scale = min(max(scale, 0.5), 2.0)
    
    return round(base_pred * scale, 2)


def save_calibration(ratios: dict):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(ratios, f)


def load_calibration() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from data_ingestion import load_raw

    print("Computing win/loss calibration ratios...\n")
    raw = load_raw()

    ratios = compute_win_loss_ratios(raw)

    print(f"{'Position':<8} {'Stat':<10} {'Win Avg':>10} {'Loss Avg':>10} {'Ratio':>8}")
    print("-" * 50)
    for pos in ["top", "jng", "mid", "bot", "sup"]:
        if pos not in ratios:
            continue
        for stat in ["kills", "deaths", "assists"]:
            r = ratios[pos][stat]
            print(f"{pos:<8} {stat:<10} {r['win_avg']:>10.2f} "
                  f"{r['loss_avg']:>10.2f} {r['ratio']:>8.3f}")
        print()

    save_calibration(ratios)
    print(f"Saved to {CACHE_PATH}")
