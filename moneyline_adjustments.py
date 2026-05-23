"""
moneyline_adjustments.py
------------------------
Adjusts per-game K/D/A predictions based on pre-game moneyline odds.

Core insight from historical data:
  - Winning teams get more kills (they teamfight ahead, clean up)
  - Winning teams get fewer deaths (less fighting from behind)
  - Assists track kills closely for winning side
  - Losing teams die more, get fewer kills but sometimes more assists (grouping/defending)

These adjustments are MULTIPLICATIVE on top of the base model prediction.
They're calibrated from historical OE data split by win probability buckets.
"""

import numpy as np


# ── Empirically-derived adjustment curves ─────────────────────────────────────
# These multipliers are estimated from OE data patterns:
# - At 50% win prob: no adjustment (multiplier = 1.0)
# - At 75% win prob: kills ~20% higher, deaths ~15% lower
# - At 25% win prob: kills ~15% lower, deaths ~20% higher
# Values are interpolated linearly between anchor points.

# Format: (win_prob, kills_mult, deaths_mult, assists_mult)
_ADJUSTMENT_TABLE = [
    # win_prob  kills   deaths  assists
    (0.10,      0.70,   1.40,   0.80),   # massive underdog (e.g. +700)
    (0.20,      0.82,   1.28,   0.88),   # heavy underdog   (e.g. +400)
    (0.30,      0.91,   1.15,   0.94),   # underdog         (e.g. +233)
    (0.40,      0.96,   1.06,   0.97),   # slight underdog  (e.g. +150)
    (0.50,      1.00,   1.00,   1.00),   # pick'em
    (0.60,      1.04,   0.94,   1.03),   # slight favourite (e.g. -150)
    (0.70,      1.10,   0.87,   1.07),   # favourite        (e.g. -233)
    (0.80,      1.18,   0.79,   1.12),   # heavy favourite  (e.g. -400)
    (0.90,      1.25,   0.72,   1.17),   # massive favourite(e.g. -900)
]


def _interpolate_adjustment(win_prob: float, stat: str) -> float:
    """
    Linearly interpolate the multiplier for a given win probability and stat.
    """
    stat_idx = {"kills": 1, "deaths": 2, "assists": 3}[stat]
    table = _ADJUSTMENT_TABLE

    # Clamp to table bounds
    win_prob = max(table[0][0], min(table[-1][0], win_prob))

    # Find surrounding anchor points
    for i in range(len(table) - 1):
        lo_wp, hi_wp = table[i][0], table[i+1][0]
        if lo_wp <= win_prob <= hi_wp:
            t = (win_prob - lo_wp) / (hi_wp - lo_wp)
            lo_val = table[i][stat_idx]
            hi_val = table[i+1][stat_idx]
            return lo_val + t * (hi_val - lo_val)

    return 1.0  # fallback


def apply_moneyline_adjustments(
    predictions: dict,
    win_prob: float,
) -> dict:
    """
    Apply win-probability-based multipliers to per-game K/D/A predictions.

    Args:
        predictions: dict with keys kills/deaths/assists, each having mid/low/high/mae
        win_prob:    Team's vig-adjusted win probability per game (0.0–1.0)

    Returns:
        Adjusted predictions dict (same structure, new values)
    """
    adjusted = {}

    for stat in ["kills", "deaths", "assists"]:
        if stat not in predictions:
            continue

        mult  = _interpolate_adjustment(win_prob, stat)
        pg    = predictions[stat]

        adjusted[stat] = {
            "mid":  round(pg["mid"] * mult, 2),
            "low":  round(pg["low"] * mult, 2),
            "high": round(pg["high"] * mult, 2),
            "mae":  pg["mae"],   # MAE stays the same (it's the model's base error)
            "ml_multiplier": round(mult, 3),
        }

    return adjusted


def moneyline_to_prob(moneyline: int) -> float:
    """American moneyline → raw implied probability."""
    if moneyline < 0:
        return (-moneyline) / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


def vig_adjusted_probs(ml_a: int, ml_b: int) -> tuple:
    """Remove vig, return (prob_a, prob_b) summing to 1.0."""
    raw_a = moneyline_to_prob(ml_a)
    raw_b = moneyline_to_prob(ml_b)
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def describe_line(win_prob: float) -> str:
    """Human-readable description of what the moneyline implies."""
    if win_prob >= 0.85:   return "massive favourite"
    elif win_prob >= 0.70: return "heavy favourite"
    elif win_prob >= 0.58: return "favourite"
    elif win_prob >= 0.52: return "slight favourite"
    elif win_prob >= 0.48: return "pick'em"
    elif win_prob >= 0.42: return "slight underdog"
    elif win_prob >= 0.30: return "underdog"
    elif win_prob >= 0.15: return "heavy underdog"
    else:                  return "massive underdog"


if __name__ == "__main__":
    # Demo: show adjustment table
    print("Moneyline Adjustment Multipliers by Win Probability\n")
    print(f"  {'Win%':>6}  {'Label':>18}  {'Kills×':>8}  {'Deaths×':>8}  {'Assists×':>8}")
    print("  " + "-"*55)
    for wp in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.748, 0.80, 0.90]:
        k = _interpolate_adjustment(wp, "kills")
        d = _interpolate_adjustment(wp, "deaths")
        a = _interpolate_adjustment(wp, "assists")
        label = describe_line(wp)
        print(f"  {wp*100:>5.1f}%  {label:>18}  {k:>8.3f}  {d:>8.3f}  {a:>8.3f}")

    print("\nExample: Red Canids (-297) player predicted 3.0 kills/game base:")
    prob_rc, _ = vig_adjusted_probs(-297, 297)
    base = {"kills": {"mid": 3.0, "low": 1.2, "high": 5.2, "mae": 1.2},
            "deaths": {"mid": 2.0, "low": 0.8, "high": 3.5, "mae": 1.0},
            "assists": {"mid": 6.0, "low": 2.8, "high": 10.1, "mae": 2.3}}
    adj = apply_moneyline_adjustments(base, prob_rc)
    for stat, v in adj.items():
        print(f"  {stat}: {base[stat]['mid']:.1f} → {v['mid']:.2f}  (×{v['ml_multiplier']})")
