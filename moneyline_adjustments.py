"""
moneyline_adjustments.py
------------------------
Adjusts per-game K/D/A predictions based on pre-game moneyline odds.

Multipliers are scaled differently for M1, M1-2, and M1-3:
  - M1 (single map): moderate adjustment — favourites do play better per map
  - M1-2 (2 maps):   smaller adjustment — less variance over 2 maps
  - M1-3 (full Bo3): smallest adjustment — expected_games already handles
                     the favourite winning faster, so we barely touch it

All multipliers are intentionally conservative to avoid over-inflating series totals.
"""

import numpy as np


# ── Adjustment tables per format ──────────────────────────────────────────────
# Format: (win_prob, kills_mult, deaths_mult, assists_mult)

# M1 — single map (moderate adjustments)
_TABLE_M1 = [
    (0.10,  0.78,  1.30,  0.84),
    (0.20,  0.86,  1.20,  0.90),
    (0.30,  0.92,  1.11,  0.95),
    (0.40,  0.97,  1.04,  0.98),
    (0.50,  1.00,  1.00,  1.00),
    (0.60,  1.02,  0.96,  1.02),
    (0.70,  1.05,  0.91,  1.04),
    (0.80,  1.09,  0.86,  1.07),
    (0.90,  1.12,  0.82,  1.09),
]

# M1-2 — 2 maps (smaller adjustments)
_TABLE_M12 = [
    (0.10,  0.84,  1.20,  0.88),
    (0.20,  0.90,  1.13,  0.93),
    (0.30,  0.94,  1.07,  0.96),
    (0.40,  0.98,  1.03,  0.99),
    (0.50,  1.00,  1.00,  1.00),
    (0.60,  1.02,  0.97,  1.01),
    (0.70,  1.04,  0.94,  1.03),
    (0.80,  1.06,  0.90,  1.05),
    (0.90,  1.08,  0.87,  1.06),
]

# M1-3 — full Bo3 (minimal adjustments — expected_games already does the work)
_TABLE_M13 = [
    (0.10,  0.90,  1.10,  0.93),
    (0.20,  0.93,  1.07,  0.95),
    (0.30,  0.96,  1.04,  0.97),
    (0.40,  0.98,  1.02,  0.99),
    (0.50,  1.00,  1.00,  1.00),
    (0.60,  1.01,  0.98,  1.01),
    (0.70,  1.03,  0.96,  1.02),
    (0.80,  1.05,  0.93,  1.03),
    (0.90,  1.07,  0.91,  1.04),
]

_TABLES = {
    "m1":  _TABLE_M1,
    "m12": _TABLE_M12,
    "m13": _TABLE_M13,
}


def _interpolate(table, win_prob: float, stat: str) -> float:
    """Linearly interpolate multiplier from a table."""
    stat_idx = {"kills": 1, "deaths": 2, "assists": 3}[stat]
    win_prob = max(table[0][0], min(table[-1][0], win_prob))
    for i in range(len(table) - 1):
        lo_wp, hi_wp = table[i][0], table[i+1][0]
        if lo_wp <= win_prob <= hi_wp:
            t = (win_prob - lo_wp) / (hi_wp - lo_wp)
            return table[i][stat_idx] + t * (table[i+1][stat_idx] - table[i][stat_idx])
    return 1.0


def _interpolate_adjustment(win_prob: float, stat: str, fmt: str = "m1") -> float:
    """
    Get the multiplier for a given win probability, stat, and format.
    fmt: 'm1', 'm12', or 'm13'
    """
    table = _TABLES.get(fmt.lower().replace("-","").replace(" ",""), _TABLE_M1)
    return _interpolate(table, win_prob, stat)


def apply_moneyline_adjustments(
    predictions: dict,
    win_prob: float,
    fmt: str = "m1",
) -> dict:
    """
    Apply win-probability multipliers to per-game K/D/A predictions.

    Args:
        predictions: dict with keys kills/deaths/assists, each having mid/low/high/mae
        win_prob:    Vig-adjusted win probability (0.0–1.0)
        fmt:         Format — 'm1', 'm12', or 'm13'

    Returns:
        Adjusted predictions dict
    """
    adjusted = {}
    for stat in ["kills", "deaths", "assists"]:
        if stat not in predictions:
            continue
        mult = _interpolate_adjustment(win_prob, stat, fmt)
        pg   = predictions[stat]
        adjusted[stat] = {
            "mid":           round(pg["mid"]  * mult, 2),
            "low":           round(pg["low"]  * mult, 2),
            "high":          round(pg["high"] * mult, 2),
            "mae":           pg["mae"],
            "ml_multiplier": round(mult, 3),
        }
    return adjusted


def moneyline_to_prob(moneyline: int) -> float:
    if moneyline < 0:
        return (-moneyline) / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


def vig_adjusted_probs(ml_a: int, ml_b: int) -> tuple:
    raw_a = moneyline_to_prob(ml_a)
    raw_b = moneyline_to_prob(ml_b)
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def describe_line(win_prob: float) -> str:
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
    print("Moneyline Multipliers by Format and Win Probability\n")
    print(f"  {'Win%':>6}  {'Label':>18}  {'M1 K×':>7}  {'M12 K×':>7}  {'M13 K×':>7}  {'M1 D×':>7}  {'M13 D×':>7}")
    print("  " + "-"*65)
    for wp in [0.25, 0.40, 0.50, 0.60, 0.70, 0.748, 0.80, 0.90]:
        m1k  = _interpolate_adjustment(wp, "kills",   "m1")
        m12k = _interpolate_adjustment(wp, "kills",   "m12")
        m13k = _interpolate_adjustment(wp, "kills",   "m13")
        m1d  = _interpolate_adjustment(wp, "deaths",  "m1")
        m13d = _interpolate_adjustment(wp, "deaths",  "m13")
        print(f"  {wp*100:>5.1f}%  {describe_line(wp):>18}  {m1k:>7.3f}  {m12k:>7.3f}  {m13k:>7.3f}  {m1d:>7.3f}  {m13d:>7.3f}")

    print("\nExample: -333 favourite (~76% win prob), base 6.0 kills/map")
    wp = 0.748
    for fmt in ["m1", "m12", "m13"]:
        mult = _interpolate_adjustment(wp, "kills", fmt)
        print(f"  {fmt.upper()}: 6.0 × {mult:.3f} = {6.0*mult:.2f} kills")
