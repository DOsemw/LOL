"""
series.py
---------
Series probability matrix for M1, M1-2, M1-3 scaling.

Based on the blueprint:
  M1:   single map prediction
  M1-2: M1 + M2 (always 2 maps in Bo3)
  M1-3: M1-2 + P(map3) × M3

P(series goes to map 3) = 1 - P(2-0 sweep)
P(2-0 sweep) = win_prob² + (1-win_prob)²
"""

import numpy as np


def p_map3(win_prob: float) -> float:
    """Probability series goes to map 3 in a Bo3."""
    p_sweep = win_prob**2 + (1 - win_prob)**2
    return 1 - p_sweep


def expected_games_bo3(win_prob: float) -> float:
    """Expected number of maps played in a Bo3."""
    p3 = p_map3(win_prob)
    return 2.0 + p3  # always 2, plus prob of map 3


def scale_to_series(
    map1_kills: float,
    map1_low:   float,
    map1_high:  float,
    map1_mae:   float,
    win_prob:   float,
) -> dict:
    """
    Scale a single-map prediction to M1-2 and M1-3 series totals.

    Key insight from blueprint:
      - M1-2 = map1 + map2 (both always played, use same per-map prediction)
      - M1-3 = M1-2 + P(map3) × map3_prediction
      - Map 3 prediction uses same per-map estimate (teams reset, fresh draft)

    Args:
        map1_kills: expected kills in map 1
        win_prob:   team win probability per map

    Returns:
        dict with m1, m1_2, m1_3 projections
    """
    p3    = p_map3(win_prob)
    exp_g = expected_games_bo3(win_prob)

    # M1-2: always 2 maps
    m12_kills = map1_kills * 2.0
    m12_low   = map1_low   * 2.0
    m12_high  = map1_high  * 2.0
    m12_mae   = map1_mae   * 2.0

    # M1-3: 2 maps certain + map 3 weighted by probability
    m13_kills = map1_kills * 2.0 + p3 * map1_kills
    m13_low   = map1_low   * 2.0 + p3 * map1_low
    m13_high  = map1_high  * 2.0 + p3 * map1_high
    m13_mae   = map1_mae   * exp_g

    return {
        "m1": {
            "expected": round(map1_kills, 2),
            "low":      round(map1_low,   2),
            "high":     round(map1_high,  2),
            "mae":      round(map1_mae,   2),
        },
        "m1_2": {
            "series_total": round(m12_kills, 1),
            "series_low":   round(m12_low,   1),
            "series_high":  round(m12_high,  1),
            "mae":          round(m12_mae,   2),
            "maps":         2.0,
        },
        "m1_3": {
            "series_total": round(m13_kills, 1),
            "series_low":   round(m13_low,   1),
            "series_high":  round(m13_high,  1),
            "mae":          round(m13_mae,   2),
            "maps":         round(exp_g,     2),
            "p_map3":       round(p3,        3),
        },
    }


def fantasy_score(kills: float, deaths: float, assists: float) -> float:
    """ParlayPlay/PrizePicks style fantasy: kills*3 + assists*1.5 - deaths."""
    return round(kills * 3 + assists * 1.5 - deaths, 1)
