"""
series_predictor.py
-------------------
Converts per-game K/D/A predictions into series totals (Bo1 / Bo3 / Bo5).

Key insight: The sportsbook line is a SERIES total, not a per-game total.
e.g. "morttheus over 15.0 kills" means across all games in the series.

We need:
  series_prediction = per_game_prediction × expected_games_played

Expected games played depends on:
  - Series format (Bo1=1, Bo3=2-3, Bo5=3-5)
  - Win probability of each team (closer matches → more games)
"""

import numpy as np
import pandas as pd
from itertools import product


# ── Expected games played calculator ─────────────────────────────────────────

def expected_games_bo3(win_prob: float) -> float:
    """
    Expected number of games in a Bo3 given team A's win probability per game.
    
    Possible outcomes:
      2 games: AA (p²) or BB ((1-p)²)
      3 games: ABA, BAA, ABB, BAB → 2*p²*(1-p) + 2*p*(1-p)²
    """
    p = win_prob
    q = 1 - p
    prob_2 = p**2 + q**2
    prob_3 = 1 - prob_2
    return 2 * prob_2 + 3 * prob_3


def expected_games_bo5(win_prob: float) -> float:
    """
    Expected number of games in a Bo5.
    Team A wins Bo5 if they get to 3 wins first.
    """
    p = win_prob
    q = 1 - p

    total_exp = 0
    for n_games in range(3, 6):  # Bo5 ends in 3, 4, or 5 games
        # n_games total: winner wins game n_games, had exactly 2 wins in first n_games-1
        from math import comb
        # Team A wins in n_games: had 2 wins in n_games-1, win game n_games
        prob_a_wins_in_n = comb(n_games - 1, 2) * (p**2) * (q**(n_games - 3)) * p
        # Team B wins in n_games: same logic
        prob_b_wins_in_n = comb(n_games - 1, 2) * (q**2) * (p**(n_games - 3)) * q
        total_exp += n_games * (prob_a_wins_in_n + prob_b_wins_in_n)

    return total_exp


def expected_games(series_format: str, win_prob: float) -> float:
    """
    Returns expected number of games given format and win probability.
    
    Args:
        series_format: "Bo1", "Bo3", or "Bo5"
        win_prob:      Team's win probability per individual game (0.0–1.0)
                       Use 0.5 if unknown (maximum variance / most games)
    """
    fmt = series_format.upper().replace(" ", "")
    if fmt == "BO1":
        return 1.0
    elif fmt == "BO3":
        return expected_games_bo3(win_prob)
    elif fmt == "BO5":
        return expected_games_bo5(win_prob)
    else:
        raise ValueError(f"Unknown format: {series_format}. Use Bo1, Bo3, or Bo5.")


# ── Win probability from match odds ──────────────────────────────────────────

def moneyline_to_prob(moneyline: int) -> float:
    """
    Convert American moneyline odds to implied win probability.
    
    Args:
        moneyline: e.g. -297 (favourite) or +297 (underdog)
    
    Returns:
        Implied probability (not vig-adjusted)
    """
    if moneyline < 0:
        return (-moneyline) / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


def vig_adjusted_probs(ml_team_a: int, ml_team_b: int) -> tuple[float, float]:
    """
    Remove the sportsbook vig to get true implied probabilities.
    Returns (prob_a, prob_b) that sum to 1.0.
    """
    raw_a = moneyline_to_prob(ml_team_a)
    raw_b = moneyline_to_prob(ml_team_b)
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


# ── Series prediction ─────────────────────────────────────────────────────────

def predict_series(
    per_game_predictions: dict,
    series_format: str,
    team_win_prob: float = 0.5,
    moneyline: int = None,
    opp_moneyline: int = None,
) -> dict:
    """
    Scale per-game K/D/A predictions to series totals.

    Args:
        per_game_predictions: Output from predict_player() — dict with kills/deaths/assists
        series_format:        "Bo1", "Bo3", or "Bo5"
        team_win_prob:        Win probability per game (0.0–1.0). Overrides moneyline if set.
        moneyline:            American ML for the player's team (e.g. -297)
        opp_moneyline:        American ML for the opponent (e.g. +297)

    Returns:
        Dict with series-scaled predictions for kills, deaths, assists
    """
    # Derive win prob from moneyline if provided
    if moneyline is not None and opp_moneyline is not None:
        team_win_prob, _ = vig_adjusted_probs(moneyline, opp_moneyline)

    exp_games = expected_games(series_format, team_win_prob)

    result = {
        "series_format":     series_format,
        "win_prob_per_game": round(team_win_prob, 3),
        "expected_games":    round(exp_games, 2),
    }

    for stat in ["kills", "deaths", "assists"]:
        if stat not in per_game_predictions:
            continue
        pg = per_game_predictions[stat]
        result[stat] = {
            "per_game":      pg["mid"],
            "series_total":  round(pg["mid"] * exp_games, 1),
            "series_low":    round(pg["low"] * exp_games, 1),
            "series_high":   round(pg["high"] * exp_games, 1),
            "mae_series":    round(pg["mae"] * exp_games, 2),
        }

    # Fantasy score: typical formula = kills*3 + assists*1.5 - deaths*1 (adjust as needed)
    if all(s in result for s in ["kills", "deaths", "assists"]):
        k = result["kills"]["series_total"]
        d = result["deaths"]["series_total"]
        a = result["assists"]["series_total"]
        result["fantasy"] = {
            "series_total": round(k * 3 + a * 1.5 - d, 1),
            "formula":      "kills×3 + assists×1.5 − deaths×1",
        }

    return result


def print_series_prediction(player_name: str, result: dict):
    fmt   = result["series_format"]
    wp    = result["win_prob_per_game"] * 100
    eg    = result["expected_games"]

    print(f"\n{'='*58}")
    print(f"  {player_name}  —  {fmt} Series Prediction")
    print(f"  Win prob/game: {wp:.1f}%  |  Expected games: {eg:.2f}")
    print(f"{'='*58}")
    print(f"  {'Stat':<10} {'Per Game':>10} {'Series Total':>14} {'±MAE':>8}  {'90% CI':>16}")
    print(f"  {'-'*55}")

    for stat in ["kills", "deaths", "assists"]:
        if stat not in result:
            continue
        s = result[stat]
        ci_str = f"[{s['series_low']:.1f}–{s['series_high']:.1f}]"
        mae_str = f"±{s['mae_series']:.1f}"
        print(f"  {stat.upper():<10} {s['per_game']:>10.2f} {s['series_total']:>14.1f} "
              f"{mae_str:>8}  {ci_str:>16}")

    if "fantasy" in result:
        f = result["fantasy"]
        print(f"\n  Fantasy Score (est.): {f['series_total']:.1f}  ({f['formula']})")

    print(f"{'='*58}\n")


# ── Batch: full match lineup ──────────────────────────────────────────────────

def predict_match(
    df,
    feature_cols: list,
    team_a_players: list[str],
    team_b_players: list[str],
    series_format: str,
    team_a_moneyline: int,
    team_b_moneyline: int,
) -> pd.DataFrame:
    """
    Predict series K/D/A for all 10 players in a match.
    Mirrors the sportsbook prop table format in your screenshot.

    Returns a DataFrame like:
      Player | Team | Kills | Assists | Deaths | Fantasy (series totals)
    """
    from predict import predict_player

    prob_a, prob_b = vig_adjusted_probs(team_a_moneyline, team_b_moneyline)

    rows = []
    for player, team_label, win_prob in (
        [(p, "Team A", prob_a) for p in team_a_players] +
        [(p, "Team B", prob_b) for p in team_b_players]
    ):
        try:
            pg = predict_player(df, feature_cols, player, verbose=False)
            series = predict_series(
                pg,
                series_format  = series_format,
                team_win_prob  = win_prob,
            )
            rows.append({
                "Player":        player,
                "Team":          team_label,
                "ML":            team_a_moneyline if team_label == "Team A" else team_b_moneyline,
                "Win%/game":     f"{win_prob*100:.1f}%",
                "Exp. Games":    series["expected_games"],
                "Kills":         series["kills"]["series_total"],
                "Assists":       series["assists"]["series_total"],
                "Deaths":        series["deaths"]["series_total"],
                "Fantasy":       series.get("fantasy", {}).get("series_total", "–"),
            })
        except Exception as e:
            rows.append({"Player": player, "Team": team_label, "Error": str(e)})

    return pd.DataFrame(rows)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: Red Canids vs Fluxo (from your screenshot)
    # Red Canids -297 / Fluxo +297

    print("=== Series Prediction Demo ===")
    print("Match: Red Canids (-297) vs Fluxo (+297) — Bo3\n")

    prob_rc, prob_fl = vig_adjusted_probs(-297, +297)
    print(f"Red Canids win prob/game: {prob_rc*100:.1f}%")
    print(f"Fluxo win prob/game:      {prob_fl*100:.1f}%")
    print(f"Expected games (Bo3):     {expected_games('Bo3', prob_rc):.2f}")

    # Show expected games across different win probs
    print("\nExpected games by win probability:")
    print(f"  {'Win%':>6}  {'Bo1':>6}  {'Bo3':>6}  {'Bo5':>6}")
    for wp in [0.35, 0.40, 0.50, 0.60, 0.65, 0.74]:
        print(f"  {wp*100:>5.0f}%  {'1.00':>6}  {expected_games('Bo3',wp):>6.2f}  {expected_games('Bo5',wp):>6.2f}")
