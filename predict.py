"""
predict.py v2
-------------
Combines Model 1 (team kills) and Model 2 (kill share) into final prediction.

Flow:
  1. Get player's recent kill share features
  2. Get team's recent kill environment features
  3. Apply win probability blend to kill share (win games vs loss games)
  4. Predict team kills → predict kill share → multiply
  5. Apply Bayesian shrinkage for low-sample players
  6. Scale to M1-2 and M1-3 using series probability matrix
"""

import numpy as np
import pandas as pd
from series import scale_to_series, fantasy_score


# Kill share limits per position (realistic bounds)
KILL_SHARE_LIMITS = {
    "top": (0.05, 0.50),
    "jng": (0.05, 0.50),
    "mid": (0.08, 0.55),
    "bot": (0.08, 0.60),
    "sup": (0.01, 0.25),
}

# Deaths and assists share limits
DEATH_SHARE_LIMITS = {
    "top": (0.10, 0.40),
    "jng": (0.08, 0.35),
    "mid": (0.10, 0.35),
    "bot": (0.08, 0.35),
    "sup": (0.10, 0.40),
}


def _get_player_row(df: pd.DataFrame, player: str, feature_cols: list) -> tuple:
    """Get most recent feature row for a player."""
    mask = df["playername"].str.lower() == player.strip().lower()
    if not mask.any():
        suggestions = [p for p in df["playername"].unique()
                       if player.lower() in p.lower()][:6]
        raise ValueError(f"Player '{player}' not found. Suggestions: {suggestions}")
    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][feature_cols].copy()
    return player_df, X


def _apply_win_prob_blend(X: pd.DataFrame, win_prob: float) -> pd.DataFrame:
    """
    Blend win/loss kill share based on win probability.
    At 70% win prob: features = 70% from winning games + 30% from losing games.
    """
    X = X.copy()
    win_col  = "kill_share_roll10_win"
    loss_col = "kill_share_roll10_loss"
    base_col = "kill_share_player_ewm"

    if win_col in X.columns and loss_col in X.columns:
        win_val  = float(X[win_col].fillna(X.get(base_col, pd.Series([np.nan])).iloc[0]).iloc[0])
        loss_val = float(X[loss_col].fillna(X.get(base_col, pd.Series([np.nan])).iloc[0]).iloc[0])

        if np.isnan(win_val):  win_val  = float(X[base_col].iloc[0]) if base_col in X.columns else 0.2
        if np.isnan(loss_val): loss_val = float(X[base_col].iloc[0]) if base_col in X.columns else 0.2

        blended = win_prob * win_val + (1 - win_prob) * loss_val
        if base_col in X.columns:
            X[base_col] = blended

    # Update win rate features to reflect actual expected win prob
    for col in ["player_winrate", "team_winrate"]:
        if col in X.columns:
            X[col] = win_prob

    return X


def _shrinkage_blend(player_pred: float, position: str, league: str,
                     games_played: int, df: pd.DataFrame) -> float:
    """
    Bayesian shrinkage: blend player prediction toward league+position average.
    Less data = more weight on league average.
    """
    if games_played >= 20:
        return player_pred  # enough data, trust the model

    player_weight = games_played / 20.0

    # League+position average kill share
    mask = (df["position"] == position) & (df["league"] == league)
    if mask.sum() >= 10:
        league_avg = float(df[mask]["kill_share"].mean())
    else:
        league_avg = float(df[df["position"] == position]["kill_share"].mean())

    return player_weight * player_pred + (1 - player_weight) * league_avg


def predict_player(
    df:            pd.DataFrame,
    tk_model:      object,
    tk_features:   list,
    ks_models:     dict,
    ks_features:   list,
    player:        str,
    win_prob:      float = 0.5,
    opponent:      str   = None,
    side:          str   = "Blue",
) -> dict:
    """
    Full prediction pipeline for one player.

    Returns dict with kills/deaths/assists for M1, M1-2, M1-3.
    """
    from model_team_kills import predict as predict_team_kills
    from model_kill_share import predict as predict_kill_share

    # Get player data
    player_df, X_player = _get_player_row(df, player, ks_features)
    position    = player_df.iloc[0]["position"]
    league      = player_df.iloc[0]["league"]
    games_played = player_df["gameid"].nunique()

    # Side encoding
    if "is_blue_side" in X_player.columns:
        X_player["is_blue_side"] = 1 if side.lower() == "blue" else 0

    # Win prob blend on kill share features
    X_player = _apply_win_prob_blend(X_player, win_prob)

    # ── Model 1: Team Kills ──────────────────────────────────────────────
    # Get team feature row
    team_mask = (df["teamname"] == player_df.iloc[0]["teamname"])
    team_df   = df[team_mask].drop_duplicates("gameid").sort_values("date", ascending=False)

    # If opponent specified, override opponent features
    if opponent:
        opp_mask = df["teamname"].str.lower().str.contains(opponent.lower(), na=False)
        opp_df   = df[opp_mask].drop_duplicates("gameid").sort_values("date", ascending=False)
        if len(opp_df) >= 3:
            opp_pos_df = opp_df[opp_df["position"] == position]
            if len(opp_pos_df) >= 3 and "opp_death_rate_ewm" in team_df.columns:
                opp_death_rate = float(opp_pos_df["deaths"].tail(10).mean())
                team_df = team_df.copy()
                team_df["opp_death_rate_ewm"] = opp_death_rate

    X_team = team_df.iloc[[0]][tk_features].copy() if len(team_df) > 0 else None

    if X_team is not None and not X_team[tk_features].isnull().all().all():
        if "is_blue_side" in X_team.columns:
            X_team["is_blue_side"] = 1 if side.lower() == "blue" else 0
        team_kills_pred = predict_team_kills(tk_model, X_team)
    else:
        # Fallback to league average team kills
        team_kills_pred = float(df[df["league"] == league]["team_kills"].mean())
        team_kills_pred = max(team_kills_pred, 5.0)

    # ── Model 2: Kill Share ──────────────────────────────────────────────
    if position not in ks_models:
        # Fallback to position average
        kill_share_pred = float(df[df["position"] == position]["kill_share"].mean())
    else:
        ks_model, _, ks_mae, pos_avg = ks_models[position]
        kill_share_pred = predict_kill_share(ks_model, X_player, pos_avg)

    # Bayesian shrinkage for low-sample players
    kill_share_pred = _shrinkage_blend(
        kill_share_pred, position, league, games_played, df
    )

    # Clamp to realistic bounds
    lo, hi = KILL_SHARE_LIMITS.get(position, (0.05, 0.55))
    kill_share_pred = float(np.clip(kill_share_pred, lo, hi))

    # ── Combine: player kills = team kills × kill share ──────────────────
    kills_per_map = team_kills_pred * kill_share_pred

    # Deaths prediction (similar approach with death share)
    # Use player's rolling death average scaled to team context
    death_col = "deaths_player_ewm"
    if death_col in X_player.columns and not pd.isna(X_player[death_col].iloc[0]):
        raw_deaths = float(X_player[death_col].iloc[0])
        if games_played < 20:
            league_death_avg = float(df[
                (df["position"] == position) & (df["league"] == league)
            ]["deaths"].mean())
            w = games_played / 20.0
            raw_deaths = w * raw_deaths + (1 - w) * league_death_avg
        deaths_per_map = max(0.3, raw_deaths)
    else:
        deaths_per_map = float(df[df["position"] == position]["deaths"].mean())

    # Assists prediction (KP% × team kills - own kills)
    kp_col = "kp_pct_player_ewm"
    if kp_col in X_player.columns and not pd.isna(X_player[kp_col].iloc[0]):
        kp = float(X_player[kp_col].iloc[0])
        if games_played < 20:
            league_kp_avg = float(df[
                (df["position"] == position) & (df["league"] == league)
            ]["kp_pct"].mean()) if "kp_pct" in df.columns else 0.5
            w = games_played / 20.0
            kp = w * kp + (1 - w) * league_kp_avg
        assists_per_map = max(0.0, team_kills_pred * kp - kills_per_map)
    else:
        assists_per_map = float(df[df["position"] == position]["assists"].mean())

    # MAE estimates from model
    kills_mae  = float(ks_models[position][2]) * team_kills_pred if position in ks_models else 1.5
    deaths_mae = 1.2
    assists_mae = 2.0

    # Confidence interval (±1 MAE)
    def ci(val, mae):
        return round(max(0, val - mae), 2), round(val + mae, 2)

    k_low, k_high = ci(kills_per_map,  kills_mae)
    d_low, d_high = ci(deaths_per_map, deaths_mae)
    a_low, a_high = ci(assists_per_map, assists_mae)

    # Scale to series
    kills_series  = scale_to_series(kills_per_map,  k_low, k_high, kills_mae,  win_prob)
    deaths_series = scale_to_series(deaths_per_map, d_low, d_high, deaths_mae, win_prob)
    assists_series = scale_to_series(assists_per_map, a_low, a_high, assists_mae, win_prob)

    # Recent form
    recent = player_df.head(5)[["date","champion","kills","deaths","assists"]].copy()
    recent["date"] = recent["date"].astype(str)

    return {
        "player":         player_df.iloc[0]["playername"],
        "position":       position,
        "league":         league,
        "win_prob":       round(win_prob, 3),
        "games_in_sample": games_played,
        "team_kills_pred": round(team_kills_pred, 2),
        "kill_share_pred": round(kill_share_pred, 3),

        "map1": {
            "kills":   kills_series["m1"],
            "deaths":  deaths_series["m1"],
            "assists": assists_series["m1"],
        },
        "m1_2": {
            "kills":   kills_series["m1_2"],
            "deaths":  deaths_series["m1_2"],
            "assists": assists_series["m1_2"],
            "fantasy": fantasy_score(
                kills_series["m1_2"]["series_total"],
                deaths_series["m1_2"]["series_total"],
                assists_series["m1_2"]["series_total"],
            ),
            "expected_games": 2.0,
        },
        "m1_3": {
            "kills":   kills_series["m1_3"],
            "deaths":  deaths_series["m1_3"],
            "assists": assists_series["m1_3"],
            "fantasy": fantasy_score(
                kills_series["m1_3"]["series_total"],
                deaths_series["m1_3"]["series_total"],
                assists_series["m1_3"]["series_total"],
            ),
            "expected_games": kills_series["m1_3"]["maps"],
            "p_map3":         kills_series["m1_3"]["p_map3"],
        },
        "recent_form": recent.to_dict(orient="records"),
    }
