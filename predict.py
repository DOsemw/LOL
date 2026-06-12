"""
predict.py v2
-------------
Combines Model 1 (team kills) and Model 2 (kill share) into final prediction.

Blueprint flow:
  Step A: devig moneyline → win probability p  (done in main.py)
  Step B: P(Map3) = 1 - (p³ + (1-p)³)         (done in series.py)
  Step C: M1-3 = Map1 + Map2 + P(Map3)×Map3   (done in series.py)
"""

import numpy as np
import pandas as pd
from series import scale_to_series

KILL_SHARE_LIMITS = {
    "top": (0.05, 0.50),
    "jng": (0.05, 0.50),
    "mid": (0.08, 0.55),
    "bot": (0.08, 0.60),
    "sup": (0.01, 0.25),
}


def _get_player_row(df, player, feature_cols):
    mask = df["playername"].str.lower() == player.strip().lower()
    if not mask.any():
        suggestions = [p for p in df["playername"].unique()
                       if player.lower() in p.lower()][:6]
        raise ValueError(f"Player '{player}' not found. Suggestions: {suggestions}")
    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][feature_cols].copy()
    return player_df, X


def _apply_win_prob_blend(X, win_prob):
    X = X.copy()
    win_col  = "kill_share_roll10_win"
    loss_col = "kill_share_roll10_loss"
    base_col = "kill_share_player_ewm"

    if win_col in X.columns and loss_col in X.columns:
        win_val  = float(X[win_col].fillna(0).iloc[0])
        loss_val = float(X[loss_col].fillna(0).iloc[0])
        if win_val == 0: win_val = float(X[base_col].iloc[0]) if base_col in X.columns else 0.2
        if loss_val == 0: loss_val = float(X[base_col].iloc[0]) if base_col in X.columns else 0.2
        blended = win_prob * win_val + (1 - win_prob) * loss_val
        if base_col in X.columns:
            X[base_col] = blended

    for col in ["player_winrate", "team_winrate"]:
        if col in X.columns:
            X[col] = win_prob
    return X


def _shrink(player_pred, position, league, games_played, df):
    if games_played >= 20:
        return player_pred
    w = games_played / 20.0
    mask = (df["position"] == position) & (df["league"] == league)
    if mask.sum() >= 10:
        avg = float(df[mask]["kill_share"].mean())
    else:
        avg = float(df[df["position"] == position]["kill_share"].mean())
    return w * player_pred + (1 - w) * avg


def predict_player(df, tk_model, tk_features, ks_models, ks_features,
                   player, win_prob=0.5, opponent=None, side="Blue",
                   pace_multipliers=None, kill_bounds=None):
    from model_team_kills import predict as predict_tk
    from model_kill_share import predict as predict_ks

    player_df, X_player = _get_player_row(df, player, ks_features)
    position    = player_df.iloc[0]["position"]
    league      = player_df.iloc[0]["league"]
    games_played = player_df["gameid"].nunique()

    if "is_blue_side" in X_player.columns:
        X_player["is_blue_side"] = 1 if side.lower() == "blue" else 0

    X_player = _apply_win_prob_blend(X_player, win_prob)

    # Model 1: Team Kills
    team_mask = df["teamname"] == player_df.iloc[0]["teamname"]
    team_df   = df[team_mask].drop_duplicates("gameid").sort_values("date", ascending=False)

    if opponent:
        opp_mask = df["teamname"].str.lower().str.contains(opponent.lower(), na=False)
        opp_df   = df[opp_mask].drop_duplicates("gameid").sort_values("date", ascending=False)
        if len(opp_df) >= 3:
            opp_pos = opp_df[opp_df["position"] == position]
            if len(opp_pos) >= 3 and len(team_df) > 0:
                team_df = team_df.copy()
                if "opp_death_rate_ewm" in team_df.columns:
                    team_df["opp_death_rate_ewm"] = float(opp_pos["deaths"].tail(10).mean())

    if len(team_df) > 0 and not team_df.iloc[[0]][tk_features].isnull().all().all():
        X_team = team_df.iloc[[0]][tk_features].copy()
        if "is_blue_side" in X_team.columns:
            X_team["is_blue_side"] = 1 if side.lower() == "blue" else 0
        team_kills_pred = predict_tk(tk_model, X_team)
    else:
        team_kills_pred = float(df[df["league"] == league]["team_kills"].mean() or 8.0)

    # Model 2: Kill Share
    if position in ks_models:
        ks_model, _, ks_mae, pos_avg = ks_models[position]
        kill_share_pred = predict_ks(ks_model, X_player, pos_avg)
    else:
        kill_share_pred = float(df[df["position"] == position]["kill_share"].mean())

    kill_share_pred = _shrink(kill_share_pred, position, league, games_played, df)
    lo, hi = KILL_SHARE_LIMITS.get(position, (0.05, 0.55))
    kill_share_pred = float(np.clip(kill_share_pred, lo, hi))

    # Final: kills = team_kills × kill_share
    kills_per_map = team_kills_pred * kill_share_pred

    # Deaths
    death_col = "deaths_player_ewm"
    if death_col in X_player.columns and not pd.isna(X_player[death_col].iloc[0]):
        deaths_per_map = float(X_player[death_col].iloc[0])
        if games_played < 20:
            w = games_played / 20.0
            league_d = float(df[(df["position"]==position)&(df["league"]==league)]["deaths"].mean() or 2.5)
            deaths_per_map = w * deaths_per_map + (1-w) * league_d
        deaths_per_map = max(0.3, deaths_per_map)
    else:
        deaths_per_map = float(df[df["position"]==position]["deaths"].mean() or 2.5)

    # Assists: KP% × team_kills - own kills
    kp_col = "kp_pct_player_ewm"
    if kp_col in X_player.columns and not pd.isna(X_player[kp_col].iloc[0]):
        kp = float(X_player[kp_col].iloc[0])
        if games_played < 20:
            w = games_played / 20.0
            league_kp = float(df[(df["position"]==position)&(df["league"]==league)]["kp_pct"].mean() if "kp_pct" in df.columns else 0.5)
            kp = w * kp + (1-w) * league_kp
        assists_per_map = max(0.0, team_kills_pred * kp - kills_per_map)
    else:
        assists_per_map = float(df[df["position"]==position]["assists"].mean() or 4.0)

    # Apply league pace multiplier (bloodier leagues = more kills)
    if pace_multipliers:
        pace_mult = pace_multipliers.get(league, 1.0)
        kills_per_map   = round(kills_per_map   * pace_mult, 2)
        deaths_per_map  = round(deaths_per_map  * pace_mult, 2)
        assists_per_map = round(assists_per_map * pace_mult, 2)

    # Steps B & C: scale to series using blueprint formula
    series = scale_to_series(kills_per_map, deaths_per_map, assists_per_map, win_prob)

    recent = player_df.head(5)[["date","champion","kills","deaths","assists"]].copy()
    recent["date"] = recent["date"].astype(str)

    return {
        "player":          player_df.iloc[0]["playername"],
        "position":        position,
        "league":          league,
        "win_prob":        round(win_prob, 3),
        "games_in_sample": games_played,
        "team_kills_pred": round(team_kills_pred, 2),
        "kill_share_pred": round(kill_share_pred, 3),
        "pace_factor":     series["pace_factor"],
        "expected_maps":   3,
        "map1": {
            "kills":   series["m1"]["kills"],
            "deaths":  series["m1"]["deaths"],
            "assists": series["m1"]["assists"],
        },
        "m1_2": {
            "kills":          series["m1_2"]["kills"],
            "deaths":         series["m1_2"]["deaths"],
            "assists":        series["m1_2"]["assists"],
            "fantasy":        series["m1_2"]["fantasy"],
            "expected_games": 2,
        },
        "m1_3": {
            "kills":          series["m1_3"]["kills"],
            "deaths":         series["m1_3"]["deaths"],
            "assists":        series["m1_3"]["assists"],
            "fantasy":        series["m1_3"]["fantasy"],
            "expected_games": 3,
        },
        "recent_form": recent.to_dict(orient="records"),
    }
