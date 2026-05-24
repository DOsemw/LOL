"""
feature_engineering.py
-----------------------
Builds predictive features for player K/D/A from OE match data.

Feature philosophy:
  - Everything uses ONLY information available BEFORE the match
  - Rolling windows are computed over past N games, never including current
  - Champion features capture role/kit tendencies (some champs just get more kills)
  - Matchup features capture opponent difficulty
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
TARGETS = ["kills", "deaths", "assists"]
ROLLING_WINDOWS = [3, 5, 10, 20]   # games lookback for rolling stats
MIN_GAMES_FOR_CHAMP = 10            # min games to trust champion-level averages


# ── Helpers ──────────────────────────────────────────────────────────────────

def _expanding_shift(series: pd.Series) -> pd.Series:
    """Exponentially weighted mean shifted by 1 — recent games count more, older games decay.
    Span=15 means the last ~15 games get the most weight, older games fade out gradually.
    """
    return series.ewm(span=15, min_periods=1).mean().shift(1)


def _rolling_shift(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over last `window` games, shifted by 1."""
    return series.rolling(window, min_periods=1).mean().shift(1)


# ── Core feature builders ─────────────────────────────────────────────────────

def add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player rolling stats over past N games.
    Groups by playername + position to avoid role-switch contamination.
    """
    df = df.sort_values(["playername", "date", "gameid"]).copy()

    for stat in TARGETS:
        grp = df.groupby(["playername", "position"])[stat]
        # Expanding (career average up to that point)
        df[f"{stat}_player_career_avg"] = grp.transform(_expanding_shift)
        # Rolling windows
        for w in ROLLING_WINDOWS:
            df[f"{stat}_player_roll{w}"] = grp.transform(
                lambda s, w=w: _rolling_shift(s, w)
            )

    # Rolling KDA ratio (kills+assists) / max(deaths,1)
    def kda_ratio(sub):
        k = sub["kills"].shift(1)
        d = sub["deaths"].shift(1).clip(lower=1)
        a = sub["assists"].shift(1)
        return (k + a) / d

    # Precompute then roll
    df["_kda"] = (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)
    for w in [5, 10]:
        df[f"player_kda_roll{w}"] = df.groupby(["playername", "position"])["_kda"].transform(
            lambda s, w=w: _rolling_shift(s, w)
        )
    df.drop(columns=["_kda"], inplace=True)

    # Kill participation proxy: kills / (kills + deaths + assists) for volatility
    df["_inv"] = 1 / (df["kills"] + df["deaths"] + df["assists"] + 1)
    df["player_kda_volatility"] = df.groupby(["playername", "position"])["_inv"].transform(
        lambda s: s.rolling(10, min_periods=3).std().shift(1)
    )
    df.drop(columns=["_inv"], inplace=True)

    # Games played (experience proxy)
    df["player_games_played"] = df.groupby(["playername", "position"]).cumcount()

    return df


def add_champion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Champion-level KDA averages across the entire dataset.
    Uses expanding window grouped by champion+position to stay leak-free.
    """
    df = df.sort_values(["champion", "position", "date"]).copy()

    for stat in TARGETS:
        grp = df.groupby(["champion", "position"])[stat]
        df[f"{stat}_champ_avg"] = grp.transform(_expanding_shift)

    # Champion kill-heavy / support flag derived from kill share
    df["champ_kill_share"] = df.groupby(["champion", "position"])["kills"].transform(
        _expanding_shift
    ) / (
        df.groupby(["champion", "position"])["kills"].transform(_expanding_shift) +
        df.groupby(["champion", "position"])["assists"].transform(_expanding_shift) + 1
    )

    # Champion pick rate (games as fraction of all games at that point in time)
    df["champ_games"] = df.groupby("champion").cumcount()

    return df


def add_team_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Team-level features: pace, kill-heaviness, and recent form.
    A team that plays fast/aggressive inflates individual K/A numbers.
    """
    df = df.sort_values(["teamname", "date", "gameid"]).copy()

    # Team total kills per game (proxy for game pace / blood-bath factor)
    team_kills = df.groupby(["gameid", "teamname"])["kills"].transform("sum")
    df["team_kills_pergame"] = team_kills

    # Rolling team kill rate per game
    team_kpg = df.drop_duplicates(["gameid", "teamname"])[["gameid", "teamname", "date", "kills"]].copy()
    team_kpg = team_kpg.groupby(["gameid", "teamname"])["kills"].sum().reset_index()
    team_kpg = team_kpg.sort_values(["teamname", "gameid"])
    for w in [5, 10]:
        team_kpg[f"team_kill_roll{w}"] = team_kpg.groupby("teamname")["kills"].transform(
            lambda s, w=w: _rolling_shift(s, w)
        )

    df = df.merge(
        team_kpg[["gameid", "teamname"] + [f"team_kill_roll{w}" for w in [5, 10]]],
        on=["gameid", "teamname"], how="left"
    )

    # Team win rate (rolling)
    if "result" in df.columns:
        win_cols = df.drop_duplicates(["gameid", "teamname"])[
            ["gameid", "teamname", "date", "result"]
        ].copy()
        win_cols = win_cols.sort_values(["teamname", "date"])
        for w in [5, 10]:
            win_cols[f"team_winrate_roll{w}"] = win_cols.groupby("teamname")["result"].transform(
                lambda s, w=w: _rolling_shift(s, w)
            )
        df = df.merge(
            win_cols[["gameid", "teamname"] + [f"team_winrate_roll{w}" for w in [5, 10]]],
            on=["gameid", "teamname"], how="left"
        )

    # Game length rolling avg (longer games = more assists typically)
    game_len = df.drop_duplicates("gameid")[["gameid", "teamname", "gamelength"]].copy()
    game_len = game_len.sort_values(["teamname", "gameid"])
    game_len["team_gamelength_roll5"] = game_len.groupby("teamname")["gamelength"].transform(
        lambda s: _rolling_shift(s, 5)
    )
    df = df.merge(
        game_len[["gameid", "teamname", "team_gamelength_roll5"]],
        on=["gameid", "teamname"], how="left"
    )

    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opponent in the same position: how many kills/deaths do they give up?
    Blue side vs red side matchup.
    """
    # Opponent side
    df["opp_side"] = df["side"].map({"Blue": "Red", "Red": "Blue"})

    # Build opponent lookup: same gameid, same position, opposite side
    opp = df[["gameid", "side", "position", "playername",
              "kills_player_roll5", "deaths_player_roll5", "assists_player_roll5"]].copy()
    opp.columns = ["gameid", "opp_side", "position",
                   "opp_playername",
                   "opp_kills_roll5", "opp_deaths_roll5", "opp_assists_roll5"]

    df = df.merge(opp, on=["gameid", "opp_side", "position"], how="left")
    df.drop(columns=["opp_side"], inplace=True)

    return df


def add_patch_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patch-level champion averages. Captures meta shifts (e.g., high-damage patches).
    """
    if "patch" not in df.columns:
        return df

    df = df.sort_values(["patch", "champion", "date"]).copy()

    for stat in TARGETS:
        patch_avg = (
            df.groupby(["patch", "champion", "position"])[stat]
            .expanding().mean().shift(1)
            .reset_index(level=[0, 1, 2], drop=True)
        )
        df[f"{stat}_patch_champ_avg"] = patch_avg

    return df


def add_opponent_defensive_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many kills/deaths does the OPPOSING TEAM give up per game, by position?
    e.g. if the enemy top laner consistently gives up 4 kills/game, that's a
    strong signal the player facing them will get more kills.

    This is different from opponent player rolling stats — it's the team's
    defensive weakness at each position.
    """
    df = df.sort_values(["teamname", "position", "date"]).copy()

    # For each player, build "kills given up by their team at this position"
    # = kills scored by the OPPONENT in that position against this team
    # We compute: for each (gameid, teamname, position) → opp_position_kills_allowed

    # Step 1: get opponent kills by position per game
    opp_kills = df[["gameid", "side", "position", "kills", "deaths", "assists"]].copy()
    opp_kills["opp_side"] = opp_kills["side"].map({"Blue": "Red", "Red": "Blue"})

    # Rename to mark as "what the opponent scored at this position"
    opp_scored = opp_kills.rename(columns={
        "kills":   "opp_pos_kills_scored",
        "deaths":  "opp_pos_deaths_scored",
        "assists": "opp_pos_assists_scored",
        "side":    "def_side",
    })[["gameid", "opp_side", "position",
        "opp_pos_kills_scored", "opp_pos_deaths_scored", "opp_pos_assists_scored"]]

    # Merge back: for each player row, get what the opponent scored at same position
    df = df.merge(
        opp_scored.rename(columns={"opp_side": "side"}),
        on=["gameid", "side", "position"], how="left"
    )

    # Step 2: rolling average of kills ALLOWED by this team at this position
    # (how soft is this team defensively at each role?)
    for stat, src_col in [
        ("kills",   "opp_pos_kills_scored"),
        ("deaths",  "opp_pos_deaths_scored"),
        ("assists", "opp_pos_assists_scored"),
    ]:
        col_name = f"team_pos_{stat}_allowed_roll5"
        df[col_name] = df.groupby(["teamname", "position"])[src_col].transform(
            lambda s: _rolling_shift(s, 5)
        )

    # Step 3: Now flip perspective — for each player, get the OPPONENT team's
    # defensive weakness at their position (how many kills do they give up there?)
    weakness = df[["gameid", "side", "position",
                   "team_pos_kills_allowed_roll5",
                   "team_pos_deaths_allowed_roll5",
                   "team_pos_assists_allowed_roll5"]].copy()
    weakness["opp_side"] = weakness["side"].map({"Blue": "Red", "Red": "Blue"})
    weakness = weakness.rename(columns={
        "team_pos_kills_allowed_roll5":   "opp_team_kills_allowed_roll5",
        "team_pos_deaths_allowed_roll5":  "opp_team_deaths_allowed_roll5",
        "team_pos_assists_allowed_roll5": "opp_team_assists_allowed_roll5",
    })

    df = df.merge(
        weakness[["gameid", "opp_side", "position",
                  "opp_team_kills_allowed_roll5",
                  "opp_team_deaths_allowed_roll5",
                  "opp_team_assists_allowed_roll5"]].rename(
            columns={"opp_side": "side"}
        ),
        on=["gameid", "side", "position"], how="left"
    )

    # Clean up temp cols
    df.drop(columns=["opp_pos_kills_scored", "opp_pos_deaths_scored",
                     "opp_pos_assists_scored"], inplace=True, errors="ignore")

    return df


def add_champion_patch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Champion performance on the CURRENT patch vs their historical average.
    Captures meta shifts: e.g. a buffed jungler suddenly getting more kills.

    Features:
      - champ_patch_kills_avg: avg kills for this champ+position on current patch
      - champ_patch_kills_delta: difference from their all-time average
        (positive = performing better on this patch than usual)
    """
    if "patch" not in df.columns:
        return df

    df = df.sort_values(["champion", "position", "patch", "date"]).copy()

    for stat in TARGETS:
        # Average for this champion+position on this specific patch (expanding, leak-free)
        df[f"{stat}_champ_patch_avg"] = df.groupby(
            ["champion", "position", "patch"]
        )[stat].transform(_expanding_shift)

        # Delta vs all-time champion average (already computed in add_champion_features)
        all_time_col = f"{stat}_champ_avg"
        if all_time_col in df.columns:
            df[f"{stat}_champ_patch_delta"] = (
                df[f"{stat}_champ_patch_avg"] - df[all_time_col]
            )

    # Position-level patch averages: is this patch good for junglers overall?
    for stat in TARGETS:
        df[f"{stat}_pos_patch_avg"] = df.groupby(
            ["position", "patch"]
        )[stat].transform(_expanding_shift)

        # Delta vs position all-time average
        pos_alltime = df.groupby("position")[stat].transform(_expanding_shift)
        df[f"{stat}_pos_patch_delta"] = df[f"{stat}_pos_patch_avg"] - pos_alltime

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Is the player trending up or down recently?
    Momentum = short-term average minus long-term average.
    Positive = hot streak, Negative = cold streak.
    """
    df = df.sort_values(["playername", "position", "date"]).copy()

    for stat in TARGETS:
        short = df.groupby(["playername", "position"])[stat].transform(
            lambda s: _rolling_shift(s, 3)
        )
        long_ = df.groupby(["playername", "position"])[stat].transform(
            lambda s: _rolling_shift(s, 10)
        )
        df[f"{stat}_momentum"] = short - long_

    # Team momentum (winning streak direction)
    if "result" in df.columns:
        team_short = df.groupby("teamname")["result"].transform(
            lambda s: _rolling_shift(s, 3)
        )
        team_long = df.groupby("teamname")["result"].transform(
            lambda s: _rolling_shift(s, 10)
        )
        df["team_momentum"] = team_short - team_long

    return df



    """
    Blue vs Red side has historically different kill distributions in some metas.
    """
    df["is_blue_side"] = (df["side"] == "Blue").astype(int)

    for stat in TARGETS:
        side_avg = df.groupby(["position", "side"])[stat].transform(_expanding_shift)
        df[f"{stat}_side_pos_avg"] = side_avg

    return df


def add_position_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode position (top/jng/mid/bot/sup)."""
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    return pd.concat([df, pos_dummies], axis=1)


# ── Master pipeline ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run all feature engineering steps in order.
    Input: cleaned player-level OE dataframe
    Output: feature-rich dataframe ready for modelling
    """
    steps = [
        ("Player rolling stats",          add_player_rolling_features),
        ("Champion averages",              add_champion_features),
        ("Champion-patch interactions",    add_champion_patch_features),
        ("Team context",                   add_team_context_features),
        ("Opponent features",              add_opponent_features),
        ("Opponent defensive strength",    add_opponent_defensive_strength),
        ("Player momentum",               add_momentum_features),
        ("Side features",                  add_side_features),
        ("Position encoding",              add_position_encoding),
    ]

    for name, fn in steps:
        if verbose:
            print(f"  [features] {name} ...")
        df = fn(df)

    if verbose:
        print(f"  [features] Done. Shape: {df.shape}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns (excludes targets and metadata)."""
    exclude = set(TARGETS) | {
        "gameid", "date", "league", "split", "patch",
        "side", "position", "playername", "teamname",
        "champion", "ban1", "ban2", "ban3", "ban4", "ban5",
        "result", "opp_playername",
    }
    cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


if __name__ == "__main__":
    from data_ingestion import load_raw, filter_major_leagues
    print("=== Feature Engineering Test ===\n")
    raw = load_raw(years=[2023])
    major = filter_major_leagues(raw)
    feat = build_features(major)
    fcols = get_feature_columns(feat)
    print(f"\nFeature columns ({len(fcols)}):")
    for c in fcols[:30]:
        print(f"  {c}")
    if len(fcols) > 30:
        print(f"  ... and {len(fcols)-30} more")
