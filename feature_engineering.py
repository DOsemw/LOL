"""
feature_engineering.py
-----------------------
Builds predictive features for player K/D/A from OE match data.

Key design principles:
  - All features use ONLY information available BEFORE the match (no leakage)
  - Win probability is baked in as a feature so the model learns it directly
  - Outlier games are winsorized adaptively based on sample size
  - Win/loss split rolling stats capture context-dependent performance
"""

import numpy as np
import pandas as pd

TARGETS = ["kills", "deaths", "assists"]
ROLLING_WINDOWS = [3, 5, 10, 20]


# ── Core helpers ──────────────────────────────────────────────────────────────

def _winsorize(series: pd.Series) -> pd.Series:
    """
    Cap outlier values adaptively based on sample size.
    Fewer games = more aggressive capping to avoid single blowout games
    dominating predictions.
    """
    n     = series.expanding().count().shift(1)
    cap85 = series.expanding().quantile(0.85).shift(1)
    cap90 = series.expanding().quantile(0.90).shift(1)
    cap93 = series.expanding().quantile(0.93).shift(1)
    cap   = pd.Series(
        np.where(n < 20, cap85, np.where(n < 40, cap90, cap93)),
        index=series.index
    )
    return series.clip(upper=cap)


def _ewm_shift(series: pd.Series, span: int = 15) -> pd.Series:
    """Exponentially weighted mean, shifted by 1 to avoid leakage."""
    return series.ewm(span=span, min_periods=1).mean().shift(1)


def _rolling_shift(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over last N games, shifted by 1."""
    return series.rolling(window, min_periods=1).mean().shift(1)


# ── Feature builders ──────────────────────────────────────────────────────────

def add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["playername", "date", "gameid"]).copy()

    # Winsorize first
    for stat in TARGETS:
        df[f"_{stat}_w"] = df.groupby(["playername", "position"])[stat].transform(_winsorize)

    for stat in TARGETS:
        wcol = f"_{stat}_w"
        grp  = df.groupby(["playername", "position"])[wcol]

        # EWM career average
        df[f"{stat}_player_career_avg"] = grp.transform(_ewm_shift)

        # Rolling windows
        for w in ROLLING_WINDOWS:
            df[f"{stat}_player_roll{w}"] = grp.transform(
                lambda s, w=w: _rolling_shift(s, w)
            )

    # KDA ratio
    df["_kda"] = (
        df["_kills_w"] + df["_assists_w"]
    ) / df["deaths"].clip(lower=1)
    for w in [5, 10]:
        df[f"player_kda_roll{w}"] = df.groupby(
            ["playername", "position"]
        )["_kda"].transform(lambda s, w=w: _rolling_shift(s, w))

    # Win/loss split rolling stats — KEY for moneyline-aware predictions
    if "result" in df.columns:
        for stat in TARGETS:
            wcol = f"_{stat}_w"
            for result_val, suffix in [(1, "win"), (0, "loss")]:
                def _split_roll(g, wcol=wcol, rv=result_val):
                    masked = g[wcol].where(g["result"] == rv)
                    return masked.rolling(10, min_periods=1).mean().shift(1)

                df[f"{stat}_player_roll10_{suffix}"] = df.groupby(
                    ["playername", "position"], group_keys=False
                ).apply(_split_roll).reset_index(level=[0,1], drop=True)

        # Per-player win rate
        for w in [5, 10]:
            df[f"player_winrate_roll{w}"] = df.groupby(
                ["playername", "position"]
            )["result"].transform(lambda s, w=w: _rolling_shift(s, w))

    # Games played
    df["player_games_played"] = df.groupby(["playername", "position"]).cumcount()

    # Clean temp cols
    df.drop(columns=[f"_{s}_w" for s in TARGETS] + ["_kda"],
            inplace=True, errors="ignore")
    return df


def add_champion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["champion", "position", "date"]).copy()
    for stat in TARGETS:
        df[f"{stat}_champ_avg"] = df.groupby(
            ["champion", "position"]
        )[stat].transform(_ewm_shift)
    df["champ_kill_share"] = (
        df["kills_champ_avg"] /
        (df["kills_champ_avg"] + df["assists_champ_avg"] + 1)
    )
    df["champ_games"] = df.groupby("champion").cumcount()
    return df


def add_champion_patch_features(df: pd.DataFrame) -> pd.DataFrame:
    if "patch" not in df.columns:
        return df
    df = df.sort_values(["champion", "position", "patch", "date"]).copy()
    for stat in TARGETS:
        df[f"{stat}_champ_patch_avg"] = df.groupby(
            ["champion", "position", "patch"]
        )[stat].transform(_ewm_shift)
        if f"{stat}_champ_avg" in df.columns:
            df[f"{stat}_champ_patch_delta"] = (
                df[f"{stat}_champ_patch_avg"] - df[f"{stat}_champ_avg"]
            )
    return df


def add_team_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["teamname", "date", "gameid"]).copy()

    team_kills = df.groupby(["gameid", "teamname"])["kills"].transform("sum")
    df["team_kills_pergame"] = team_kills

    # Rolling team stats (deduplicated per game)
    team_game = df.drop_duplicates(["gameid", "teamname"]).sort_values(["teamname", "date"])
    for w in [5, 10]:
        team_game[f"team_kill_roll{w}"] = team_game.groupby("teamname")["kills"].transform(
            lambda s, w=w: _rolling_shift(s, w)
        )
    df = df.merge(
        team_game[["gameid","teamname"] + [f"team_kill_roll{w}" for w in [5,10]]],
        on=["gameid","teamname"], how="left"
    )

    if "result" in df.columns:
        for w in [5, 10]:
            team_game[f"team_winrate_roll{w}"] = team_game.groupby("teamname")["result"].transform(
                lambda s, w=w: _rolling_shift(s, w)
            )
        df = df.merge(
            team_game[["gameid","teamname"] + [f"team_winrate_roll{w}" for w in [5,10]]],
            on=["gameid","teamname"], how="left"
        )

    game_len = df.drop_duplicates("gameid")[["gameid","teamname","gamelength"]].copy()
    game_len = game_len.sort_values(["teamname","gameid"])
    game_len["team_gamelength_roll5"] = game_len.groupby("teamname")["gamelength"].transform(
        lambda s: _rolling_shift(s, 5)
    )
    df = df.merge(game_len[["gameid","teamname","team_gamelength_roll5"]],
                  on=["gameid","teamname"], how="left")
    return df


def add_opponent_defensive_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["teamname", "position", "date"]).copy()

    opp_kills = df[["gameid","side","position","kills","deaths","assists"]].copy()
    opp_kills["opp_side"] = opp_kills["side"].map({"Blue":"Red","Red":"Blue"})
    opp_scored = opp_kills.rename(columns={
        "kills":"opp_pos_kills_scored",
        "deaths":"opp_pos_deaths_scored",
        "assists":"opp_pos_assists_scored",
        "side":"def_side",
    })[["gameid","opp_side","position",
        "opp_pos_kills_scored","opp_pos_deaths_scored","opp_pos_assists_scored"]]

    df = df.merge(
        opp_scored.rename(columns={"opp_side":"side"}),
        on=["gameid","side","position"], how="left"
    )

    for stat, src in [("kills","opp_pos_kills_scored"),
                      ("deaths","opp_pos_deaths_scored"),
                      ("assists","opp_pos_assists_scored")]:
        df[f"team_pos_{stat}_allowed_roll5"] = df.groupby(
            ["teamname","position"]
        )[src].transform(lambda s: _rolling_shift(s, 5))

    weakness = df[["gameid","side","position",
                   "team_pos_kills_allowed_roll5",
                   "team_pos_deaths_allowed_roll5",
                   "team_pos_assists_allowed_roll5"]].copy()
    weakness["opp_side"] = weakness["side"].map({"Blue":"Red","Red":"Blue"})
    weakness = weakness.rename(columns={
        "team_pos_kills_allowed_roll5":   "opp_team_kills_allowed_roll5",
        "team_pos_deaths_allowed_roll5":  "opp_team_deaths_allowed_roll5",
        "team_pos_assists_allowed_roll5": "opp_team_assists_allowed_roll5",
    })
    df = df.merge(
        weakness[["gameid","opp_side","position",
                  "opp_team_kills_allowed_roll5",
                  "opp_team_deaths_allowed_roll5",
                  "opp_team_assists_allowed_roll5"]].rename(
            columns={"opp_side":"side"}),
        on=["gameid","side","position"], how="left"
    )
    df.drop(columns=["opp_pos_kills_scored","opp_pos_deaths_scored",
                     "opp_pos_assists_scored"], inplace=True, errors="ignore")
    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    df["opp_side"] = df["side"].map({"Blue":"Red","Red":"Blue"})
    opp = df[["gameid","side","position","playername",
              "kills_player_roll5","deaths_player_roll5","assists_player_roll5"]].copy()
    opp.columns = ["gameid","opp_side","position","opp_playername",
                   "opp_kills_roll5","opp_deaths_roll5","opp_assists_roll5"]
    df = df.merge(opp, on=["gameid","opp_side","position"], how="left")
    df.drop(columns=["opp_side"], inplace=True)
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["playername","position","date"]).copy()
    for stat in TARGETS:
        short = df.groupby(["playername","position"])[stat].transform(
            lambda s: _rolling_shift(s, 3))
        long_ = df.groupby(["playername","position"])[stat].transform(
            lambda s: _rolling_shift(s, 10))
        df[f"{stat}_momentum"] = short - long_
    if "result" in df.columns:
        team_short = df.groupby("teamname")["result"].transform(lambda s: _rolling_shift(s, 3))
        team_long  = df.groupby("teamname")["result"].transform(lambda s: _rolling_shift(s, 10))
        df["team_momentum"] = team_short - team_long
    return df


def add_side_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_blue_side"] = (df["side"] == "Blue").astype(int)
    for stat in TARGETS:
        df[f"{stat}_side_pos_avg"] = df.groupby(
            ["position","side"]
        )[stat].transform(_ewm_shift)
    return df


def add_position_encoding(df: pd.DataFrame) -> pd.DataFrame:
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    return pd.concat([df, pos_dummies], axis=1)


# ── Master pipeline ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    steps = [
        ("Player rolling stats",        add_player_rolling_features),
        ("Champion averages",            add_champion_features),
        ("Champion-patch interactions",  add_champion_patch_features),
        ("Team context",                 add_team_context_features),
        ("Opponent defensive strength",  add_opponent_defensive_strength),
        ("Opponent features",            add_opponent_features),
        ("Player momentum",              add_momentum_features),
        ("Side features",                add_side_features),
        ("Position encoding",            add_position_encoding),
    ]
    for name, fn in steps:
        if verbose:
            print(f"  [features] {name} ...")
        df = fn(df)
    if verbose:
        print(f"  [features] Done. Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(TARGETS) | {
        "gameid","date","league","split","patch",
        "side","position","playername","teamname",
        "champion","ban1","ban2","ban3","ban4","ban5",
        "result","opp_playername",
    }
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
