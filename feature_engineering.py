"""
feature_engineering.py v2
--------------------------
Builds features for two separate models:
  1. Team kills model  — how many total kills will happen this game?
  2. Kill share model  — what % of team kills does this player get?

Final prediction: team_kills × kill_share = player kills

Key features per the blueprint:
  - Kill Participation % (KP%) = (kills + assists) / team_kills
  - Kill Share % = kills / team_kills
  - Team Bloody Rate = (kills + deaths) / gamelength
  - Avg Game Length
  - Opponent Death Rate
  - Post-15 Kill Share
  - Win probability (from moneyline, as a feature)
"""

import numpy as np
import pandas as pd

ROLLING = 10  # primary rolling window
ROLLING_SHORT = 5


def _shift_ewm(series: pd.Series, span: int = 10) -> pd.Series:
    """EWM shifted by 1 to avoid leakage."""
    return series.ewm(span=span, min_periods=1).mean().shift(1)


def _shift_roll(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean shifted by 1."""
    return series.rolling(window, min_periods=1).mean().shift(1)


def _winsorize(series: pd.Series, pct: float = 0.95) -> pd.Series:
    """Cap at expanding percentile to handle outliers."""
    cap = series.expanding().quantile(pct).shift(1)
    return series.clip(upper=cap)


# ── Step 1: compute per-game derived stats ─────────────────────────────────

def add_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute kill share, KP%, bloody rate etc. per game row."""
    df = df.copy()

    # Team kills per game (sum all players on same team in same game)
    team_kills = df.groupby(["gameid", "teamname"])["kills"].transform("sum")
    team_deaths = df.groupby(["gameid", "teamname"])["deaths"].transform("sum")
    team_assists = df.groupby(["gameid", "teamname"])["assists"].transform("sum")

    df["team_kills"]   = team_kills
    df["team_deaths"]  = team_deaths

    # Kill share: what % of team kills does this player get?
    df["kill_share"] = np.where(team_kills > 0, df["kills"] / team_kills, 0)

    # Kill participation: (kills + assists) / team kills
    df["kp_pct"] = np.where(
        team_kills > 0,
        (df["kills"] + df["assists"]) / team_kills,
        0
    )

    # Bloody rate: (total kills + total deaths) per minute
    # Use both teams combined for full game pace
    game_total_kills  = df.groupby("gameid")["kills"].transform("sum")
    game_total_deaths = df.groupby("gameid")["deaths"].transform("sum")
    game_len_min = df["gamelength"] / 60.0
    df["bloody_rate"] = np.where(
        game_len_min > 0,
        (game_total_kills + game_total_deaths) / game_len_min,
        0
    )

    # Post-15 kill share (kills after 15 mins / total kills)
    # OE has killsat15 = kills up to 15 min
    if "killsat15" in df.columns:
        post15_kills = (df["kills"] - df["killsat15"].fillna(0)).clip(lower=0)
        df["post15_kill_share"] = np.where(
            team_kills > 0, post15_kills / team_kills, 0
        )
    else:
        df["post15_kill_share"] = df["kill_share"]

    # Damage share (already in OE)
    if "damageshare" not in df.columns:
        df["damageshare"] = 0.2  # default

    return df


# ── Step 2: player rolling features ───────────────────────────────────────

def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player rolling kill share and KP% features."""
    df = df.sort_values(["playername", "position", "date"]).copy()

    for stat in ["kill_share", "kp_pct", "post15_kill_share", "kills", "deaths", "assists"]:
        if stat not in df.columns:
            continue
        # Winsorize first
        w = df.groupby(["playername", "position"])[stat].transform(_winsorize)
        # EWM rolling average
        df[f"{stat}_player_ewm"] = df.groupby(
            ["playername", "position"]
        )[stat].transform(_shift_ewm)
        # Short rolling (recent form)
        df[f"{stat}_player_roll5"] = df.groupby(
            ["playername", "position"]
        )[stat].transform(lambda s: _shift_roll(s, 5))

    # Win/loss split kill share
    if "result" in df.columns:
        for rv, suffix in [(1, "win"), (0, "loss")]:
            df[f"kill_share_roll10_{suffix}"] = df.groupby(
                ["playername", "position"]
            )["kill_share"].transform(
                lambda s, rv=rv: s.where(
                    df.loc[s.index, "result"] == rv
                ).rolling(10, min_periods=1).mean().shift(1)
            )
        df["player_winrate"] = df.groupby(
            ["playername", "position"]
        )["result"].transform(lambda s: _shift_roll(s, 10))

    # Games played (for shrinkage)
    df["player_games"] = df.groupby(["playername", "position"]).cumcount()

    return df


# ── Step 3: team features ─────────────────────────────────────────────────

def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level pace and kill environment features."""
    df = df.sort_values(["teamname", "date"]).copy()

    # Deduplicate to one row per team per game
    team_game = df.drop_duplicates(["gameid", "teamname"]).copy()
    team_game = team_game.sort_values(["teamname", "date"])

    for stat, col in [
        ("team_kills",  "team_kills_ewm"),
        ("bloody_rate", "team_bloody_rate_ewm"),
        ("gamelength",  "team_gamelength_ewm"),
    ]:
        if stat not in team_game.columns:
            continue
        team_game[col] = team_game.groupby("teamname")[stat].transform(_shift_ewm)

    if "result" in team_game.columns:
        team_game["team_winrate"] = team_game.groupby("teamname")["result"].transform(
            lambda s: _shift_roll(s, 10)
        )

    merge_cols = ["gameid", "teamname"] + [
        c for c in ["team_kills_ewm", "team_bloody_rate_ewm",
                    "team_gamelength_ewm", "team_winrate"]
        if c in team_game.columns
    ]
    df = df.merge(team_game[merge_cols], on=["gameid", "teamname"], how="left")
    return df


# ── Step 4: opponent features ──────────────────────────────────────────────

def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Opponent team defensive stats — how many kills do they give up?"""
    df = df.sort_values(["teamname", "date"]).copy()

    # Opponent team = other team in same game
    opp_side = df["side"].map({"Blue": "Red", "Red": "Blue"})

    # Get opponent team kills allowed (= kills scored against them)
    opp_stats = df[["gameid", "side", "team_kills", "team_deaths",
                    "bloody_rate", "team_kills_ewm"]].copy()
    opp_stats["opp_side"] = opp_stats["side"].map({"Blue": "Red", "Red": "Blue"})
    opp_stats = opp_stats.rename(columns={
        "team_kills":     "opp_team_kills",
        "team_deaths":    "opp_team_deaths",
        "bloody_rate":    "opp_bloody_rate",
        "team_kills_ewm": "opp_team_kills_ewm",
    })

    df = df.merge(
        opp_stats[["gameid", "opp_side", "opp_team_kills",
                   "opp_team_deaths", "opp_bloody_rate", "opp_team_kills_ewm"]].rename(
            columns={"opp_side": "side"}
        ),
        on=["gameid", "side"], how="left"
    )

    # Opponent death rate rolling (how many kills do they give up per game?)
    team_game = df.drop_duplicates(["gameid", "teamname"]).sort_values(["teamname", "date"])
    team_game["opp_death_rate_ewm"] = team_game.groupby("teamname")["team_deaths"].transform(
        _shift_ewm
    )
    df = df.merge(
        team_game[["gameid", "teamname", "opp_death_rate_ewm"]],
        on=["gameid", "teamname"], how="left"
    )

    return df


# ── Step 5: position and side encoding ────────────────────────────────────

def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df["is_blue_side"] = (df["side"] == "Blue").astype(int)
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    return pd.concat([df, pos_dummies], axis=1)


# ── Master pipeline ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    steps = [
        ("Derived stats (kill share, KP%, bloody rate)", add_derived_stats),
        ("Player rolling features",                       add_player_features),
        ("Team pace features",                            add_team_features),
        ("Opponent features",                             add_opponent_features),
        ("Encodings",                                     add_encodings),
    ]
    for name, fn in steps:
        if verbose:
            print(f"  [features] {name} ...")
        df = fn(df)
    if verbose:
        print(f"  [features] Done. Shape: {df.shape}")
    return df


def get_team_kill_features(df: pd.DataFrame) -> list[str]:
    """Features for Model 1 — predicting total team kills."""
    candidates = [
        "team_kills_ewm", "team_bloody_rate_ewm", "team_gamelength_ewm",
        "team_winrate", "opp_team_kills_ewm", "opp_bloody_rate",
        "opp_death_rate_ewm", "is_blue_side",
        # early game indicators
        "golddiffat15", "xpdiffat15",
    ]
    return [c for c in candidates if c in df.columns]


def get_kill_share_features(df: pd.DataFrame) -> list[str]:
    """Features for Model 2 — predicting player kill share %."""
    candidates = [
        # Player kill share history
        "kill_share_player_ewm", "kill_share_player_roll5",
        "kp_pct_player_ewm", "kp_pct_player_roll5",
        "post15_kill_share_player_ewm",
        "kill_share_roll10_win", "kill_share_roll10_loss",
        "player_winrate", "player_games",
        # Raw kill history (winsorized via ewm)
        "kills_player_ewm", "kills_player_roll5",
        "deaths_player_ewm", "assists_player_ewm",
        # Damage share
        "damageshare",
        # Team context
        "team_kills_ewm", "team_winrate",
        # Position encoding
        "pos_top", "pos_jng", "pos_mid", "pos_bot", "pos_sup",
        "is_blue_side",
    ]
    return [c for c in candidates if c in df.columns]
