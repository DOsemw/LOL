"""
main.py — LoL Props Prediction API
Deploy all files flat (no subfolders) to Railway.
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATE = {"df": None, "feature_cols": None, "ready": False,
         "n_players": 0, "date_range": ""}


def load_everything():
    from data_ingestion      import load_raw
    from feature_engineering import build_features, get_feature_columns
    from model               import train_all, load_model

    years = [int(y) for y in os.getenv("YEARS", "2026").split(",")]
    log.info(f"Loading data for years: {years}")

    raw  = load_raw(years=years)
    feat = build_features(raw, verbose=False)
    fcols = get_feature_columns(feat)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    models_exist = all(
        (model_dir / f"{t}_model.pkl").exists()
        for t in ["kills","deaths","assists"]
    )
    if not models_exist:
        log.info("Training models (first run — ~5 min) ...")
        train_all(feat, fcols)
    else:
        log.info("Loaded existing models from disk.")

    # Compute win/loss calibration ratios from data
    from calibration import compute_win_loss_ratios, save_calibration
    ratios = compute_win_loss_ratios(raw)
    save_calibration(ratios)
    log.info(f"Calibration computed for {len(ratios)} positions")

    STATE.update({
        "df": feat, "feature_cols": fcols, "ready": True,
        "n_players": feat["playername"].nunique(),
        "date_range": f"{feat['date'].min().date()} → {feat['date'].max().date()}",
        "calibration": ratios,
    })
    log.info(f"Ready — {STATE['n_players']} players, {STATE['date_range']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_everything()
    yield

app = FastAPI(title="LoL Props API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_ready():
    if not STATE["ready"]:
        raise HTTPException(503, detail="Model still loading — try again in a minute.")


def _get_player_row(player: str):
    df    = STATE["df"]
    fcols = STATE["feature_cols"]
    mask  = df["playername"].str.lower() == player.strip().lower()
    if not mask.any():
        suggestions = [p for p in df["playername"].unique()
                       if player.lower() in p.lower()][:6]
        raise HTTPException(404, detail={
            "message": f"Player '{player}' not found.",
            "suggestions": suggestions
        })
    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][fcols].copy()
    return player_df, X


def _predict_stat(model, X: pd.DataFrame, mae: float) -> dict:
    base = max(0.0, float(model.predict(X)[0]))
    preds = []
    for _ in range(150):
        noisy = X.copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noisy[col] += np.random.normal(
                0, abs(float(noisy[col].iloc[0])) * 0.05 + 0.01
            )
        preds.append(max(0.0, float(model.predict(noisy)[0])))
    return {
        "per_game": round(base, 2),
        "low":      round(max(0.0, float(np.quantile(preds, 0.05))), 2),
        "high":     round(float(np.quantile(preds, 0.95)), 2),
        "mae":      mae,
    }


def _build_series(pg: dict, exp_games: float, win_prob: float) -> dict:
    result = {"expected_games": round(exp_games, 2), "win_prob": round(win_prob, 3)}
    for stat in ["kills","deaths","assists"]:
        s = pg[stat]
        result[stat] = {
            "per_game":     s["per_game"],
            "series_total": round(s["per_game"] * exp_games, 1),
            "series_low":   round(s["low"]       * exp_games, 1),
            "series_high":  round(s["high"]      * exp_games, 1),
            "mae":          round(s["mae"]        * exp_games, 2),
        }
    k = result["kills"]["series_total"]
    d = result["deaths"]["series_total"]
    a = result["assists"]["series_total"]
    result["fantasy"] = round(k * 3 + a * 1.5 - d, 1)
    return result


def _apply_win_prob_blend(X: pd.DataFrame, win_prob: float,
                           df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend win-game and loss-game rolling stats based on win probability.
    This is the core accuracy mechanism — if team is 75% favourite,
    features = 75% from winning games + 25% from losing games.
    Also directly overrides player_winrate features.
    """
    X = X.copy()

    for stat in ["kills","deaths","assists"]:
        win_col  = f"{stat}_player_roll10_win"
        loss_col = f"{stat}_player_roll10_loss"
        base_col = f"{stat}_player_roll10"

        if win_col in X.columns and loss_col in X.columns:
            win_val  = float(X[win_col].fillna(
                X.get(base_col, pd.Series([np.nan])).iloc[0]
            ).iloc[0])
            loss_val = float(X[loss_col].fillna(
                X.get(base_col, pd.Series([np.nan])).iloc[0]
            ).iloc[0])

            # Fill NaN with overall average if no win/loss history
            if np.isnan(win_val):
                win_val = float(X[base_col].iloc[0]) if base_col in X.columns else loss_val
            if np.isnan(loss_val):
                loss_val = float(X[base_col].iloc[0]) if base_col in X.columns else win_val

            blended = win_prob * win_val + (1 - win_prob) * loss_val

            if base_col in X.columns:
                X[base_col] = blended
            career_col = f"{stat}_player_career_avg"
            if career_col in X.columns:
                X[career_col] = blended

    # Override win rate features with the actual expected win prob
    for col in ["player_winrate_roll5", "player_winrate_roll10"]:
        if col in X.columns:
            X[col] = win_prob
    for col in ["team_winrate_roll5", "team_winrate_roll10"]:
        if col in X.columns:
            X[col] = win_prob

    return X


def _apply_opponent_adjustment(X: pd.DataFrame, opponent: str,
                                position: str, df: pd.DataFrame) -> tuple:
    """
    Override opponent defensive strength features with actual upcoming opponent stats.
    Returns (adjusted X, adjustment info dict).
    """
    opp_mask = df["teamname"].str.lower() == opponent.strip().lower()
    if not opp_mask.any():
        opp_mask = df["teamname"].str.lower().str.contains(
            opponent.strip().lower(), na=False
        )
    if not opp_mask.any():
        return X, {}

    opp_df  = df[opp_mask & (df["position"] == position)].sort_values("date", ascending=False)
    if len(opp_df) < 3:
        return X, {}

    # Get opponent's recent kills allowed at this position
    col = "opp_team_kills_allowed_roll5"
    if col in opp_df.columns and opp_df[col].notna().any():
        opp_kills_allowed = float(opp_df[col].dropna().iloc[0])
    else:
        opp_kills_allowed = float(opp_df["kills"].tail(10).mean())

    avg_kills = float(df[df["position"] == position]["kills"].mean())
    avg_deaths = float(df[df["position"] == position]["deaths"].mean())
    opp_deaths_allowed = float(opp_df["deaths"].tail(10).mean())

    # Clamp ratios to prevent extreme adjustments
    kills_ratio  = min(max(opp_kills_allowed / max(avg_kills, 0.1),  0.6), 1.6)
    deaths_ratio = min(max(opp_deaths_allowed / max(avg_deaths, 0.1), 0.6), 1.6)

    # Override opponent features in X
    for col in ["opp_team_kills_allowed_roll5"]:
        if col in X.columns:
            X[col] = opp_kills_allowed
    for col in ["opp_team_deaths_allowed_roll5"]:
        if col in X.columns:
            X[col] = opp_deaths_allowed

    return X, {
        "kills_ratio":  round(kills_ratio, 3),
        "deaths_ratio": round(deaths_ratio, 3),
        "opp_team":     opp_df.iloc[0]["teamname"],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":     "ok" if STATE["ready"] else "loading",
        "players":    STATE["n_players"],
        "date_range": STATE["date_range"],
    }


@app.get("/search")
def search_players(q: str = Query(...)):
    _check_ready()
    df = STATE["df"]
    matches = df[df["playername"].str.lower().str.contains(q.strip().lower(), na=False)]
    players = (
        matches.groupby("playername")
               .agg(league=("league","last"), position=("position","last"),
                    games=("gameid","nunique"))
               .reset_index().sort_values("games", ascending=False).head(10)
    )
    return players.to_dict(orient="records")


@app.get("/players")
def list_players(league: str = Query(None), q: str = Query(None)):
    _check_ready()
    df = STATE["df"]
    if league: df = df[df["league"].str.upper() == league.upper()]
    if q:      df = df[df["playername"].str.lower().str.contains(q.lower(), na=False)]
    players = (
        df.groupby("playername")
          .agg(league=("league","last"), position=("position","last"),
               games=("gameid","nunique"))
          .reset_index().sort_values("games", ascending=False)
    )
    return players.to_dict(orient="records")


@app.get("/teams")
def list_teams(league: str = Query(None), q: str = Query(None)):
    _check_ready()
    df = STATE["df"]
    if league: df = df[df["league"].str.upper() == league.upper()]
    if q:      df = df[df["teamname"].str.lower().str.contains(q.lower(), na=False)]
    teams = (
        df.groupby("teamname")
          .agg(league=("league","last"), games=("gameid","nunique"))
          .reset_index().sort_values("games", ascending=False)
    )
    return teams.to_dict(orient="records")


@app.get("/predict")
def predict_player(
    player:    str   = Query(...),
    side:      str   = Query("Blue"),
    moneyline: int   = Query(None),
    opp_ml:    int   = Query(None),
    opponent:  str   = Query(None),
):
    _check_ready()
    from model          import load_model
    from series_predictor import vig_adjusted_probs, expected_games

    df    = STATE["df"]
    fcols = STATE["feature_cols"]

    player_df, X = _get_player_row(player)
    position      = player_df.iloc[0]["position"]

    if "is_blue_side" in X.columns:
        X["is_blue_side"] = 1 if side.lower() == "blue" else 0

    # Win probability
    if moneyline is not None and opp_ml is not None:
        win_prob, _ = vig_adjusted_probs(moneyline, opp_ml)
    else:
        win_prob = 0.5

    # 1. Apply opponent defensive adjustment to features
    opp_info = {}
    if opponent:
        X, opp_info = _apply_opponent_adjustment(X, opponent, position, df)

    # 2. Get base per-game predictions from model
    pg = {}
    for stat in ["kills","deaths","assists"]:
        model, _, metrics = load_model(stat)
        pg[stat] = _predict_stat(model, X, metrics["mae"])

    # 3. Scale predictions using data-driven win/loss calibration
    # This replaces hand-crafted multipliers with empirical ratios from actual data
    if win_prob != 0.5:
        from calibration import scale_prediction, load_calibration
        ratios = STATE.get("calibration") or load_calibration()
        if ratios:
            for stat in ["kills","deaths","assists"]:
                scaled = scale_prediction(
                    pg[stat]["per_game"], stat, position, win_prob, ratios
                )
                ratio = scaled / pg[stat]["per_game"] if pg[stat]["per_game"] > 0 else 1.0
                ratio = min(max(ratio, 0.5), 2.0)
                pg[stat]["per_game"] = round(pg[stat]["per_game"] * ratio, 2)
                pg[stat]["low"]      = round(pg[stat]["low"]      * ratio, 2)
                pg[stat]["high"]     = round(pg[stat]["high"]      * ratio, 2)

    # 4. Apply opponent kill ratio adjustment
    if opp_info:
        kills_ratio   = min(max(opp_info.get("kills_ratio",  1.0), 0.6), 1.6)
        deaths_ratio  = min(max(opp_info.get("deaths_ratio", 1.0), 0.6), 1.6)
        assists_ratio = (kills_ratio + 1.0) / 2.0
        for stat, ratio in [("kills", kills_ratio),
                             ("deaths", deaths_ratio),
                             ("assists", assists_ratio)]:
            pg[stat]["per_game"] = round(pg[stat]["per_game"] * ratio, 2)
            pg[stat]["low"]      = round(pg[stat]["low"]      * ratio, 2)
            pg[stat]["high"]     = round(pg[stat]["high"]     * ratio, 2)

    # 5. Series projections
    exp_bo3 = expected_games("Bo3", win_prob)
    exp_bo2 = 2.0  # M1-2 always 2 maps

    # Recent form
    recent = player_df.head(5)[["date","champion","kills","deaths","assists"]].copy()
    recent["date"] = recent["date"].astype(str)

    return {
        "player":    player_df.iloc[0]["playername"],
        "position":  position,
        "league":    player_df.iloc[0]["league"],
        "win_prob":  round(win_prob, 3),
        "moneyline": moneyline,
        "opponent":  opp_info.get("opp_team", opponent),

        "map1": {stat: {
            "expected": pg[stat]["per_game"],
            "low":      pg[stat]["low"],
            "high":     pg[stat]["high"],
            "mae":      pg[stat]["mae"],
        } for stat in ["kills","deaths","assists"]},

        "m1_2": _build_series(pg, exp_bo2, win_prob),
        "m1_3": _build_series(pg, exp_bo3, win_prob),

        "recent_form": recent.to_dict(orient="records"),
    }


@app.get("/refresh")
def refresh():
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
