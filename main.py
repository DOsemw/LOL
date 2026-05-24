"""
main.py — LoL Props API
Deploy ALL files in this folder to Railway (flat, no subfolders).
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure current directory is on path so local modules are found
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATE = {"df": None, "feature_cols": None, "ready": False, "n_players": 0, "date_range": ""}


def load_everything():
    from data_ingestion      import load_raw, filter_major_leagues
    from feature_engineering import build_features, get_feature_columns
    from model               import train_all, load_model

    years = [int(y) for y in os.getenv("YEARS", "2023,2024").split(",")]
    log.info(f"Loading data for years: {years}")

    raw   = load_raw(years=years)
    major = filter_major_leagues(raw)
    feat  = build_features(major, verbose=False)
    fcols = get_feature_columns(feat)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    models_exist = all((model_dir / f"{t}_model.pkl").exists() for t in ["kills","deaths","assists"])
    if not models_exist:
        log.info("Training models (first run — takes ~5 min) ...")
        train_all(feat, fcols)
    else:
        log.info("Models loaded from disk.")

    STATE.update({
        "df": feat, "feature_cols": fcols, "ready": True,
        "n_players": feat["playername"].nunique(),
        "date_range": f"{feat['date'].min().date()} → {feat['date'].max().date()}",
    })
    log.info(f"Ready — {STATE['n_players']} players, {STATE['date_range']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_everything()
    yield

app = FastAPI(title="LoL Props API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_ready():
    if not STATE["ready"]:
        raise HTTPException(503, detail="Model still loading — try again in a minute.")

def _get_player_row(player: str):
    df    = STATE["df"]
    fcols = STATE["feature_cols"]
    mask  = df["playername"].str.lower() == player.strip().lower()
    if not mask.any():
        suggestions = [p for p in df["playername"].unique() if player.lower() in p.lower()][:6]
        raise HTTPException(404, detail={"message": f"Player '{player}' not found.", "suggestions": suggestions})
    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][fcols].copy()
    return player_df, X

def _predict_stat(model, X: pd.DataFrame, mae: float) -> dict:
    base = max(0.0, float(model.predict(X)[0]))
    preds = []
    for _ in range(150):
        noisy = X.copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noisy[col] += np.random.normal(0, abs(float(noisy[col].iloc[0])) * 0.05 + 0.01)
        preds.append(max(0.0, float(model.predict(noisy)[0])))
    alpha = 0.05
    return {
        "per_game": round(base, 2),
        "low":      round(max(0.0, float(np.quantile(preds, alpha))), 2),
        "high":     round(float(np.quantile(preds, 1 - alpha)), 2),
        "mae":      mae,
    }

def _build_series_result(pg: dict, exp_games: float, win_prob: float) -> dict:
    """Scale per-game predictions to a series total."""
    result = {"expected_games": round(exp_games, 2), "win_prob": round(win_prob, 3)}
    for stat in ["kills", "deaths", "assists"]:
        s = pg[stat]
        result[stat] = {
            "per_game":     s["per_game"],
            "series_total": round(s["per_game"] * exp_games, 1),
            "series_low":   round(s["low"]      * exp_games, 1),
            "series_high":  round(s["high"]     * exp_games, 1),
            "mae":          round(s["mae"]       * exp_games, 2),
        }
    # Fantasy: kills*3 + assists*1.5 - deaths
    k = result["kills"]["series_total"]
    d = result["deaths"]["series_total"]
    a = result["assists"]["series_total"]
    result["fantasy"] = round(k * 3 + a * 1.5 - d, 1)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok" if STATE["ready"] else "loading",
            "players": STATE["n_players"], "date_range": STATE["date_range"]}


@app.get("/search")
def search_players(q: str = Query(..., description="Player name fragment, e.g. 'fak'")):
    """Search players by name — used for autocomplete in Google Sheets."""
    _check_ready()
    df = STATE["df"]
    matches = df[df["playername"].str.lower().str.contains(q.strip().lower(), na=False)]
    players = (
        matches.groupby("playername")
               .agg(league=("league", "last"), position=("position", "last"), games=("gameid", "nunique"))
               .reset_index()
               .sort_values("games", ascending=False)
               .head(10)
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
          .agg(league=("league","last"), position=("position","last"), games=("gameid","nunique"))
          .reset_index().sort_values("games", ascending=False)
    )
    return players.to_dict(orient="records")


@app.get("/predict")
def predict_player(
    player:     str   = Query(...),
    side:       str   = Query("Blue"),
    moneyline:  int   = Query(None, description="Team ML e.g. -297"),
    opp_ml:     int   = Query(None, description="Opponent ML e.g. +297"),
):
    """
    Predict per-game AND series (M1-2 and M1-3) K/D/A for a player.
    Returns per_game, bo3 (M1-3), and bo2 (M1-2) projections.
    """
    _check_ready()
    from model                  import load_model
    from series_predictor       import vig_adjusted_probs, expected_games
    from moneyline_adjustments  import apply_moneyline_adjustments

    player_df, X = _get_player_row(player)
    if "is_blue_side" in X.columns:
        X["is_blue_side"] = 1 if side.lower() == "blue" else 0

    # Win probability
    if moneyline is not None and opp_ml is not None:
        win_prob, _ = vig_adjusted_probs(moneyline, opp_ml)
    else:
        win_prob = 0.5

    # Per-game base predictions
    pg = {}
    for stat in ["kills", "deaths", "assists"]:
        model, _, metrics = load_model(stat)
        pg[stat] = _predict_stat(model, X, metrics["mae"])

    # Apply moneyline adjustment to per-game predictions
    if moneyline is not None and opp_ml is not None:
        adj = apply_moneyline_adjustments(pg, win_prob)
        for stat in ["kills", "deaths", "assists"]:
            if stat in adj:
                pg[stat]["per_game"] = adj[stat]["mid"]
                pg[stat]["low"]      = adj[stat]["low"]
                pg[stat]["high"]     = adj[stat]["high"]

    # Recent form
    recent = player_df.head(5)[["date","champion","kills","deaths","assists"]].copy()
    recent["date"] = recent["date"].astype(str)

    # Series projections
    exp_bo3 = expected_games("Bo3", win_prob)   # M1-3
    exp_bo2 = expected_games_bo2(win_prob)       # M1-2

    return {
        "player":    player_df.iloc[0]["playername"],
        "position":  player_df.iloc[0]["position"],
        "league":    player_df.iloc[0]["league"],
        "win_prob":  round(win_prob, 3),
        "moneyline": moneyline,

        # Per-game (single map)
        "map1": {stat: {"expected": pg[stat]["per_game"],
                        "low": pg[stat]["low"],
                        "high": pg[stat]["high"],
                        "mae": pg[stat]["mae"]}
                 for stat in ["kills","deaths","assists"]},

        # M1-2 (first 2 maps of a Bo3, series ends after map 2 if one team up 2-0)
        "m1_2": _build_series_result(pg, exp_bo2, win_prob),

        # M1-3 (full Bo3)
        "m1_3": _build_series_result(pg, exp_bo3, win_prob),

        "recent_form": recent.to_dict(orient="records"),
    }


def expected_games_bo2(win_prob: float) -> float:
    """
    Expected games across the first 2 maps of a Bo3.
    (Some books offer M1-2 = maps 1+2 regardless of series score.)
    Since these are always played: expected = 2.0
    But if the book means 'total across all maps if series ends in 2':
    we return 2.0 exactly since both maps are always played in M1-2 props.
    """
    return 2.0


@app.get("/refresh")
def refresh():
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
