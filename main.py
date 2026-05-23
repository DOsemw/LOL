"""
main.py
-------
FastAPI server for LoL player props predictions.
Deploy this to Railway. Google Sheets calls it via Apps Script.

Endpoints:
  GET  /                        health check
  GET  /predict?player=Faker    predict K/D/A for a player
  GET  /players                 list all known players
  GET  /refresh                 re-download latest OE data & retrain (slow)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── App state (loaded once at startup) ───────────────────────────────────────
STATE = {
    "df":         None,   # feature-engineered dataframe
    "feature_cols": None,
    "ready":      False,
    "n_players":  0,
    "date_range": "",
}


def load_everything():
    """Download data, build features, train/load models. Called once at startup."""
    log.info("Loading pipeline ...")
    from data_ingestion      import load_raw, filter_major_leagues
    from feature_engineering import build_features, get_feature_columns
    from model               import train_all, load_model
    from pathlib             import Path

    years = [int(y) for y in os.getenv("YEARS", "2023,2024").split(",")]
    log.info(f"Using years: {years}")

    raw   = load_raw(years=years)
    major = filter_major_leagues(raw)
    feat  = build_features(major, verbose=False)
    fcols = get_feature_columns(feat)

    # Train if models don't exist yet
    model_dir = Path("models")
    models_exist = all((model_dir / f"{t}_model.pkl").exists() for t in ["kills","deaths","assists"])
    if not models_exist:
        log.info("No saved models found — training now (this takes a few minutes) ...")
        train_all(feat, fcols)
    else:
        log.info("Loaded existing models from disk.")

    STATE["df"]           = feat
    STATE["feature_cols"] = fcols
    STATE["ready"]        = True
    STATE["n_players"]    = feat["playername"].nunique()
    STATE["date_range"]   = f"{feat['date'].min().date()} → {feat['date'].max().date()}"
    log.info(f"Ready. {STATE['n_players']} players, {STATE['date_range']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_everything()
    yield

app = FastAPI(title="LoL Props API", lifespan=lifespan)

# Allow requests from Google Apps Script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────────

class StatPrediction(BaseModel):
    expected: float
    low:      float
    high:     float
    mae:      float

class PlayerPrediction(BaseModel):
    player:  str
    kills:   StatPrediction
    deaths:  StatPrediction
    assists: StatPrediction
    games_in_sample: int
    recent_form: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_ready():
    if not STATE["ready"]:
        raise HTTPException(503, "Model not ready yet — check back in a minute.")


def _predict_stat(model, X: pd.DataFrame, ci: float = 0.9) -> StatPrediction:
    import numpy as np
    base = float(model.predict(X)[0])
    base = max(0.0, base)

    # Light bootstrap for CI
    preds = []
    for _ in range(150):
        noisy = X.copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noisy[col] += np.random.normal(0, abs(float(noisy[col].iloc[0])) * 0.05 + 0.01)
        preds.append(max(0.0, float(model.predict(noisy)[0])))

    alpha = (1 - ci) / 2
    return StatPrediction(
        expected = round(base, 2),
        low      = round(max(0, float(np.quantile(preds, alpha))), 2),
        high     = round(float(np.quantile(preds, 1 - alpha)), 2),
        mae      = 0.0,  # filled in below from saved metrics
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":     "ok" if STATE["ready"] else "loading",
        "players":    STATE["n_players"],
        "date_range": STATE["date_range"],
    }


@app.get("/predict", response_model=PlayerPrediction)
def predict(
    player: str = Query(..., description="Player name, e.g. Faker"),
    side:   str = Query("Blue", description="Blue or Red"),
):
    _check_ready()
    from model import load_model

    df    = STATE["df"]
    fcols = STATE["feature_cols"]

    # Find player
    mask = df["playername"].str.lower() == player.lower()
    if not mask.any():
        # Fuzzy suggestion
        all_players = df["playername"].unique()
        suggestions = [p for p in all_players if player.lower() in p.lower()][:5]
        raise HTTPException(
            404,
            detail=f"Player '{player}' not found. Did you mean: {suggestions}"
        )

    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][fcols].copy()

    # Side override
    if "is_blue_side" in X.columns:
        X["is_blue_side"] = 1 if side.lower() == "blue" else 0

    results = {}
    for target in ["kills", "deaths", "assists"]:
        model, _, metrics = load_model(target)
        stat = _predict_stat(model, X)
        stat.mae = metrics["mae"]
        results[target] = stat

    # Recent form (last 5 games)
    recent = player_df.head(5)[["date", "champion", "kills", "deaths", "assists"]].copy()
    recent["date"] = recent["date"].astype(str)
    recent_list = recent.to_dict(orient="records")

    return PlayerPrediction(
        player          = player_df.iloc[0]["playername"],
        kills           = results["kills"],
        deaths          = results["deaths"],
        assists         = results["assists"],
        games_in_sample = int(mask.sum()),
        recent_form     = recent_list,
    )


@app.get("/players")
def list_players(
    league: str = Query(None, description="Filter by league, e.g. LCK"),
    q:      str = Query(None, description="Search by name fragment"),
):
    _check_ready()
    df = STATE["df"]

    if league:
        df = df[df["league"].str.upper() == league.upper()]
    if q:
        df = df[df["playername"].str.lower().str.contains(q.lower())]

    players = (
        df.groupby("playername")
          .agg(league=("league", "last"), games=("gameid", "nunique"))
          .reset_index()
          .sort_values("games", ascending=False)
    )
    return players.to_dict(orient="records")


@app.get("/predict/series")
def predict_series_endpoint(
    player:      str = Query(...,    description="Player name, e.g. Faker"),
    format:      str = Query("Bo3", description="Bo1, Bo3, or Bo5"),
    moneyline:   int = Query(None,  description="Player's team moneyline, e.g. -297"),
    opp_ml:      int = Query(None,  description="Opponent moneyline, e.g. +297"),
    win_prob:    float = Query(0.5, description="Win prob per game (used if no moneyline)"),
    side:        str = Query("Blue", description="Blue or Red"),
):
    _check_ready()
    from model            import load_model
    from series_predictor import predict_series

    df    = STATE["df"]
    fcols = STATE["feature_cols"]

    mask = df["playername"].str.lower() == player.lower()
    if not mask.any():
        raise HTTPException(404, detail=f"Player '{player}' not found.")

    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][fcols].copy()
    if "is_blue_side" in X.columns:
        X["is_blue_side"] = 1 if side.lower() == "blue" else 0

    # Build per-game predictions
    pg_preds = {}
    for target in ["kills", "deaths", "assists"]:
        model, _, metrics = load_model(target)
        from main import _predict_stat
        stat = _predict_stat(model, X)
        stat.mae = metrics["mae"]
        pg_preds[target] = {"mid": stat.expected, "low": stat.low, "high": stat.high, "mae": stat.mae}

    series = predict_series(
        pg_preds,
        series_format  = format,
        team_win_prob  = win_prob,
        moneyline      = moneyline,
        opp_moneyline  = opp_ml,
    )
    series["player"] = player_df.iloc[0]["playername"]
    return series


@app.get("/refresh")
def refresh():
    """Re-download latest OE data and retrain. Takes several minutes."""
    _check_ready()
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
