"""
main.py v2 — LoL Props API
Clean rebuild. Two-model architecture:
  Model 1: team kills
  Model 2: kill share per position
  Final:   team_kills × kill_share = player kills
"""

import os, sys, logging
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATE = {
    "df":               None,
    "tk_model":         None,
    "tk_features":      None,
    "ks_models":        None,
    "ks_features":      None,
    "pace_multipliers": {},
    "ready":            False,
    "n_players":        0,
    "date_range":       "",
}


def load_everything():
    from data_ingestion      import load_raw
    from feature_engineering import (build_features,
                                      get_team_kill_features,
                                      get_kill_share_features)
    from model_team_kills    import train as train_tk, load as load_tk
    from model_kill_share    import train_all as train_ks, load as load_ks, POSITIONS

    years = [int(y) for y in os.getenv("YEARS", "2026").split(",")]
    log.info(f"Loading data: {years}")

    raw  = load_raw(years=years)
    feat = build_features(raw, verbose=False)

    tk_features = get_team_kill_features(feat)
    ks_features = get_kill_share_features(feat)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Train or load team kills model
    tk_path = model_dir / "team_kills_model.pkl"
    if not tk_path.exists():
        log.info("Training team kills model...")
        train_tk(feat, tk_features)
    tk_model, tk_features, tk_mae = load_tk()
    log.info(f"Team kills model ready (MAE={tk_mae:.2f})")

    # Train or load kill share models
    ks_paths_exist = all(
        (model_dir / f"kill_share_{p}.pkl").exists()
        for p in POSITIONS
    )
    if not ks_paths_exist:
        log.info("Training kill share models...")
        train_ks(feat, ks_features)

    ks_models = {}
    for pos in POSITIONS:
        try:
            ks_models[pos] = load_ks(pos)
            log.info(f"  Kill share {pos}: MAE={ks_models[pos][2]:.4f}, avg={ks_models[pos][3]:.3f}")
        except FileNotFoundError:
            log.warning(f"  No kill share model for {pos}")

    from series import compute_league_pace
    pace_multipliers = compute_league_pace(feat)
    log.info(f"League pace multipliers: {pace_multipliers}")

    STATE.update({
        "df":               feat,
        "tk_model":         tk_model,
        "tk_features":      tk_features,
        "ks_models":        ks_models,
        "ks_features":      ks_features,
        "pace_multipliers": pace_multipliers,
        "ready":            True,
        "n_players":        feat["playername"].nunique(),
        "date_range":       f"{feat['date'].min().date()} → {feat['date'].max().date()}",
    })
    log.info(f"Ready — {STATE['n_players']} players, {STATE['date_range']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_everything()
    yield

app = FastAPI(title="LoL Props API v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def _check_ready():
    if not STATE["ready"]:
        raise HTTPException(503, detail="Model loading — try again shortly.")


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":     "ok" if STATE["ready"] else "loading",
        "version":    "2.0",
        "players":    STATE["n_players"],
        "date_range": STATE["date_range"],
    }


@app.get("/search")
def search(q: str = Query(...)):
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
def predict(
    player:    str   = Query(...),
    side:      str   = Query("Blue"),
    moneyline: int   = Query(None),
    opp_ml:    int   = Query(None),
    opponent:  str   = Query(None),
):
    _check_ready()
    from predict          import predict_player
    from series           import scale_to_series
    from model_kill_share import POSITIONS

    df = STATE["df"]

    # Win probability — Step A from blueprint
    if moneyline is not None and opp_ml is not None:
        from series import devig
        win_prob = devig(moneyline, opp_ml)
    else:
        win_prob = 0.5

    try:
        result = predict_player(
            df                = df,
            tk_model          = STATE["tk_model"],
            tk_features       = STATE["tk_features"],
            ks_models         = STATE["ks_models"],
            ks_features       = STATE["ks_features"],
            pace_multipliers  = STATE["pace_multipliers"],
            player            = player,
            win_prob          = win_prob,
            opponent          = opponent,
            side              = side,
        )
        result["moneyline"] = moneyline
        result["opponent"]  = opponent
        return result

    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        log.error(f"Prediction error for {player}: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.get("/refresh")
def refresh():
    STATE["ready"] = False
    # Delete saved models to force retrain
    for p in Path("models").glob("*.pkl"):
        p.unlink()
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
