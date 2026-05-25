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
    from data_ingestion      import load_raw
    from feature_engineering import build_features, get_feature_columns
    from model               import train_all, load_model

    years = [int(y) for y in os.getenv("YEARS", "2023,2024").split(",")]
    log.info(f"Loading data for years: {years}")

    raw   = load_raw(years=years)
    major = raw  # use all players/leagues
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

def _build_series_result(pg: dict, exp_games: float, win_prob: float, fmt_mult: str = "mult_m13") -> dict:
    """Scale per-game predictions to a series total with format-specific ML adjustment."""
    result = {"expected_games": round(exp_games, 2), "win_prob": round(win_prob, 3)}
    for stat in ["kills", "deaths", "assists"]:
        s = pg[stat]
        # Use format-specific multiplier if available, else 1.0
        extra_mult = s.get(fmt_mult, 1.0) / s.get("mult_m1_applied", s.get(fmt_mult, 1.0))
        # Simpler: just use per_game (already has M1 mult) and apply ratio
        series_mult = s.get(fmt_mult, 1.0)
        base_pg = s["per_game"]
        result[stat] = {
            "per_game":     base_pg,
            "series_total": round(base_pg * exp_games, 1),
            "series_low":   round(s["low"]  * exp_games, 1),
            "series_high":  round(s["high"] * exp_games, 1),
            "mae":          round(s["mae"]  * exp_games, 2),
        }
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


@app.get("/teams")
def list_teams(league: str = Query(None), q: str = Query(None)):
    """List all teams, optionally filtered by league or name fragment."""
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
    player:    str = Query(...),
    side:      str = Query("Blue"),
    moneyline: int = Query(None, description="Team ML e.g. -297"),
    opp_ml:    int = Query(None, description="Opponent ML e.g. +297"),
    opponent:  str = Query(None, description="Opposing team name e.g. PCFIC"),
):
    """
    Predict per-game AND series (M1-2 and M1-3) K/D/A for a player.
    Optionally pass opponent team name to adjust for their defensive strength.
    """
    _check_ready()
    from model                  import load_model
    from series_predictor       import vig_adjusted_probs, expected_games
    from moneyline_adjustments  import _interpolate_adjustment

    player_df, X = _get_player_row(player)
    position = player_df.iloc[0]["position"]

    if "is_blue_side" in X.columns:
        X["is_blue_side"] = 1 if side.lower() == "blue" else 0

    # Win probability
    if moneyline is not None and opp_ml is not None:
        win_prob, _ = vig_adjusted_probs(moneyline, opp_ml)
    else:
        win_prob = 0.5

    # ── Opponent defensive adjustment ──────────────────────────────────────────
    # Override opponent defensive strength features with the actual upcoming opponent
    opp_adj = {}
    if opponent:
        df = STATE["df"]
        opp_mask = df["teamname"].str.lower() == opponent.strip().lower()
        if not opp_mask.any():
            # Try partial match
            opp_mask = df["teamname"].str.lower().str.contains(opponent.strip().lower(), na=False)

        if opp_mask.any():
            opp_df = df[opp_mask].sort_values("date", ascending=False)
            # Get how many kills this opponent gives up at the player's position
            opp_pos = opp_df[opp_df["position"] == position]
            if len(opp_pos) >= 3:
                # kills_allowed = kills scored BY opponent's laner (what they give up to enemy)
                # We use opp_team_kills_allowed_roll5 if available, else compute from raw
                col = "opp_team_kills_allowed_roll5"
                if col in opp_pos.columns and opp_pos[col].notna().any():
                    opp_kills_allowed = float(opp_pos[col].dropna().iloc[0])
                else:
                    # Fallback: compute from raw kills at that position
                    opp_kills_allowed = float(opp_pos["kills"].tail(10).mean())

                opp_deaths_allowed = float(opp_pos["deaths"].tail(10).mean()) if len(opp_pos) >= 3 else None

                # Adjustment ratio: how does this opponent compare to average?
                avg_kills_at_pos = float(df[df["position"] == position]["kills"].mean())
                avg_deaths_at_pos = float(df[df["position"] == position]["deaths"].mean())

                if avg_kills_at_pos > 0:
                    opp_adj["kills_ratio"]   = opp_kills_allowed / avg_kills_at_pos
                if avg_deaths_at_pos and avg_deaths_at_pos > 0:
                    opp_adj["deaths_ratio"]  = opp_deaths_allowed / avg_deaths_at_pos if opp_deaths_allowed else 1.0

    # Per-game base predictions
    pg = {}
    for stat in ["kills", "deaths", "assists"]:
        model, _, metrics = load_model(stat)
        pg[stat] = _predict_stat(model, X, metrics["mae"])

    # Apply opponent defensive adjustment
    if opp_adj:
        kills_ratio  = min(max(opp_adj.get("kills_ratio",  1.0), 0.5), 2.0)  # clamp 0.5–2.0
        deaths_ratio = min(max(opp_adj.get("deaths_ratio", 1.0), 0.5), 2.0)
        assists_ratio = (kills_ratio + 1.0) / 2.0  # assists partially correlated with kills

        for stat, ratio in [("kills", kills_ratio), ("deaths", deaths_ratio), ("assists", assists_ratio)]:
            pg[stat]["per_game"] = round(pg[stat]["per_game"] * ratio, 2)
            pg[stat]["low"]      = round(pg[stat]["low"]      * ratio, 2)
            pg[stat]["high"]     = round(pg[stat]["high"]     * ratio, 2)
            pg[stat]["opp_adj_ratio"] = round(ratio, 3)

    # Apply format-specific moneyline adjustment multipliers
    if moneyline is not None and opp_ml is not None:
        for stat in ["kills", "deaths", "assists"]:
            mult_m1  = _interpolate_adjustment(win_prob, stat, "m1")
            mult_m12 = _interpolate_adjustment(win_prob, stat, "m12")
            mult_m13 = _interpolate_adjustment(win_prob, stat, "m13")
            pg[stat]["per_game"]   = round(pg[stat]["per_game"] * mult_m1,  2)
            pg[stat]["low"]        = round(pg[stat]["low"]       * mult_m1,  2)
            pg[stat]["high"]       = round(pg[stat]["high"]      * mult_m1,  2)
            pg[stat]["mult_m12"]   = mult_m12
            pg[stat]["mult_m13"]   = mult_m13

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
