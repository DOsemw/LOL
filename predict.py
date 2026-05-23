"""
predict.py
----------
Inference layer: given a player name + upcoming match context,
return expected kills / deaths / assists with confidence intervals.

Usage:
    python predict.py --player "Faker" --champion "Azir" --opponent "Chovy" --league "LCK"
"""

import argparse
import numpy as np
import pandas as pd
from model import load_model, TARGETS
from feature_engineering import build_features, get_feature_columns
from data_ingestion import load_raw, filter_major_leagues


# ── Confidence intervals via quantile estimation ──────────────────────────────

def bootstrap_ci(model, X_row: pd.DataFrame, n_boot: int = 200, ci: float = 0.9) -> tuple:
    """
    Estimate prediction uncertainty by perturbing features slightly.
    Returns (low, mid, high) at the given confidence interval.
    Quick proxy — proper uncertainty requires quantile regression or conformal prediction.
    """
    base_pred = float(model.predict(X_row)[0])
    noise_scale = 0.05  # 5% feature noise

    preds = []
    for _ in range(n_boot):
        noisy = X_row.copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noisy[col] += np.random.normal(0, abs(noisy[col].values[0]) * noise_scale + 0.01)
        preds.append(float(model.predict(noisy)[0]))

    alpha = (1 - ci) / 2
    low  = max(0, np.quantile(preds, alpha))
    high = np.quantile(preds, 1 - alpha)
    return round(low, 2), round(base_pred, 2), round(high, 2)


# ── Lookup player's recent game rows ─────────────────────────────────────────

def get_player_context(df: pd.DataFrame, player_name: str, n_recent: int = 5) -> pd.DataFrame:
    """
    Pull the most recent N games for a player to display form.
    df must already have features built.
    """
    player_df = df[df["playername"].str.lower() == player_name.lower()].copy()
    player_df = player_df.sort_values("date", ascending=False)
    return player_df.head(n_recent)[["date", "champion", "kills", "deaths", "assists", "result"]]


def get_player_feature_row(df: pd.DataFrame, player_name: str, feature_cols: list[str]) -> pd.DataFrame:
    """
    Get the most recent feature row for a player (represents their current form state).
    This is what gets passed to the model for inference.
    """
    player_df = df[df["playername"].str.lower() == player_name.lower()].copy()
    player_df = player_df.sort_values("date", ascending=False)
    if len(player_df) == 0:
        raise ValueError(f"Player '{player_name}' not found in dataset.")

    row = player_df.iloc[[0]][feature_cols].copy()
    return row


# ── Main prediction function ──────────────────────────────────────────────────

def predict_player(
    df: pd.DataFrame,
    feature_cols: list[str],
    player_name: str,
    override_features: dict = None,
    ci: float = 0.9,
    verbose: bool = True,
) -> dict:
    """
    Predict expected K/D/A for a player in their next game.

    Args:
        df:               Feature-engineered dataframe
        feature_cols:     List of feature column names
        player_name:      Player name (must match OE data)
        override_features: Dict of feature overrides (e.g., {"is_blue_side": 1})
        ci:               Confidence interval width (default 90%)
        verbose:          Print formatted output

    Returns:
        Dict with predictions for kills, deaths, assists
    """
    # Get feature row from most recent game
    X = get_player_feature_row(df, player_name, feature_cols)

    # Apply any manual overrides (e.g., different champion, side)
    if override_features:
        for k, v in override_features.items():
            if k in X.columns:
                X[k] = v

    results = {}
    for target in TARGETS:
        model, _, metrics = load_model(target)
        low, mid, high = bootstrap_ci(model, X, ci=ci)
        results[target] = {"low": low, "mid": mid, "high": high, "mae": metrics["mae"]}

    if verbose:
        _print_prediction(player_name, results, ci, df)

    return results


def _print_prediction(player_name: str, results: dict, ci: float, df: pd.DataFrame):
    ci_pct = int(ci * 100)
    print(f"\n{'='*55}")
    print(f"  Player Props Prediction: {player_name}")
    print(f"{'='*55}")
    print(f"  {'Stat':<10} {'Expected':>10}  {'±MAE':>8}  {f'{ci_pct}% CI':>16}")
    print(f"  {'-'*50}")
    for stat, r in results.items():
        mae = r["mae"]
        ci_str = f"[{r['low']:.1f} – {r['high']:.1f}]"
        print(f"  {stat.upper():<10} {r['mid']:>10.2f}  {f'±{mae:.2f}':>8}  {ci_str:>16}")
    print(f"{'='*55}\n")

    # Recent form
    recent = df[df["playername"].str.lower() == player_name.lower()].sort_values(
        "date", ascending=False
    ).head(5)[["date", "champion", "kills", "deaths", "assists"]]
    if len(recent) > 0:
        print("  Recent form (last 5 games):")
        print("  " + recent.to_string(index=False).replace("\n", "\n  "))
        print()


# ── Batch prediction for a full lineup ───────────────────────────────────────

def predict_lineup(
    df: pd.DataFrame,
    feature_cols: list[str],
    players: list[str],
) -> pd.DataFrame:
    """Predict K/D/A for a list of players (e.g., a full team)."""
    rows = []
    for player in players:
        try:
            r = predict_player(df, feature_cols, player, verbose=False)
            rows.append({
                "player": player,
                "kills_exp":   r["kills"]["mid"],
                "deaths_exp":  r["deaths"]["mid"],
                "assists_exp": r["assists"]["mid"],
                "kills_ci":    f"[{r['kills']['low']:.1f}–{r['kills']['high']:.1f}]",
                "deaths_ci":   f"[{r['deaths']['low']:.1f}–{r['deaths']['high']:.1f}]",
                "assists_ci":  f"[{r['assists']['low']:.1f}–{r['assists']['high']:.1f}]",
            })
        except ValueError as e:
            print(f"  [warn] {e}")
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict LoL player K/D/A props")
    parser.add_argument("--player", required=True, help="Player name (OE format)")
    parser.add_argument("--league", default=None, help="Filter to specific league")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--side", default=None, choices=["Blue", "Red"])
    args = parser.parse_args()

    print(f"\nLoading data ({args.years}) ...")
    raw   = load_raw(years=args.years)
    major = filter_major_leagues(raw)

    if args.league:
        major = major[major["league"] == args.league]

    print("Building features ...")
    feat  = build_features(major, verbose=False)
    fcols = get_feature_columns(feat)

    overrides = {}
    if args.side:
        overrides["is_blue_side"] = 1 if args.side == "Blue" else 0

    predict_player(feat, fcols, args.player, override_features=overrides or None)
