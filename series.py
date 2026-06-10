"""
series.py v2
------------
Series math for ParlayPlay LoL props.

KEY FACT: ParlayPlay M1-3 = maps 1+2+3 of a Bo5.
In a Bo5, minimum maps played = 3 (a 3-0 sweep).
Therefore maps 1, 2, AND 3 are ALWAYS played.
M1-3 = flat sum of 3 maps. No probability discount needed.

M1-2 = maps 1+2, also always played in a Bo5.
M1   = single map baseline.

The moneyline still matters for:
  - Win/loss kill share blending (favoured teams have different kill share patterns)
  - Game pace (heavy favourites win faster = fewer kills per map)
"""


def american_to_ip(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return (-odds) / (-odds + 100)
    else:
        return 100 / (odds + 100)


def devig(team_ml: int, opp_ml: int) -> float:
    """
    Strip bookmaker margin to get true win probability p.
    Step A from blueprint.
    """
    ip_team = american_to_ip(team_ml)
    ip_opp  = american_to_ip(opp_ml)
    return ip_team / (ip_team + ip_opp)


def game_pace_factor(win_prob: float) -> float:
    """
    Heavy favourites win faster → fewer kills per map.
    A -5000 team wins in ~25 min, average team wins in ~32 min.
    Scale kills down for heavy favourites, up for underdogs.
    
    At 50% win prob: factor = 1.0 (no adjustment)
    At 90% win prob: factor = 0.88 (faster games, fewer kills)
    At 10% win prob: factor = 1.08 (longer games, more kills)
    """
    # Linear interpolation: centred at 0.5
    deviation = win_prob - 0.5
    # Slope: -0.24 means at 100% win prob, factor = 0.88
    factor = 1.0 - (0.24 * deviation)
    return round(max(0.80, min(1.15, factor)), 3)


def scale_to_series(
    map_kills:   float,
    map_deaths:  float,
    map_assists: float,
    win_prob:    float = 0.5,
) -> dict:
    """
    Scale per-map predictions to M1, M1-2, M1-3.

    Bo5 context:
      M1   = 1 map  (always played)
      M1-2 = 2 maps (always played)
      M1-3 = 3 maps (always played — minimum in a Bo5)

    Apply game pace factor based on win probability:
    Heavy favourites win faster → fewer kills per map.
    """
    pace = game_pace_factor(win_prob)

    # Apply pace adjustment to per-map baseline
    adj_kills   = map_kills   * pace
    adj_deaths  = map_deaths  * pace
    adj_assists = map_assists * pace

    # Fearless draft decay for maps 2 and 3
    # In Bo5 fearless draft, players are forced off comfort picks
    # Map 2 = 96% of Map 1 baseline
    # Map 3 = 92% of Map 1 baseline
    m2_kills   = adj_kills   * 0.96
    m2_deaths  = adj_deaths  * 0.96
    m2_assists = adj_assists * 0.96

    m3_kills   = adj_kills   * 0.92
    m3_deaths  = adj_deaths  * 0.92
    m3_assists = adj_assists * 0.92

    # M1-2 = map1 + map2
    m12_kills   = adj_kills   + m2_kills
    m12_deaths  = adj_deaths  + m2_deaths
    m12_assists = adj_assists + m2_assists

    # M1-3 = map1 + map2 + map3 (all guaranteed in Bo5)
    m13_kills   = adj_kills   + m2_kills   + m3_kills
    m13_deaths  = adj_deaths  + m2_deaths  + m3_deaths
    m13_assists = adj_assists + m2_assists + m3_assists

    def fantasy(k, d, a):
        return round(k * 3 + a * 1.5 - d, 1)

    return {
        "pace_factor":   pace,
        "expected_maps": 3,  # always 3 in Bo5 M1-3 context

        "m1": {
            "kills":   round(adj_kills,   2),
            "deaths":  round(adj_deaths,  2),
            "assists": round(adj_assists, 2),
        },
        "m1_2": {
            "kills":   round(m12_kills,   1),
            "deaths":  round(m12_deaths,  1),
            "assists": round(m12_assists, 1),
            "fantasy": fantasy(m12_kills, m12_deaths, m12_assists),
        },
        "m1_3": {
            "kills":   round(m13_kills,   1),
            "deaths":  round(m13_deaths,  1),
            "assists": round(m13_assists, 1),
            "fantasy": fantasy(m13_kills, m13_deaths, m13_assists),
        },
    }


if __name__ == "__main__":
    # Demo
    p = devig(-333, 220)
    print(f"Win prob (-333/+220): {p:.3f} ({p*100:.1f}%)")
    print(f"Pace factor: {game_pace_factor(p):.3f}")
    print()

    # Show pace factor at various moneylines
    print(f"{'Win%':>6}  {'Pace':>8}  {'M1 kills (base 5)':>20}  {'M1-3 kills':>12}")
    print("-" * 55)
    for wp in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pace = game_pace_factor(wp)
        m1   = 5.0 * pace
        s    = scale_to_series(5.0, 2.0, 4.0, wp)
        print(f"{wp*100:>5.0f}%  {pace:>8.3f}  {m1:>20.2f}  {s['m1_3']['kills']:>12.1f}")


# ── League pace multiplier ────────────────────────────────────────────────────

def compute_league_pace(df) -> dict:
    """
    Compute average bloody rate (kills+deaths per minute) per league.
    Returns dict of {league: pace_multiplier} where multiplier = 
    league_pace / global_pace.
    
    > 1.0 = bloodier than average (scale predictions up)
    < 1.0 = slower than average (scale predictions down)
    """
    import pandas as pd
    import numpy as np

    if "bloody_rate" not in df.columns or "league" not in df.columns:
        return {}

    # One row per game (not per player)
    game_df = df.drop_duplicates("gameid")[["gameid", "league", "bloody_rate"]].copy()
    game_df = game_df[game_df["bloody_rate"] > 0]

    global_avg = float(game_df["bloody_rate"].mean())
    if global_avg <= 0:
        return {}

    league_avg = game_df.groupby("league")["bloody_rate"].mean()
    multipliers = (league_avg / global_avg).round(3).to_dict()

    # Clamp to prevent extreme adjustments
    multipliers = {k: float(np.clip(v, 0.75, 1.30)) for k, v in multipliers.items()}
    return multipliers


def apply_league_pace(kills: float, deaths: float, assists: float,
                      league: str, pace_multipliers: dict) -> tuple:
    """
    Apply league pace multiplier to predictions.
    Final Projection = Base × (league_pace / global_pace)
    """
    mult = pace_multipliers.get(league, 1.0)
    return (
        round(kills   * mult, 2),
        round(deaths  * mult, 2),
        round(assists * mult, 2),
        mult,
    )
