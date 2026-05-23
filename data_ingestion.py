"""
data_ingestion.py
-----------------
Downloads Oracle's Elixir match data CSVs directly from their public S3 bucket.
Caches locally so repeat runs are instant.
"""

import os
import requests
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
S3_BASE = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com"
CACHE_DIR = Path(__file__).parent / "data"

# Major leagues to keep (filter out minor/academy leagues for cleaner data)
MAJOR_LEAGUES = {
    "LCS", "LEC", "LCK", "LPL",
    "LCS Challengers", "LEC Proving Grounds",
    "LCK CL", "LDL",
    "Worlds", "MSI",
}

# Columns we actually need (keeps memory low)
KEEP_COLS = [
    "gameid", "date", "league", "split", "patch",
    "side", "position", "playername", "teamname",
    "champion", "ban1", "ban2", "ban3", "ban4", "ban5",
    "gamelength",
    "kills", "deaths", "assists",
    "damagetochampions", "damageshare",
    "wardsplaced", "wardskilled", "controlwardsbought",
    "totalgold", "earnedgold", "goldspent",
    "minionkills", "monsterkills",
    "vspm", "dpm", "cspm", "gpm",
    "killsat10", "assistsat10", "deathsat10",
    "killsat15", "assistsat15", "deathsat15",
    "goldat10", "goldat15",
    "xpat10", "xpat15",
    "csat10", "csat15",
    "golddiffat10", "golddiffat15",
    "xpdiffat10", "xpdiffat15",
    "csdiffat10", "csdiffat15",
    "firstblood", "firstbloodkill", "firstbloodassist", "firstbloodvictim",
    "result",
]


def _csv_url(year: int) -> str:
    return f"{S3_BASE}/{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def _cache_path(year: int) -> Path:
    return CACHE_DIR / f"oe_{year}.csv"


def download_year(year: int, force: bool = False) -> pd.DataFrame:
    """Download (or load from cache) one year of OE data."""
    path = _cache_path(year)

    if path.exists() and not force:
        print(f"  [cache] Loading {year} from {path}")
        return pd.read_csv(path, low_memory=False)

    url = _csv_url(year)
    print(f"  [download] Fetching {year} data from S3 ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    print(f"  [download] Saved to {path}")
    return pd.read_csv(path, low_memory=False)


def load_raw(years: list[int] = None, force: bool = False) -> pd.DataFrame:
    """
    Load multiple years of OE data, concatenated into one DataFrame.
    Keeps only player-level rows (drops team summary rows).
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]

    frames = []
    for year in years:
        try:
            df = download_year(year, force=force)
            frames.append(df)
            print(f"  [ok] {year}: {len(df):,} rows")
        except requests.HTTPError as e:
            print(f"  [warn] {year} not available: {e}")

    raw = pd.concat(frames, ignore_index=True)

    # Keep only player rows (position != 'team')
    raw = raw[raw["position"].notna() & (raw["position"] != "team")].copy()

    # Filter to columns that exist in this dataset
    cols = [c for c in KEEP_COLS if c in raw.columns]
    raw = raw[cols]

    # Parse date
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")

    print(f"\n[load_raw] Total player-game rows: {len(raw):,}")
    print(f"[load_raw] Date range: {raw['date'].min().date()} → {raw['date'].max().date()}")
    return raw


def filter_major_leagues(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows from major leagues."""
    mask = df["league"].isin(MAJOR_LEAGUES)
    filtered = df[mask].copy()
    print(f"[filter] Major leagues only: {len(filtered):,} rows "
          f"({len(filtered)/len(df)*100:.1f}% of total)")
    return filtered


if __name__ == "__main__":
    print("=== Downloading Oracle's Elixir Data ===\n")
    raw = load_raw(years=[2022, 2023, 2024])
    major = filter_major_leagues(raw)
    print("\nLeague breakdown:")
    print(major["league"].value_counts().to_string())
    print("\nPositions:")
    print(major["position"].value_counts().to_string())
