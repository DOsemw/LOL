"""
data_ingestion.py
-----------------
Loads Oracle's Elixir match data.

Oracle's Elixir S3 bucket went private in 2024/2025.
Data must be downloaded manually and either:
  A) Uploaded to your Railway volume / repo as data/oe_YYYY.csv
  B) Hosted somewhere publicly accessible (Google Drive, Dropbox, etc.)

HOW TO GET THE DATA:
  1. Go to https://oracleselixir.com/tools/downloads
  2. Download the yearly CSV files (2023, 2024, 2025)
  3. Rename them: oe_2023.csv, oe_2024.csv, oe_2025.csv
  4. Upload them to your GitHub repo inside a /data folder
     OR set PUBLIC_DATA_URL_YYYY env vars pointing to direct download links

MAJOR LEAGUES:
"""

import os
import io
import logging
import requests
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

MAJOR_LEAGUES = {
    "LCS", "LEC", "LCK", "LPL",
    "LCS Challengers", "LCK CL", "LDL",
    "Worlds", "MSI",
}

KEEP_COLS = [
    "gameid", "date", "league", "split", "patch",
    "side", "position", "playername", "teamname",
    "champion",
    "gamelength",
    "kills", "deaths", "assists",
    "damagetochampions", "damageshare",
    "wardsplaced", "wardskilled", "controlwardsbought",
    "totalgold", "earnedgold",
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


def _load_year_from_env(year: int) -> pd.DataFrame | None:
    """Try loading from a public URL set in environment variables."""
    url = os.getenv(f"DATA_URL_{year}")
    if not url:
        return None
    log.info(f"  [env] Downloading {year} from {url[:60]}...")
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text), low_memory=False)
    except Exception as e:
        log.warning(f"  [env] Failed to load {year} from env URL: {e}")
        return None


def _load_year_from_disk(year: int) -> pd.DataFrame | None:
    """Try loading from local data/ directory."""
    path = DATA_DIR / f"oe_{year}.csv"
    if not path.exists():
        # Also try without prefix
        path2 = DATA_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        if path2.exists():
            path = path2
        else:
            return None
    log.info(f"  [disk] Loading {year} from {path}")
    return pd.read_csv(path, low_memory=False)


def load_year(year: int) -> pd.DataFrame | None:
    """Load one year of OE data — tries env URL first, then disk."""
    df = _load_year_from_env(year)
    if df is not None:
        return df
    df = _load_year_from_disk(year)
    if df is not None:
        return df
    log.warning(f"  [warn] No data found for {year}. "
                f"Upload data/oe_{year}.csv or set DATA_URL_{year} env var.")
    return None


def load_raw(years: list[int] = None) -> pd.DataFrame:
    years = years or [int(y) for y in os.getenv("YEARS", "2026").split(",")]
    DATA_DIR.mkdir(exist_ok=True)

    frames = []
    for year in years:
        df = load_year(year)
        if df is not None:
            frames.append(df)
            log.info(f"  [ok] {year}: {len(df):,} rows")

    if not frames:
        raise RuntimeError(
            "No data loaded! You need to upload OE CSV files.\n"
            "See data_ingestion.py for instructions.\n"
            "Quick fix: set DATA_URL_2024 and DATA_URL_2025 in Railway environment variables "
            "pointing to direct-download links of the OE CSV files."
        )

    raw = pd.concat(frames, ignore_index=True)

    # Player rows only
    raw = raw[raw["position"].notna() & (raw["position"] != "team")].copy()

    # Keep only cols that exist
    cols = [c for c in KEEP_COLS if c in raw.columns]
    raw = raw[cols]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")

    log.info(f"[load_raw] {len(raw):,} player-game rows | "
             f"{raw['date'].min().date()} → {raw['date'].max().date()}")
    return raw


def filter_major_leagues(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["league"].isin(MAJOR_LEAGUES)].copy()
    log.info(f"[filter] {len(filtered):,} rows from major leagues")
    return filtered
