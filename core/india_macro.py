"""
core/india_macro.py
Fetches India macroeconomic data from FRED API (live, no extra key needed
beyond the existing FRED_API_KEY used for US macro).

Series used:
  - INDCPIALLMINMEI : India CPI All Items (OECD, monthly index, 2015=100)
                      → we compute YoY % change in code
  - INTDSRINM193N   : India Discount Rate / Repo Rate proxy (IMF, monthly %)

Falls back to local CSV if FRED key is absent or the request fails.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from core.data_fred import fetch_fred_series, _get_fred_api_key

logger = logging.getLogger(__name__)

# FRED series IDs for India
INDIA_CPI_SERIES  = "INDCPIALLMINMEI"   # Index (2015=100), monthly
INDIA_RATE_SERIES = "INTDSRINM193N"     # Discount/repo rate %, monthly

FALLBACK_CSV = Path(__file__).parent.parent / "data" / "india_macro.csv"


# ─────────────────────────────────────────────────────────────────
# FRED-based fetch (primary)
# ─────────────────────────────────────────────────────────────────

def _fetch_india_cpi_yoy() -> pd.Series:
    """
    Fetch India CPI index from FRED and compute YoY % change.
    Returns pd.Series with DatetimeIndex and name 'cpi_yoy', or empty Series.
    """
    df = fetch_fred_series(INDIA_CPI_SERIES)
    if df.empty:
        return pd.Series(dtype=float, name="cpi_yoy")
    cpi = df["value"].resample("MS").last()
    yoy = cpi.pct_change(periods=12) * 100
    yoy.name = "cpi_yoy"
    return yoy.dropna()


def _fetch_india_repo_rate() -> pd.Series:
    """
    Fetch India discount/repo rate from FRED.
    Returns pd.Series with DatetimeIndex and name 'repo_rate', or empty Series.
    """
    df = fetch_fred_series(INDIA_RATE_SERIES)
    if df.empty:
        return pd.Series(dtype=float, name="repo_rate")
    rate = df["value"].resample("MS").last()
    rate.name = "repo_rate"
    return rate.dropna()


# ─────────────────────────────────────────────────────────────────
# CSV fallback (secondary)
# ─────────────────────────────────────────────────────────────────

def _load_csv_fallback() -> pd.DataFrame:
    """Load India macro from local CSV (used when FRED key is absent)."""
    try:
        if not FALLBACK_CSV.exists():
            return pd.DataFrame()
        df = pd.read_csv(FALLBACK_CSV, parse_dates=["date"])
        df = df.sort_values("date").set_index("date")
        df["cpi_yoy"]   = pd.to_numeric(df.get("cpi_yoy"),   errors="coerce")
        df["repo_rate"] = pd.to_numeric(df.get("repo_rate"), errors="coerce")
        return df.dropna(how="all")[["cpi_yoy", "repo_rate"]]
    except Exception as e:
        logger.warning(f"CSV fallback failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_india_macro() -> pd.DataFrame:
    """
    Load India CPI YoY and Repo Rate.

    Priority:
      1. FRED API (live, updates monthly)  — requires FRED_API_KEY
      2. Local CSV fallback                — bundled static data

    Returns:
        DataFrame with DatetimeIndex and columns [cpi_yoy, repo_rate].
        Empty DataFrame if both sources fail.
    """
    has_key = bool(_get_fred_api_key())

    if has_key:
        cpi_yoy   = _fetch_india_cpi_yoy()
        repo_rate = _fetch_india_repo_rate()

        if not cpi_yoy.empty or not repo_rate.empty:
            df = pd.concat([cpi_yoy, repo_rate], axis=1).sort_index()
            if "repo_rate" in df.columns:
                df["repo_rate"] = df["repo_rate"].ffill()
            df = df.dropna(how="all")
            if not df.empty:
                latest_cpi  = cpi_yoy.index[-1].date()  if not cpi_yoy.empty  else "N/A"
                latest_rate = repo_rate.index[-1].date() if not repo_rate.empty else "N/A"
                logger.info(f"India macro from FRED: CPI→{latest_cpi}, Rate→{latest_rate}")
                return df

        st.warning("⚠️ India macro FRED fetch returned empty — falling back to bundled CSV.")

    # No key or FRED failed → CSV
    df = _load_csv_fallback()
    if not df.empty:
        if not has_key:
            st.info(
                "ℹ️ India macro is using **bundled CSV data** (static, last updated Feb 2025). "
                "Add a FRED_API_KEY to get live monthly updates automatically."
            )
        return df

    st.warning("⚠️ India macro data unavailable from both FRED and CSV fallback.")
    return pd.DataFrame()


def load_india_macro_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Backwards-compatible alias → delegates to load_india_macro()."""
    return load_india_macro()
