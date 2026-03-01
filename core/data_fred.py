"""
core/data_fred.py
Fetches macroeconomic time series from FRED API with retry logic.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs used across the app
FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "fed_funds": "FEDFUNDS",
    "t2y": "DGS2",
    "t10y": "DGS10",
    "breakeven_10y": "T10YIE",
    "unemployment": "UNRATE",
}


def _get_fred_api_key() -> Optional[str]:
    """Retrieve FRED API key from Streamlit secrets or environment variables."""
    # Try environment variable first (works locally without secrets.toml)
    env_key = os.environ.get("FRED_API_KEY")
    if env_key:
        return env_key

    # Try Streamlit secrets (works on Streamlit Cloud and with secrets.toml)
    try:
        key = st.secrets.get("FRED_API_KEY")
        return key if key else None
    except Exception:
        return None


def fetch_fred_series(
    series_id: str,
    start_date: Optional[str] = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch a single FRED series as a DataFrame with columns [date, value].

    Args:
        series_id: FRED series identifier (e.g., 'CPIAUCSL').
        start_date: ISO date string 'YYYY-MM-DD'; defaults to 6 years ago.
        retries: Number of retry attempts on failure.
        backoff: Exponential backoff multiplier.

    Returns:
        DataFrame with DatetimeIndex and a 'value' column, or empty DataFrame on error.
    """
    api_key = _get_fred_api_key()
    if not api_key:
        return pd.DataFrame()  # Warning shown once by load_all_fred_series()

    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(FRED_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            if not observations:
                return pd.DataFrame()

            df = pd.DataFrame(observations)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).set_index("date").sort_index()
            return df

        except requests.exceptions.RequestException as e:
            wait = backoff ** attempt
            logger.warning(f"FRED fetch attempt {attempt+1} failed for {series_id}: {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected error fetching FRED {series_id}: {e}")
            break

    st.warning(f"⚠️ Could not fetch FRED series: {series_id}")
    return pd.DataFrame()


def compute_yoy(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """
    Compute YoY % change for monthly series (shift by 12 periods).

    Args:
        df: DataFrame with DatetimeIndex and a numeric column.
        col: Column name to transform.

    Returns:
        DataFrame with additional 'yoy' column.
    """
    result = df.copy()
    result["yoy"] = result[col].pct_change(periods=12) * 100
    return result


def compute_spread(df_10y: pd.DataFrame, df_2y: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 10Y–2Y yield spread aligned on dates.

    Returns:
        DataFrame with DatetimeIndex and 'spread' column.
    """
    combined = pd.concat(
        [df_10y["value"].rename("t10y"), df_2y["value"].rename("t2y")], axis=1
    ).dropna()
    combined["spread"] = combined["t10y"] - combined["t2y"]
    return combined


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_all_fred_series() -> dict:
    """
    Load all required FRED series and compute derived metrics.
    Cached for 6 hours.

    Returns:
        Dictionary of DataFrames keyed by series name + derived keys.
    """
    result = {}

    # Check key once — warn once
    if not _get_fred_api_key():
        st.warning(
            "⚠️ **FRED API key not set** — Global Macro data (CPI, yields, Fed Funds) is unavailable. "
            "To enable it: create `D:\\MacroTerminal\\.streamlit\\secrets.toml` with `FRED_API_KEY = \"your_key\"` "
            "or set the `FRED_API_KEY` environment variable. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return result

    for name, sid in FRED_SERIES.items():
        df = fetch_fred_series(sid)
        result[name] = df

    # YoY for inflation series
    if not result.get("cpi", pd.DataFrame()).empty:
        result["cpi_yoy"] = compute_yoy(result["cpi"])
    if not result.get("core_cpi", pd.DataFrame()).empty:
        result["core_cpi_yoy"] = compute_yoy(result["core_cpi"])

    # Yield spread
    t10y = result.get("t10y", pd.DataFrame())
    t2y = result.get("t2y", pd.DataFrame())
    if not t10y.empty and not t2y.empty:
        result["yield_spread"] = compute_spread(t10y, t2y)

    return result
