"""
core/data_markets.py
Fetches market OHLCV data via yfinance and computes derived metrics.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)

# Ticker maps
GLOBAL_TICKERS = {
    "spx": "^GSPC",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
}

INDIA_TICKERS = {
    "nifty": "^NSEI",
    "india_vix": "^INDIAVIX",
    "usdinr": "INR=X",
}

PERIOD_MAP = {
    "1M": "1mo",
    "3M": "3mo",
    "1Y": "1y",
    "5Y": "5y",
}


def fetch_yfinance(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker using yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol.
        period: Data period string (e.g., '1y', '2y').
        interval: Data interval (e.g., '1d').

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or empty DataFrame on error.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            logger.warning(f"No data returned for ticker: {ticker}")
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}")
        return pd.DataFrame()


def compute_returns(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Compute daily log returns from a price series."""
    return np.log(df[col] / df[col].shift(1)).dropna()


def compute_rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute annualized rolling volatility from daily returns."""
    return returns.rolling(window).std() * np.sqrt(252)


def compute_drawdown_from_high(prices: pd.Series, lookback_days: int = 21) -> float:
    """
    Compute current drawdown from the high over the last N trading days.

    Returns:
        Drawdown as a negative percentage float.
    """
    recent = prices.iloc[-lookback_days:]
    if recent.empty:
        return 0.0
    peak = recent.max()
    current = prices.iloc[-1]
    if peak == 0:
        return 0.0
    return ((current - peak) / peak) * 100


def compute_percentile_rank(series: pd.Series, lookback_days: int = 252) -> float:
    """
    Compute the percentile rank of the latest value within the last N observations.

    Returns:
        Percentile rank between 0 and 100.
    """
    recent = series.dropna().iloc[-lookback_days:]
    if len(recent) < 2:
        return 50.0
    latest = recent.iloc[-1]
    rank = (recent < latest).mean() * 100
    return float(rank)


def compute_ma(prices: pd.Series, window: int) -> pd.Series:
    """Compute simple moving average."""
    return prices.rolling(window).mean()


def compute_rolling_beta(
    y_returns: pd.Series, x_returns: pd.Series, window: int = 252
) -> float:
    """
    Compute rolling beta of y vs x over last `window` observations.

    Returns:
        Beta coefficient or NaN if insufficient data.
    """
    aligned = pd.concat([y_returns, x_returns], axis=1).dropna()
    aligned.columns = ["y", "x"]
    recent = aligned.iloc[-window:]
    if len(recent) < 30:
        return float("nan")
    cov = recent.cov()
    if cov.loc["x", "x"] == 0:
        return float("nan")
    return float(cov.loc["y", "x"] / cov.loc["x", "x"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_market_data() -> dict:
    """
    Load all required market tickers and compute derived series.
    Cached for 1 hour.

    Returns:
        Dictionary of DataFrames and scalar metrics.
    """
    result = {}

    all_tickers = {**GLOBAL_TICKERS, **INDIA_TICKERS}
    for name, ticker in all_tickers.items():
        df = fetch_yfinance(ticker, period="2y")
        result[name] = df
        if not df.empty and "Close" in df.columns:
            result[f"{name}_returns"] = compute_returns(df)
            result[f"{name}_vol20d"] = compute_rolling_vol(result[f"{name}_returns"])

    return result
