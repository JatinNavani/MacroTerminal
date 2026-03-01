"""
core/regime.py
Implements regime classification, insight feed generation, and alert computation.
"""

from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# GLOBAL REGIME
# ─────────────────────────────────────────────

def global_inflation_regime(cpi_yoy_df: pd.DataFrame) -> str:
    """
    Compare 3-month average YoY vs 12-month average YoY.
    Returns 'Inflation Rising' or 'Inflation Falling'.
    """
    if cpi_yoy_df is None or cpi_yoy_df.empty:
        return "Inflation Unknown"
    series = cpi_yoy_df["yoy"].dropna()
    if len(series) < 12:
        return "Inflation Unknown"
    avg_3m = series.iloc[-3:].mean()
    avg_12m = series.iloc[-12:].mean()
    return "Inflation Rising" if avg_3m > avg_12m else "Inflation Falling"


def global_rates_regime(t2y_df: pd.DataFrame) -> str:
    """
    Assess 3-month slope of the 2Y yield.
    Returns 'Rates Tightening' or 'Rates Easing'.
    """
    if t2y_df is None or t2y_df.empty:
        return "Rates Unknown"
    series = t2y_df["value"].dropna()
    if len(series) < 63:
        return "Rates Unknown"
    change = series.iloc[-1] - series.iloc[-63]  # ~3 months trading days
    return "Rates Tightening" if change > 0.1 else ("Rates Easing" if change < -0.1 else "Rates Stable")


def global_risk_regime(
    vix_df: pd.DataFrame,
    spx_df: pd.DataFrame,
) -> str:
    """
    Combine VIX percentile rank and SPX 200D MA trend.
    Returns 'Risk-Off' or 'Risk-On'.
    """
    risk_off_signals = 0
    total_signals = 0

    # VIX percentile
    if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
        vix_series = vix_df["Close"].dropna()
        if len(vix_series) > 50:
            total_signals += 1
            lookback = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
            pct = (lookback < vix_series.iloc[-1]).mean() * 100
            if pct > 80:
                risk_off_signals += 1

    # SPX vs 200D MA
    if spx_df is not None and not spx_df.empty and "Close" in spx_df.columns:
        spx_prices = spx_df["Close"].dropna()
        if len(spx_prices) >= 200:
            total_signals += 1
            ma200 = spx_prices.rolling(200).mean().iloc[-1]
            if spx_prices.iloc[-1] < ma200:
                risk_off_signals += 1

    if total_signals == 0:
        return "Risk Unknown"
    return "Risk-Off" if risk_off_signals > total_signals / 2 else "Risk-On"


def compute_global_regime(fred_data: dict, market_data: dict) -> str:
    """
    Combine all global regime signals into a single label string.
    """
    inflation = global_inflation_regime(fred_data.get("cpi_yoy"))
    rates = global_rates_regime(fred_data.get("t2y"))
    risk = global_risk_regime(
        market_data.get("vix"),
        market_data.get("spx"),
    )
    return f"{inflation} • {rates} • {risk}"


# ─────────────────────────────────────────────
# INDIA REGIME
# ─────────────────────────────────────────────

def india_fx_regime(usdinr_df: pd.DataFrame) -> str:
    """
    Assess INR trend via 1-month change and 20D vol.
    Returns 'INR Weakening' or 'INR Strengthening'.
    """
    if usdinr_df is None or usdinr_df.empty or "Close" not in usdinr_df.columns:
        return "FX Unknown"
    prices = usdinr_df["Close"].dropna()
    if len(prices) < 25:
        return "FX Unknown"
    change_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
    return "INR Weakening" if change_1m > 0.5 else ("INR Strengthening" if change_1m < -0.5 else "INR Stable")


def india_risk_regime(
    india_vix_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
) -> str:
    """
    Combine India VIX percentile and NIFTY 50D MA trend.
    Returns 'Risk-Off' or 'Risk-On'.
    """
    risk_off_signals = 0
    total_signals = 0

    if india_vix_df is not None and not india_vix_df.empty and "Close" in india_vix_df.columns:
        vix_series = india_vix_df["Close"].dropna()
        if len(vix_series) > 50:
            total_signals += 1
            lookback = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
            pct = (lookback < vix_series.iloc[-1]).mean() * 100
            if pct > 80:
                risk_off_signals += 1

    if nifty_df is not None and not nifty_df.empty and "Close" in nifty_df.columns:
        nifty_prices = nifty_df["Close"].dropna()
        if len(nifty_prices) >= 50:
            total_signals += 1
            ma50 = nifty_prices.rolling(50).mean().iloc[-1]
            if nifty_prices.iloc[-1] < ma50:
                risk_off_signals += 1

    if total_signals == 0:
        return "Risk Unknown"
    return "Risk-Off" if risk_off_signals > total_signals / 2 else "Risk-On"


def india_inflation_regime(india_macro_df: pd.DataFrame) -> str:
    """Compute India CPI YoY momentum from CSV data."""
    if india_macro_df is None or india_macro_df.empty or "cpi_yoy" not in india_macro_df.columns:
        return ""
    series = india_macro_df["cpi_yoy"].dropna()
    if len(series) < 4:
        return ""
    avg_3m = series.iloc[-3:].mean()
    avg_12m = series.iloc[-12:].mean() if len(series) >= 12 else series.mean()
    return "Inflation Rising" if avg_3m > avg_12m else "Inflation Falling"


def compute_india_regime(market_data: dict, india_macro_df: pd.DataFrame) -> str:
    """
    Combine all India regime signals into a single label string.
    """
    fx = india_fx_regime(market_data.get("usdinr"))
    risk = india_risk_regime(market_data.get("india_vix"), market_data.get("nifty"))
    inflation = india_inflation_regime(india_macro_df)
    parts = [s for s in [inflation, fx, risk] if s]
    return " • ".join(parts) if parts else "Data Insufficient"


# ─────────────────────────────────────────────
# INSIGHTS FEED
# ─────────────────────────────────────────────

def _safe_last(series: pd.Series, n: int = 1) -> Optional[float]:
    """Return the nth-from-last value of a series, or None."""
    s = series.dropna()
    if len(s) < n:
        return None
    return float(s.iloc[-n])


def compute_insights_feed(fred_data: dict, market_data: dict, india_macro_df: pd.DataFrame) -> list:
    """
    Generate a list of deterministic insight strings based on latest data.

    Returns:
        List of insight strings (up to 12).
    """
    insights = []

    # --- 10Y yield WoW change ---
    t10y = fred_data.get("t10y", pd.DataFrame())
    if not t10y.empty:
        v_now = _safe_last(t10y["value"], 1)
        v_1w = _safe_last(t10y["value"], 6)
        if v_now is not None and v_1w is not None:
            delta_bps = (v_now - v_1w) * 100
            direction = "↑" if delta_bps > 0 else "↓"
            impulse = "tightening impulse" if delta_bps > 5 else ("easing signal" if delta_bps < -5 else "stable")
            insights.append(f"🇺🇸 US 10Y {direction} {abs(delta_bps):.1f} bps WoW → {impulse}")

    # --- Yield curve inversion ---
    spread_df = fred_data.get("yield_spread", pd.DataFrame())
    if not spread_df.empty:
        latest_spread = _safe_last(spread_df["spread"])
        if latest_spread is not None:
            if latest_spread < 0:
                insights.append(f"🔴 Yield curve INVERTED (10Y–2Y = {latest_spread:.2f}%) → recession risk elevated")
            elif latest_spread < 0.3:
                insights.append(f"⚠️ Yield curve near-flat (10Y–2Y = {latest_spread:.2f}%) → watch inversion risk")
            else:
                insights.append(f"✅ Yield curve positive (10Y–2Y = {latest_spread:.2f}%) → no immediate inversion")

    # --- VIX regime ---
    vix_df = market_data.get("vix", pd.DataFrame())
    if not vix_df.empty and "Close" in vix_df.columns:
        vix_series = vix_df["Close"].dropna()
        if len(vix_series) > 50:
            lookback = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
            vix_pct = (lookback < vix_series.iloc[-1]).mean() * 100
            vix_val = vix_series.iloc[-1]
            if vix_pct > 80:
                insights.append(f"🔴 VIX at {vix_val:.1f} (>{vix_pct:.0f}th pct of 1Y) → risk-off conditions")
            elif vix_pct < 20:
                insights.append(f"🟢 VIX at {vix_val:.1f} (<{vix_pct:.0f}th pct of 1Y) → complacency risk")
            else:
                insights.append(f"🟡 VIX at {vix_val:.1f} ({vix_pct:.0f}th pct of 1Y) → moderate risk environment")

    # --- DXY 5D change ---
    dxy_df = market_data.get("dxy", pd.DataFrame())
    if not dxy_df.empty and "Close" in dxy_df.columns:
        dxy = dxy_df["Close"].dropna()
        if len(dxy) > 6:
            chg_5d = (dxy.iloc[-1] / dxy.iloc[-6] - 1) * 100
            if abs(chg_5d) > 0.8:
                direction = "strengthening" if chg_5d > 0 else "weakening"
                insights.append(f"💵 DXY {'+' if chg_5d > 0 else ''}{chg_5d:.2f}% in 5D → USD {direction}")

    # --- SPX trend ---
    spx_df = market_data.get("spx", pd.DataFrame())
    if not spx_df.empty and "Close" in spx_df.columns:
        spx = spx_df["Close"].dropna()
        if len(spx) >= 200:
            ma200 = spx.rolling(200).mean().iloc[-1]
            ma50 = spx.rolling(50).mean().iloc[-1]
            spx_now = spx.iloc[-1]
            if spx_now < ma200:
                insights.append(f"🔴 SPX below 200D MA ({spx_now:,.0f} vs {ma200:,.0f}) → bearish trend")
            elif spx_now < ma50:
                insights.append(f"⚠️ SPX below 50D MA ({spx_now:,.0f} vs {ma50:,.0f}) → near-term weakness")
            else:
                insights.append(f"🟢 SPX above 50D & 200D MA → trend intact")

    # --- USD/INR stress ---
    usdinr_df = market_data.get("usdinr", pd.DataFrame())
    if not usdinr_df.empty and "Close" in usdinr_df.columns:
        inr = usdinr_df["Close"].dropna()
        if len(inr) > 6:
            chg_5d = (inr.iloc[-1] / inr.iloc[-6] - 1) * 100
            if chg_5d > 0.5:
                insights.append(f"🇮🇳 USD/INR +{chg_5d:.2f}% in 5D → FX stress building")
            elif chg_5d < -0.5:
                insights.append(f"🇮🇳 USD/INR {chg_5d:.2f}% in 5D → INR recovery underway")

    # --- NIFTY trend ---
    nifty_df = market_data.get("nifty", pd.DataFrame())
    if not nifty_df.empty and "Close" in nifty_df.columns:
        nifty = nifty_df["Close"].dropna()
        if len(nifty) >= 50:
            ma50 = nifty.rolling(50).mean().iloc[-1]
            nifty_now = nifty.iloc[-1]
            if nifty_now < ma50:
                insights.append(f"🇮🇳 NIFTY below 50D MA ({nifty_now:,.0f} vs {ma50:,.0f}) → risk appetite weakening")
            else:
                insights.append(f"🇮🇳 NIFTY above 50D MA ({nifty_now:,.0f} vs {ma50:,.0f}) → bullish structure intact")

    # --- India CPI trend ---
    if india_macro_df is not None and not india_macro_df.empty and "cpi_yoy" in india_macro_df.columns:
        cpi_series = india_macro_df["cpi_yoy"].dropna()
        if len(cpi_series) >= 3:
            latest = cpi_series.iloc[-1]
            prev = cpi_series.iloc[-2]
            trend = "↑" if latest > prev else "↓"
            insights.append(f"🇮🇳 India CPI YoY {latest:.2f}% {trend} from {prev:.2f}% prior month")

    return insights[:12]


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────

def compute_alerts(fred_data: dict, market_data: dict) -> list:
    """
    Compute threshold and percentile-based alerts.

    Returns:
        List of dicts with keys: alert, status, latest, threshold, interpretation.
    """
    alerts = []

    # VIX above threshold
    vix_threshold = 25.0
    vix_df = market_data.get("vix", pd.DataFrame())
    if not vix_df.empty and "Close" in vix_df.columns:
        vix_val = vix_df["Close"].dropna().iloc[-1]
        status = "🔴 BREACH" if vix_val > vix_threshold else "🟢 OK"
        alerts.append({
            "Alert": "VIX Spike",
            "Status": status,
            "Latest": f"{vix_val:.2f}",
            "Threshold": f"> {vix_threshold}",
            "Interpretation": "Elevated fear / risk-off" if vix_val > vix_threshold else "Normal volatility",
        })

    # Yield curve inversion
    spread_df = fred_data.get("yield_spread", pd.DataFrame())
    if not spread_df.empty:
        spread_val = spread_df["spread"].dropna().iloc[-1]
        status = "🔴 INVERTED" if spread_val < 0 else ("⚠️ FLAT" if spread_val < 0.3 else "🟢 OK")
        alerts.append({
            "Alert": "10Y–2Y Inversion",
            "Status": status,
            "Latest": f"{spread_val:.2f}%",
            "Threshold": "< 0%",
            "Interpretation": "Recession signal active" if spread_val < 0 else "No inversion",
        })

    # DXY 5D change
    dxy_threshold = 1.5
    dxy_df = market_data.get("dxy", pd.DataFrame())
    if not dxy_df.empty and "Close" in dxy_df.columns:
        dxy = dxy_df["Close"].dropna()
        if len(dxy) > 6:
            chg = (dxy.iloc[-1] / dxy.iloc[-6] - 1) * 100
            status = "⚠️ WATCH" if abs(chg) > dxy_threshold else "🟢 OK"
            alerts.append({
                "Alert": "DXY 5D Surge",
                "Status": status,
                "Latest": f"{chg:+.2f}%",
                "Threshold": f"|ΔΔ| > {dxy_threshold}%",
                "Interpretation": "Strong USD move — watch EM stress" if abs(chg) > dxy_threshold else "DXY stable",
            })

    # India VIX percentile
    india_vix_threshold = 80
    india_vix_df = market_data.get("india_vix", pd.DataFrame())
    if not india_vix_df.empty and "Close" in india_vix_df.columns:
        ivix = india_vix_df["Close"].dropna()
        if len(ivix) > 50:
            lookback = ivix.iloc[-252:] if len(ivix) >= 252 else ivix
            pct = (lookback < ivix.iloc[-1]).mean() * 100
            status = "🔴 HIGH" if pct > india_vix_threshold else "🟢 OK"
            alerts.append({
                "Alert": "India VIX Percentile",
                "Status": status,
                "Latest": f"{pct:.0f}th pct",
                "Threshold": f"> {india_vix_threshold}th pct",
                "Interpretation": "Elevated India volatility" if pct > india_vix_threshold else "Normal range",
            })

    # USD/INR 5D change
    usdinr_threshold = 1.0
    usdinr_df = market_data.get("usdinr", pd.DataFrame())
    if not usdinr_df.empty and "Close" in usdinr_df.columns:
        inr = usdinr_df["Close"].dropna()
        if len(inr) > 6:
            chg = (inr.iloc[-1] / inr.iloc[-6] - 1) * 100
            status = "⚠️ WATCH" if chg > usdinr_threshold else "🟢 OK"
            alerts.append({
                "Alert": "USD/INR 5D Move",
                "Status": status,
                "Latest": f"{chg:+.2f}%",
                "Threshold": f"> +{usdinr_threshold}%",
                "Interpretation": "INR depreciation pressure" if chg > usdinr_threshold else "FX stable",
            })

    # NIFTY drawdown from 1M high
    nifty_threshold = -5.0
    nifty_df = market_data.get("nifty", pd.DataFrame())
    if not nifty_df.empty and "Close" in nifty_df.columns:
        nifty = nifty_df["Close"].dropna()
        if len(nifty) >= 21:
            drawdown = ((nifty.iloc[-1] / nifty.iloc[-21:].max()) - 1) * 100
            status = "⚠️ DRAWDOWN" if drawdown < nifty_threshold else "🟢 OK"
            alerts.append({
                "Alert": "NIFTY 1M Drawdown",
                "Status": status,
                "Latest": f"{drawdown:.2f}%",
                "Threshold": f"< {nifty_threshold}%",
                "Interpretation": "Material correction underway" if drawdown < nifty_threshold else "Within normal range",
            })

    return alerts
