"""
core/regime.py
Implements regime classification, insight feed generation, and alert computation.
"""

from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _safe_last(series: pd.Series, n: int = 1) -> Optional[float]:
    """Return the nth-from-last value of a series, or None."""
    s = series.dropna()
    if len(s) < n:
        return None
    return float(s.iloc[-n])


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
    avg_3m  = series.iloc[-3:].mean()
    avg_12m = series.iloc[-12:].mean()
    return "Inflation Rising" if avg_3m > avg_12m else "Inflation Falling"


def global_rates_regime(t2y_df: pd.DataFrame) -> str:
    """
    Assess 3-month slope of the 2Y yield.
    Returns 'Rates Tightening', 'Rates Easing', or 'Rates Stable'.
    """
    if t2y_df is None or t2y_df.empty:
        return "Rates Unknown"
    series = t2y_df["value"].dropna()
    if len(series) < 63:
        return "Rates Unknown"
    change = series.iloc[-1] - series.iloc[-63]
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
    total_signals    = 0

    if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
        vix_series = vix_df["Close"].dropna()
        if len(vix_series) > 50:
            total_signals += 1
            lookback = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
            pct = (lookback < vix_series.iloc[-1]).mean() * 100
            if pct > 80:
                risk_off_signals += 1

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
    """Combine all global regime signals into a single label string."""
    inflation = global_inflation_regime(fred_data.get("cpi_yoy"))
    rates     = global_rates_regime(fred_data.get("t2y"))
    risk      = global_risk_regime(
        market_data.get("vix"),
        market_data.get("spx"),
    )
    return f"{inflation} • {rates} • {risk}"


# ─────────────────────────────────────────────
# INDIA REGIME
# ─────────────────────────────────────────────

def india_fx_regime(usdinr_df: pd.DataFrame) -> str:
    """
    Assess INR trend via 1-month change.
    Returns 'INR Weakening', 'INR Strengthening', or 'INR Stable'.
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
    total_signals    = 0

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
    """Compute India CPI YoY momentum from CSV/FRED data."""
    if india_macro_df is None or india_macro_df.empty or "cpi_yoy" not in india_macro_df.columns:
        return ""
    series = india_macro_df["cpi_yoy"].dropna()
    if len(series) < 4:
        return ""
    avg_3m  = series.iloc[-3:].mean()
    avg_12m = series.iloc[-12:].mean() if len(series) >= 12 else series.mean()
    return "Inflation Rising" if avg_3m > avg_12m else "Inflation Falling"


def compute_india_regime(market_data: dict, india_macro_df: pd.DataFrame) -> str:
    """Combine all India regime signals into a single label string."""
    fx        = india_fx_regime(market_data.get("usdinr"))
    risk      = india_risk_regime(market_data.get("india_vix"), market_data.get("nifty"))
    inflation = india_inflation_regime(india_macro_df)
    parts = [s for s in [inflation, fx, risk] if s]
    return " • ".join(parts) if parts else "Data Insufficient"


# ─────────────────────────────────────────────
# INSIGHTS FEED
# ─────────────────────────────────────────────

def compute_insights_feed(fred_data: dict, market_data: dict, india_macro_df: pd.DataFrame) -> list:
    """
    Generate a list of deterministic insight strings based on latest data.
    Covers: rates, yield curve, volatility, FX, equities, credit, commodities,
            real yields, copper/gold ratio, India G-Sec, Nifty Bank, Brent crude.

    Returns:
        List of insight strings (up to 18).
    """
    insights = []

    # ── US 10Y yield WoW change ──
    t10y = fred_data.get("t10y", pd.DataFrame())
    if not t10y.empty:
        v_now = _safe_last(t10y["value"], 1)
        v_1w  = _safe_last(t10y["value"], 6)
        if v_now is not None and v_1w is not None:
            delta_bps = (v_now - v_1w) * 100
            direction = "↑" if delta_bps > 0 else "↓"
            impulse   = "tightening impulse" if delta_bps > 5 else ("easing signal" if delta_bps < -5 else "stable")
            insights.append(f"🇺🇸 US 10Y {direction} {abs(delta_bps):.1f} bps WoW → {impulse}")

    # ── 10Y Real Yield ──
    real_yield_df = fred_data.get("real_yield_10y", pd.DataFrame())
    if not real_yield_df.empty:
        rv = _safe_last(real_yield_df["value"])
        rv_1m = _safe_last(real_yield_df["value"], 22)
        if rv is not None:
            if rv > 2.0:
                insights.append(f"📈 10Y Real Yield at {rv:.2f}% → tight financial conditions, headwind for gold & growth")
            elif rv < 0:
                insights.append(f"📉 10Y Real Yield negative ({rv:.2f}%) → accommodative real rates, supportive for gold")
            else:
                direction = "rising" if (rv_1m is not None and rv > rv_1m) else "falling"
                insights.append(f"📊 10Y Real Yield at {rv:.2f}% ({direction}) → neutral financial conditions")

    # ── Yield curve inversion ──
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

    # ── VIX regime ──
    vix_df = market_data.get("vix", pd.DataFrame())
    if not vix_df.empty and "Close" in vix_df.columns:
        vix_series = vix_df["Close"].dropna()
        if len(vix_series) > 50:
            lookback = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
            vix_pct  = (lookback < vix_series.iloc[-1]).mean() * 100
            vix_val  = vix_series.iloc[-1]
            if vix_pct > 80:
                insights.append(f"🔴 VIX at {vix_val:.1f} (>{vix_pct:.0f}th pct of 1Y) → risk-off conditions")
            elif vix_pct < 20:
                insights.append(f"🟢 VIX at {vix_val:.1f} (<{vix_pct:.0f}th pct of 1Y) → complacency risk")
            else:
                insights.append(f"🟡 VIX at {vix_val:.1f} ({vix_pct:.0f}th pct of 1Y) → moderate risk environment")

    # ── HYG (Credit Spreads proxy) ──
    hyg_price = market_data.get("hyg_price", pd.Series(dtype=float))
    if hyg_price is not None and not hyg_price.empty and len(hyg_price) > 21:
        chg_1m = (hyg_price.iloc[-1] / hyg_price.iloc[-21] - 1) * 100
        if chg_1m < -2.0:
            insights.append(f"🔴 HYG (HY Bond ETF) down {chg_1m:.2f}% in 1M → credit spreads widening, stress building")
        elif chg_1m > 1.5:
            insights.append(f"🟢 HYG up {chg_1m:.2f}% in 1M → credit spreads tightening, risk appetite healthy")
        else:
            insights.append(f"🟡 HYG {chg_1m:+.2f}% in 1M → credit markets broadly stable")

    # ── Gold ──
    gold_df = market_data.get("gold", pd.DataFrame())
    if not gold_df.empty and "Close" in gold_df.columns:
        gold = gold_df["Close"].dropna()
        if len(gold) > 6:
            chg_5d = (gold.iloc[-1] / gold.iloc[-6] - 1) * 100
            level  = gold.iloc[-1]
            if abs(chg_5d) > 1.0:
                direction = "surging" if chg_5d > 0 else "falling"
                driver    = "safe-haven demand or USD weakness" if chg_5d > 0 else "risk-on rotation or USD strength"
                insights.append(f"🥇 Gold {'+' if chg_5d > 0 else ''}{chg_5d:.2f}% in 5D (${level:,.0f}) → {driver}")

    # ── Gold vs DXY divergence ──
    dxy_df = market_data.get("dxy", pd.DataFrame())
    if not gold_df.empty and not dxy_df.empty and "Close" in gold_df.columns and "Close" in dxy_df.columns:
        gold = gold_df["Close"].dropna()
        dxy  = dxy_df["Close"].dropna()
        if len(gold) > 6 and len(dxy) > 6:
            gold_5d = (gold.iloc[-1] / gold.iloc[-6] - 1) * 100
            dxy_5d  = (dxy.iloc[-1]  / dxy.iloc[-6]  - 1) * 100
            if gold_5d > 1.0 and dxy_5d > 0.5:
                insights.append(
                    f"⚠️ Gold & DXY both rising (Gold {gold_5d:+.2f}%, DXY {dxy_5d:+.2f}%) "
                    f"→ fear-driven safe-haven, not just USD effect"
                )
            elif gold_5d < -1.0 and dxy_5d < -0.5:
                insights.append(f"🟢 Gold & DXY both falling → broad risk-on, safe-havens being sold")

    # ── WTI Crude Oil ──
    oil_df = market_data.get("oil_wti", pd.DataFrame())
    if not oil_df.empty and "Close" in oil_df.columns:
        oil = oil_df["Close"].dropna()
        if len(oil) > 6:
            chg_5d = (oil.iloc[-1] / oil.iloc[-6] - 1) * 100
            level  = oil.iloc[-1]
            if chg_5d > 3.0:
                insights.append(f"🛢️ WTI Crude +{chg_5d:.2f}% in 5D (${level:.1f}) → supply shock risk, watch CPI")
            elif chg_5d < -3.0:
                insights.append(f"🛢️ WTI Crude {chg_5d:.2f}% in 5D (${level:.1f}) → demand concerns or supply surge")

    # ── Copper/Gold ratio (growth barometer) ──
    cg_ratio = market_data.get("copper_gold_ratio", pd.Series(dtype=float))
    if cg_ratio is not None and not cg_ratio.empty and len(cg_ratio) > 21:
        chg_1m = (cg_ratio.iloc[-1] / cg_ratio.iloc[-21] - 1) * 100
        if chg_1m > 3.0:
            insights.append(f"🟢 Copper/Gold ratio up {chg_1m:.1f}% in 1M → market pricing in growth optimism")
        elif chg_1m < -3.0:
            insights.append(f"🔴 Copper/Gold ratio down {abs(chg_1m):.1f}% in 1M → growth fears rising, metals signaling slowdown")

    # ── DXY 5D change ──
    if not dxy_df.empty and "Close" in dxy_df.columns:
        dxy = dxy_df["Close"].dropna()
        if len(dxy) > 6:
            chg_5d = (dxy.iloc[-1] / dxy.iloc[-6] - 1) * 100
            if abs(chg_5d) > 0.8:
                direction = "strengthening" if chg_5d > 0 else "weakening"
                insights.append(f"💵 DXY {'+' if chg_5d > 0 else ''}{chg_5d:.2f}% in 5D → USD {direction}")

    # ── SPX trend ──
    spx_df = market_data.get("spx", pd.DataFrame())
    if not spx_df.empty and "Close" in spx_df.columns:
        spx = spx_df["Close"].dropna()
        if len(spx) >= 200:
            ma200    = spx.rolling(200).mean().iloc[-1]
            ma50     = spx.rolling(50).mean().iloc[-1]
            spx_now  = spx.iloc[-1]
            if spx_now < ma200:
                insights.append(f"🔴 SPX below 200D MA ({spx_now:,.0f} vs {ma200:,.0f}) → bearish trend")
            elif spx_now < ma50:
                insights.append(f"⚠️ SPX below 50D MA ({spx_now:,.0f} vs {ma50:,.0f}) → near-term weakness")
            else:
                insights.append(f"🟢 SPX above 50D & 200D MA → trend intact")

    # ── USD/INR stress ──
    usdinr_df = market_data.get("usdinr", pd.DataFrame())
    if not usdinr_df.empty and "Close" in usdinr_df.columns:
        inr = usdinr_df["Close"].dropna()
        if len(inr) > 6:
            chg_5d = (inr.iloc[-1] / inr.iloc[-6] - 1) * 100
            if chg_5d > 0.5:
                insights.append(f"🇮🇳 USD/INR +{chg_5d:.2f}% in 5D → FX stress building")
            elif chg_5d < -0.5:
                insights.append(f"🇮🇳 USD/INR {chg_5d:.2f}% in 5D → INR recovery underway")

    # ── Brent Crude impact on India ──
    brent_df = market_data.get("brent", pd.DataFrame())
    if not brent_df.empty and "Close" in brent_df.columns:
        brent = brent_df["Close"].dropna()
        if len(brent) > 6:
            chg_5d = (brent.iloc[-1] / brent.iloc[-6] - 1) * 100
            level  = brent.iloc[-1]
            if chg_5d > 3.0:
                insights.append(
                    f"🇮🇳 Brent Crude +{chg_5d:.2f}% in 5D (${level:.1f}) "
                    f"→ India import bill rising, CAD & INR under pressure"
                )
            elif chg_5d < -3.0:
                insights.append(
                    f"🇮🇳 Brent Crude {chg_5d:.2f}% in 5D (${level:.1f}) "
                    f"→ relief for India CAD & inflation outlook"
                )

    # ── NIFTY trend ──
    nifty_df = market_data.get("nifty", pd.DataFrame())
    if not nifty_df.empty and "Close" in nifty_df.columns:
        nifty = nifty_df["Close"].dropna()
        if len(nifty) >= 50:
            ma50      = nifty.rolling(50).mean().iloc[-1]
            nifty_now = nifty.iloc[-1]
            if nifty_now < ma50:
                insights.append(f"🇮🇳 NIFTY below 50D MA ({nifty_now:,.0f} vs {ma50:,.0f}) → risk appetite weakening")
            else:
                insights.append(f"🇮🇳 NIFTY above 50D MA ({nifty_now:,.0f} vs {ma50:,.0f}) → bullish structure intact")

    # ── Nifty Bank vs Nifty divergence ──
    nifty_bank_df = market_data.get("nifty_bank", pd.DataFrame())
    if not nifty_bank_df.empty and not nifty_df.empty and "Close" in nifty_bank_df.columns and "Close" in nifty_df.columns:
        nb  = nifty_bank_df["Close"].dropna()
        nif = nifty_df["Close"].dropna()
        if len(nb) > 6 and len(nif) > 6:
            nb_5d  = (nb.iloc[-1]  / nb.iloc[-6]  - 1) * 100
            nif_5d = (nif.iloc[-1] / nif.iloc[-6] - 1) * 100
            spread = nb_5d - nif_5d
            if spread > 1.5:
                insights.append(f"🇮🇳 Nifty Bank outperforming Nifty by {spread:.1f}% in 5D → financials leading, credit expanding")
            elif spread < -1.5:
                insights.append(f"🇮🇳 Nifty Bank underperforming Nifty by {abs(spread):.1f}% in 5D → financial sector stress, watch NPAs")

    # ── India G-Sec 10Y ──
    india_gsec_df = fred_data.get("india_gsec_10y", pd.DataFrame())
    if not india_gsec_df.empty:
        gsec_val  = _safe_last(india_gsec_df["value"])
        gsec_prev = _safe_last(india_gsec_df["value"], 2)
        if gsec_val is not None and gsec_prev is not None:
            delta = gsec_val - gsec_prev
            direction = "↑" if delta > 0 else "↓"
            insights.append(f"🇮🇳 India 10Y G-Sec at {gsec_val:.2f}% {direction} → {'bond yields rising, rates tightening' if delta > 0 else 'yields easing, monetary policy loosening'}")

    # ── India CPI trend ──
    if india_macro_df is not None and not india_macro_df.empty and "cpi_yoy" in india_macro_df.columns:
        cpi_series = india_macro_df["cpi_yoy"].dropna()
        if len(cpi_series) >= 3:
            latest = cpi_series.iloc[-1]
            prev   = cpi_series.iloc[-2]
            trend  = "↑" if latest > prev else "↓"
            insights.append(f"🇮🇳 India CPI YoY {latest:.2f}% {trend} from {prev:.2f}% prior month")

    return insights[:18]


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────

def compute_alerts(fred_data: dict, market_data: dict) -> list:
    """
    Compute threshold and percentile-based alerts for all tracked indicators.

    Returns:
        List of dicts with keys: Alert, Status, Latest, Threshold, Interpretation.
    """
    alerts = []

    # ── VIX Spike ──
    vix_threshold = 25.0
    vix_df = market_data.get("vix", pd.DataFrame())
    if not vix_df.empty and "Close" in vix_df.columns:
        vix_val = vix_df["Close"].dropna().iloc[-1]
        status  = "🔴 BREACH" if vix_val > vix_threshold else "🟢 OK"
        alerts.append({
            "Alert": "VIX Spike",
            "Status": status,
            "Latest": f"{vix_val:.2f}",
            "Threshold": f"> {vix_threshold}",
            "Interpretation": "Elevated fear / risk-off" if vix_val > vix_threshold else "Normal volatility",
        })

    # ── Yield Curve Inversion ──
    spread_df = fred_data.get("yield_spread", pd.DataFrame())
    if not spread_df.empty:
        spread_val = spread_df["spread"].dropna().iloc[-1]
        status     = "🔴 INVERTED" if spread_val < 0 else ("⚠️ FLAT" if spread_val < 0.3 else "🟢 OK")
        alerts.append({
            "Alert": "10Y–2Y Inversion",
            "Status": status,
            "Latest": f"{spread_val:.2f}%",
            "Threshold": "< 0%",
            "Interpretation": "Recession signal active" if spread_val < 0 else "No inversion",
        })

    # ── 10Y Real Yield Extremes ──
    real_yield_df = fred_data.get("real_yield_10y", pd.DataFrame())
    if not real_yield_df.empty:
        rv     = real_yield_df["value"].dropna().iloc[-1]
        status = "🔴 BREACH" if rv > 2.5 else ("⚠️ WATCH" if rv > 2.0 else ("⚠️ WATCH" if rv < 0 else "🟢 OK"))
        alerts.append({
            "Alert": "10Y Real Yield",
            "Status": status,
            "Latest": f"{rv:.2f}%",
            "Threshold": "> 2.0% or < 0%",
            "Interpretation": (
                "Real rates restrictive — headwind for equities & gold" if rv > 2.0
                else ("Negative real rates — inflationary, supportive for gold" if rv < 0 else "Neutral real rates")
            ),
        })

    # ── HYG Credit Spread Proxy ──
    hyg_price = market_data.get("hyg_price", pd.Series(dtype=float))
    hyg_threshold_pct = -2.0
    if hyg_price is not None and not hyg_price.empty and len(hyg_price) > 21:
        chg_1m = (hyg_price.iloc[-1] / hyg_price.iloc[-21] - 1) * 100
        status = "🔴 BREACH" if chg_1m < hyg_threshold_pct else "🟢 OK"
        alerts.append({
            "Alert": "HYG Credit Proxy",
            "Status": status,
            "Latest": f"{chg_1m:+.2f}% (1M)",
            "Threshold": f"< {hyg_threshold_pct}% in 1M",
            "Interpretation": "HY bond ETF falling → credit spreads widening, stress building" if chg_1m < hyg_threshold_pct else "Credit markets stable",
        })

    # ── Gold 5D Spike ──
    gold_threshold = 2.0
    gold_df = market_data.get("gold", pd.DataFrame())
    if not gold_df.empty and "Close" in gold_df.columns:
        gold = gold_df["Close"].dropna()
        if len(gold) > 6:
            chg   = (gold.iloc[-1] / gold.iloc[-6] - 1) * 100
            level = gold.iloc[-1]
            status = "⚠️ WATCH" if abs(chg) > gold_threshold else "🟢 OK"
            alerts.append({
                "Alert": "Gold 5D Spike",
                "Status": status,
                "Latest": f"${level:,.0f} ({chg:+.2f}%)",
                "Threshold": f"|Δ| > {gold_threshold}%",
                "Interpretation": (
                    "Safe-haven demand surge — watch for risk-off or stagflation signals" if chg > gold_threshold
                    else ("Gold selloff — risk-on rotation or USD strength" if chg < -gold_threshold else "Gold stable")
                ),
            })

    # ── Gold / DXY Divergence ──
    dxy_df = market_data.get("dxy", pd.DataFrame())
    if not gold_df.empty and not dxy_df.empty and "Close" in gold_df.columns and "Close" in dxy_df.columns:
        gold = gold_df["Close"].dropna()
        dxy  = dxy_df["Close"].dropna()
        if len(gold) > 6 and len(dxy) > 6:
            gold_chg = (gold.iloc[-1] / gold.iloc[-6] - 1) * 100
            dxy_chg  = (dxy.iloc[-1]  / dxy.iloc[-6]  - 1) * 100
            diverging = gold_chg > 1.0 and dxy_chg > 0.5
            status    = "🔴 BREACH" if diverging else "🟢 OK"
            alerts.append({
                "Alert": "Gold/DXY Divergence",
                "Status": status,
                "Latest": f"Gold {gold_chg:+.2f}%, DXY {dxy_chg:+.2f}%",
                "Threshold": "Gold >+1% & DXY >+0.5%",
                "Interpretation": "Both rising → fear-driven flight to safety, not purely USD-driven" if diverging else "Normal co-movement",
            })

    # ── WTI Crude Oil Spike ──
    oil_threshold = 5.0
    oil_df = market_data.get("oil_wti", pd.DataFrame())
    if not oil_df.empty and "Close" in oil_df.columns:
        oil = oil_df["Close"].dropna()
        if len(oil) > 6:
            chg   = (oil.iloc[-1] / oil.iloc[-6] - 1) * 100
            level = oil.iloc[-1]
            status = "🔴 BREACH" if abs(chg) > oil_threshold else "🟢 OK"
            alerts.append({
                "Alert": "WTI Crude 5D Move",
                "Status": status,
                "Latest": f"${level:.1f} ({chg:+.2f}%)",
                "Threshold": f"|Δ| > {oil_threshold}%",
                "Interpretation": (
                    "Sharp oil spike → supply shock risk, inflationary" if chg > oil_threshold
                    else ("Sharp oil drop → demand destruction fears" if chg < -oil_threshold else "Oil stable")
                ),
            })

    # ── Copper/Gold Ratio ──
    cg_ratio = market_data.get("copper_gold_ratio", pd.Series(dtype=float))
    if cg_ratio is not None and not cg_ratio.empty and len(cg_ratio) > 21:
        chg_1m = (cg_ratio.iloc[-1] / cg_ratio.iloc[-21] - 1) * 100
        status = "⚠️ WATCH" if chg_1m < -3.0 else "🟢 OK"
        alerts.append({
            "Alert": "Copper/Gold Ratio",
            "Status": status,
            "Latest": f"{chg_1m:+.2f}% (1M)",
            "Threshold": "< -3% in 1M",
            "Interpretation": "Growth slowdown signal from metals market" if chg_1m < -3.0 else "Metals ratio stable — growth intact",
        })

    # ── DXY 5D Surge ──
    dxy_threshold = 1.5
    if not dxy_df.empty and "Close" in dxy_df.columns:
        dxy = dxy_df["Close"].dropna()
        if len(dxy) > 6:
            chg    = (dxy.iloc[-1] / dxy.iloc[-6] - 1) * 100
            status = "⚠️ WATCH" if abs(chg) > dxy_threshold else "🟢 OK"
            alerts.append({
                "Alert": "DXY 5D Surge",
                "Status": status,
                "Latest": f"{chg:+.2f}%",
                "Threshold": f"|Δ| > {dxy_threshold}%",
                "Interpretation": "Strong USD move — watch EM stress" if abs(chg) > dxy_threshold else "DXY stable",
            })

    # ── India VIX Percentile ──
    india_vix_threshold = 80
    india_vix_df = market_data.get("india_vix", pd.DataFrame())
    if not india_vix_df.empty and "Close" in india_vix_df.columns:
        ivix = india_vix_df["Close"].dropna()
        if len(ivix) > 50:
            lookback = ivix.iloc[-252:] if len(ivix) >= 252 else ivix
            pct      = (lookback < ivix.iloc[-1]).mean() * 100
            status   = "🔴 HIGH" if pct > india_vix_threshold else "🟢 OK"
            alerts.append({
                "Alert": "India VIX Percentile",
                "Status": status,
                "Latest": f"{pct:.0f}th pct",
                "Threshold": f"> {india_vix_threshold}th pct",
                "Interpretation": "Elevated India volatility" if pct > india_vix_threshold else "Normal range",
            })

    # ── USD/INR 5D Move ──
    usdinr_threshold = 1.0
    usdinr_df = market_data.get("usdinr", pd.DataFrame())
    if not usdinr_df.empty and "Close" in usdinr_df.columns:
        inr = usdinr_df["Close"].dropna()
        if len(inr) > 6:
            chg    = (inr.iloc[-1] / inr.iloc[-6] - 1) * 100
            status = "⚠️ WATCH" if chg > usdinr_threshold else "🟢 OK"
            alerts.append({
                "Alert": "USD/INR 5D Move",
                "Status": status,
                "Latest": f"{chg:+.2f}%",
                "Threshold": f"> +{usdinr_threshold}%",
                "Interpretation": "INR depreciation pressure" if chg > usdinr_threshold else "FX stable",
            })

    # ── Brent Crude (India CAD risk) ──
    brent_threshold = 5.0
    brent_df = market_data.get("brent", pd.DataFrame())
    if not brent_df.empty and "Close" in brent_df.columns:
        brent = brent_df["Close"].dropna()
        if len(brent) > 6:
            chg   = (brent.iloc[-1] / brent.iloc[-6] - 1) * 100
            level = brent.iloc[-1]
            status = "🔴 BREACH" if chg > brent_threshold else "🟢 OK"
            alerts.append({
                "Alert": "Brent Crude (India CAD)",
                "Status": status,
                "Latest": f"${level:.1f} ({chg:+.2f}%)",
                "Threshold": f"> +{brent_threshold}% in 5D",
                "Interpretation": "Brent spike → India import cost rising, CAD & INR at risk" if chg > brent_threshold else "Brent stable — India CAD not immediately threatened",
            })

    # ── Nifty Bank vs Nifty (financial sector stress) ──
    nifty_df      = market_data.get("nifty", pd.DataFrame())
    nifty_bank_df = market_data.get("nifty_bank", pd.DataFrame())
    if (not nifty_bank_df.empty and not nifty_df.empty
            and "Close" in nifty_bank_df.columns and "Close" in nifty_df.columns):
        nb  = nifty_bank_df["Close"].dropna()
        nif = nifty_df["Close"].dropna()
        if len(nb) > 6 and len(nif) > 6:
            nb_5d   = (nb.iloc[-1]  / nb.iloc[-6]  - 1) * 100
            nif_5d  = (nif.iloc[-1] / nif.iloc[-6] - 1) * 100
            spread  = nb_5d - nif_5d
            status  = "⚠️ WATCH" if spread < -1.5 else "🟢 OK"
            alerts.append({
                "Alert": "Nifty Bank vs Nifty",
                "Status": status,
                "Latest": f"Bank {nb_5d:+.2f}%, Nifty {nif_5d:+.2f}%",
                "Threshold": "Bank lag > 1.5%",
                "Interpretation": "Financial sector underperforming — watch credit quality & RBI signals" if spread < -1.5 else "Banking sector in line with broad market",
            })

    # ── NIFTY 1M Drawdown ──
    nifty_threshold = -5.0
    if not nifty_df.empty and "Close" in nifty_df.columns:
        nifty = nifty_df["Close"].dropna()
        if len(nifty) >= 21:
            drawdown = ((nifty.iloc[-1] / nifty.iloc[-21:].max()) - 1) * 100
            status   = "⚠️ DRAWDOWN" if drawdown < nifty_threshold else "🟢 OK"
            alerts.append({
                "Alert": "NIFTY 1M Drawdown",
                "Status": status,
                "Latest": f"{drawdown:.2f}%",
                "Threshold": f"< {nifty_threshold}%",
                "Interpretation": "Material correction underway" if drawdown < nifty_threshold else "Within normal range",
            })

    return alerts