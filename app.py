"""
app.py — MacroTerminal: Global + India Macro Dashboard
Streamlit-based dark terminal-style finance dashboard.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from core.data_fred import load_all_fred_series
from core.data_markets import load_all_market_data, compute_rolling_beta
from core.india_macro import load_india_macro
from core.regime import compute_global_regime, compute_india_regime, compute_insights_feed, compute_alerts
from core.ui_components import (
    kpi_card, regime_badge, section_header, line_chart, apply_chart_theme,
    fmt_val, fmt_pct, CHART_LAYOUT, LINE_COLORS
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="MacroTerminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    ::-webkit-scrollbar {width:6px;height:6px;}
    ::-webkit-scrollbar-track {background:#0A0E1A;}
    ::-webkit-scrollbar-thumb {background:#1E293B;border-radius:3px;}
    .stTabs [data-baseweb="tab-list"] {gap:4px;border-bottom:1px solid #1E293B;}
    .stTabs [data-baseweb="tab"] {
        background:#111827;border-radius:6px 6px 0 0;color:#64748B;
        padding:8px 18px;font-size:0.82rem;font-family:monospace;letter-spacing:0.04em;
    }
    .stTabs [aria-selected="true"] {background:#0A0E1A;color:#00D4AA;border-bottom:2px solid #00D4AA;}
    div[data-testid="metric-container"] {background:#111827;border-radius:8px;padding:12px;}
    .stDataFrame {font-family:monospace;font-size:0.82rem;}
    section[data-testid="stSidebar"] {background:#0D1117;border-right:1px solid #1E293B;}
    .main .block-container {padding-top:1rem;}
    .insight-item {
        background:#111827;border-left:3px solid #00D4AA;
        padding:8px 14px;margin:4px 0;border-radius:0 6px 6px 0;
        font-size:0.85rem;color:#CBD5E1;font-family:monospace;
    }
</style>
""")

# ─────────────────────────────────────────────
# TERMINAL HEADER
# ─────────────────────────────────────────────

st.html("""
<div style="
    display:flex;align-items:center;gap:16px;
    border-bottom:1px solid #1E293B;padding-bottom:14px;margin-bottom:4px
">
    <span style="font-size:1.6rem">📡</span>
    <div>
        <div style="font-family:monospace;font-size:1.4rem;font-weight:700;
                    color:#00D4AA;letter-spacing:0.06em;">MACROTERMINAL</div>
        <div style="font-family:monospace;font-size:0.72rem;color:#475569;letter-spacing:0.1em;">
            GLOBAL + INDIA MACRO INTELLIGENCE DASHBOARD
        </div>
    </div>
</div>
""")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.html('<div style="font-family:monospace;color:#00D4AA;font-weight:700;font-size:0.9rem;letter-spacing:0.08em;margin-bottom:16px">⚙ CONTROLS</div>')

    region     = st.selectbox("Region", ["Global", "India", "Both"], index=2)
    time_range = st.selectbox("Time Range", ["1M", "3M", "1Y", "5Y"], index=2)

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.html(f'<div style="font-family:monospace;font-size:0.72rem;color:#475569;margin-top:8px">Last updated:<br>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>')

    st.markdown("---")
    st.html("""
    <div style="font-family:monospace;font-size:0.7rem;color:#334155;line-height:1.7">
        Data sources:<br>
        • FRED API (US + India Macro)<br>
        • yfinance (Markets, Commodities)<br>
        • CSV fallback (India, if no key)<br><br>
        <span style="color:#475569">FRED key unlocks live India CPI,<br>
        Repo Rate & 10Y G-Sec yield.</span>
    </div>
    """)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

PERIOD_DAYS = {"1M": 21, "3M": 63, "1Y": 252, "5Y": 1260}
N_DAYS = PERIOD_DAYS[time_range]

with st.spinner("Loading data..."):
    fred_data      = load_all_fred_series()
    market_data    = load_all_market_data()
    india_macro_df = load_india_macro()


def latest_val(df: Optional[pd.DataFrame], col: str = "value") -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def change_vs_n(df: Optional[pd.DataFrame], n: int, col: str = "value") -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    if len(s) < n + 1:
        return None
    return float(s.iloc[-1] - s.iloc[-(n + 1)])


def pct_change_vs_n(df: Optional[pd.DataFrame], n: int, col: str = "Close") -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    if len(s) < n + 1:
        return None
    prev = s.iloc[-(n + 1)]
    if prev == 0:
        return None
    return float((s.iloc[-1] - prev) / prev * 100)


def pct_change_series_n(s: Optional[pd.Series], n: int) -> Optional[float]:
    """pct_change_vs_n for a bare Series instead of a DataFrame."""
    if s is None or s.empty:
        return None
    s = s.dropna()
    if len(s) < n + 1:
        return None
    prev = s.iloc[-(n + 1)]
    if prev == 0:
        return None
    return float((s.iloc[-1] - prev) / prev * 100)


def filter_series(s: Optional[pd.Series], n_days: int) -> Optional[pd.Series]:
    if s is None or (hasattr(s, "empty") and s.empty):
        return s
    return s.dropna().iloc[-n_days:]


# ─────────────────────────────────────────────
# BUILD KPI DICTS
# ─────────────────────────────────────────────

def get_global_kpis() -> list:
    kpis = []

    # CPI YoY
    cpi_yoy_df = fred_data.get("cpi_yoy", pd.DataFrame())
    if not cpi_yoy_df.empty:
        v = latest_val(cpi_yoy_df, "yoy")
        kpis.append(dict(label="US CPI YoY", value=fmt_pct(v),
                         c1d=None, c1w=change_vs_n(cpi_yoy_df, 1, "yoy"), unit="%", invert=True))

    # Core CPI YoY
    core_yoy_df = fred_data.get("core_cpi_yoy", pd.DataFrame())
    if not core_yoy_df.empty:
        v = latest_val(core_yoy_df, "yoy")
        kpis.append(dict(label="Core CPI YoY", value=fmt_pct(v),
                         c1d=None, c1w=change_vs_n(core_yoy_df, 1, "yoy"), unit="%", invert=True))

    # Fed Funds
    ff_df = fred_data.get("fed_funds", pd.DataFrame())
    if not ff_df.empty:
        v = latest_val(ff_df)
        kpis.append(dict(label="Fed Funds", value=fmt_pct(v),
                         c1d=None, c1w=change_vs_n(ff_df, 1), unit="%", invert=True))

    # 2Y Treasury
    t2y_df = fred_data.get("t2y", pd.DataFrame())
    if not t2y_df.empty:
        v = latest_val(t2y_df)
        kpis.append(dict(label="2Y Treasury", value=fmt_pct(v),
                         c1d=change_vs_n(t2y_df, 1), c1w=change_vs_n(t2y_df, 5), unit="%", invert=True))

    # 10Y Treasury
    t10y_df = fred_data.get("t10y", pd.DataFrame())
    if not t10y_df.empty:
        v = latest_val(t10y_df)
        kpis.append(dict(label="10Y Treasury", value=fmt_pct(v),
                         c1d=change_vs_n(t10y_df, 1), c1w=change_vs_n(t10y_df, 5), unit="%", invert=True))

    # 10Y Real Yield (TIPS)
    real_yield_df = fred_data.get("real_yield_10y", pd.DataFrame())
    if not real_yield_df.empty:
        v = latest_val(real_yield_df)
        kpis.append(dict(label="10Y Real Yield", value=fmt_pct(v),
                         c1d=change_vs_n(real_yield_df, 1), c1w=change_vs_n(real_yield_df, 5),
                         unit="%", invert=True))

    # 10Y–2Y Spread
    spread_df = fred_data.get("yield_spread", pd.DataFrame())
    if not spread_df.empty:
        v = latest_val(spread_df, "spread")
        kpis.append(dict(label="10Y–2Y Spread", value=fmt_pct(v),
                         c1d=change_vs_n(spread_df, 1, "spread"), c1w=change_vs_n(spread_df, 5, "spread"),
                         unit="%", invert=False))

    # 10Y Breakeven
    be_df = fred_data.get("breakeven_10y", pd.DataFrame())
    if not be_df.empty:
        v = latest_val(be_df)
        kpis.append(dict(label="10Y Breakeven", value=fmt_pct(v),
                         c1d=change_vs_n(be_df, 1), c1w=change_vs_n(be_df, 5), unit="%", invert=True))

    # Unemployment
    unemp_df = fred_data.get("unemployment", pd.DataFrame())
    if not unemp_df.empty:
        v = latest_val(unemp_df)
        kpis.append(dict(label="Unemployment", value=fmt_pct(v),
                         c1d=None, c1w=change_vs_n(unemp_df, 1), unit="%", invert=True))

    # S&P 500
    spx_df = market_data.get("spx", pd.DataFrame())
    if not spx_df.empty:
        v = latest_val(spx_df, "Close")
        kpis.append(dict(label="S&P 500", value=f"{v:,.0f}" if v else "—",
                         c1d=pct_change_vs_n(spx_df, 1), c1w=pct_change_vs_n(spx_df, 5),
                         unit="%", invert=False))

    # VIX
    vix_df = market_data.get("vix", pd.DataFrame())
    if not vix_df.empty:
        v = latest_val(vix_df, "Close")
        kpis.append(dict(label="VIX", value=fmt_val(v, 2),
                         c1d=pct_change_vs_n(vix_df, 1), c1w=pct_change_vs_n(vix_df, 5),
                         unit="%", invert=True))

    # Gold
    gold_df = market_data.get("gold", pd.DataFrame())
    if not gold_df.empty:
        v = latest_val(gold_df, "Close")
        kpis.append(dict(label="Gold (USD/oz)", value=f"${v:,.0f}" if v else "—",
                         c1d=pct_change_vs_n(gold_df, 1), c1w=pct_change_vs_n(gold_df, 5),
                         unit="%", invert=False))

    # WTI Crude Oil
    oil_df = market_data.get("oil_wti", pd.DataFrame())
    if not oil_df.empty:
        v = latest_val(oil_df, "Close")
        kpis.append(dict(label="WTI Crude (USD/bbl)", value=f"${v:.1f}" if v else "—",
                         c1d=pct_change_vs_n(oil_df, 1), c1w=pct_change_vs_n(oil_df, 5),
                         unit="%", invert=True))

    # DXY
    dxy_df = market_data.get("dxy", pd.DataFrame())
    if not dxy_df.empty:
        v = latest_val(dxy_df, "Close")
        kpis.append(dict(label="DXY", value=fmt_val(v, 2),
                         c1d=pct_change_vs_n(dxy_df, 1), c1w=pct_change_vs_n(dxy_df, 5),
                         unit="%", invert=False))

    # HYG (Credit proxy)
    hyg_price = market_data.get("hyg_price", pd.Series(dtype=float))
    if hyg_price is not None and not hyg_price.empty:
        v = float(hyg_price.iloc[-1])
        kpis.append(dict(label="HYG (Credit)", value=f"${v:.2f}",
                         c1d=pct_change_series_n(hyg_price, 1),
                         c1w=pct_change_series_n(hyg_price, 5),
                         unit="%", invert=False))

    return kpis


def get_india_kpis() -> list:
    kpis = []

    # NIFTY 50
    nifty_df = market_data.get("nifty", pd.DataFrame())
    if not nifty_df.empty:
        v = latest_val(nifty_df, "Close")
        kpis.append(dict(label="NIFTY 50", value=f"{v:,.0f}" if v else "—",
                         c1d=pct_change_vs_n(nifty_df, 1), c1w=pct_change_vs_n(nifty_df, 5),
                         unit="%", invert=False))

    # Nifty Bank
    nifty_bank_df = market_data.get("nifty_bank", pd.DataFrame())
    if not nifty_bank_df.empty:
        v = latest_val(nifty_bank_df, "Close")
        kpis.append(dict(label="Nifty Bank", value=f"{v:,.0f}" if v else "—",
                         c1d=pct_change_vs_n(nifty_bank_df, 1), c1w=pct_change_vs_n(nifty_bank_df, 5),
                         unit="%", invert=False))

    # India VIX
    ivix_df = market_data.get("india_vix", pd.DataFrame())
    if not ivix_df.empty:
        v = latest_val(ivix_df, "Close")
        kpis.append(dict(label="India VIX", value=fmt_val(v, 2),
                         c1d=pct_change_vs_n(ivix_df, 1), c1w=pct_change_vs_n(ivix_df, 5),
                         unit="%", invert=True))

    # USD/INR
    usdinr_df = market_data.get("usdinr", pd.DataFrame())
    if not usdinr_df.empty:
        v = latest_val(usdinr_df, "Close")
        kpis.append(dict(label="USD/INR", value=fmt_val(v, 2),
                         c1d=pct_change_vs_n(usdinr_df, 1), c1w=pct_change_vs_n(usdinr_df, 5),
                         unit="%", invert=True))

    # Brent Crude
    brent_df = market_data.get("brent", pd.DataFrame())
    if not brent_df.empty:
        v = latest_val(brent_df, "Close")
        kpis.append(dict(label="Brent (USD/bbl)", value=f"${v:.1f}" if v else "—",
                         c1d=pct_change_vs_n(brent_df, 1), c1w=pct_change_vs_n(brent_df, 5),
                         unit="%", invert=True))

    # India CPI YoY from FRED/CSV
    if not india_macro_df.empty and "cpi_yoy" in india_macro_df.columns:
        cpi_series = india_macro_df["cpi_yoy"].dropna()
        if not cpi_series.empty:
            v   = float(cpi_series.iloc[-1])
            c1w = float(cpi_series.iloc[-1] - cpi_series.iloc[-2]) if len(cpi_series) >= 2 else None
            kpis.append(dict(label="India CPI YoY", value=fmt_pct(v),
                             c1d=None, c1w=c1w, unit="%", invert=True))

    # RBI Repo Rate
    if not india_macro_df.empty and "repo_rate" in india_macro_df.columns:
        repo_series = india_macro_df["repo_rate"].dropna()
        if not repo_series.empty:
            v = float(repo_series.iloc[-1])
            kpis.append(dict(label="RBI Repo Rate", value=fmt_pct(v),
                             c1d=None, c1w=None, unit="%", invert=True))

    # India 10Y G-Sec
    india_gsec_df = fred_data.get("india_gsec_10y", pd.DataFrame())
    if not india_gsec_df.empty:
        v = latest_val(india_gsec_df)
        kpis.append(dict(label="India 10Y G-Sec", value=fmt_pct(v),
                         c1d=None, c1w=change_vs_n(india_gsec_df, 1), unit="%", invert=True))

    return kpis


def render_kpi_grid(kpis: list, cols_per_row: int = 4) -> None:
    for i in range(0, len(kpis), cols_per_row):
        row  = kpis[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, k in zip(cols, row):
            with col:
                kpi_card(
                    label=k["label"], value=k["value"],
                    change_1d=k.get("c1d"), change_1w=k.get("c1w"),
                    unit=k.get("unit", ""), invert_color=k.get("invert", False),
                )


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_overview, tab_inflation, tab_risk, tab_commodities, tab_stress, tab_alerts = st.tabs([
    "📊  Overview",
    "🔥  Inflation & Rates",
    "⚡  Risk & Cross-Asset",
    "🛢️  Commodities",
    "🧪  Stress Lab",
    "🚨  Alerts",
])


# ══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════

with tab_overview:

    # ── Regime Banners ──
    if region in ("Global", "Both"):
        global_regime = compute_global_regime(fred_data, market_data)
        regime_badge(global_regime, title="🇺🇸 Global Macro Regime")

    if region in ("India", "Both"):
        india_regime = compute_india_regime(market_data, india_macro_df)
        regime_badge(india_regime, title="🇮🇳 India Macro Regime")

    st.markdown("")

    # ── KPI Cards ──
    if region == "Both":
        col_g, col_i = st.columns(2)
        with col_g:
            section_header("🇺🇸 Global KPIs")
            render_kpi_grid(get_global_kpis(), cols_per_row=2)
        with col_i:
            section_header("🇮🇳 India KPIs")
            render_kpi_grid(get_india_kpis(), cols_per_row=2)
    elif region == "Global":
        section_header("🇺🇸 Global KPIs")
        render_kpi_grid(get_global_kpis(), cols_per_row=4)
    else:
        section_header("🇮🇳 India KPIs")
        render_kpi_grid(get_india_kpis(), cols_per_row=3)

    st.markdown("---")

    # ── What Changed Feed ──
    section_header("💡 What Changed", subtitle="Auto-generated insights from latest data")
    insights = compute_insights_feed(fred_data, market_data, india_macro_df)
    if insights:
        for insight in insights:
            st.html(f'<div class="insight-item">{insight}</div>')
    else:
        st.info("No insights available — check data sources.")


# ══════════════════════════════════════════════
# TAB 2: INFLATION & RATES
# ══════════════════════════════════════════════

with tab_inflation:

    if region in ("Global", "Both"):
        section_header("🇺🇸 US Inflation", subtitle="CPI YoY and Core CPI YoY")

        cpi_yoy_df  = fred_data.get("cpi_yoy", pd.DataFrame())
        core_yoy_df = fred_data.get("core_cpi_yoy", pd.DataFrame())
        cpi_yoy_s   = filter_series(cpi_yoy_df["yoy"], N_DAYS * 30) if not cpi_yoy_df.empty else None
        core_yoy_s  = filter_series(core_yoy_df["yoy"], N_DAYS * 30) if not core_yoy_df.empty else None

        if cpi_yoy_s is not None or core_yoy_s is not None:
            series = {}
            if cpi_yoy_s is not None:
                series["CPI YoY (%)"] = cpi_yoy_s
            if core_yoy_s is not None:
                series["Core CPI YoY (%)"] = core_yoy_s
            fig = line_chart(series, title="US Inflation (YoY %)", yaxis_title="%", height=300)
            fig.add_hline(y=2.0, line_dash="dot", line_color="#475569",
                          annotation_text="2% target", annotation_font_color="#475569")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ US Inflation data unavailable (FRED API key required).")

        section_header("🇺🇸 US Treasury Yields & Real Rates",
                       subtitle="Nominal yields, real yield (TIPS), breakeven, and spread")

        t2y_df        = fred_data.get("t2y", pd.DataFrame())
        t10y_df       = fred_data.get("t10y", pd.DataFrame())
        spread_df     = fred_data.get("yield_spread", pd.DataFrame())
        be_df         = fred_data.get("breakeven_10y", pd.DataFrame())
        real_yield_df = fred_data.get("real_yield_10y", pd.DataFrame())

        yields_series = {}
        spread_series = {}
        if not t2y_df.empty:
            yields_series["2Y Treasury (%)"]   = filter_series(t2y_df["value"], N_DAYS * 5)
        if not t10y_df.empty:
            yields_series["10Y Treasury (%)"]  = filter_series(t10y_df["value"], N_DAYS * 5)
        if not be_df.empty:
            yields_series["10Y Breakeven (%)"] = filter_series(be_df["value"], N_DAYS * 5)
        if not real_yield_df.empty:
            yields_series["10Y Real Yield (%)"] = filter_series(real_yield_df["value"], N_DAYS * 5)
        if not spread_df.empty:
            spread_series["10Y–2Y Spread (%)"] = filter_series(spread_df["spread"], N_DAYS * 5)

        if yields_series:
            fig = line_chart(
                yields_series,
                secondary_series=spread_series if spread_series else None,
                secondary_yaxis_title="Spread (%)",
                title="US Treasury Yields + Real Yield",
                yaxis_title="%",
                height=340,
            )
            if spread_series:
                fig.add_hline(y=0, line_dash="dash", line_color="#EF4444",
                              annotation_text="Inversion", annotation_font_color="#EF4444",
                              secondary_y=True)
            # Zero line for real yield
            fig.add_hline(y=0, line_dash="dot", line_color="#334155")
            st.plotly_chart(fig, use_container_width=True)

    if region in ("India", "Both"):
        section_header("🇮🇳 India Macro", subtitle="CPI YoY, Repo Rate & 10Y G-Sec Yield")

        if not india_macro_df.empty:
            india_gsec_df  = fred_data.get("india_gsec_10y", pd.DataFrame())
            secondary_dict = {"Repo Rate (%)": india_macro_df.get("repo_rate")}
            if not india_gsec_df.empty:
                secondary_dict["India 10Y G-Sec (%)"] = filter_series(india_gsec_df["value"], N_DAYS * 30)

            fig_india = line_chart(
                {"India CPI YoY (%)": india_macro_df.get("cpi_yoy")},
                secondary_series=secondary_dict,
                secondary_yaxis_title="Rate (%)",
                title="India Inflation, Repo Rate & G-Sec Yield",
                yaxis_title="CPI YoY (%)",
                height=320,
            )
            st.plotly_chart(fig_india, use_container_width=True)
        else:
            st.warning("⚠️ India macro CSV data not available.")


# ══════════════════════════════════════════════
# TAB 3: RISK & CROSS-ASSET
# ══════════════════════════════════════════════

with tab_risk:

    if region in ("Global", "Both"):
        section_header("🇺🇸 US Equities & Volatility")

        col1, col2 = st.columns(2)
        with col1:
            spx_df = market_data.get("spx", pd.DataFrame())
            if not spx_df.empty:
                spx_s  = filter_series(spx_df["Close"], N_DAYS)
                ma50   = spx_df["Close"].rolling(50).mean()
                ma200  = spx_df["Close"].rolling(200).mean()
                fig = line_chart(
                    {
                        "S&P 500":  spx_s,
                        "50D MA":   filter_series(ma50, N_DAYS),
                        "200D MA":  filter_series(ma200, N_DAYS),
                    },
                    title="S&P 500 + Moving Averages", yaxis_title="Price", height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("SPX data unavailable.")

        with col2:
            vix_df = market_data.get("vix", pd.DataFrame())
            if not vix_df.empty:
                vix_s = filter_series(vix_df["Close"], N_DAYS)
                fig = line_chart({"VIX": vix_s}, title="VIX — Fear Index", yaxis_title="VIX", height=300)
                fig.add_hrect(
                    y0=30, y1=vix_s.max() * 1.05 if vix_s is not None and not vix_s.empty else 80,
                    fillcolor="rgba(239,68,68,0.07)", line_width=0,
                    annotation_text="Extreme Fear", annotation_font_color="#EF4444",
                )
                fig.add_hline(y=20, line_dash="dot", line_color="#F59E0B",
                              annotation_text="Elevated", annotation_font_color="#F59E0B")
                st.plotly_chart(fig, use_container_width=True)

        # HYG Credit Proxy
        section_header("💳 HYG — High Yield Credit Proxy", subtitle="Falling HYG = widening credit spreads = stress")
        hyg_price = market_data.get("hyg_price", pd.Series(dtype=float))
        hyg_vol   = market_data.get("hyg_vol20d")
        if hyg_price is not None and not hyg_price.empty:
            fig = line_chart(
                {"HYG Price": filter_series(hyg_price, N_DAYS)},
                secondary_series={"20D Vol (ann.)": filter_series(hyg_vol, N_DAYS)} if hyg_vol is not None else None,
                secondary_yaxis_title="Annualized Vol",
                title="HYG — iShares HY Bond ETF",
                yaxis_title="Price (USD)",
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("HYG data unavailable.")

        section_header("💵 DXY — US Dollar Index")
        dxy_df  = market_data.get("dxy", pd.DataFrame())
        dxy_vol = market_data.get("dxy_vol20d")
        if not dxy_df.empty:
            fig = line_chart(
                {"DXY": filter_series(dxy_df["Close"], N_DAYS)},
                secondary_series={"20D Vol (ann.)": filter_series(dxy_vol, N_DAYS)} if dxy_vol is not None else None,
                secondary_yaxis_title="Annualized Vol",
                title="DXY — US Dollar Index",
                yaxis_title="DXY",
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

    if region in ("India", "Both"):
        section_header("🇮🇳 India Equities & Volatility")

        col1, col2 = st.columns(2)
        with col1:
            nifty_df = market_data.get("nifty", pd.DataFrame())
            if not nifty_df.empty:
                nifty_s = filter_series(nifty_df["Close"], N_DAYS)
                ma50_n  = nifty_df["Close"].rolling(50).mean()
                fig = line_chart(
                    {"NIFTY 50": nifty_s, "50D MA": filter_series(ma50_n, N_DAYS)},
                    title="NIFTY 50", yaxis_title="Price", height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            ivix_df = market_data.get("india_vix", pd.DataFrame())
            if not ivix_df.empty:
                fig = line_chart(
                    {"India VIX": filter_series(ivix_df["Close"], N_DAYS)},
                    title="India VIX", yaxis_title="India VIX", height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Nifty Bank
        section_header("🏦 Nifty Bank vs Nifty 50", subtitle="Financial sector relative strength — credit cycle indicator")
        nifty_bank_df = market_data.get("nifty_bank", pd.DataFrame())
        if not nifty_bank_df.empty and not nifty_df.empty:
            nb_s  = filter_series(nifty_bank_df["Close"], N_DAYS)
            nif_s = filter_series(nifty_df["Close"], N_DAYS)
            # Normalize to 100 at start for easy comparison
            nb_norm  = (nb_s  / nb_s.iloc[0]  * 100) if nb_s  is not None and not nb_s.empty  else None
            nif_norm = (nif_s / nif_s.iloc[0] * 100) if nif_s is not None and not nif_s.empty else None
            fig = line_chart(
                {"Nifty Bank (rebased)": nb_norm, "NIFTY 50 (rebased)": nif_norm},
                title="Nifty Bank vs Nifty 50 (rebased to 100)",
                yaxis_title="Rebased", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nifty Bank data unavailable.")

        section_header("₹ USD/INR Exchange Rate")
        usdinr_df = market_data.get("usdinr", pd.DataFrame())
        if not usdinr_df.empty:
            inr_vol = market_data.get("usdinr_vol20d")
            fig = line_chart(
                {"USD/INR": filter_series(usdinr_df["Close"], N_DAYS)},
                secondary_series={"20D Vol (ann.)": filter_series(inr_vol, N_DAYS)} if inr_vol is not None else None,
                secondary_yaxis_title="Annualized Vol",
                title="USD/INR", yaxis_title="USD/INR", height=260,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4: COMMODITIES
# ══════════════════════════════════════════════

with tab_commodities:

    section_header("🥇 Gold & Real Rates", subtitle="Safe-haven dynamics + 10Y Real Yield (true gold driver)")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gold_df  = market_data.get("gold", pd.DataFrame())
        gold_vol = market_data.get("gold_vol20d")
        if not gold_df.empty:
            fig = line_chart(
                {"Gold (USD/oz)": filter_series(gold_df["Close"], N_DAYS)},
                secondary_series={"20D Vol (ann.)": filter_series(gold_vol, N_DAYS)} if gold_vol is not None else None,
                secondary_yaxis_title="Vol",
                title="Gold Price", yaxis_title="USD/oz", height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Gold data unavailable.")

    with col_g2:
        real_yield_df = fred_data.get("real_yield_10y", pd.DataFrame())
        if not real_yield_df.empty:
            ry_s = filter_series(real_yield_df["value"], N_DAYS * 5)
            fig = line_chart(
                {"10Y Real Yield (%)": ry_s},
                title="10Y TIPS Real Yield", yaxis_title="%", height=300,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#EF4444",
                          annotation_text="Zero real rate", annotation_font_color="#EF4444")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Real yield data unavailable (FRED key required).")

    st.markdown("---")
    section_header("🛢️ WTI & Brent Crude", subtitle="Global growth proxy + India CAD driver")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        oil_df  = market_data.get("oil_wti", pd.DataFrame())
        oil_vol = market_data.get("oil_wti_vol20d")
        if not oil_df.empty:
            fig = line_chart(
                {"WTI Crude (USD/bbl)": filter_series(oil_df["Close"], N_DAYS)},
                secondary_series={"20D Vol (ann.)": filter_series(oil_vol, N_DAYS)} if oil_vol is not None else None,
                secondary_yaxis_title="Vol",
                title="WTI Crude Oil", yaxis_title="USD/bbl", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("WTI Crude data unavailable.")

    with col_o2:
        brent_df  = market_data.get("brent", pd.DataFrame())
        brent_vol = market_data.get("brent_vol20d")
        if not brent_df.empty:
            fig = line_chart(
                {"Brent Crude (USD/bbl)": filter_series(brent_df["Close"], N_DAYS)},
                secondary_series={"20D Vol (ann.)": filter_series(brent_vol, N_DAYS)} if brent_vol is not None else None,
                secondary_yaxis_title="Vol",
                title="Brent Crude (India CAD driver)", yaxis_title="USD/bbl", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Brent Crude data unavailable.")

    st.markdown("---")
    section_header("🔴 Copper & Copper/Gold Ratio", subtitle="Dr. Copper = growth barometer. Ratio vs Gold = growth vs fear")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        copper_df = market_data.get("copper", pd.DataFrame())
        if not copper_df.empty:
            fig = line_chart(
                {"Copper (USD/lb)": filter_series(copper_df["Close"], N_DAYS)},
                title="Copper Futures", yaxis_title="USD/lb", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Copper data unavailable.")

    with col_c2:
        cg_ratio = market_data.get("copper_gold_ratio", pd.Series(dtype=float))
        if cg_ratio is not None and not cg_ratio.empty:
            fig = line_chart(
                {"Copper/Gold Ratio": filter_series(cg_ratio, N_DAYS)},
                title="Copper/Gold Ratio (growth vs fear)", yaxis_title="Ratio", height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Copper/Gold ratio unavailable.")

    # Commodity heatmap summary
    st.markdown("---")
    section_header("📋 Commodity Summary Table", subtitle="5D performance across all tracked commodities")

    commodity_summary = []
    for name, label, df_key, col in [
        ("Gold",    "🥇 Gold (USD/oz)",     "gold",    "Close"),
        ("WTI",     "🛢️ WTI Crude ($/bbl)", "oil_wti", "Close"),
        ("Brent",   "🛢️ Brent ($/bbl)",     "brent",   "Close"),
        ("Copper",  "🔴 Copper ($/lb)",     "copper",  "Close"),
    ]:
        df = market_data.get(df_key, pd.DataFrame())
        if not df.empty and col in df.columns:
            s   = df[col].dropna()
            val = float(s.iloc[-1]) if not s.empty else None
            p5d = (float((s.iloc[-1] / s.iloc[-6] - 1) * 100) if len(s) > 6 else None)
            p1m = (float((s.iloc[-1] / s.iloc[-22] - 1) * 100) if len(s) > 22 else None)
            commodity_summary.append({
                "Asset":    label,
                "Price":    f"{val:,.2f}" if val else "—",
                "5D %":     f"{p5d:+.2f}%" if p5d is not None else "—",
                "1M %":     f"{p1m:+.2f}%" if p1m is not None else "—",
            })

    if commodity_summary:
        st.dataframe(
            pd.DataFrame(commodity_summary).set_index("Asset"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════
# TAB 5: STRESS LAB
# ══════════════════════════════════════════════

with tab_stress:
    section_header("🧪 Stress Lab", subtitle="Scenario analysis — deterministic rule-based shocks")

    st.html("""
    <div style="background:#111827;border:1px solid #1E293B;border-radius:8px;
                padding:14px 18px;margin-bottom:20px;font-family:monospace;
                font-size:0.8rem;color:#64748B;line-height:1.7">
        ⚠️ Stress scenarios are estimated using rolling OLS beta regressions.
        These are analytical approximations — not financial forecasts.
    </div>
    """)

    # ── Scenario 1: 10Y yield shock → SPX impact ──
    section_header("Scenario 1: Rate Shock → SPX Impact")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        yield_shock_bps = st.slider(
            "Shock to 10Y Yield (bps)",
            min_value=-200, max_value=200, value=50, step=10,
            help="Parallel shift in 10-year US Treasury yield"
        )

    spx_df  = market_data.get("spx", pd.DataFrame())
    t10y_df = fred_data.get("t10y", pd.DataFrame())

    with col_s2:
        if not spx_df.empty and not t10y_df.empty:
            spx_returns  = market_data.get("spx_returns", pd.Series())
            t10y_daily   = t10y_df["value"].resample("B").last().ffill()
            t10y_changes = t10y_daily.diff().dropna()
            aligned      = pd.concat([spx_returns, t10y_changes], axis=1).dropna()
            aligned.columns = ["spx_ret", "t10y_chg"]
            last_252 = aligned.iloc[-252:] if len(aligned) >= 252 else aligned

            if len(last_252) >= 30:
                slope, intercept, r, p, se = stats.linregress(last_252["t10y_chg"], last_252["spx_ret"])
                shock_pct  = float(yield_shock_bps) / 100.0
                spx_impact = slope * shock_pct * 100
                r_sq       = r ** 2
                color      = "#10B981" if spx_impact > 0 else "#EF4444"
                st.html(f"""
                <div style="background:#0A0E1A;border:1px solid #1E293B;border-radius:8px;padding:18px 24px">
                    <div style="color:#64748B;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px">Estimated SPX Impact</div>
                    <div style="font-size:2rem;font-weight:700;font-family:monospace;color:{color}">{spx_impact:+.2f}%</div>
                    <div style="color:#475569;font-size:0.78rem;margin-top:8px;font-family:monospace">
                        Beta (SPX/10Y): {slope:.2f} | R²: {r_sq:.3f} | N: {len(last_252)} obs
                    </div>
                    <div style="color:#334155;font-size:0.75rem;margin-top:4px">
                        Based on 1Y rolling OLS regression of daily SPX returns vs 10Y yield Δ
                    </div>
                </div>
                """)
            else:
                st.warning("Insufficient data for regression.")
        else:
            st.warning("⚠️ SPX or 10Y data unavailable for stress calculation.")

    st.markdown("---")

    # ── Scenario 2: Oil Shock → India Macro Impact ──
    section_header("Scenario 2: Oil Shock → India CAD & INR Impact")

    col_o1, col_o2 = st.columns([1, 2])
    with col_o1:
        oil_shock_pct = st.slider(
            "Brent Crude Shock (%)",
            min_value=-30.0, max_value=50.0, value=10.0, step=2.5,
            help="Instantaneous % change in Brent Crude price"
        )

    brent_df  = market_data.get("brent", pd.DataFrame())
    usdinr_df = market_data.get("usdinr", pd.DataFrame())

    with col_o2:
        if not brent_df.empty and not usdinr_df.empty:
            brent_now = float(brent_df["Close"].dropna().iloc[-1])
            inr_now   = float(usdinr_df["Close"].dropna().iloc[-1])
            shocked_brent = brent_now * (1 + oil_shock_pct / 100)
            # India crude import bill ≈ 4.5M bpd; rough CAD sensitivity: ~$1/bbl Brent = ~$1.6B/yr CAD impact
            brent_delta   = shocked_brent - brent_now
            cad_impact_bn = brent_delta * 1.6   # $Billion/yr
            # Empirical: ~$10/bbl Brent = ~0.4% USD/INR depreciation
            inr_impact_pct = (oil_shock_pct / 10.0) * 0.4
            shocked_inr   = inr_now * (1 + inr_impact_pct / 100)
            stress_flag   = oil_shock_pct > 15 or oil_shock_pct < -15
            color = "#EF4444" if oil_shock_pct > 0 else "#10B981"
            flag_text = "🔴 HIGH STRESS — Significant CAD pressure" if oil_shock_pct > 15 else \
                        ("🟢 POSITIVE — CAD relief, INR tailwind" if oil_shock_pct < -15 else "🟡 MODERATE IMPACT")

            st.html(f"""
            <div style="background:#0A0E1A;border:1px solid #1E293B;border-radius:8px;padding:18px 24px">
                <div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
                    <div>
                        <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Brent: Current → Shocked</div>
                        <div style="font-size:1.3rem;font-weight:700;font-family:monospace;color:#E2E8F0">
                            ${brent_now:.1f} → <span style="color:{color}">${shocked_brent:.1f}</span>
                        </div>
                    </div>
                    <div>
                        <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">CAD Impact (ann.)</div>
                        <div style="font-size:1.3rem;font-weight:700;font-family:monospace;color:{color}">
                            {'+' if cad_impact_bn > 0 else ''}${cad_impact_bn:.1f}B
                        </div>
                    </div>
                    <div>
                        <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">USD/INR Impact</div>
                        <div style="font-size:1.3rem;font-weight:700;font-family:monospace;color:{color}">
                            {inr_now:.2f} → {shocked_inr:.2f}
                        </div>
                    </div>
                </div>
                <div style="margin-top:14px;padding:8px 14px;background:{color}22;border:1px solid {color}44;
                            border-radius:6px;font-family:monospace;font-size:0.85rem;color:{color};font-weight:600">
                    {flag_text}
                </div>
                <div style="color:#334155;font-size:0.73rem;margin-top:10px;font-family:monospace">
                    CAD sensitivity: ~$1.6B/yr per $1/bbl Brent (India ~4.5M bpd import) |
                    INR rule-of-thumb: ~0.4% depreciation per +10% Brent
                </div>
            </div>
            """)
        else:
            st.warning("⚠️ Brent or USD/INR data unavailable.")

    st.markdown("---")

    # ── Scenario 3: USD/INR FX Shock ──
    section_header("Scenario 3: FX Shock → INR Stress")

    col_s3, col_s4 = st.columns([1, 2])
    with col_s3:
        inr_shock_pct = st.slider(
            "USD/INR Shock (%)",
            min_value=-5.0, max_value=10.0, value=2.0, step=0.5,
            help="Instantaneous depreciation (+) or appreciation (-) of USD/INR"
        )

    usdinr_df = market_data.get("usdinr", pd.DataFrame())
    inr_vol   = market_data.get("usdinr_vol20d")

    with col_s4:
        if not usdinr_df.empty and "Close" in usdinr_df.columns:
            current_rate = float(usdinr_df["Close"].dropna().iloc[-1])
            shocked_rate = current_rate * (1 + inr_shock_pct / 100)
            current_vol  = float(inr_vol.dropna().iloc[-1]) * 100 if inr_vol is not None and not inr_vol.empty else None
            stress_flag  = inr_shock_pct > 1.5 or (current_vol and current_vol > 8)
            color        = "#EF4444" if stress_flag else "#10B981"
            flag_text    = "🔴 FX STRESS FLAG ACTIVE" if stress_flag else "🟢 Within Normal Range"

            st.html(f"""
            <div style="background:#0A0E1A;border:1px solid #1E293B;border-radius:8px;padding:18px 24px">
                <div style="display:flex;gap:24px;align-items:flex-start">
                    <div>
                        <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Current Rate</div>
                        <div style="font-size:1.5rem;font-weight:700;font-family:monospace;color:#E2E8F0">{current_rate:.2f}</div>
                    </div>
                    <div style="color:#475569;font-size:1.2rem;padding-top:12px">→</div>
                    <div>
                        <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Shocked Rate</div>
                        <div style="font-size:1.5rem;font-weight:700;font-family:monospace;color:{color}">{shocked_rate:.2f}</div>
                    </div>
                    <div>
                        <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">20D Ann. Vol</div>
                        <div style="font-size:1.5rem;font-weight:700;font-family:monospace;color:#E2E8F0">{f"{current_vol:.2f}%" if current_vol else "—"}</div>
                    </div>
                </div>
                <div style="margin-top:14px;padding:8px 14px;background:{color}22;border:1px solid {color}44;border-radius:6px;
                            font-family:monospace;font-size:0.85rem;color:{color};font-weight:600">{flag_text}</div>
            </div>
            """)
        else:
            st.warning("⚠️ USD/INR data unavailable.")

    st.markdown("---")

    # ── Rolling Beta Chart ──
    section_header("Rolling 1Y Beta: SPX vs 10Y Yield", subtitle="How sensitive is SPX to rate changes?")
    if not spx_df.empty and not t10y_df.empty:
        spx_returns = market_data.get("spx_returns", pd.Series(dtype=float))
        t10y_daily  = t10y_df["value"].resample("B").last().ffill()
        t10y_ch     = t10y_daily.diff().dropna()
        aligned     = pd.concat([spx_returns, t10y_ch], axis=1).dropna()
        aligned.columns = ["spx", "t10y"]
        if len(aligned) >= 252:
            window = 126
            betas, dates = [], []
            for i in range(window, len(aligned)):
                sub = aligned.iloc[i - window:i]
                s, _, _, _, _ = stats.linregress(sub["t10y"], sub["spx"])
                betas.append(s)
                dates.append(aligned.index[i])
            beta_s = pd.Series(betas, index=dates)
            fig = line_chart(
                {"SPX/10Y Beta (126D rolling)": filter_series(beta_s, N_DAYS)},
                title="Rolling SPX β to 10Y Yield Changes",
                yaxis_title="Beta", height=260,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#475569")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 6: ALERTS
# ══════════════════════════════════════════════

with tab_alerts:
    section_header("🚨 Alert Monitor", subtitle="Threshold & percentile-based signal alerts")

    alerts = compute_alerts(fred_data, market_data)

    if alerts:
        for alert in alerts:
            status = alert["Status"]
            if "BREACH" in status or "INVERTED" in status or "HIGH" in status:
                border = "#EF4444"
            elif "WATCH" in status or "FLAT" in status or "DRAWDOWN" in status:
                border = "#F59E0B"
            else:
                border = "#10B981"

            st.html(f"""
            <div style="
                background:#111827;border-left:4px solid {border};
                border-radius:0 8px 8px 0;padding:12px 18px;margin-bottom:8px;
                display:flex;gap:24px;align-items:center;flex-wrap:wrap
            ">
                <div style="min-width:200px">
                    <div style="color:#94A3B8;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px">Alert</div>
                    <div style="color:#E2E8F0;font-family:monospace;font-size:0.9rem;font-weight:600">{alert['Alert']}</div>
                </div>
                <div style="min-width:140px">
                    <div style="color:#94A3B8;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px">Status</div>
                    <div style="font-family:monospace;font-size:0.85rem;font-weight:600">{alert['Status']}</div>
                </div>
                <div style="min-width:130px">
                    <div style="color:#94A3B8;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px">Latest</div>
                    <div style="color:#CBD5E1;font-family:monospace;font-size:0.85rem">{alert['Latest']}</div>
                </div>
                <div style="min-width:130px">
                    <div style="color:#94A3B8;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px">Threshold</div>
                    <div style="color:#CBD5E1;font-family:monospace;font-size:0.85rem">{alert['Threshold']}</div>
                </div>
                <div style="flex:1;min-width:200px">
                    <div style="color:#94A3B8;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px">Interpretation</div>
                    <div style="color:#94A3B8;font-size:0.82rem">{alert['Interpretation']}</div>
                </div>
            </div>
            """)
    else:
        st.info("No alerts available — check data sources.")

    # Alert summary counters
    if alerts:
        n_breach = sum(1 for a in alerts if "BREACH" in a["Status"] or "INVERTED" in a["Status"] or "HIGH" in a["Status"])
        n_watch  = sum(1 for a in alerts if "WATCH"  in a["Status"] or "FLAT"     in a["Status"] or "DRAWDOWN" in a["Status"])
        n_ok     = len(alerts) - n_breach - n_watch

        st.html(f"""
        <div style="
            background:#111827;border:1px solid #1E293B;border-radius:8px;
            padding:14px 24px;margin-top:16px;display:flex;gap:32px
        ">
            <div>
                <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase">Breach</div>
                <div style="color:#EF4444;font-size:1.6rem;font-weight:700;font-family:monospace">{n_breach}</div>
            </div>
            <div>
                <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase">Watch</div>
                <div style="color:#F59E0B;font-size:1.6rem;font-weight:700;font-family:monospace">{n_watch}</div>
            </div>
            <div>
                <div style="color:#64748B;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase">OK</div>
                <div style="color:#10B981;font-size:1.6rem;font-weight:700;font-family:monospace">{n_ok}</div>
            </div>
            <div style="margin-left:auto;display:flex;align-items:center">
                <div style="color:#475569;font-size:0.75rem;font-family:monospace">
                    {len(alerts)} total signals monitored
                </div>
            </div>
        </div>
        """)