# 📡 MacroTerminal
### Global + India Macro Intelligence Dashboard

A dark, terminal-style macroeconomic dashboard built with **Python + Streamlit** that provides real-time macro regime classification, cross-asset risk signals, stress testing, and automated insights across US (Global) and India markets.

Designed as a portfolio-grade analytics product — combining data engineering, financial analysis, and clean UI design.

---

## 🔥 Live Features

### 🖥 Terminal-Style Overview
- Real-time KPI cards (CPI, Rates, Yields, SPX, VIX, DXY, NIFTY, USD/INR)
- Yield curve spread (10Y–2Y) monitoring
- Region toggle: **Global / India / Both**
- Dark, finance-terminal aesthetic

### 🧠 Regime Engine
Rule-based macro classification across four dimensions:
- **Inflation Momentum** — 3M vs 12M CPI trend
- **Rates Impulse** — 2Y yield slope / policy rate change
- **Risk Regime** — VIX percentile + equity trend vs moving averages
- **FX Stress Detection** — INR momentum + 20D annualized volatility

Example output:
```
Disinflation • Tight Rates • Risk-Off
```

### 📊 Inflation & Rates
- US CPI & Core CPI (YoY %)
- Fed Funds Rate
- 2Y / 10Y Treasury Yields
- Yield Curve Spread (10Y–2Y)
- India CPI YoY & RBI Repo Rate (live via FRED, CSV fallback)

### 🌍 Risk & Cross-Asset
- SPX vs VIX vs DXY
- NIFTY vs India VIX vs USD/INR
- Rolling volatility (20D annualized)
- 50D / 200D moving average trend signals

### 🧪 Stress Lab
Interactive scenario testing:
- **+N bps rate shock** → Estimated SPX impact via rolling OLS beta regression
- **+N% INR shock** → FX stress flag + live volatility display
- Rolling beta chart (SPX sensitivity to 10Y yield changes)

### 🚨 Alert Engine
Color-coded alerts (🔴 Breach / ⚠️ Watch / 🟢 OK) for:
- Yield curve inversion (10Y–2Y < 0)
- VIX percentile breaches (>80th pct of 1Y)
- DXY 5-day momentum spikes
- India VIX percentile elevation
- USD/INR 5-day depreciation
- NIFTY drawdown from 1-month high

All alerts are rule-based and derived directly from live data — no manual inputs.

---

## 🏗 Architecture

```
MacroTerminal/
├── app.py                        # Main app — orchestrates all 5 tabs
├── requirements.txt
├── README.md
├── setup_key.py                  # One-click FRED key setup helper
├── .streamlit/
│   ├── config.toml               # Dark terminal theme
│   └── secrets.toml.example      # Key template (never commit real key)
├── core/
│   ├── data_fred.py              # FRED API fetcher, YoY/spread transforms
│   ├── data_markets.py           # yfinance OHLCV, returns, vol, beta
│   ├── india_macro.py            # Live FRED fetch for India + CSV fallback
│   ├── regime.py                 # Regime classifier, insights feed, alerts
│   └── ui_components.py          # KPI cards, badges, Plotly chart theme
└── data/
    └── india_macro.csv           # Fallback static data (if no FRED key)
```

**Core design principles:**
- Modular data pipelines with clean separation of concerns
- Cached API layers (`st.cache_data`) — FRED 6h, markets 1h
- Graceful degradation — app never crashes if a data source fails
- Deployment-ready for Streamlit Community Cloud

---

## 📡 Data Sources

| Source | Data | Frequency | Notes |
|--------|------|-----------|-------|
| [FRED API](https://fred.stlouisfed.org) | US CPI, Core CPI, Fed Funds, 2Y/10Y, Breakeven, Unemployment | Monthly / Daily | Free key required |
| [FRED API](https://fred.stlouisfed.org) | India CPI (`INDCPIALLMINMEI`), Repo Rate (`INTDSRINM193N`) | Monthly | Same key, live data |
| [yfinance](https://github.com/ranaroussi/yfinance) | SPX, VIX, DXY, Gold, NIFTY, India VIX, USD/INR | Daily | No key required |
| Local CSV | India CPI YoY + Repo Rate | Static (Feb 2025) | Auto-used if no FRED key |

**Caching:**
- FRED data: 6-hour cache
- Market data: 1-hour cache
- Manual refresh available via sidebar button

---

## 📈 Why This Project

MacroTerminal demonstrates end-to-end applied finance + data engineering:

- **Financial data ingestion** — REST API integration with retry logic and caching
- **Time-series transformations** — YoY computation, spreads, rolling stats, drawdowns
- **Cross-asset analytics** — correlating equity, rates, FX, and volatility regimes
- **Regime classification logic** — deterministic rule engine across multiple macro signals
- **Risk detection systems** — percentile-based alert thresholds with live data
- **Interactive scenario modeling** — OLS regression-based stress testing
- **Production deployment** — Streamlit Cloud with secrets management and graceful fallbacks

It bridges quantitative finance, data engineering, and product UI design into a single deployable system.

---

## 📌 Roadmap

- [ ] Automated India macro ingestion (live RBI data feed)
- [ ] Macro calendar integration (CPI release / FOMC meeting markers on charts)
- [ ] Daily auto-generated macro brief (PDF export)
- [ ] Composite Macro Risk Index (0–100 scoring model)
- [ ] Backtesting: regime labels → forward asset performance
- [ ] Email/Slack alert delivery for threshold breaches

---

## ⚠ Disclaimers

- `yfinance` is an unofficial market data wrapper and may occasionally return stale or missing data.
- India macro data falls back to a bundled static CSV (Feb 2025) when no FRED key is present.
- Stress scenarios use simplified OLS regression models and are illustrative only — not financial forecasts.
- Regime labels are rule-based heuristics, not machine-learning driven.
- This project is for educational and research purposes only. Nothing here constitutes financial advice.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.
