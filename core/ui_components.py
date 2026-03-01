"""
core/ui_components.py
Reusable UI components: KPI cards, regime badges, chart helpers.
"""

from typing import Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────

def fmt_val(val: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    """Format a float value with optional suffix. Returns '—' for None/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:,.{decimals}f}{suffix}"


def fmt_pct(val: Optional[float], decimals: int = 2) -> str:
    """Format value as percentage string."""
    return fmt_val(val, decimals, "%")


def arrow_indicator(change: Optional[float], threshold: float = 0.01) -> str:
    """Return arrow emoji based on change direction."""
    if change is None or (isinstance(change, float) and pd.isna(change)):
        return "➡️"
    if change > threshold:
        return "▲"
    elif change < -threshold:
        return "▼"
    return "➡️"


def change_color(change: Optional[float], invert: bool = False) -> str:
    """Return CSS color class for positive/negative change."""
    if change is None or (isinstance(change, float) and pd.isna(change)):
        return "#94A3B8"
    positive = change > 0
    if invert:
        positive = not positive
    return "#10B981" if positive else "#EF4444"


# ─────────────────────────────────────────────
# KPI CARD
# ─────────────────────────────────────────────

def kpi_card(
    label: str,
    value: str,
    change_1d: Optional[float] = None,
    change_1w: Optional[float] = None,
    unit: str = "",
    invert_color: bool = False,
) -> None:
    """
    Render a styled KPI card using HTML.

    Args:
        label: Metric name.
        value: Formatted current value string.
        change_1d: Absolute or percentage change over 1 day.
        change_1w: Absolute or percentage change over 1 week.
        unit: Optional unit appended to changes.
        invert_color: If True, positive change shows red (e.g., for VIX, inflation).
    """
    c1d_str = ""
    c1w_str = ""

    if change_1d is not None and not pd.isna(change_1d):
        color = change_color(change_1d, invert=invert_color)
        arrow = arrow_indicator(change_1d)
        c1d_str = f'<span style="color:{color};font-size:0.75rem">{arrow} {change_1d:+.2f}{unit} 1D</span>'

    if change_1w is not None and not pd.isna(change_1w):
        color = change_color(change_1w, invert=invert_color)
        arrow = arrow_indicator(change_1w)
        c1w_str = f'<span style="color:{color};font-size:0.75rem">{arrow} {change_1w:+.2f}{unit} 1W</span>'

    html = f"""
    <div style="
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 0 0 1px rgba(0,212,170,0.08), 0 2px 8px rgba(0,0,0,0.4);
        transition: box-shadow 0.2s;
    ">
        <div style="color:#64748B;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">
            {label}
        </div>
        <div style="color:#F1F5F9;font-size:1.45rem;font-weight:700;font-family:monospace;margin-bottom:6px;">
            {value}
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
            {c1d_str}
            {c1w_str}
        </div>
    </div>
    """
    st.html(html)


# ─────────────────────────────────────────────
# REGIME BADGE
# ─────────────────────────────────────────────

def regime_badge(regime_str: str, title: str = "Regime") -> None:
    """Render a regime label as a styled badge strip."""
    parts = [p.strip() for p in regime_str.split("•")]

    color_map = {
        "Rising": "#EF4444",
        "Falling": "#10B981",
        "Tightening": "#F59E0B",
        "Easing": "#10B981",
        "Stable": "#94A3B8",
        "Off": "#EF4444",
        "On": "#10B981",
        "Weakening": "#EF4444",
        "Strengthening": "#10B981",
        "Unknown": "#64748B",
        "Insufficient": "#64748B",
    }

    badges = ""
    for part in parts:
        color = "#94A3B8"
        for keyword, c in color_map.items():
            if keyword.lower() in part.lower():
                color = c
                break
        badges += f"""
        <span style="
            background:{color}22;
            color:{color};
            border:1px solid {color}55;
            border-radius:4px;
            padding:3px 10px;
            font-size:0.78rem;
            font-weight:600;
            font-family:monospace;
            margin-right:6px;
        ">{part}</span>
        """

    html = f"""
    <div style="margin:12px 0 8px">
        <div style="color:#64748B;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">{title}</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">{badges}</div>
    </div>
    """
    st.html(html)


# ─────────────────────────────────────────────
# SECTION HEADER
# ─────────────────────────────────────────────

def section_header(text: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    sub_html = f'<div style="color:#64748B;font-size:0.8rem;margin-top:2px">{subtitle}</div>' if subtitle else ""
    st.html(f"""
    <div style="border-left:3px solid #00D4AA;padding-left:12px;margin:18px 0 12px">
        <div style="color:#E2E8F0;font-size:1rem;font-weight:700;letter-spacing:0.04em">{text}</div>
        {sub_html}
    </div>
    """)


# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#0A0E1A",
    font=dict(family="monospace", color="#94A3B8", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="#1E293B",
        showline=False,
        tickfont=dict(color="#64748B"),
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        showline=False,
        tickfont=dict(color="#64748B"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    hovermode="x unified",
)

LINE_COLORS = ["#00D4AA", "#6366F1", "#F59E0B", "#EF4444", "#10B981", "#EC4899"]


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the terminal dark theme to a Plotly figure."""
    fig.update_layout(**CHART_LAYOUT)
    return fig


def line_chart(
    series_dict: dict,
    title: str = "",
    yaxis_title: str = "",
    secondary_series: Optional[dict] = None,
    secondary_yaxis_title: str = "",
    height: int = 320,
    date_filter_days: Optional[int] = None,
) -> go.Figure:
    """
    Build a multi-line Plotly chart.

    Args:
        series_dict: {name: pd.Series} for primary y-axis.
        title: Chart title.
        yaxis_title: Primary y-axis label.
        secondary_series: {name: pd.Series} for secondary y-axis.
        secondary_yaxis_title: Secondary y-axis label.
        height: Chart height in pixels.
        date_filter_days: If set, only show last N days.

    Returns:
        Plotly Figure.
    """
    has_secondary = secondary_series and len(secondary_series) > 0
    if has_secondary:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    color_idx = 0
    for name, series in series_dict.items():
        if series is None or (hasattr(series, "empty") and series.empty):
            continue
        s = series.dropna()
        if date_filter_days:
            s = s.iloc[-date_filter_days:]
        kwargs = dict(secondary_y=False) if has_secondary else {}
        trace = go.Scatter(
            x=s.index,
            y=s.values,
            name=name,
            line=dict(color=LINE_COLORS[color_idx % len(LINE_COLORS)], width=1.8),
            hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<extra></extra>",
        )
        if has_secondary:
            fig.add_trace(trace, **kwargs)
        else:
            fig.add_trace(trace)
        color_idx += 1

    if has_secondary:
        for name, series in secondary_series.items():
            if series is None or (hasattr(series, "empty") and series.empty):
                continue
            s = series.dropna()
            if date_filter_days:
                s = s.iloc[-date_filter_days:]
            trace = go.Scatter(
                x=s.index,
                y=s.values,
                name=name,
                line=dict(color=LINE_COLORS[color_idx % len(LINE_COLORS)], width=1.5, dash="dot"),
                hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<extra></extra>",
            )
            fig.add_trace(trace, secondary_y=True)
            color_idx += 1

    layout = {**CHART_LAYOUT, "title": dict(text=title, font=dict(size=13, color="#E2E8F0")), "height": height}
    if has_secondary:
        layout["yaxis2"] = dict(
            title=secondary_yaxis_title,
            gridcolor="#1E293B",
            showline=False,
            tickfont=dict(color="#64748B"),
            overlaying="y",
            side="right",
        )
    fig.update_layout(**layout)
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title, secondary_y=False if has_secondary else None)
    return fig
