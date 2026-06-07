"""
visualization.py
All Plotly chart functions for the DCF Valuation Tool.
Sensitivity analysis uses real parameter perturbation data from DCFEngine,
not hardcoded placeholder values.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

# Shared layout defaults applied to every chart
_LAYOUT_DEFAULTS = {
    'template': 'plotly_dark',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
}

_PURPLE = '#c584f7'
_RED    = '#f87171'
_GREEN  = '#4ade80'
_BLUE   = '#60a5fa'


# ---------------------------------------------------------------------------
# Distribution chart
# ---------------------------------------------------------------------------

def plot_distribution(vals: np.ndarray, price: float, curr: str = 'USD') -> go.Figure:
    """Histogram + KDE overlay with reference lines for price, mean, median."""
    fig = go.Figure()

    fig.add_histogram(
        x=vals, nbinsx=50,
        marker_color='rgba(197,132,247,0.6)',
        name='Value Distribution',
    )

    kde_x = np.linspace(vals.min(), vals.max(), 200)
    bin_width = (vals.max() - vals.min()) / 50
    kde_y = stats.gaussian_kde(vals)(kde_x) * len(vals) * bin_width
    fig.add_scatter(
        x=kde_x, y=kde_y, mode='lines',
        line=dict(color=_PURPLE, width=3),
        name='Probability Density',
    )

    for value, color, dash, label in [
        (price,           _RED,   'dash', 'Current'),
        (vals.mean(),     _GREEN, 'dot',  'Mean'),
        (np.median(vals), _BLUE,  'dot',  'Median'),
    ]:
        fig.add_vline(
            x=value, line_dash=dash, line_color=color,
            annotation_text=f'{label}: {value:,.2f}',
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=500,
        title='Fair Value Distribution',
        xaxis_title=f'Fair Value ({curr})',
        yaxis_title='Frequency',
    )
    return fig


# ---------------------------------------------------------------------------
# Percentile chart
# ---------------------------------------------------------------------------

def plot_percentiles(res: dict, price: float, curr: str = 'USD') -> go.Figure:
    """Line chart of valuation across key percentiles."""
    pctls = [5, 10, 25, 50, 75, 90, 95]
    values = [res['percentiles'][f'p{p}'] for p in pctls]

    fig = go.Figure()
    fig.add_scatter(
        x=pctls, y=values, mode='lines+markers',
        line=dict(color=_PURPLE, width=3),
        marker=dict(size=10),
    )
    fig.add_hline(
        y=price, line_dash='dash', line_color=_RED,
        annotation_text=f'Current: {price:.2f}',
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=400,
        title='Valuation Percentiles',
        xaxis_title='Percentile',
        yaxis_title=f'Fair Value ({curr})',
    )
    return fig


# ---------------------------------------------------------------------------
# Scenario waterfall
# ---------------------------------------------------------------------------

def plot_waterfall(base_price: float, scenarios: dict, curr: str = 'USD') -> go.Figure:
    """
    Waterfall chart: current price → bear → base → bull → expected.

    Parameters
    ----------
    scenarios : dict with keys bear, base, bull, expected (per-share values)
    """
    labels = ['Current', 'Bear', 'Base', 'Bull', 'Expected']
    values = [
        base_price,
        scenarios['bear'] - base_price,
        scenarios['base'] - scenarios['bear'],
        scenarios['bull'] - scenarios['base'],
        scenarios['expected'],
    ]
    measures = ['absolute', 'relative', 'relative', 'relative', 'total']

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        decreasing={'marker': {'color': _RED}},
        increasing={'marker': {'color': _GREEN}},
        totals={'marker': {'color': _PURPLE}},
    ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=400,
        title='Scenario Analysis',
        yaxis_title=f'Price ({curr})',
    )
    return fig


# ---------------------------------------------------------------------------
# Sensitivity tornado — uses real perturbation data
# ---------------------------------------------------------------------------

def plot_sensitivity(sensitivity_data: dict) -> go.Figure:
    """
    Horizontal tornado chart built from actual DCFEngine sensitivity analysis.

    Parameters
    ----------
    sensitivity_data : dict
        Output of DCFEngine.sensitivity_analysis().
        Maps parameter label -> (downside_impact, upside_impact).
    """
    # Sort by absolute magnitude so largest driver is at top
    sorted_items = sorted(
        sensitivity_data.items(),
        key=lambda x: max(abs(x[1][0]), abs(x[1][1])),
        reverse=True,
    )

    labels = [item[0] for item in sorted_items]
    downsides = [item[1][0] for item in sorted_items]
    upsides = [item[1][1] for item in sorted_items]

    fig = go.Figure()

    fig.add_bar(
        y=labels, x=downsides, orientation='h',
        marker_color=_RED, name='Downside (−1pp)',
    )
    fig.add_bar(
        y=labels, x=upsides, orientation='h',
        marker_color=_GREEN, name='Upside (+1pp)',
    )

    fig.add_vline(x=0, line_dash='solid', line_color='white', line_width=2)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=400,
        title='Sensitivity Analysis (Impact of ±1pp Change per Parameter)',
        xaxis_title='Change in Mean Fair Value (absolute)',
        barmode='overlay',
    )
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard (two-panel inline chart)
# ---------------------------------------------------------------------------

def display_summary(res: dict, price: float, curr: str = 'USD') -> None:
    """Render 4 key metrics and a two-panel dashboard chart in Streamlit."""
    from styles import fmt_curr, fmt_pct  # local import to avoid circular dependency

    vals = res['per_share_values']
    upside_probability = float((vals > price).mean() * 100)

    metrics = [
        ('Mean Fair Value',    fmt_curr(res['mean'], curr),                   f"{(res['mean'] / price - 1) * 100:+.1f}%"),
        ('Median Fair Value',  fmt_curr(res['median'], curr),                 f"{(res['median'] / price - 1) * 100:+.1f}%"),
        ('Upside Probability', f"{upside_probability:.1f}%",                  'vs current price'),
        ('Value at Risk (P10)', fmt_curr(res['percentiles']['p10'], curr),    f"{(res['percentiles']['p10'] / price - 1) * 100:+.1f}%"),
    ]

    cols = st.columns(4)
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)

    # Two-panel dashboard
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fair Value Distribution', 'Risk-Return Profile'),
    )

    fig.add_histogram(x=vals, nbinsx=50, marker_color='rgba(197,132,247,0.6)', row=1, col=1)

    confidence_levels = np.arange(10, 91, 10)
    fair_values_at_confidence = [np.percentile(vals, p) for p in confidence_levels]
    fig.add_scatter(
        x=list(confidence_levels), y=fair_values_at_confidence,
        mode='lines+markers',
        marker=dict(size=8, color=_PURPLE),
        row=1, col=2,
    )

    fig.update_layout(**_LAYOUT_DEFAULTS, height=400, showlegend=False)
    fig.update_xaxes(title_text=f'Fair Value ({curr})', row=1, col=1)
    fig.update_xaxes(title_text='Confidence Level (%)', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text=f'Fair Value ({curr})', row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)