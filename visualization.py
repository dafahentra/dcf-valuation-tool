"""
visualization.py
Plotly chart functions for the DCF Valuation Tool.
Sensitivity analysis uses real perturbation data from DCFEngine.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

_LAYOUT = {'template': 'plotly_dark', 'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'}
_PURPLE, _RED, _GREEN, _BLUE = '#c584f7', '#f87171', '#4ade80', '#60a5fa'


def plot_distribution(vals: np.ndarray, price: float, curr: str = 'USD') -> go.Figure:
    """Histogram + KDE overlay with reference lines for price, mean, and median."""
    fig = go.Figure()
    fig.add_histogram(x=vals, nbinsx=50, marker_color='rgba(197,132,247,0.6)', name='Value Distribution')

    kde_x = np.linspace(vals.min(), vals.max(), 200)
    kde_y = stats.gaussian_kde(vals)(kde_x) * len(vals) * (vals.max() - vals.min()) / 50
    fig.add_scatter(x=kde_x, y=kde_y, mode='lines', line=dict(color=_PURPLE, width=3), name='Probability Density')

    for value, color, dash, label in [
        (price,           _RED,   'dash', 'Current'),
        (vals.mean(),     _GREEN, 'dot',  'Mean'),
        (np.median(vals), _BLUE,  'dot',  'Median'),
    ]:
        fig.add_vline(x=value, line_dash=dash, line_color=color, annotation_text=f'{label}: {value:,.2f}')

    fig.update_layout(**_LAYOUT, height=500, title='Fair Value Distribution',
                      xaxis_title=f'Fair Value ({curr})', yaxis_title='Frequency')
    return fig


def plot_percentiles(res: dict, price: float, curr: str = 'USD') -> go.Figure:
    """Line chart of valuation across key percentiles."""
    pctls = [5, 10, 25, 50, 75, 90, 95]
    fig = go.Figure()
    fig.add_scatter(x=pctls, y=[res['percentiles'][f'p{p}'] for p in pctls],
                    mode='lines+markers', line=dict(color=_PURPLE, width=3), marker=dict(size=10))
    fig.add_hline(y=price, line_dash='dash', line_color=_RED, annotation_text=f'Current: {price:.2f}')
    fig.update_layout(**_LAYOUT, height=400, title='Valuation Percentiles',
                      xaxis_title='Percentile', yaxis_title=f'Fair Value ({curr})')
    return fig


def plot_waterfall(base_price: float, scenarios: dict, curr: str = 'USD') -> go.Figure:
    """Waterfall: current price → bear → base → bull → expected."""
    fig = go.Figure(go.Waterfall(
        x=['Current', 'Bear', 'Base', 'Bull', 'Expected'],
        y=[base_price,
           scenarios['bear'] - base_price,
           scenarios['base'] - scenarios['bear'],
           scenarios['bull'] - scenarios['base'],
           scenarios['expected']],
        measure=['absolute', 'relative', 'relative', 'relative', 'total'],
        decreasing={'marker': {'color': _RED}},
        increasing={'marker': {'color': _GREEN}},
        totals={'marker': {'color': _PURPLE}},
    ))
    fig.update_layout(**_LAYOUT, height=400, title='Scenario Analysis', yaxis_title=f'Price ({curr})')
    return fig


def plot_sensitivity(sensitivity_data: dict) -> go.Figure:
    """
    Tornado chart from real DCFEngine.sensitivity_analysis() data.
    Sorted by absolute magnitude — largest driver at top.
    """
    items = sorted(sensitivity_data.items(), key=lambda x: max(abs(x[1][0]), abs(x[1][1])), reverse=True)
    labels    = [i[0] for i in items]
    downsides = [i[1][0] for i in items]
    upsides   = [i[1][1] for i in items]

    fig = go.Figure()
    fig.add_bar(y=labels, x=downsides, orientation='h', marker_color=_RED,   name='Downside (−1pp)')
    fig.add_bar(y=labels, x=upsides,   orientation='h', marker_color=_GREEN, name='Upside (+1pp)')
    fig.add_vline(x=0, line_dash='solid', line_color='white', line_width=2)
    fig.update_layout(**_LAYOUT, height=400, barmode='overlay',
                      title='Sensitivity Analysis (±1pp per parameter)',
                      xaxis_title='Change in Mean Fair Value')
    return fig


def display_summary(res: dict, price: float, curr: str = 'USD') -> None:
    """4 key metric cards + two-panel distribution/risk-return chart."""
    from styles import fmt_curr, fmt_pct

    vals    = res['per_share_values']
    up_prob = float((vals > price).mean() * 100)

    for col, (label, value, delta) in zip(st.columns(4), [
        ('Mean Fair Value',     fmt_curr(res['mean'], curr),                 f"{(res['mean']   / price - 1) * 100:+.1f}%"),
        ('Median Fair Value',   fmt_curr(res['median'], curr),               f"{(res['median'] / price - 1) * 100:+.1f}%"),
        ('Upside Probability',  f"{up_prob:.1f}%",                           'vs current price'),
        ('Value at Risk (P10)', fmt_curr(res['percentiles']['p10'], curr),   f"{(res['percentiles']['p10'] / price - 1) * 100:+.1f}%"),
    ]):
        col.metric(label, value, delta)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Fair Value Distribution', 'Risk-Return Profile'))
    fig.add_histogram(x=vals, nbinsx=50, marker_color='rgba(197,132,247,0.6)', row=1, col=1)

    pct_levels = np.arange(10, 91, 10)
    fig.add_scatter(x=list(pct_levels), y=[np.percentile(vals, p) for p in pct_levels],
                    mode='lines+markers', marker=dict(size=8, color=_PURPLE), row=1, col=2)

    fig.update_layout(**_LAYOUT, height=400, showlegend=False)
    fig.update_xaxes(title_text=f'Fair Value ({curr})', row=1, col=1)
    fig.update_xaxes(title_text='Confidence Level (%)', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text=f'Fair Value ({curr})', row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)