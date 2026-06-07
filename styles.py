"""
styles.py
UI formatting helpers: CSS, metric cards, currency/percentage formatting.
"""

CURR_SYMBOLS: dict[str, str] = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
    'CNY': '¥', 'INR': '₹', 'KRW': '₩', 'IDR': 'Rp',
}


def get_custom_css() -> str:
    return """
    <style>
        #MainMenu, footer, header, .stDeployButton, .stToolbar, ._profileContainer {
            visibility: hidden; display: none;
        }

        .block-container { padding-top: 1rem; }

        .main-header {
            font-size: 2.5rem; font-weight: 300; margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #c584f7 0%, #a068d8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }

        .sub-header { font-size: 1.1rem; color: #888; margin-bottom: 2rem; }

        .metric-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            padding: 1.5rem; border-radius: 12px;
        }

        .summary-box {
            background: rgba(197,132,247,0.1);
            border: 1px solid rgba(197,132,247,0.3);
            padding: 1.5rem; border-radius: 12px; margin: 2rem 0;
        }

        .input-section {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        }
    </style>
    """


def metric_card(label: str, value: str, delta: tuple | None = None) -> str:
    """Render a styled metric card as HTML. delta = (sentiment, text)."""
    delta_html = ''
    if delta:
        sentiment, text = delta
        color = {'positive': '#4ade80', 'negative': '#f87171'}.get(sentiment, '#888')
        delta_html = f'<div style="color:{color};font-size:0.875rem">{text}</div>'
    return (
        f'<div class="metric-card">'
        f'<div style="color:#888;font-size:0.875rem">{label}</div>'
        f'<div style="font-size:1.75rem;font-weight:600">{value}</div>'
        f'{delta_html}</div>'
    )


def summary_box(title: str, content: str) -> str:
    return (
        f'<div class="summary-box">'
        f'<div style="font-size:1.25rem;font-weight:600;color:#c584f7;margin-bottom:1rem">{title}</div>'
        f'<div>{content}</div></div>'
    )


def fmt_curr(amount: float | None, curr: str = 'USD') -> str:
    """Format a number as currency with magnitude suffix (K/M/B/T)."""
    if amount is None:
        return 'N/A'
    symbol = CURR_SYMBOLS.get(curr, '$')
    for magnitude, suffix in [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')]:
        if abs(amount) >= magnitude:
            return f'{symbol}{amount / magnitude:,.2f}{suffix}'
    return f'{symbol}{amount:,.2f}'


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Format a decimal as a percentage string."""
    if value is None:
        return 'N/A'
    return f'{value * 100:.{decimals}f}%'