"""
beta_fetcher.py
Fetches stock beta from yfinance. Falls back to calculating from
historical returns if provider beta is unavailable.
"""

import numpy as np
import yfinance as yf
import streamlit as st

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CFFI = True
except ImportError:
    _HAS_CFFI = False

# Market configurations: index ticker, suffix, market premium, risk-free rate
MARKETS: dict[str, dict] = {
    'US': {'idx': '^GSPC',     'sfx': '',    'mp': 0.065, 'rf': 0.045},
    'UK': {'idx': '^FTSE',     'sfx': '.L',  'mp': 0.060, 'rf': 0.040},
    'DE': {'idx': '^GDAXI',    'sfx': '.DE', 'mp': 0.055, 'rf': 0.025},
    'JP': {'idx': '^N225',     'sfx': '.T',  'mp': 0.050, 'rf': 0.001},
    'HK': {'idx': '^HSI',      'sfx': '.HK', 'mp': 0.065, 'rf': 0.040},
    'IN': {'idx': '^BSESN',    'sfx': '.NS', 'mp': 0.080, 'rf': 0.070},
    'CN': {'idx': '000001.SS', 'sfx': '.SS', 'mp': 0.070, 'rf': 0.025},
}

BETA_MIN = 0.1
BETA_MAX = 3.0
MIN_DATA_POINTS = 60
OUTLIER_THRESHOLD = 0.5


def _create_session():
    """Return a curl_cffi session if available, otherwise None."""
    if _HAS_CFFI:
        try:
            return cffi_requests.Session(impersonate="chrome120")
        except Exception:
            pass
    return None


def _detect_market(ticker: str) -> str:
    """Detect market from ticker suffix. Defaults to US."""
    for market_key, config in MARKETS.items():
        if config['sfx'] and ticker.endswith(config['sfx']):
            return market_key
    return 'US'


def _calculate_beta_from_history(
    stock_ticker: yf.Ticker,
    market_ticker: yf.Ticker,
    period: str,
) -> tuple[float | None, str | None]:
    """Calculate beta from historical returns. Returns (beta, error)."""
    stock_hist  = stock_ticker.history(period=period)['Close']
    market_hist = market_ticker.history(period=period)['Close']

    if len(stock_hist) < MIN_DATA_POINTS or len(market_hist) < MIN_DATA_POINTS:
        return None, f"Insufficient data (need {MIN_DATA_POINTS} points)"

    s_ret, m_ret = stock_hist.pct_change().dropna(), market_hist.pct_change().dropna()
    s_ret, m_ret = s_ret.align(m_ret, join='inner')

    # Remove outliers (data errors or extreme events)
    mask = (abs(s_ret) < OUTLIER_THRESHOLD) & (abs(m_ret) < OUTLIER_THRESHOLD)
    s_clean, m_clean = s_ret[mask], m_ret[mask]

    if len(s_clean) < MIN_DATA_POINTS:
        return None, f"Insufficient clean data after outlier removal ({len(s_clean)} points)"

    var_m = m_clean.var()
    if var_m == 0:
        return None, "Market variance is zero — cannot compute beta"

    return float(np.clip(s_clean.cov(m_clean) / var_m, BETA_MIN, BETA_MAX)), None


@st.cache_data(ttl=3600)
def fetch_stock_beta(
    ticker: str,
    period: str = "2y",
) -> tuple[float | None, str | None, dict | None]:
    """
    Fetch beta for a given ticker.
    1. Try provider beta from yfinance info.
    2. Fall back to calculating from historical returns.
    Returns (beta, error, market_info) — one of beta/error is always None.
    """
    ticker = ticker.upper().strip()
    session = _create_session()

    try:
        stock = yf.Ticker(ticker, session=session)
        market_key = _detect_market(ticker)
        cfg = MARKETS[market_key]
        market_info = {'market': market_key, 'market_premium': cfg['mp'], 'risk_free': cfg['rf']}

        # Attempt 1: provider beta
        info = stock.info or {}
        provider_beta = info.get('beta')
        if provider_beta and BETA_MIN <= provider_beta <= BETA_MAX:
            return float(provider_beta), None, market_info

        # Attempt 2: calculate from history
        beta, error = _calculate_beta_from_history(stock, yf.Ticker(cfg['idx'], session=session), period)
        return (beta, None, market_info) if beta else (None, error, None)

    except (ValueError, KeyError) as e:
        return None, f"Data error: {e}", None
    except Exception as e:
        return None, f"Unexpected error for '{ticker}': {e}", None