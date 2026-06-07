"""
beta_fetcher.py
Fetches stock beta coefficients from market data using yfinance.
Falls back to manual calculation if provider beta is unavailable.
"""

from __future__ import annotations

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
    """Create a curl_cffi session if available, otherwise return None."""
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
    """
    Calculate beta from historical price data.
    Returns (beta, error_message). One of them will always be None.
    """
    stock_hist = stock_ticker.history(period=period)['Close']
    market_hist = market_ticker.history(period=period)['Close']

    if len(stock_hist) < MIN_DATA_POINTS or len(market_hist) < MIN_DATA_POINTS:
        return None, f"Insufficient data: need {MIN_DATA_POINTS} points, got stock={len(stock_hist)}, market={len(market_hist)}"

    stock_returns = stock_hist.pct_change().dropna()
    market_returns = market_hist.pct_change().dropna()
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner')

    # Remove outliers to reduce noise from data errors or extreme events
    clean_mask = (abs(stock_returns) < OUTLIER_THRESHOLD) & (abs(market_returns) < OUTLIER_THRESHOLD)
    stock_clean = stock_returns[clean_mask]
    market_clean = market_returns[clean_mask]

    if len(stock_clean) < MIN_DATA_POINTS:
        return None, f"Insufficient clean data after outlier removal: {len(stock_clean)} points"

    market_variance = market_clean.var()
    if market_variance == 0:
        return None, "Market variance is zero — cannot compute beta"

    beta = np.clip(stock_clean.cov(market_clean) / market_variance, BETA_MIN, BETA_MAX)
    return float(beta), None


@st.cache_data(ttl=3600)
def fetch_stock_beta(
    ticker: str,
    period: str = "2y",
) -> tuple[float | None, str | None, dict | None]:
    """
    Fetch beta coefficient for a given stock ticker.

    Strategy:
    1. Try provider-supplied beta from yfinance info (fast).
    2. Fall back to calculating beta from historical returns (slower).

    Returns:
        (beta, error_message, market_info)
        On success: (float, None, dict)
        On failure: (None, str, None)
    """
    ticker = ticker.upper().strip()
    session = _create_session()

    try:
        stock = yf.Ticker(ticker, session=session)
        market_key = _detect_market(ticker)
        market_config = MARKETS[market_key]
        market_info = {
            'market': market_key,
            'market_premium': market_config['mp'],
            'risk_free': market_config['rf'],
        }

        # Attempt 1: use provider beta if plausible
        info = stock.info or {}
        provider_beta = info.get('beta')
        if provider_beta and BETA_MIN <= provider_beta <= BETA_MAX:
            return float(provider_beta), None, market_info

        # Attempt 2: calculate from historical returns
        market_stock = yf.Ticker(market_config['idx'], session=session)
        beta, error = _calculate_beta_from_history(stock, market_stock, period)

        if error:
            return None, error, None

        return beta, None, market_info

    except (ValueError, KeyError) as e:
        return None, f"Data error: {e}", None
    except Exception as e:
        return None, f"Unexpected error fetching beta for '{ticker}': {e}", None