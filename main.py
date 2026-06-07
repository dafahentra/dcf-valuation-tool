"""
main.py — DCF Valuation Tool, Streamlit entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

from dcf_engine import DCFEngine
from beta_fetcher import fetch_stock_beta
from visualization import plot_distribution, plot_percentiles, plot_waterfall, plot_sensitivity, display_summary
from styles import get_custom_css, metric_card, summary_box, fmt_curr, fmt_pct

st.set_page_config(page_title="DCF Valuation Tool", page_icon="📈", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

if 'valuation_results' not in st.session_state:
    st.session_state.valuation_results = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def growth_stats(vals: tuple) -> dict:
    """Compute mean and std of growth rates from historical values. Cached."""
    arr = np.array(vals, dtype=float)
    if len(arr) < 2:
        return {'mean': 0.05, 'std': 0.03}
    gr = np.clip(np.diff(arr) / arr[:-1], -0.5, 1.0)
    return {'mean': float(gr.mean()), 'std': float(gr.std()) if len(gr) > 1 else 0.03}


def validate_inputs(price: float, shares: float, fcfs: list[float], proj_growth: list) -> list[str]:
    """Return list of validation errors. Empty = all valid."""
    errors = []
    if price <= 0:    errors.append("Stock price must be greater than 0.")
    if shares <= 0:   errors.append("Shares outstanding must be greater than 0.")
    if all(f <= 0 for f in fcfs): errors.append("At least one year of Free Cash Flow must be positive.")
    for i, g in enumerate(proj_growth):
        if isinstance(g, (tuple, list)) and g[0] > g[1]:
            errors.append(f"Year {i+1} growth range: min ({g[0]:.1%}) cannot exceed max ({g[1]:.1%}).")
    return errors


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.markdown('<h1 class="main-header">Stocks Valuation Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Monte Carlo for intrinsic value estimation</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Market Parameters")
        rf = st.number_input("Risk-Free Rate", 0.0, 0.10, 0.045, 0.005, format="%.3f")
        mp = st.number_input("Market Risk Premium", 0.03, 0.12, st.session_state.get('mp_calc', 0.065), 0.005, format="%.3f")

        st.subheader("Simulation Settings")
        n_sims  = st.select_slider("Monte Carlo Simulations", [1_000, 5_000, 10_000, 25_000, 50_000], value=10_000)
        n_years = st.slider("Projection Years", 3, 10, 5)

        st.subheader("Reproducibility")
        use_fixed_seed = st.checkbox("Fix random seed", value=False, help="Enable for reproducible results across runs")
        random_seed    = st.number_input("Seed", 0, 9999, 42, disabled=not use_fixed_seed)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Company Information</div>', unsafe_allow_html=True)
        company  = st.text_input("Company Name", "Example Corp")
        ticker   = st.text_input("Ticker Symbol", "AAPL")
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "KRW", "IDR"])
        price    = st.number_input("Current Stock Price", min_value=0.01, value=100.00, format="%.2f")
        shares   = st.number_input("Shares Outstanding (millions)", min_value=0.1, value=100.00, format="%.2f")

        if st.button("Fetch Beta", use_container_width=True):
            with st.spinner("Fetching market data…"):
                beta_val, err, mkt_info = fetch_stock_beta(ticker)
                if beta_val is not None:
                    st.session_state.fetched_beta = beta_val
                    st.session_state.mp_calc = mkt_info['market_premium']
                    st.success(f"Beta fetched: {beta_val:.3f} ({mkt_info['market']} market)")
                else:
                    st.error(f"Could not fetch beta: {err}")

        beta = st.number_input("Beta Coefficient", 0.1, 3.0, st.session_state.get('fetched_beta', 1.0), 0.01, format="%.3f")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Financial Structure</div>', unsafe_allow_html=True)
        debt     = st.number_input("Total Debt (millions)", min_value=0.0, value=200.0, format="%.1f")
        cash     = st.number_input("Cash & Equivalents (millions)", min_value=0.0, value=50.0, format="%.1f")
        net_debt = debt - cash
        d2e      = debt / (price * shares) if price * shares > 0 else 0.0
        cod      = st.number_input("Cost of Debt", 0.0, 0.15, 0.04, 0.005, format="%.3f")
        tax      = st.number_input("Effective Tax Rate", 0.0, 0.50, 0.25, 0.01, format="%.3f")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Historical Financials</div>', unsafe_allow_html=True)
        n_hist_years = st.slider("Years of historical data", 3, 5, 5)
        curr_yr = datetime.now().year
        revs, fcfs = [], []

        st.markdown("**Revenue (millions)**")
        for i, col in enumerate(st.columns(n_hist_years)):
            yr = curr_yr - i
            revs.append(col.number_input(f"{yr}", 0.0, value=round(1000.0 * (1.05 ** (n_hist_years - i - 1)), 1), format="%.1f", key=f"r{yr}"))

        st.markdown("**Free Cash Flow (millions)**")
        for i, col in enumerate(st.columns(n_hist_years)):
            yr = curr_yr - i
            fcfs.append(col.number_input(f"{yr}", -1000.0, value=round(100.0 * (1.05 ** (n_hist_years - i - 1)), 1), format="%.1f", key=f"f{yr}"))

        revs.reverse(); fcfs.reverse()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Calculated Metrics</div>', unsafe_allow_html=True)
        rev_g = growth_stats(tuple(revs))
        fcf_g = growth_stats(tuple(fcfs))
        fcf_margin = float(np.mean([f / r for f, r in zip(fcfs, revs) if r > 0])) if any(r > 0 for r in revs) else 0.0

        we   = 1.0 / (1.0 + d2e)
        wacc = we * (rf + beta * mp) + (1.0 - we) * cod * (1.0 - tax)

        for col, args in zip(st.columns(4), [
            ("Avg Revenue Growth", fmt_pct(rev_g['mean']), ('neutral', f"σ: {fmt_pct(rev_g['std'])}")),
            ("Avg FCF Growth",     fmt_pct(fcf_g['mean']), ('neutral', f"σ: {fmt_pct(fcf_g['std'])}")),
            ("Avg FCF Margin",     fmt_pct(fcf_margin), None),
            ("WACC (indicative)",  fmt_pct(wacc), None),
        ]):
            col.markdown(metric_card(*args), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section"><div style="font-size:1.3rem;font-weight:500;color:#c584f7">Growth Assumptions</div>', unsafe_allow_html=True)
        use_range  = st.checkbox("Use growth rate ranges", value=True)
        proj_growth = []
        base_g = rev_g['mean']

        if use_range:
            st.markdown("**Projected Growth Rate Ranges**")
            for i in range(n_years):
                c1, c2, c3 = st.columns([2, 2, 2])
                center = max(base_g * (0.9 ** i), 0.02)
                with c1: st.markdown(f"**Year {i+1}**")
                min_g = c2.number_input("Min", -0.5, 1.0, max(center - 0.05, -0.1), 0.01, format="%.3f", key=f"ming{i}")
                max_g = c3.number_input("Max", -0.5, 1.0, min(center + 0.05, 0.3),  0.01, format="%.3f", key=f"maxg{i}")
                proj_growth.append((min(min_g, max_g), max(min_g, max_g)))
        else:
            st.markdown("**Projected Growth Rates**")
            for i, col in enumerate(st.columns(n_years)):
                g = col.number_input(f"Year {i+1}", -0.5, 1.0, max(base_g * (0.9 ** i), 0.02), 0.01, format="%.3f", key=f"g{i}")
                proj_growth.append(g)

        st.markdown("**Terminal Value**")
        if use_range:
            c1, c2 = st.columns(2)
            term_growth = (c1.number_input("Min Terminal", 0.0, 0.04, 0.02, 0.005, format="%.3f"),
                           c2.number_input("Max Terminal", 0.0, 0.05, 0.03, 0.005, format="%.3f"))
        else:
            term_growth = st.number_input("Terminal Growth", 0.0, 0.05, 0.025, 0.005, format="%.3f")
        st.markdown('</div>', unsafe_allow_html=True)

    # Run valuation
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run DCF Valuation", type="primary", use_container_width=True):
        errors = validate_inputs(price, shares, fcfs, proj_growth)
        if errors:
            for e in errors: st.error(e)
        else:
            params = {
                'base_fcf':           fcfs[-1] * 1e6,
                'growth_rates':       proj_growth,
                'terminal_growth':    term_growth,
                'beta':               beta,
                'debt_to_equity':     d2e,
                'cost_of_debt':       cod,
                'tax_rate':           tax,
                'net_debt':           net_debt * 1e6,
                'shares_outstanding': shares * 1e6,
            }
            unc = {'fcf_growth_std': max(fcf_g['std'], 0.03), 'terminal_growth_std': 0.005, 'beta_std': 0.1}
            rng = np.random.default_rng(int(random_seed) if use_fixed_seed else None)

            with st.spinner(f"Running {n_sims:,} simulations…"):
                dcf = DCFEngine(rf, mp)
                results = dcf.monte_carlo(params, n_sims, unc, rng=rng)
            with st.spinner("Computing sensitivity analysis…"):
                sensitivity = dcf.sensitivity_analysis(params)

            st.session_state.valuation_results = {
                'results': results, 'sensitivity': sensitivity,
                'current_price': price, 'currency': currency,
                'company_name': company, 'parameters': params,
            }

    # Display results
    if st.session_state.valuation_results:
        data    = st.session_state.valuation_results
        res     = data['results']
        price   = data['current_price']
        curr    = data['currency']
        company = data['company_name']

        st.markdown(f"## Valuation Results for {company}")
        display_summary(res, price, curr)

        tab1, tab2, tab3 = st.tabs(["Distribution", "Scenarios", "Sensitivity"])

        with tab1:
            st.plotly_chart(plot_distribution(res['per_share_values'], price, curr), use_container_width=True)
            st.plotly_chart(plot_percentiles(res, price, curr), use_container_width=True)

        with tab2:
            scenarios = {
                'bear': res['percentiles']['p10'], 'base': res['percentiles']['p50'],
                'bull': res['percentiles']['p90'], 'expected': res['mean'],
            }
            st.plotly_chart(plot_waterfall(price, scenarios, curr), use_container_width=True)
            st.markdown("#### Scenario Summary")
            st.dataframe(pd.DataFrame([
                {'Scenario': name, 'Fair Value': fmt_curr(scenarios[key], curr), 'Upside/Downside': fmt_pct(scenarios[key] / price - 1)}
                for name, key in [('Bear (P10)', 'bear'), ('Base (P50)', 'base'), ('Bull (P90)', 'bull'), ('Expected (Mean)', 'expected')]
            ]), use_container_width=True, hide_index=True)

        with tab3:
            st.plotly_chart(plot_sensitivity(data['sensitivity']), use_container_width=True)
            up_prob = float((res['per_share_values'] > price).mean() * 100)
            insight = (
                "Strong buy signal"   if up_prob > 70 else
                "Moderate buy signal" if up_prob > 50 else
                "Fairly valued"       if up_prob > 30 else
                "Potentially overvalued"
            )
            st.markdown(summary_box("Key Insight",
                f"<strong>{up_prob:.0f}%</strong> probability of upside vs current price. {insight}."),
                unsafe_allow_html=True)

        # Export
        st.markdown("### Export Results")
        up_prob = float((res['per_share_values'] > price).mean() * 100)
        st.download_button(
            "Download Results (JSON)",
            data=json.dumps({
                'summary': {'company': company, 'current_price': price, 'mean_fair_value': res['mean'],
                            'upside_probability': f"{up_prob:.1f}%", 'n_simulations': res['n_simulations']},
                'percentiles': res['percentiles'],
                'statistics':  {'std': res['std'], 'skew': res['skew'], 'kurtosis': res['kurtosis']},
            }, indent=2),
            file_name=f"{company.replace(' ', '_')}_DCF_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()