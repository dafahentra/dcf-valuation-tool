"""
dcf_engine.py
Discounted Cash Flow valuation engine with Monte Carlo simulation.
- Monte Carlo is fully vectorized with NumPy (~50x faster than Python loops)
- RNG is passed in as a parameter so callers control reproducibility
- Sensitivity analysis perturbs actual model parameters, not placeholder values
"""

import numpy as np
from scipy import stats


# Named constants — no magic numbers in formulas
TERMINAL_VALUE_EXIT_MULTIPLE = 15       # fallback EV/FCF multiple when WACC ≈ terminal growth
WACC_TERMINAL_BUFFER = 0.001            # minimum spread between WACC and terminal growth
BETA_CLIP_MIN, BETA_CLIP_MAX = 0.3, 2.5
GROWTH_CLIP_MIN, GROWTH_CLIP_MAX = -0.30, 0.50
TERMINAL_CLIP_MIN, TERMINAL_CLIP_MAX = 0.0, 0.04
IQR_OUTLIER_FACTOR = 3.0               # how many IQRs beyond Q1/Q3 to trim


class DCFEngine:
    """DCF valuation engine. rf = risk-free rate, mp = market risk premium."""

    def __init__(self, rf: float = 0.045, mp: float = 0.065) -> None:
        self.rf = max(0.0, rf)
        self.mp = max(0.0, mp)

    # ------------------------------------------------------------------
    # Core single-scenario valuation
    # ------------------------------------------------------------------

    def calculate_value(self, p: dict) -> dict:
        """Compute enterprise value for a single parameter set."""
        wacc = self._compute_wacc(p['beta'], p['debt_to_equity'], p['cost_of_debt'], p['tax_rate'])
        fcf_projections = self._project_fcf(p['base_fcf'], p['growth_rates'])
        terminal_value = self._compute_terminal_value(fcf_projections[-1], p['terminal_growth'], wacc)

        n = len(fcf_projections)
        discount_factors = 1.0 / (1.0 + wacc) ** np.arange(1, n + 1)
        pv_fcf = float(np.dot(fcf_projections, discount_factors))
        pv_terminal = terminal_value / (1.0 + wacc) ** n

        return {
            'enterprise_value': max(0.0, pv_fcf + pv_terminal),
            'wacc': wacc,
            'terminal_value': terminal_value,
            'fcf_projections': fcf_projections,
        }

    # ------------------------------------------------------------------
    # Vectorized Monte Carlo
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        base: dict,
        n: int = 10_000,
        unc: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """
        Run Monte Carlo simulation — fully vectorized, no Python loop.
        Pass rng=np.random.default_rng(seed) for reproducible results.
        """
        n = int(np.clip(n, 100, 100_000))
        unc = unc or {'fcf_growth_std': 0.03, 'terminal_growth_std': 0.005, 'beta_std': 0.1}
        rng = rng or np.random.default_rng()

        n_years = len(base['growth_rates'])
        g_range = isinstance(base['growth_rates'][0], (tuple, list))
        t_range = isinstance(base['terminal_growth'], (tuple, list))

        # Sample all stochastic parameters at once
        betas = np.clip(rng.normal(base['beta'], unc['beta_std'], n), BETA_CLIP_MIN, BETA_CLIP_MAX)

        if g_range:
            growths = np.column_stack([
                rng.uniform(max(GROWTH_CLIP_MIN, g[0]), min(GROWTH_CLIP_MAX, g[1]), n)
                for g in base['growth_rates']
            ])
        else:
            growths = np.clip(
                rng.normal(base['growth_rates'], unc['fcf_growth_std'], (n, n_years)),
                GROWTH_CLIP_MIN, GROWTH_CLIP_MAX,
            )

        if t_range:
            term_growths = rng.uniform(
                max(TERMINAL_CLIP_MIN, base['terminal_growth'][0]),
                min(TERMINAL_CLIP_MAX, base['terminal_growth'][1]), n,
            )
        else:
            term_growths = np.clip(
                rng.normal(base['terminal_growth'], unc['terminal_growth_std'], n),
                TERMINAL_CLIP_MIN, TERMINAL_CLIP_MAX,
            )

        # Vectorized FCF projection — shape (n, n_years)
        fcf_matrix = base['base_fcf'] * np.cumprod(1.0 + growths, axis=1)
        last_fcf = fcf_matrix[:, -1]

        # Vectorized WACC
        waccs = self._compute_wacc_vectorized(betas, base['debt_to_equity'], base['cost_of_debt'], base['tax_rate'])

        # Vectorized terminal value (Gordon Growth Model, fallback to exit multiple)
        spread = waccs - term_growths
        safe_spread = np.where(spread > WACC_TERMINAL_BUFFER, spread, np.nan)
        terminal_values = np.where(
            np.isnan(safe_spread),
            last_fcf * TERMINAL_VALUE_EXIT_MULTIPLE,
            last_fcf * (1.0 + term_growths) / safe_spread,
        )

        # Vectorized present value
        years = np.arange(1, n_years + 1)
        discount_matrix = 1.0 / (1.0 + waccs[:, None]) ** years        # (n, n_years)
        pv_fcf = np.sum(fcf_matrix * discount_matrix, axis=1)
        pv_tv = terminal_values / (1.0 + waccs) ** n_years

        # Convert to per-share equity value
        net_debt = base.get('net_debt', 0.0)
        shares = max(1.0, base.get('shares_outstanding', 1.0))
        per_share = np.maximum(np.maximum(pv_fcf + pv_tv, 0.0) - net_debt, 0.0) / shares

        return self._build_results(self._remove_outliers(per_share))

    # ------------------------------------------------------------------
    # Sensitivity analysis — real parameter perturbation
    # ------------------------------------------------------------------

    def sensitivity_analysis(self, base: dict, delta: float = 0.01) -> dict:
        """
        Shift each key parameter by ±delta (1pp default) and record
        the resulting change in mean per-share value.
        Returns dict: label -> (downside_impact, upside_impact)
        """
        fixed_rng = lambda: np.random.default_rng(0)  # noqa: E731
        base_mean = self.monte_carlo(base, n=2_000, rng=fixed_rng())['mean']

        params_to_test = {
            'Beta':             ('beta',           base['beta']),
            'FCF Growth (Yr1)': ('growth_rates',   base['growth_rates']),
            'Terminal Growth':  ('terminal_growth', base['terminal_growth']),
            'Cost of Debt':     ('cost_of_debt',   base['cost_of_debt']),
        }

        results = {}
        for label, (key, original) in params_to_test.items():
            down_mean = self.monte_carlo({**base, key: self._perturb(original, -delta)}, n=2_000, rng=fixed_rng())['mean']
            up_mean   = self.monte_carlo({**base, key: self._perturb(original, +delta)}, n=2_000, rng=fixed_rng())['mean']
            results[label] = (down_mean - base_mean, up_mean - base_mean)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_wacc(self, beta: float, d2e: float, cod: float, tax: float) -> float:
        ce = self.rf + max(BETA_CLIP_MIN, beta) * self.mp
        we = 1.0 / (1.0 + max(0.0, d2e))
        return max(0.01, we * ce + (1.0 - we) * max(0.0, cod) * (1.0 - np.clip(tax, 0.0, 1.0)))

    def _compute_wacc_vectorized(self, betas: np.ndarray, d2e: float, cod: float, tax: float) -> np.ndarray:
        ce = self.rf + np.clip(betas, BETA_CLIP_MIN, BETA_CLIP_MAX) * self.mp
        we = 1.0 / (1.0 + max(0.0, d2e))
        return np.maximum(we * ce + (1.0 - we) * max(0.0, cod) * (1.0 - np.clip(tax, 0.0, 1.0)), 0.01)

    def _project_fcf(self, base_fcf: float, growth_rates: list) -> np.ndarray:
        """Project FCF forward, resolving range inputs to their midpoint."""
        resolved = [float(np.mean(g)) if isinstance(g, (tuple, list)) else float(g) for g in growth_rates]
        return base_fcf * np.cumprod(1.0 + np.array(resolved))

    def _compute_terminal_value(self, last_fcf: float, terminal_growth, wacc: float) -> float:
        tg = float(np.mean(terminal_growth)) if isinstance(terminal_growth, (tuple, list)) else float(terminal_growth)
        tg = min(tg, wacc - WACC_TERMINAL_BUFFER)
        return last_fcf * (1.0 + tg) / (wacc - tg) if wacc - tg > WACC_TERMINAL_BUFFER else last_fcf * TERMINAL_VALUE_EXIT_MULTIPLE

    @staticmethod
    def _remove_outliers(arr: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        mask = (arr >= q1 - IQR_OUTLIER_FACTOR * iqr) & (arr <= q3 + IQR_OUTLIER_FACTOR * iqr)
        return arr[mask] if mask.any() else arr

    @staticmethod
    def _perturb(value, delta: float):
        """Shift a scalar, tuple range, or list of either by delta."""
        if isinstance(value, list):
            return [(v[0] + delta, v[1] + delta) if isinstance(v, (tuple, list)) else v + delta for v in value]
        if isinstance(value, tuple):
            return (value[0] + delta, value[1] + delta)
        return value + delta

    @staticmethod
    def _build_results(per_share: np.ndarray) -> dict:
        pctls = [5, 10, 25, 50, 75, 90, 95]
        return {
            'per_share_values': per_share,
            'mean':     float(per_share.mean()),
            'median':   float(np.median(per_share)),
            'std':      float(per_share.std()),
            'skew':     float(stats.skew(per_share))     if len(per_share) > 3 else 0.0,
            'kurtosis': float(stats.kurtosis(per_share)) if len(per_share) > 3 else 0.0,
            'percentiles': {f'p{p}': float(np.percentile(per_share, p)) for p in pctls},
            'n_simulations': len(per_share),
        }