"""
dcf_engine.py
Discounted Cash Flow valuation engine with Monte Carlo simulation.

Key design decisions:
- Monte Carlo is fully vectorized with NumPy for performance (~50x faster than Python loops)
- RNG is passed in as a parameter so callers control reproducibility
- Sensitivity analysis perturbs actual model parameters, not placeholder values
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --- Constants (named, not magic numbers) ---
TERMINAL_VALUE_EXIT_MULTIPLE = 15       # fallback EV/FCF multiple when WACC ≈ terminal growth
WACC_TERMINAL_BUFFER = 0.001            # minimum spread between WACC and terminal growth
BETA_CLIP_MIN, BETA_CLIP_MAX = 0.3, 2.5
GROWTH_CLIP_MIN, GROWTH_CLIP_MAX = -0.30, 0.50
TERMINAL_CLIP_MIN, TERMINAL_CLIP_MAX = 0.0, 0.04
IQR_OUTLIER_FACTOR = 3.0               # how many IQRs beyond Q1/Q3 to trim


class DCFEngine:
    """
    Discounted Cash Flow valuation engine.

    Parameters
    ----------
    rf : float
        Risk-free rate (e.g. 0.045 for 4.5%)
    mp : float
        Equity market risk premium (e.g. 0.065 for 6.5%)
    """

    def __init__(self, rf: float = 0.045, mp: float = 0.065) -> None:
        self.rf = max(0.0, rf)
        self.mp = max(0.0, mp)

    # ------------------------------------------------------------------
    # Core single-scenario valuation
    # ------------------------------------------------------------------

    def calculate_value(self, p: dict) -> dict:
        """
        Compute enterprise value for a single parameter set.

        Parameters
        ----------
        p : dict with keys:
            beta, debt_to_equity, cost_of_debt, tax_rate,
            base_fcf, growth_rates (list), terminal_growth

        Returns
        -------
        dict: enterprise_value, wacc, terminal_value, fcf_projections
        """
        wacc = self._compute_wacc(
            beta=p['beta'],
            debt_to_equity=p['debt_to_equity'],
            cost_of_debt=p['cost_of_debt'],
            tax_rate=p['tax_rate'],
        )

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
        Run a Monte Carlo simulation by sampling uncertain parameters.

        Fully vectorized — no Python loop over simulations.

        Parameters
        ----------
        base : dict
            Base-case parameters (same structure as calculate_value).
        n : int
            Number of simulations (clamped to [100, 100_000]).
        unc : dict, optional
            Uncertainty (std dev) for each stochastic parameter.
            Keys: fcf_growth_std, terminal_growth_std, beta_std
        rng : np.random.Generator, optional
            Random number generator. Pass np.random.default_rng(seed)
            for reproducible results. Defaults to a fresh generator.

        Returns
        -------
        dict with per_share_values, statistics, and percentiles.
        """
        n = int(np.clip(n, 100, 100_000))
        unc = unc or {'fcf_growth_std': 0.03, 'terminal_growth_std': 0.005, 'beta_std': 0.1}
        rng = rng or np.random.default_rng()

        n_years = len(base['growth_rates'])
        g_range = isinstance(base['growth_rates'][0], (tuple, list))
        t_range = isinstance(base['terminal_growth'], (tuple, list))

        # --- Sample stochastic parameters (all at once) ---
        betas = np.clip(
            rng.normal(base['beta'], unc['beta_std'], n),
            BETA_CLIP_MIN, BETA_CLIP_MAX,
        )

        if g_range:
            growths = np.column_stack([
                rng.uniform(
                    max(GROWTH_CLIP_MIN, g[0]),
                    min(GROWTH_CLIP_MAX, g[1]),
                    n,
                )
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
                min(TERMINAL_CLIP_MAX, base['terminal_growth'][1]),
                n,
            )
        else:
            term_growths = np.clip(
                rng.normal(base['terminal_growth'], unc['terminal_growth_std'], n),
                TERMINAL_CLIP_MIN, TERMINAL_CLIP_MAX,
            )

        # --- Vectorized FCF projection ---
        # Shape: (n, n_years). Each row is a cumulative growth path.
        cumulative_growth = np.cumprod(1.0 + growths, axis=1)          # (n, n_years)
        fcf_matrix = base['base_fcf'] * cumulative_growth               # (n, n_years)
        last_fcf = fcf_matrix[:, -1]                                    # (n,)

        # --- Vectorized WACC ---
        waccs = self._compute_wacc_vectorized(
            betas=betas,
            debt_to_equity=base['debt_to_equity'],
            cost_of_debt=base['cost_of_debt'],
            tax_rate=base['tax_rate'],
        )

        # --- Vectorized terminal value ---
        spread = waccs - term_growths
        safe_spread = np.where(spread > WACC_TERMINAL_BUFFER, spread, np.nan)
        tv_gordon = last_fcf * (1.0 + term_growths) / safe_spread
        tv_multiple = last_fcf * TERMINAL_VALUE_EXIT_MULTIPLE
        terminal_values = np.where(np.isnan(tv_gordon), tv_multiple, tv_gordon)

        # --- Vectorized present value ---
        years = np.arange(1, n_years + 1)
        discount_matrix = 1.0 / (1.0 + waccs[:, None]) ** years        # (n, n_years)
        pv_fcf = np.sum(fcf_matrix * discount_matrix, axis=1)           # (n,)
        pv_tv = terminal_values / (1.0 + waccs) ** n_years              # (n,)

        enterprise_values = np.maximum(pv_fcf + pv_tv, 0.0)

        # --- Convert to per-share equity value ---
        net_debt = base.get('net_debt', 0.0)
        shares = max(1.0, base.get('shares_outstanding', 1.0))
        per_share = np.maximum(enterprise_values - net_debt, 0.0) / shares

        # Remove extreme outliers (beyond 3×IQR)
        per_share = self._remove_outliers(per_share)

        return self._build_results(per_share)

    # ------------------------------------------------------------------
    # Sensitivity analysis (real parameter perturbation)
    # ------------------------------------------------------------------

    def sensitivity_analysis(self, base: dict, delta: float = 0.01) -> dict:
        """
        Compute sensitivity of mean equity value to each key parameter.

        Each parameter is shifted ±delta independently (all else equal),
        and the resulting change in mean per-share value is recorded.

        Parameters
        ----------
        base : dict
            Base-case parameters.
        delta : float
            Absolute perturbation size (default 1 percentage point).

        Returns
        -------
        dict mapping parameter name -> (downside_impact, upside_impact)
        """
        base_rng = np.random.default_rng(0)
        base_result = self.monte_carlo(base, n=2_000, rng=base_rng)
        base_mean = base_result['mean']

        parameters_to_test = {
            'Beta': ('beta', base['beta']),
            'FCF Growth (Yr1)': ('growth_rates', base['growth_rates']),
            'Terminal Growth': ('terminal_growth', base['terminal_growth']),
            'Cost of Debt': ('cost_of_debt', base['cost_of_debt']),
        }

        results = {}
        for label, (key, original) in parameters_to_test.items():
            down_val = self._perturb(original, -delta)
            up_val = self._perturb(original, +delta)

            down_mean = self.monte_carlo(
                {**base, key: down_val}, n=2_000, rng=np.random.default_rng(0)
            )['mean']
            up_mean = self.monte_carlo(
                {**base, key: up_val}, n=2_000, rng=np.random.default_rng(0)
            )['mean']

            results[label] = (down_mean - base_mean, up_mean - base_mean)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_wacc(
        self,
        beta: float,
        debt_to_equity: float,
        cost_of_debt: float,
        tax_rate: float,
    ) -> float:
        cost_of_equity = self.rf + max(BETA_CLIP_MIN, beta) * self.mp
        weight_equity = 1.0 / (1.0 + max(0.0, debt_to_equity))
        weight_debt = 1.0 - weight_equity
        after_tax_debt_cost = max(0.0, cost_of_debt) * (1.0 - np.clip(tax_rate, 0.0, 1.0))
        return max(0.01, weight_equity * cost_of_equity + weight_debt * after_tax_debt_cost)

    def _compute_wacc_vectorized(
        self,
        betas: np.ndarray,
        debt_to_equity: float,
        cost_of_debt: float,
        tax_rate: float,
    ) -> np.ndarray:
        cost_of_equity = self.rf + np.clip(betas, BETA_CLIP_MIN, BETA_CLIP_MAX) * self.mp
        weight_equity = 1.0 / (1.0 + max(0.0, debt_to_equity))
        after_tax_debt = max(0.0, cost_of_debt) * (1.0 - np.clip(tax_rate, 0.0, 1.0))
        waccs = weight_equity * cost_of_equity + (1.0 - weight_equity) * after_tax_debt
        return np.maximum(waccs, 0.01)

    def _project_fcf(self, base_fcf: float, growth_rates: list) -> np.ndarray:
        """Project FCF forward, resolving range inputs to their midpoint."""
        resolved = [
            float(np.mean(g)) if isinstance(g, (tuple, list)) else float(g)
            for g in growth_rates
        ]
        cumulative = np.cumprod(1.0 + np.array(resolved))
        return base_fcf * cumulative

    def _compute_terminal_value(
        self, last_fcf: float, terminal_growth: float, wacc: float
    ) -> float:
        tg = float(np.mean(terminal_growth)) if isinstance(terminal_growth, (tuple, list)) else float(terminal_growth)
        tg = min(tg, wacc - WACC_TERMINAL_BUFFER)
        if wacc - tg > WACC_TERMINAL_BUFFER:
            return last_fcf * (1.0 + tg) / (wacc - tg)
        return last_fcf * TERMINAL_VALUE_EXIT_MULTIPLE

    @staticmethod
    def _remove_outliers(arr: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        mask = (arr >= q1 - IQR_OUTLIER_FACTOR * iqr) & (arr <= q3 + IQR_OUTLIER_FACTOR * iqr)
        return arr[mask] if mask.any() else arr

    @staticmethod
    def _perturb(value, delta: float):
        """
        Add delta to a scalar or uniformly shift a (min, max) range tuple.
        For list-of-ranges (yearly growth), shifts every year's midpoint.
        """
        if isinstance(value, list):
            # List of growth rates or (min, max) tuples — shift each element
            return [
                (v[0] + delta, v[1] + delta) if isinstance(v, (tuple, list)) else v + delta
                for v in value
            ]
        if isinstance(value, tuple):
            return (value[0] + delta, value[1] + delta)
        return value + delta

    @staticmethod
    def _build_results(per_share: np.ndarray) -> dict:
        pctls = [5, 10, 25, 50, 75, 90, 95]
        return {
            'per_share_values': per_share,
            'mean': float(per_share.mean()),
            'median': float(np.median(per_share)),
            'std': float(per_share.std()),
            'skew': float(stats.skew(per_share)) if len(per_share) > 3 else 0.0,
            'kurtosis': float(stats.kurtosis(per_share)) if len(per_share) > 3 else 0.0,
            'percentiles': {f'p{p}': float(np.percentile(per_share, p)) for p in pctls},
            'n_simulations': len(per_share),
        }