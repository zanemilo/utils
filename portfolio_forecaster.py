"""
portfolio_forecaster.py

A well-documented, inflation-aware portfolio forecasting tool with Monte Carlo simulation,
optional regime switching, and flexible contribution/withdrawal schedules.

Key features
------------
- **Real vs. nominal accounting:** Model nominal market returns and inflation, compute real balances.
- **Monthly granularity:** Compounds monthly for realism; converts annual inputs appropriately.
- **Regime switching (optional):** Simple two-state (Bull/Bear) Markov model to shift return mean/vol.
- **Custom cashflow schedules:** Monthly contributions and withdrawals as constants or callables.
- **Summary stats:** Median/mean and selected percentiles of ending balances; pathwise real CAGR.
- **Plotting helpers:** Time-series fan chart (optional) and percentile bands.

Dependencies
------------
- numpy
- matplotlib (for plotting)

Usage (quick start)
-------------------
>>> from portfolio_forecaster import PortfolioForecaster
>>> pf = PortfolioForecaster(
...     initial_investment=30_000,
...     monthly_contribution=1_000,
...     years=20,
...     simulations=2000,
...     annual_return=0.07,
...     annual_vol=0.16,
...     annual_inflation_mean=0.025,
...     annual_inflation_vol=0.01,
...     use_regimes=True
... )
>>> pf.simulate(seed=42)
>>> stats = pf.summary_statistics()
>>> pf.plot_fan_chart()


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RegimeConfig:
    """Configuration for a simple two-state (Bull/Bear) Markov regime model.

    Attributes
    ----------
    p_bb : float
        P(Bull → Bull) monthly persistence probability.
    p_aa : float
        P(Bear → Bear) monthly persistence probability.
    bull_mu_shift : float
        Additive shift to **monthly** mean return in Bull.
    bear_mu_shift : float
        Additive shift to **monthly** mean return in Bear (typically negative).
    bull_sigma_mult : float
        Multiplier on **monthly** volatility in Bull (e.g., 0.8 for calmer markets).
    bear_sigma_mult : float
        Multiplier on **monthly** volatility in Bear (e.g., 1.3 for more turbulence).
    start_state : int
        Initial regime state (1 = Bull, 0 = Bear).
    """

    p_bb: float = 0.90
    p_aa: float = 0.85
    bull_mu_shift: float = 0.001  # +0.10% monthly
    bear_mu_shift: float = -0.003  # -0.30% monthly
    bull_sigma_mult: float = 0.85
    bear_sigma_mult: float = 1.25
    start_state: int = 1


class PortfolioForecaster:
    """Inflation-aware portfolio forecaster with Monte Carlo simulation.

    The model draws **nominal** monthly market returns from a normal distribution (or a
    regime-adjusted one) and draws monthly inflation from a normal distribution, then
    converts to **real** returns using:

        real_return = (1 + nominal_return) / (1 + inflation) - 1

    The balance update in real terms is:

        balance_real = balance_real * (1 + real_return) + contribution_real - withdrawal_real

    Parameters
    ----------
    initial_investment : float
        Starting balance in today's dollars.
    monthly_contribution : float | Callable[[int], float]
        Constant monthly contribution (real) or a function f(t_month) → amount (real).
    monthly_withdrawal : float | Callable[[int], float] | None
        Constant monthly withdrawal (real) or a function; set None to disable.
    years : int
        Horizon in years.
    simulations : int
        Number of Monte Carlo paths.
    annual_return : float
        Expected **annual** nominal market return (e.g., 0.07 for 7%).
    annual_vol : float
        Expected **annual** nominal volatility (e.g., 0.16 for 16% std dev).
    annual_inflation_mean : float
        Expected **annual** inflation mean.
    annual_inflation_vol : float
        Expected **annual** inflation volatility (std dev).
    use_regimes : bool
        If True, activate a two-state regime model that shifts monthly mean/vol.
    regime_config : RegimeConfig | None
        Regime parameters; defaults are reasonable if None.
    clip_returns : Tuple[float, float] | None
        Optional min/max clip on nominal monthly returns to avoid crazy outliers.
    clip_inflation : Tuple[float, float] | None
        Optional min/max clip on monthly inflation.
    """

    def __init__(
        self,
        initial_investment: float = 30_000.0,
        monthly_contribution: float | Callable[[int], float] = 1_000.0,
        monthly_withdrawal: Optional[float | Callable[[int], float]] = None,
        years: int = 20,
        simulations: int = 1_000,
        annual_return: float = 0.07,
        annual_vol: float = 0.16,
        annual_inflation_mean: float = 0.025,
        annual_inflation_vol: float = 0.010,
        use_regimes: bool = False,
        regime_config: Optional[RegimeConfig] = None,
        clip_returns: Optional[Tuple[float, float]] = (-0.5, 0.5),
        clip_inflation: Optional[Tuple[float, float]] = (-0.02, 0.02),
    ) -> None:
        self.initial_investment = float(initial_investment)
        self.monthly_contribution = monthly_contribution
        self.monthly_withdrawal = monthly_withdrawal
        self.years = int(years)
        self.simulations = int(simulations)
        self.annual_return = float(annual_return)
        self.annual_vol = float(annual_vol)
        self.annual_inflation_mean = float(annual_inflation_mean)
        self.annual_inflation_vol = float(annual_inflation_vol)
        self.use_regimes = bool(use_regimes)
        self.regime_config = regime_config or RegimeConfig()
        self.clip_returns = clip_returns
        self.clip_inflation = clip_inflation

        # Derived monthly parameters
        self.mu_m = (1 + self.annual_return) ** (1 / 12) - 1
        # Convert annual vol to monthly (assuming i.i.d. monthly increments)
        self.sigma_m = self.annual_vol / np.sqrt(12)
        self.inf_mu_m = (1 + self.annual_inflation_mean) ** (1 / 12) - 1
        self.inf_sigma_m = self.annual_inflation_vol / np.sqrt(12)

        # Results containers
        self.results_real: Optional[np.ndarray] = None  # shape: (sim, months+1)
        self.regimes: Optional[np.ndarray] = None       # shape: (sim, months)

    # ------------------------------ utilities ------------------------------ #
    @staticmethod
    def _to_callable(x: float | Callable[[int], float]) -> Callable[[int], float]:
        """Wrap a constant or function as a callable schedule f(t_month) → amount."""
        if callable(x):
            return x  # type: ignore[return-value]
        return lambda t: float(x)

    def _draw_regime_path(self, months: int, rng: np.random.Generator) -> np.ndarray:
        """Generate a single Bull/Bear regime path.

        Returns an array of 0/1 where 1=**Bull**, 0=**Bear**.
        """
        cfg = self.regime_config
        states = np.empty(months, dtype=np.int8)
        states[0] = cfg.start_state
        for t in range(1, months):
            if states[t - 1] == 1:  # Bull
                states[t] = 1 if rng.random() < cfg.p_bb else 0
            else:  # Bear
                states[t] = 0 if rng.random() < cfg.p_aa else 1
        return states

    # ------------------------------ simulation ---------------------------- #
    def simulate(self, seed: Optional[int] = None) -> None:
        """Run Monte Carlo simulation and store **real** balance paths.

        Parameters
        ----------
        seed : int | None
            Optional seed for reproducibility.
        """
        months = self.years * 12
        rng = np.random.default_rng(seed)

        contrib_fn = self._to_callable(self.monthly_contribution)
        withdraw_fn = None if self.monthly_withdrawal is None else self._to_callable(self.monthly_withdrawal)

        results = np.empty((self.simulations, months + 1), dtype=float)
        regimes_store = np.empty((self.simulations, months), dtype=np.int8) if self.use_regimes else None

        for s in range(self.simulations):
            balance = self.initial_investment  # real dollars
            results[s, 0] = balance

            # Regime path for this simulation (if enabled)
            if self.use_regimes:
                reg_path = self._draw_regime_path(months, rng)
                regimes_store[s, :] = reg_path  # type: ignore[index]
            else:
                reg_path = None

            for t in range(months):
                # Inflation (nominal → real conversion)
                inf = rng.normal(self.inf_mu_m, self.inf_sigma_m)
                if self.clip_inflation is not None:
                    inf = float(np.clip(inf, self.clip_inflation[0], self.clip_inflation[1]))

                # Nominal market return — possibly regime-adjusted
                mu_t = self.mu_m
                sigma_t = self.sigma_m
                if reg_path is not None:
                    if reg_path[t] == 1:  # Bull
                        mu_t += self.regime_config.bull_mu_shift
                        sigma_t *= self.regime_config.bull_sigma_mult
                    else:  # Bear
                        mu_t += self.regime_config.bear_mu_shift
                        sigma_t *= self.regime_config.bear_sigma_mult

                nominal_r = rng.normal(mu_t, sigma_t)
                if self.clip_returns is not None:
                    nominal_r = float(np.clip(nominal_r, self.clip_returns[0], self.clip_returns[1]))

                # Convert to real monthly return
                real_r = (1.0 + nominal_r) / (1.0 + inf) - 1.0

                # Real cashflows this month
                c = contrib_fn(t)
                w = 0.0 if withdraw_fn is None else withdraw_fn(t)

                # Update real balance
                balance = balance * (1.0 + real_r) + c - w
                # Guard against negative drift if modeling withdrawals
                balance = max(balance, 0.0)
                results[s, t + 1] = balance

        self.results_real = results
        self.regimes = regimes_store

    # ------------------------------ analytics ----------------------------- #
    def summary_statistics(self, percentiles: Tuple[int, int] = (10, 90)) -> dict:
        """Compute summary statistics on ending **real** balances.

        Parameters
        ----------
        percentiles : tuple[int, int]
            Two percentiles to report (default 10th and 90th).

        Returns
        -------
        dict
            Dictionary with Median, Mean, and requested percentile endpoints.
        """
        if self.results_real is None:
            raise ValueError("Run simulate() first.")
        end_bal = self.results_real[:, -1]
        p_lo, p_hi = np.percentile(end_bal, percentiles[0]), np.percentile(end_bal, percentiles[1])
        out = {
            "Median": float(np.median(end_bal)),
            "Mean": float(np.mean(end_bal)),
            f"P{percentiles[0]}": float(p_lo),
            f"P{percentiles[1]}": float(p_hi),
        }
        return out

    def real_cagr_distribution(self) -> np.ndarray:
        """Compute pathwise **real** CAGR across the horizon.

        Returns
        -------
        np.ndarray
            Array of length `simulations` with per-path CAGR values.
        """
        if self.results_real is None:
            raise ValueError("Run simulate() first.")
        start = np.maximum(self.results_real[:, 0], 1e-9)
        end = self.results_real[:, -1]
        years = self.years
        cagr = (end / start) ** (1.0 / years) - 1.0
        return cagr

    # ------------------------------ plotting ------------------------------ #
    def plot_fan_chart(self, max_paths: int = 200, show: bool = True) -> None:
        """Plot sample paths and percentile bands in **real** dollars.

        Notes
        -----
        - Uses matplotlib only (no seaborn). One figure; single plot.
        - Does not specify colors to comply with style constraints.
        """
        if self.results_real is None:
            raise ValueError("Run simulate() first.")
        T = self.results_real.shape[1]
        x = np.arange(T)

        # Fan percentiles
        p10 = np.percentile(self.results_real, 10, axis=0)
        p50 = np.percentile(self.results_real, 50, axis=0)
        p90 = np.percentile(self.results_real, 90, axis=0)

        plt.figure(figsize=(12, 6))
        # Plot a subset of paths for clarity
        nplot = min(max_paths, self.results_real.shape[0])
        for i in range(nplot):
            plt.plot(x, self.results_real[i], alpha=0.08)
        plt.plot(x, p50, linestyle='--', label='Median (real)')
        plt.fill_between(x, p10, p90, alpha=0.15, label='P10–P90 band (real)')
        plt.title("Portfolio Monte Carlo — Real Dollars")
        plt.xlabel("Months")
        plt.ylabel("Real Portfolio Value")
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()


# ------------------------------ CLI / interactive runner ------------------------------ #

def _pretty_print_summary(stats: dict, cagr_dist: np.ndarray) -> None:
    """Pretty-print summary statistics and CAGR percentiles.

    Parameters
    ----------
    stats : dict
        Output from `summary_statistics()`.
    cagr_dist : np.ndarray
        Real CAGR values per simulation.
    """
    p10_cagr, p50_cagr, p90_cagr = np.percentile(cagr_dist, [10, 50, 90])
    print("" + "=" * 66)
    print(" Portfolio Forecast — Real (Inflation-Adjusted) Dollars ")
    print("=" * 66)
    print(f"  Median end balance : ${stats['Median']:,.2f}")
    print(f"  Mean end balance   : ${stats['Mean']:,.2f}")
    p_keys = [k for k in stats.keys() if k.startswith('P')]
    for k in sorted(p_keys, key=lambda x: int(x[1:])):
        print(f"  {k:>4} end balance : ${stats[k]:,.2f}")
    print("-" * 66)
    print("  Real CAGR distribution (per year)")
    print(f"    P10: {p10_cagr*100:5.2f}%   P50 (median): {p50_cagr*100:5.2f}%   P90: {p90_cagr*100:5.2f}%")
    print("=" * 66 + "")


def run_interactive() -> None:
    """Prompt for inputs, run simulation, print pretty summary, and plot.

    The inputs are interpreted in **today's dollars** (real terms). Contributions
    are treated as real monthly amounts; the model handles inflation internally.
    """
    try:
        init = float(input("Initial investment (e.g., 30000): ").strip() or 30000)
        contrib = float(input("Monthly contribution (e.g., 1000): ").strip() or 1000)
        years = int(input("Years to simulate (e.g., 20): ").strip() or 20)
        sims = int(input("Number of simulations (e.g., 2000): ").strip() or 2000)
    except ValueError:
        print("Invalid input detected. Using defaults: 30000, 1000, 20, 2000.")
        init, contrib, years, sims = 30000.0, 1000.0, 20, 2000

    pf = PortfolioForecaster(
        initial_investment=init,
        monthly_contribution=contrib,
        years=years,
        simulations=sims,
        annual_return=0.07,
        annual_vol=0.16,
        annual_inflation_mean=0.025,
        annual_inflation_vol=0.01,
        use_regimes=True,
    )
    pf.simulate(seed=123)
    stats = pf.summary_statistics()
    cagr = pf.real_cagr_distribution()
    _pretty_print_summary(stats, cagr)
    pf.plot_fan_chart()


if __name__ == "__main__":
    run_interactive()
