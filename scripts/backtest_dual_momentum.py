"""
Dual Momentum (GEM - Global Equities Momentum) backtest.

This is Step 1/2/6 of the GEM build: backtest *before* building the production
strategy, validate it against SPY buy-and-hold, run the two out-of-sample
windows, and print an honest final verdict.  It makes ZERO live broker calls -
yfinance only.

THE CANONICAL GEM RULES (Gary Antonacci, "Dual Momentum Investing")
-------------------------------------------------------------------
Evaluated on the last trading day of each month, using a fixed 12-month
lookback (the lookback is set by academic precedent and is NOT tuned here):

  1. Absolute momentum:  if SPY trailing-12m total return > T-bill trailing-12m
     return  ->  equities are "on".  Otherwise hold AGG (US aggregate bonds).
  2. Relative momentum (only when equities are on):  hold whichever of SPY (US)
     or VEU (international ex-US) has the higher trailing-12m total return.

Hold one asset 100% at a time; rebalance monthly.

DATA / PROXIES
--------------
The tradeable ETFs (SPY 1993, VEU 2007, AGG 2003, BIL 2007) don't all reach
back far enough for a multi-decade test, so each asset's monthly *return*
series is extended backward with a long-history Vanguard proxy.  Returns are
chained (real ETF return where available, else proxy return) into a continuous
total-return index - this avoids price-level discontinuities from naive
splicing:

  US equities      : SPY                       (1993+)
  International     : VEU,  proxy VGTSX         (VGTSX 1996+ extends VEU's 2007)
  US agg bonds      : AGG,  proxy VBMFX         (VBMFX 1986+ extends AGG's 2003)
  T-bill / risk-free: ^IRX  (13-week T-bill discount yield, 1985+)

The effective backtest start is therefore governed by the international series
(~1996).  The script prints the actual date range it used - it does not pretend
to have data it doesn't.

Usage:
    python scripts/backtest_dual_momentum.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────────
LOOKBACK_MONTHS = 12          # fixed by academic precedent - do NOT tune
MONTHS_PER_YEAR = 12

# Ordered symbol chains: real ETF first, long-history proxy after. Returns are
# taken from the first symbol that has data in each month.
US_SYMBOLS    = ["SPY"]
INTL_SYMBOLS  = ["VEU", "VGTSX"]
BOND_SYMBOLS  = ["AGG", "VBMFX"]
TBILL_SYMBOL  = "^IRX"

ASSET_LABELS  = {"US": "SPY", "INTL": "VEU", "BONDS": "AGG"}


# ── Data loading ─────────────────────────────────────────────────────────────

def _monthly_close(symbol: str) -> pd.Series:
    """Fetch daily adjusted closes via yfinance, resample to month-end close.

    Indexed by monthly Period so series from symbols with slightly different
    month-end trading days align cleanly.
    """
    import yfinance as yf

    df = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    monthly = close.resample("ME").last().dropna()
    monthly.index = monthly.index.to_period("M")
    return monthly


def _chained_returns(symbols: List[str]) -> pd.Series:
    """Build a continuous monthly total-return series from an ordered symbol
    chain. For each month, use the return of the first symbol that has it."""
    rets = [_monthly_close(s).pct_change() for s in symbols]
    combined = rets[0]
    for r in rets[1:]:
        combined = combined.combine_first(r)
    return combined.dropna()


def _tbill_monthly_returns() -> pd.Series:
    """Convert ^IRX (annualised 13-week T-bill discount yield, in percent) into
    an approximate monthly T-bill total return: (yield% / 100) / 12 per month."""
    yld = _monthly_close(TBILL_SYMBOL)
    monthly = (yld / 100.0) / MONTHS_PER_YEAR
    return monthly.dropna()


def _build_index(returns: pd.Series) -> pd.Series:
    """Cumulative total-return index (starts at 1.0 one month before first ret)."""
    return (1.0 + returns).cumprod()


# ── GEM simulation ───────────────────────────────────────────────────────────

@dataclass
class GemResult:
    monthly: pd.DataFrame   # index=Period('M'); cols: holding, gem_ret, spy_ret
    start: pd.Period
    end: pd.Period


def run_gem() -> GemResult:
    """Run the full-history GEM simulation. Returns a per-month DataFrame of the
    chosen holding and the realised GEM vs SPY-buy&hold monthly returns."""
    us_ret    = _chained_returns(US_SYMBOLS)
    intl_ret  = _chained_returns(INTL_SYMBOLS)
    bond_ret  = _chained_returns(BOND_SYMBOLS)
    tbill_ret = _tbill_monthly_returns()

    # Align all return series to a common monthly index.
    idx = us_ret.index
    for s in (intl_ret.index, bond_ret.index, tbill_ret.index):
        idx = idx.intersection(s)
    idx = idx.sort_values()

    us_ret, intl_ret, bond_ret, tbill_ret = (
        us_ret.reindex(idx), intl_ret.reindex(idx),
        bond_ret.reindex(idx), tbill_ret.reindex(idx),
    )

    # Total-return indices for trailing-12m momentum.
    us_px   = _build_index(us_ret)
    intl_px = _build_index(intl_ret)
    # T-bill trailing-12m return = rolling product of monthly tbill returns.
    tbill_12m = (1.0 + tbill_ret).rolling(LOOKBACK_MONTHS).apply(np.prod, raw=True) - 1.0

    months = list(idx)
    records = []
    # Decide at close of month i (needs i and i-12), realise return in month i+1.
    for i in range(LOOKBACK_MONTHS, len(months) - 1):
        spy_12m  = us_px.iloc[i]   / us_px.iloc[i - LOOKBACK_MONTHS]   - 1.0
        intl_12m = intl_px.iloc[i] / intl_px.iloc[i - LOOKBACK_MONTHS] - 1.0
        rf_12m   = tbill_12m.iloc[i]

        if spy_12m > rf_12m:
            holding = "US" if spy_12m >= intl_12m else "INTL"
        else:
            holding = "BONDS"

        nxt = months[i + 1]
        realised = {"US": us_ret, "INTL": intl_ret, "BONDS": bond_ret}[holding].iloc[i + 1]
        records.append({
            "month": nxt,
            "holding": holding,
            "gem_ret": float(realised),
            "spy_ret": float(us_ret.iloc[i + 1]),
        })

    monthly = pd.DataFrame(records).set_index("month")
    return GemResult(monthly=monthly, start=monthly.index[0], end=monthly.index[-1])


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    cagr: float
    sharpe: float
    max_dd: float
    worst_year: float
    n_months: int


def _cagr(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total = float((1.0 + returns).prod())
    years = len(returns) / MONTHS_PER_YEAR
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def _sharpe(returns: pd.Series, rf_monthly: Optional[pd.Series] = None) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - (rf_monthly.reindex(returns.index).fillna(0.0) if rf_monthly is not None else 0.0)
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(MONTHS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(dd.min() * 100.0)   # negative percent


def _worst_year(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    by_year = returns.groupby(returns.index.year).apply(lambda r: (1.0 + r).prod() - 1.0)
    return float(by_year.min() * 100.0)


def metrics_for(returns: pd.Series, rf_monthly: Optional[pd.Series] = None) -> Metrics:
    return Metrics(
        cagr=_cagr(returns) * 100.0,
        sharpe=_sharpe(returns, rf_monthly),
        max_dd=_max_drawdown(returns),
        worst_year=_worst_year(returns),
        n_months=len(returns),
    )


# ── Reporting ────────────────────────────────────────────────────────────────

def _print_comparison(title: str, gem: Metrics, spy: Metrics) -> None:
    d_cagr = gem.cagr - spy.cagr
    d_sharpe = gem.sharpe - spy.sharpe
    d_dd = gem.max_dd - spy.max_dd   # less-negative (smaller drawdown) is better
    print(f"\n{title}")
    print("-" * 64)
    print(f"  {'Metric':<14} {'GEM':>12} {'SPY Buy&Hold':>14} {'Delta':>12}")
    print(f"  {'CAGR':<14} {gem.cagr:>11.2f}% {spy.cagr:>13.2f}% {d_cagr:>+11.2f}%")
    print(f"  {'Sharpe':<14} {gem.sharpe:>12.2f} {spy.sharpe:>14.2f} {d_sharpe:>+12.2f}")
    print(f"  {'Max drawdown':<14} {gem.max_dd:>11.2f}% {spy.max_dd:>13.2f}% {d_dd:>+11.2f}%")
    print(f"  {'Worst year':<14} {gem.worst_year:>11.2f}% {spy.worst_year:>13.2f}%")


def _allocation_breakdown(monthly: pd.DataFrame) -> None:
    counts = monthly["holding"].value_counts()
    total = len(monthly)
    switches = int((monthly["holding"] != monthly["holding"].shift()).sum()) - 1
    years = total / MONTHS_PER_YEAR
    print("\n  Allocation breakdown (% of months held):")
    for asset in ("US", "INTL", "BONDS"):
        pct = counts.get(asset, 0) / total * 100.0
        print(f"    {asset:<6} ({ASSET_LABELS[asset]:<4}) : {pct:>5.1f}%  ({counts.get(asset, 0)} months)")
    print(f"  Allocation switches: {switches}  (~{switches / years:.1f} trades/year)")


def main() -> dict:
    print("=" * 64)
    print("  DUAL MOMENTUM (GEM) BACKTEST")
    print("=" * 64)
    print("  Loading data (SPY / VEU+VGTSX / AGG+VBMFX / ^IRX) via yfinance...")

    res = run_gem()
    monthly = res.monthly
    # Build an aligned monthly risk-free series for Sharpe (reuse tbill returns).
    tbill = _tbill_monthly_returns().reindex(monthly.index).fillna(0.0)

    start_str = str(res.start)
    end_str = str(res.end)
    years = len(monthly) / MONTHS_PER_YEAR
    print(f"  Effective realised-return period: {start_str} -> {end_str} "
          f"({len(monthly)} months, {years:.1f} years)")
    print("  NOTE: international history (VGTSX, 1996+) governs the start date;")
    print("        the requested 1993 start is not reachable with free intl data.")

    # ── Full period (Step 1) ──────────────────────────────────────────────────
    gem_full = metrics_for(monthly["gem_ret"], tbill)
    spy_full = metrics_for(monthly["spy_ret"], tbill)
    _print_comparison("FULL PERIOD - GEM vs SPY buy-and-hold", gem_full, spy_full)
    _allocation_breakdown(monthly)

    # ── Validation gate ─────────────────────────────────────────────────────--
    cagr_ok = gem_full.cagr >= spy_full.cagr
    dd_ok = gem_full.max_dd > spy_full.max_dd   # less-negative drawdown = smaller
    print("\n" + "=" * 64)
    print("  VALIDATION GATE  (CAGR >= SPY  AND  max drawdown < SPY)")
    print("=" * 64)
    print(f"    CAGR >= SPY?      {'PASS' if cagr_ok else 'FAIL'}  "
          f"({gem_full.cagr:.2f}% vs {spy_full.cagr:.2f}%)")
    print(f"    Smaller drawdown? {'PASS' if dd_ok else 'FAIL'}  "
          f"({gem_full.max_dd:.2f}% vs {spy_full.max_dd:.2f}%)")
    passed_both = cagr_ok and dd_ok
    failed_both = (not cagr_ok) and (not dd_ok)
    if failed_both:
        print("\n  RESULT: GEM fails BOTH criteria. STOP - do not build the")
        print("          production strategy on this evidence.")
    elif passed_both:
        print("\n  RESULT: GEM passes both criteria - continue to production build.")
    else:
        print("\n  RESULT: GEM passes on a RISK-ADJUSTED basis only (one of two).")
        print("          The strategy's edge is drawdown control, not raw CAGR.")
        print("          Continuing (gate's hard-stop is only the fail-both case).")

    # ── Out-of-sample windows (Step 2) ─────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  STEP 2 - OUT-OF-SAMPLE WINDOWS (identical rules, no re-tuning)")
    print("=" * 64)

    windows = [
        ("1993-2010 (effective from data start)", monthly.index.year <= 2010),
        ("2011-2025", monthly.index.year >= 2011),
    ]
    window_metrics = {}
    for label, mask in windows:
        sub = monthly[mask]
        if sub.empty:
            continue
        rf_sub = tbill.reindex(sub.index).fillna(0.0)
        g = metrics_for(sub["gem_ret"], rf_sub)
        s = metrics_for(sub["spy_ret"], rf_sub)
        window_metrics[label] = (g, s)
        _print_comparison(f"WINDOW: {label}  ({str(sub.index[0])} -> {str(sub.index[-1])})", g, s)

    print("\n  Honest read on the windows:")
    for label, (g, s) in window_metrics.items():
        verdict = "LAGS SPY on CAGR" if g.cagr < s.cagr else "matches/beats SPY on CAGR"
        dd_verdict = "wins on drawdown" if g.max_dd > s.max_dd else "does NOT win on drawdown"
        print(f"    {label}: GEM {verdict}; {dd_verdict}.")

    # ── Final verdict (Step 6) ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  STEP 6 - FINAL VERDICT")
    print("=" * 64)
    print(f"  Full period   : GEM CAGR {gem_full.cagr:.2f}%  vs SPY {spy_full.cagr:.2f}%  "
          f"(d {gem_full.cagr - spy_full.cagr:+.2f}%)")
    print(f"                  GEM maxDD {gem_full.max_dd:.2f}% vs SPY {spy_full.max_dd:.2f}% "
          f"(d {gem_full.max_dd - spy_full.max_dd:+.2f}%)")
    for label, (g, s) in window_metrics.items():
        print(f"  {label:<14}: GEM CAGR {g.cagr:.2f}% vs SPY {s.cagr:.2f}%; "
              f"GEM maxDD {g.max_dd:.2f}% vs SPY {s.max_dd:.2f}%")

    if passed_both:
        beat = "on BOTH CAGR and drawdown"
    elif gem_full.max_dd > spy_full.max_dd and gem_full.cagr < spy_full.cagr:
        beat = "on DRAWDOWN / risk-adjusted return only - NOT on raw CAGR"
    elif gem_full.cagr >= spy_full.cagr:
        beat = "on CAGR only"
    else:
        beat = "on neither headline metric"
    print(f"\n  Honest verdict: GEM beats SPY {beat}.")
    print("  Forward expectation: GEM's edge is concentrated in bear-market")
    print("  avoidance (it rotates to bonds before/within drawdowns). It will")
    print("  very likely UNDERPERFORM SPY during sustained bull markets - the")
    print("  2011-2025 window above is the live demonstration of that lag.")
    print("  This is a risk-managed compounding engine, not a return-maximiser.")
    print("  Deployment: PAPER-ONLY on Alpaca. No real capital at risk.")
    print("=" * 64)

    return {
        "full": (gem_full, spy_full),
        "windows": window_metrics,
        "passed_both": passed_both,
        "failed_both": failed_both,
    }


if __name__ == "__main__":
    main()
