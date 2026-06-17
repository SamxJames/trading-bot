"""
Tests for the Dual Momentum (GEM) strategy.

GEM is a monthly portfolio-allocation engine, so these tests exercise the
allocation decision (compute_target), the monthly cadence (should_rebalance),
and graceful degradation when there isn't enough lookback history — not the
bar-by-bar machinery the EMA strategies use.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from bot.strategies.dual_momentum import DualMomentumStrategy
from bot.strategies.registry import REGISTRY, get_strategy


def _strategy() -> DualMomentumStrategy:
    return DualMomentumStrategy(
        gem_lookback_months=12,
        gem_assets={"us": "SPY", "intl": "VEU", "bonds": "AGG"},
    )


# ── Allocation decision ─────────────────────────────────────────────────────

def test_returns_us_when_equities_on_and_us_beats_intl():
    """SPY 12mo > T-bill (equities on) AND SPY 12mo > VEU 12mo -> hold SPY."""
    s = _strategy()
    target = s.compute_target(us_12m=0.20, intl_12m=0.10, tbill_12m=0.05)
    assert target == "SPY"
    assert s.target_allocation == "SPY"


def test_returns_intl_when_equities_on_and_intl_beats_us():
    """Equities on (SPY 12mo > T-bill) but VEU 12mo > SPY 12mo -> hold VEU."""
    s = _strategy()
    target = s.compute_target(us_12m=0.12, intl_12m=0.25, tbill_12m=0.05)
    assert target == "VEU"
    assert s.target_allocation == "VEU"


def test_returns_bonds_when_absolute_momentum_negative():
    """SPY 12mo < T-bill -> absolute momentum negative -> hold AGG, regardless
    of how strong international looks."""
    s = _strategy()
    target = s.compute_target(us_12m=0.01, intl_12m=0.30, tbill_12m=0.04)
    assert target == "AGG"
    assert s.target_allocation == "AGG"


def test_us_wins_ties_against_intl():
    """On an exact relative-momentum tie, US (home market) is held — matches the
    >= tie-break in the backtest."""
    s = _strategy()
    assert s.compute_target(us_12m=0.15, intl_12m=0.15, tbill_12m=0.02) == "SPY"


def test_last_readings_recorded():
    s = _strategy()
    s.compute_target(us_12m=0.2, intl_12m=0.1, tbill_12m=0.05)
    assert s.last_readings == {"us_12m": 0.2, "intl_12m": 0.1, "tbill_12m": 0.05}


# ── Monthly cadence ──────────────────────────────────────────────────────────

def test_should_rebalance_true_on_month_end_weekday():
    """2024-01-31 is a Wednesday and the last weekday of January -> rebalance."""
    s = _strategy()
    assert s.should_rebalance(date(2024, 1, 31)) is True


def test_should_rebalance_false_mid_month():
    s = _strategy()
    assert s.should_rebalance(date(2024, 1, 15)) is False


def test_should_rebalance_rolls_back_over_weekend():
    """Aug 2024 ends on Sat 31st; the last trading day is Fri the 30th."""
    s = _strategy()
    assert s.should_rebalance(date(2024, 8, 30)) is True   # Friday
    assert s.should_rebalance(date(2024, 8, 31)) is False  # Saturday


def test_should_rebalance_accepts_datetime():
    s = _strategy()
    assert s.should_rebalance(datetime(2024, 1, 31, 20, 45)) is True


# ── Graceful degradation on insufficient lookback ────────────────────────────

def test_trailing_return_none_with_insufficient_history():
    """Need lookback + 1 (=13) monthly points; fewer -> None, not a crash."""
    s = _strategy()
    prices = [100.0 + i for i in range(10)]  # only 10 months
    assert s.trailing_return(prices) is None


def test_trailing_return_computed_with_enough_history():
    s = _strategy()
    prices = [100.0] * 13
    prices[-1] = 120.0
    # 13 points -> compares prices[-1] (120) vs prices[0] (100) = +20%
    assert s.trailing_return(prices) == pytest.approx(0.20)


def test_compute_target_defaults_to_bonds_when_data_missing():
    """If the US / T-bill reading can't be computed, GEM preserves capital in
    bonds rather than guessing at an equity allocation."""
    s = _strategy()
    assert s.compute_target(us_12m=None, intl_12m=0.3, tbill_12m=0.04) == "AGG"
    assert s.compute_target(us_12m=0.2, intl_12m=0.1, tbill_12m=None) == "AGG"


# ── Registry wiring ──────────────────────────────────────────────────────────

def test_registered_in_registry():
    assert REGISTRY.get("dual_momentum") is DualMomentumStrategy


def test_get_strategy_constructs_with_config_kwargs():
    """get_strategy forwards all config kwargs; the GEM strategy must ignore the
    EMA-specific ones (fast_period, etc.) without erroring."""
    s = get_strategy(
        "dual_momentum",
        fast_period=20,
        slow_period=50,
        gem_lookback_months=12,
        gem_assets={"us": "SPY", "intl": "VEU", "bonds": "AGG"},
    )
    assert isinstance(s, DualMomentumStrategy)
    assert s.gem_lookback_months == 12
