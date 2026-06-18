"""
Tests for bot/health_monitor.py — the strategy degradation kill-switch layer.

These drive the pure check functions and the StrategyHealthMonitor directly with
synthetic trade streams; no filesystem or network access is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.health_monitor import (
    CRITICAL,
    EMA_BASELINE,
    MONITORING,
    OK,
    WARNING,
    StrategyHealthMonitor,
    check_drawdown_breach,
    check_execution_drag,
    check_losing_streak,
    check_win_rate_drift,
    format_alert,
)

MIN_TRADES = 20


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        health_monitoring=True,
        health_auto_pause=False,
        health_min_trades_for_check=MIN_TRADES,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Win-rate drift ───────────────────────────────────────────────────────────

class TestWinRateDrift:
    def test_drift_below_threshold_triggers_warning(self):
        # baseline 47.7%, floor = 47.7 - 15 = 32.7%. 6/20 wins = 30% < floor.
        pnls = [100.0] * 6 + [-100.0] * 14
        check = check_win_rate_drift(pnls, EMA_BASELINE, MIN_TRADES)
        assert check.status == WARNING
        assert check.observed == 30.0
        assert check.expected == EMA_BASELINE.expected_win_rate

    def test_win_rate_just_above_floor_is_ok(self):
        # 7/20 = 35% > 32.7% floor → OK (no false alarm right at the boundary).
        pnls = [100.0] * 7 + [-100.0] * 13
        check = check_win_rate_drift(pnls, EMA_BASELINE, MIN_TRADES)
        assert check.status == OK

    def test_healthy_win_rate_is_ok(self):
        pnls = [100.0] * 12 + [-100.0] * 8  # 60%
        assert check_win_rate_drift(pnls, EMA_BASELINE, MIN_TRADES).status == OK


# ── Drawdown breach ──────────────────────────────────────────────────────────

class TestDrawdownBreach:
    def test_drawdown_beyond_limit_triggers_critical(self):
        # baseline 8% → limit 12%. Build a peak then a ~13% trough.
        pnls = [100.0] * 7 + [-1000.0] * 13
        check = check_drawdown_breach(pnls, EMA_BASELINE, MIN_TRADES)
        assert check.status == CRITICAL
        assert check.observed > check.expected  # observed DD exceeds the 12% limit

    def test_shallow_drawdown_is_ok(self):
        pnls = [100.0] * 10 + [-50.0] * 10  # tiny drawdown on a $100k book
        assert check_drawdown_breach(pnls, EMA_BASELINE, MIN_TRADES).status == OK


# ── Losing streak ────────────────────────────────────────────────────────────

class TestLosingStreak:
    def test_streak_beyond_backtest_worst_triggers_warning(self):
        # worst backtest streak = 7; 8 consecutive losers should warn.
        pnls = [100.0] * 12 + [-100.0] * 8
        check = check_losing_streak(pnls, EMA_BASELINE, MIN_TRADES)
        assert check.status == WARNING
        assert check.observed == 8

    def test_streak_within_backtest_worst_is_ok(self):
        # max streak of 5, interleaved so nothing exceeds 7.
        pnls = ([-100.0] * 5 + [100.0]) * 4  # 24 trades, longest run = 5
        check = check_losing_streak(pnls, EMA_BASELINE, MIN_TRADES)
        assert check.status == OK
        assert check.observed == 5


# ── Insufficient data ────────────────────────────────────────────────────────

class TestInsufficientData:
    def test_few_trades_returns_monitoring_not_alarm(self):
        pnls = [-100.0] * 5  # all losers, but only 5 trades
        for check in (
            check_win_rate_drift(pnls, EMA_BASELINE, MIN_TRADES),
            check_drawdown_breach(pnls, EMA_BASELINE, MIN_TRADES),
            check_losing_streak(pnls, EMA_BASELINE, MIN_TRADES),
        ):
            assert check.status == MONITORING
            assert "insufficient data" in check.explanation.lower()

    def test_monitor_aggregate_is_monitoring_with_no_alerts(self):
        monitor = StrategyHealthMonitor(_settings())
        health = monitor.evaluate_ema([-100.0] * 5, shadow_stats=None,
                                      days_since_last_signal=3)
        assert health.status == MONITORING
        assert health.sufficient_data is False
        assert health.alerts == []  # no WARNING/CRITICAL despite a terrible win rate


# ── Execution drag breach ────────────────────────────────────────────────────

class TestExecutionDrag:
    def test_negative_realistic_while_ideal_positive_is_critical(self):
        shadow = {
            "total_trades": 30,
            "ideal":     {"total_return_pct": 4.0},
            "realistic": {"total_return_pct": -1.5},
        }
        check = check_execution_drag(shadow)
        assert check.status == CRITICAL
        assert "does not survive" in check.explanation

    def test_both_positive_is_ok(self):
        shadow = {
            "total_trades": 30,
            "ideal":     {"total_return_pct": 4.0},
            "realistic": {"total_return_pct": 3.1},
        }
        assert check_execution_drag(shadow).status == OK

    def test_no_shadow_trades_is_monitoring(self):
        assert check_execution_drag(None).status == MONITORING
        assert check_execution_drag({"total_trades": 0}).status == MONITORING

    def test_drag_breach_drives_strategy_critical(self):
        monitor = StrategyHealthMonitor(_settings())
        shadow = {
            "total_trades": 30,
            "ideal":     {"total_return_pct": 4.0},
            "realistic": {"total_return_pct": -1.5},
        }
        # 25 healthy trades so other checks pass; drag check should dominate.
        pnls = [100.0] * 13 + [-100.0] * 12
        health = monitor.evaluate_ema(pnls, shadow_stats=shadow, days_since_last_signal=2)
        assert health.status == CRITICAL
        assert any(c.name == "execution_drag" and c.status == CRITICAL for c in health.alerts)


# ── Auto-pause is off by default ─────────────────────────────────────────────

class TestAutoPauseDefault:
    def test_auto_pause_disabled_by_default(self):
        monitor = StrategyHealthMonitor(_settings())
        assert monitor.auto_pause is False

    def test_default_settings_object_does_not_enable_auto_pause(self):
        # A bare settings object (no health_* attrs) must still default to alert-only.
        monitor = StrategyHealthMonitor(SimpleNamespace())
        assert monitor.auto_pause is False
        assert monitor.enabled is True
        assert monitor.min_trades == MIN_TRADES

    def test_critical_health_does_not_mutate_any_strategy_state(self):
        # The monitor only reports; evaluating a CRITICAL result must not flip an
        # enabled flag on the settings/strategy. Alert-only discipline.
        settings = _settings()
        monitor = StrategyHealthMonitor(settings)
        shadow = {"total_trades": 30,
                  "ideal": {"total_return_pct": 4.0},
                  "realistic": {"total_return_pct": -1.5}}
        health = monitor.evaluate_ema([100.0] * 13 + [-100.0] * 12,
                                      shadow_stats=shadow, days_since_last_signal=1)
        assert health.status == CRITICAL          # alert is raised
        assert monitor.auto_pause is False          # but auto-pause stayed off
        assert not hasattr(settings, "enabled")     # nothing disabled the strategy

    def test_format_alert_renders_for_alert_statuses_only(self):
        monitor = StrategyHealthMonitor(_settings())
        # Healthy strategy → no alert payload. Interleaved so no long loss streak.
        healthy = monitor.evaluate_ema([100.0, -100.0] * 12 + [100.0],
                                       shadow_stats={"total_trades": 30,
                                                     "ideal": {"total_return_pct": 4.0},
                                                     "realistic": {"total_return_pct": 3.1}},
                                       days_since_last_signal=2)
        assert healthy.status == OK
        assert format_alert(healthy) is None

        # Degraded strategy → a titled, coloured alert.
        degraded = monitor.evaluate_ema([100.0] * 6 + [-100.0] * 14,
                                        shadow_stats=None, days_since_last_signal=2)
        title, message, colour = format_alert(degraded)
        assert "EMA" in title
        assert colour in ("amber", "red")
        assert "Review before continuing live trading." in message
