"""
Tests for bot/weekly_summary.py — Filter Activity (30d) and Signal Funnel
(30d) sections of the weekly Discord message.

All tests mock the analytics.json / signal_log.jsonl loaders directly
(`_load_filter_blocks_30d`, `_count_signal_evaluations_30d`,
`_load_signal_funnel`) — no real filesystem or network access.
"""

from __future__ import annotations

from unittest.mock import patch

import bot.weekly_summary as ws
from bot.weekly_summary import FILTER_BAR_WIDTH, FILTER_BLOCK_KEYS


def _empty_filter_blocks() -> dict[str, int]:
    return {k: 0 for k in FILTER_BLOCK_KEYS}


# ── build_filter_activity_lines() ───────────────────────────────────────────

class TestBuildFilterActivityLines:
    def test_filter_blocks_render_as_bar_chart(self):
        filter_blocks = _empty_filter_blocks()
        filter_blocks["volume"] = 10
        filter_blocks["trend_sma"] = 5

        with (
            patch.object(ws, "_load_filter_blocks_30d", return_value=filter_blocks),
            patch.object(ws, "_count_signal_evaluations_30d", return_value=100),
        ):
            lines = ws.build_filter_activity_lines()

        assert len(lines) == len(FILTER_BLOCK_KEYS)

        volume_line = next(l for l in lines if "volume" in l)
        trend_line = next(l for l in lines if "trend_sma" in l)

        # volume has the max count -> fully filled bar
        assert f"`{'█' * FILTER_BAR_WIDTH}`" in volume_line
        assert "volume: 10" in volume_line

        # trend_sma is half of the max -> half-filled bar
        assert f"`{'█' * 5}{'░' * 5}`" in trend_line
        assert "trend_sma: 5" in trend_line

    def test_block_rate_over_60_percent_flagged(self):
        filter_blocks = _empty_filter_blocks()
        filter_blocks["volume"] = 70

        with (
            patch.object(ws, "_load_filter_blocks_30d", return_value=filter_blocks),
            patch.object(ws, "_count_signal_evaluations_30d", return_value=100),
        ):
            lines = ws.build_filter_activity_lines()

        volume_line = next(l for l in lines if "volume" in l)
        assert "⚠️ potentially over-active" in volume_line

    def test_block_rate_under_60_percent_not_flagged(self):
        filter_blocks = _empty_filter_blocks()
        filter_blocks["volume"] = 50

        with (
            patch.object(ws, "_load_filter_blocks_30d", return_value=filter_blocks),
            patch.object(ws, "_count_signal_evaluations_30d", return_value=100),
        ):
            lines = ws.build_filter_activity_lines()

        volume_line = next(l for l in lines if "volume" in l)
        assert "⚠️" not in volume_line

    def test_analytics_json_missing_fallback(self):
        with patch.object(ws, "_load_filter_blocks_30d", return_value={}):
            lines = ws.build_filter_activity_lines()

        assert lines == ["No filter data yet — check back after first live run"]

    def test_signal_log_missing_does_not_crash(self):
        filter_blocks = _empty_filter_blocks()
        filter_blocks["volume"] = 5

        with (
            patch.object(ws, "_load_filter_blocks_30d", return_value=filter_blocks),
            patch.object(ws, "_count_signal_evaluations_30d", return_value=0),
        ):
            lines = ws.build_filter_activity_lines()

        assert len(lines) == len(FILTER_BLOCK_KEYS)
        # total_evaluated == 0 -> block_rate is 0.0 for every filter, never flagged
        assert all("⚠️" not in l for l in lines)


# ── build_signal_funnel_lines() ──────────────────────────────────────────────

class TestBuildSignalFunnelLines:
    def test_funnel_renders_with_crossover_data(self):
        funnel = {
            "crossovers_30d": 14,
            "signals_fired_30d": 2,
            "signals_blocked_30d": 12,
            "filter_funnel": {
                "volume": 7,
                "weekly_ema": 3,
                "trend_sma": 2,
                "rsi_overbought": 0,
            },
        }

        with patch.object(ws, "_load_signal_funnel", return_value=funnel):
            lines = ws.build_signal_funnel_lines()

        joined = "\n".join(lines)
        assert "EMA crossovers detected: 14" in joined
        assert "Passed all filters:      2  (14%)" in joined
        assert "Blocked by filters:      12  (86%)" in joined
        assert "Top blockers:" in joined

        # Top blockers sorted descending, zero-count filters excluded
        assert f"  {'volume':<12} 7 blocks" in lines
        assert f"  {'weekly_ema':<12} 3 blocks" in lines
        assert f"  {'trend_sma':<12} 2 blocks" in lines
        assert not any("rsi_overbought" in l for l in lines)

    def test_no_crossovers_message(self):
        funnel = {
            "crossovers_30d": 0,
            "signals_fired_30d": 0,
            "signals_blocked_30d": 0,
            "filter_funnel": {},
        }

        with patch.object(ws, "_load_signal_funnel", return_value=funnel):
            lines = ws.build_signal_funnel_lines()

        assert lines == [
            "No EMA crossovers detected in last 30 days — market may be trending sideways."
        ]

    def test_over_blocking_warning(self):
        funnel = {
            "crossovers_30d": 5,
            "signals_fired_30d": 0,
            "signals_blocked_30d": 5,
            "filter_funnel": {"volume": 5},
        }

        with patch.object(ws, "_load_signal_funnel", return_value=funnel):
            lines = ws.build_signal_funnel_lines()

        assert "⚠️ All crossovers blocked by filters — consider reviewing thresholds." in lines
