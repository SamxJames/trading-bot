"""Shared pytest fixtures.

Isolates every test from the production trade-journal files. `bot.job` writes
`signal_evaluation` / `job_complete` records to a module-level
`SIGNAL_LOG_PATH` (bot/trade_journal/signal_log.jsonl); without isolation the
job tests append real rows to that file on every `pytest` run. The autouse
fixture below redirects the journal write paths into a per-test `tmp_path` so
the production journal on disk is never touched.

(test_rebalance.py already patches `bot.rebalance.REBALANCE_LOG_PATH`
per-test, so that writer is covered by its own tests.)
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_trade_journal(tmp_path, monkeypatch):
    """Point every trade-journal write path at a temp dir for the test's life."""
    journal = tmp_path / "trade_journal"
    journal.mkdir()

    import bot.job as job_mod
    monkeypatch.setattr(job_mod, "SIGNAL_LOG_PATH", journal / "signal_log.jsonl")

    # Shadow-trading writers share the same journal dir — redirect them too so
    # no test can ever append to the real files, even if one is added later.
    try:
        import bot.shadow as shadow_mod
    except ImportError:
        pass
    else:
        monkeypatch.setattr(shadow_mod, "SHADOW_TRADES_PATH", journal / "shadow_trades.csv")
        monkeypatch.setattr(shadow_mod, "SHADOW_LOG_PATH", journal / "shadow_log.jsonl")
        monkeypatch.setattr(shadow_mod, "SHADOW_STATE_PATH", journal / "shadow_state.json")

    yield
