# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Personal algorithmic **paper-trading** bot (Alpaca paper API). Pure Python,
async I/O via `asyncio.to_thread` over the synchronous `alpaca-py` SDK.
Backtesting uses yfinance and makes **zero** live broker calls.

## Commands

All commands assume the working directory is the repo root and use the
project's Python (`C:\Python312\python.exe` on the dev machine; `python` on
CI/Linux).

```powershell
# Install deps
python -m pip install -r requirements.txt

# Run all tests
python -m pytest

# Run a single test file / test
python -m pytest tests/test_job.py
python -m pytest tests/test_job.py::test_buy_signal_approved_places_order_and_notifies

# Backtest (zero Alpaca order calls — historical data only)
python -m bot backtest --strategy ema_cross_filtered --tickers AAPL --from 2024-01-01 --to 2024-06-01
python -m bot backtest --strategy ema_cross_filtered --tickers AAPL,MSFT,SPY --from 2020-01-01 --to 2024-06-01

# Compare strategies side by side
python -m bot compare --tickers AAPL --from 2024-01-01 --to 2024-06-01

# Daily scheduled job (run-once, <60s) — what GitHub Actions runs
python -m bot job --dry-run     # no orders placed, full log output
python -m bot job                # places real paper orders if a signal fires

# Continuous live paper-trading loop (WebSocket)
python -m bot live --dry-run

# Weekly Discord performance summary
python -m bot weekly --dry-run
```

## Architecture

```
config.yaml + .env  →  bot/config.py (Settings)
                              │
                       bot/job.py (_run)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  RegimeFilter          fetch_bars (per ticker)   RiskManager
  (VIX / SPY gate,      → strategy.on_bar()       (positions, drawdown,
   fetched once)        → Signal                   max_notional)
        │                     │                     │
        └─────────► allow_buy(ticker) gate ◄────────┘
                              │
                    atr_position_size (optional)
                              │
                    BrokerClient.place_market_order
                              │
                          notify.send (Discord)
```

### Config (`bot/config.py`)

`Settings` (pydantic-settings) is the **single source of truth** — nothing
else should read `os.environ` or `config.yaml` directly. Source priority
(highest first): init kwargs → env vars → `.env` → `config.yaml` → field
defaults.

**Important gotcha**: `YamlConfigSource` only passes through `config.yaml`
keys that match a declared field on `Settings` (`k in known` filter, see
`bot/config.py`). Adding a new tunable to `config.yaml` does nothing unless
a matching field is also added to the `Settings` class — it will be silently
dropped, and any `getattr(settings, "new_key", default)` call elsewhere will
silently fall back to its default.

### Daily job (`bot/job.py`)

Entry point `run_job(dry_run)` → `_run`. Per-run sequence:
1. `BrokerClient.get_account()` — environment/credentials sanity check.
2. `RegimeFilter.from_config(settings)` then `regime.fetch()` — fetches
   VIX + SPY/SMA(200) **once** for the whole run (not per ticker).
3. `broker.is_trading_day(today)` — exits silently (exit 0) on weekends/holidays.
4. Per ticker (`_process_ticker`): fetch ~60 days warm-up + today's bar →
   feed warm-up bars through the strategy → read the signal from today's bar.
5. `RiskManager.evaluate()` — position-count and notional checks.
6. For BUY signals only: `regime.allow_buy(ticker)` — VIX-too-high or
   SPY-below-SMA200 (non-SPY) blocks the entry (regime fails open/permissive
   if VIX/SPY data is unavailable).
7. Order sizing: if `settings.atr_sizing` is true, `atr_position_size()`
   (in `bot/risk/sizing.py`) scales notional inversely with the ticker's
   recent volatility (close-to-close ATR%); otherwise flat
   `max_notional_per_trade` is used. Result is converted to `qty`.
8. `broker.place_market_order(...)` (or just logged in `--dry-run`), then
   `notify.send(...)` to Discord.

Any unhandled exception is caught in `run_job`, sent to Discord, and
re-raised so GitHub Actions marks the run as failed.

### Strategies (`bot/strategies/`)

Strategies implement the `Strategy` protocol in `base.py`
(`on_start` / `on_bar` → `Signal | None` / `on_stop`) and are looked up by
name via `bot/strategies/registry.py::REGISTRY` /
`get_strategy(name, **kwargs)`. The engine (`job.py`, `main.py`,
`backtest.py`) never imports a concrete strategy class — only the registry.
`get_strategy` forwards all settings as kwargs; each strategy's `**kwargs`
constructor ignores params it doesn't use. To add a strategy: create the
class, register it in `REGISTRY`, set `strategy: <name>` in `config.yaml`.

### Risk (`bot/risk/`)

- `manager.py::RiskManager` — stateful per-run gate: max concurrent
  positions, max notional per trade, drawdown halt (`is_halted`), and
  `emergency_flatten()` (kill switch — closes all positions, cancels all
  orders).
- `sizing.py::atr_position_size` — volatility-scaled notional, clamped to
  `[min_notional, max_notional]`, falling back to `base_notional` if ATR
  can't be computed (insufficient history).

### Regime filter (`bot/filters/regime.py`)

`RegimeFilter` fetches VIX and SPY (via yfinance) once per job run and gates
BUY entries: VIX above `vix_threshold` blocks all BUYs; SPY below its
`spy_sma_period`-bar SMA blocks non-SPY BUYs (SPY itself is exempt). Always
**fails open** — if yfinance/network fails, `allow_buy()` returns `True` for
everything. `adjusted_stop_pct()` can tighten the stop loss when VIX is
elevated but below the hard threshold.

### Execution (`bot/execution/broker.py`)

Thin async wrapper over `alpaca-py`'s synchronous client (every call goes
through `asyncio.to_thread`). Provides `get_account`, `is_trading_day`,
`get_positions`, `get_open_orders`, `place_market_order`,
`place_limit_order`, `close_all_positions`, `cancel_all_orders`.

### Data (`bot/data/`)

`historical.py::fetch_bars` fetches OHLCV bars (Alpaca, with a SQLite cache
under `cache/`); `feed.py` defines the `Bar` dataclass and the live WebSocket
feed; `yfinance_historical.py` is the alternative source used by
`backtest.py`.

### Logging & notifications

`bot/logging/logger.py` configures `structlog` (`get_logger(__name__)`) and
a `TradeJournal` that writes `trades.csv`. `bot/notify.py::send(title,
message, colour)` posts Discord embeds; a blank `discord_webhook_url`
disables notifications without error.

## Testing conventions

- All Alpaca/network calls are mocked (`unittest.mock.AsyncMock`/`MagicMock`)
  — see `tests/test_job.py::_fake_settings` / `_fake_broker` / `_fake_df` for
  the standard fixtures and patch points (`bot.job.get_settings`,
  `bot.job.BrokerClient`, `bot.job.fetch_bars`, `bot.job.get_strategy`,
  `bot.job.RegimeFilter.from_config`, `bot.job.atr_position_size`,
  `bot.notify.send`).
- `_fake_settings()` is a `MagicMock` — every `Settings` field that `job.py`
  reads via plain attribute access (not `getattr(..., default)`) **must**
  be set explicitly there, otherwise it returns a `MagicMock` instead of a
  real value and arithmetic/comparisons downstream will raise `TypeError`.
- `pytest.ini` sets `asyncio_mode = auto`, so async tests just need
  `@pytest.mark.asyncio`.

## CI (GitHub Actions)

- `.github/workflows/daily_job.yml` — runs `python -m bot job` on a Mon–Fri
  cron (20:45 UTC), then regenerates `docs/analytics.json`
  (`scripts/analyse_trades.py`) and commits it back, and syncs
  `trades_live.csv` to Google Drive.
- `.github/workflows/weekly_summary.yml` — runs `python -m bot weekly` every
  Monday and regenerates/commits `docs/analytics.json`.
