# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Personal algorithmic **paper-trading** bot for Alpaca's paper API, running a
9-filter EMA-crossover strategy (`ema_cross_filtered`) across a 5-ticker
universe (SPY, QQQ, GLD, AAPL, NVDA). A GitHub Actions cron job runs
`python -m bot job` once per weekday after market close — it evaluates each
ticker's signal, applies risk/regime/earnings/correlation gates, optionally
places a paper order, and posts the outcome to Discord. Backtesting
(`python -m bot backtest`) and the weekly summary use yfinance/CSV data only
and make **zero** live broker calls.

Alongside the daily EMA job, the repo also runs a shadow trader and a strategy
health monitor (both inside the daily job), a separate **monthly** GEM / dual-
momentum rebalance, and a **quarterly** walk-forward parameter re-optimisation.
See "Beyond the daily EMA job" below.

## Critical rules

- Never hardcode credentials or API keys.
- Never commit `trades_live.csv` or `signal_log.jsonl` with real or synthetic data.
- Never modify `config.yaml` without running the full backtest to validate the change.
- Never add a new filter without updating `last_filter_snapshot` in `_compute_filter_snapshot()`.
- Never change exit logic (stop loss, trailing stop, take profit) without updating the relevant tests in `test_ema_cross.py`.
- Always run pytest before committing — 87 tests must pass.
- Always use `[skip ci]` in commit messages to prevent recursive Actions triggers.
- Always run `git pull` before starting a session to avoid rebase conflicts with Actions commits.

## Architecture

### Config — `bot/config.py`

`Settings` (pydantic-settings) is the **single source of truth**. Priority
(highest first): init kwargs → env vars → `.env` → `config.yaml` → field
defaults. Credential fields use `AliasChoices` so `APCA_API_KEY_ID` etc. map
unambiguously to `settings.apca_api_key_id`.

**Gotcha**: `YamlConfigSource.__call__()` only passes through `config.yaml`
keys that match a declared field on `Settings` (`k in known` filter). Adding
a new tunable to `config.yaml` does nothing unless a matching field is also
added to `Settings` — it's silently dropped, and `getattr(settings, "x",
default)` elsewhere silently falls back to `default`.

### Data flow: cron → Discord

1. `.github/workflows/daily_job.yml` — Mon–Fri 20:45 UTC cron (or manual
   `workflow_dispatch`) checks out the repo, installs deps, verifies
   `APCA_*` secrets are set (exits 1 if not; warns but continues if
   `DISCORD_WEBHOOK_URL` is unset), then runs `python -m bot job`.
2. `bot/job.py::run_job` → `_run`:
   - `BrokerClient.get_account()` — credential/connectivity sanity check.
   - `RegimeFilter.fetch()` — VIX + SPY/SMA(200), once for the whole run.
   - `EarningsFilter.fetch()` — earnings dates for all tickers, once.
   - `CorrelationGuard.fetch_dynamic()` — correlation matrix, once (if `dynamic_correlation`).
   - `broker.is_trading_day(today)` — exit silently (exit 0) on weekends/holidays.
   - Per ticker (`_process_ticker`):
     - Fetch ~60 days warm-up + today's bar (`fetch_bars`).
     - If today's bar isn't published yet → log `bar_not_ready`, return (exit 0).
     - Build a fresh strategy instance (`get_strategy`), `on_start()`, feed
       every bar through `on_bar()` — only the signal from *today's* bar is used.
     - Write a `signal_evaluation` record to `bot/trade_journal/signal_log.jsonl`
       (filter snapshot + `blocked_by`), regardless of whether a signal fired.
     - No signal/HOLD → "Daily Heartbeat" to Discord, return.
     - Signal fired → "Signal Detected" to Discord (shows regime/earnings/correlation gate results).
     - `RiskManager.evaluate()` — max positions / notional / drawdown halt.
     - BUY only: regime (`allow_buy`), earnings (`is_blackout`), correlation
       (`is_blocked`) gates — any block → "Signal Blocked" to Discord, return.
     - Size the order: `atr_position_size()` if `atr_sizing`, else flat
       `max_notional_per_trade`.
     - `broker.place_market_order(...)` (or just logged in `--dry-run`).
       On exception or `order.status == "rejected"`: call
       `strategy.force_exit_position()`, notify "Order Failed"/"Order
       Rejected", return.
     - Notify "Trade Opened" (BUY) or "Trade Closed" (SELL).
   - After the ticker loop, still inside `_run`: `ShadowTrader.run()` shadow-
     trades every ticker against real bars (records to the shadow journals,
     places no orders), then — if `health_monitoring` (default true) —
     `_run_health_monitor()` compares live/shadow metrics vs backtest baselines
     and Discord-alerts on WARNING/CRITICAL.
3. `job_complete` is appended to `signal_log.jsonl` with elapsed time.
4. Any unhandled exception anywhere in `_run` → `notify.send("Job Failed", ...)`,
   write a `job_failed` record, then **re-raise** so GitHub Actions marks the run red.
5. After the job step, the workflow runs `scripts/analyse_trades.py --out docs/`,
   commits `docs/analytics.json` back to the repo (`[skip ci]`), and syncs
   `trades_live.csv` to Google Drive.

### The 9-filter stack (`ema_cross_filtered`)

Filters 1, 2, 4, 9 and both exits live in
`bot/strategies/ema_cross_filtered.py`; filters 5–8 are job-level gates
evaluated by `bot/job.py` after a BUY is emitted.

| # | Filter | Where | Gate |
|---|--------|-------|------|
| 1 | Trend SMA | strategy | `close > SMA(trend_sma_period)` |
| 2 | RSI overbought | strategy (lazy) | `RSI(rsi_period) < rsi_overbought`, computed only after Filter 1 passes |
| 3 | *(retired)* static stop loss | — | superseded by Exit 1; number kept retired so 4–9 match the 2005-2025 sweep (`scripts/tune_filters.py`) |
| 4 | Volume confirmation | strategy | `volume >= volume_multiplier × N-bar avg volume` |
| 5 | VIX regime gate | `bot/filters/regime.py` | blocks all BUYs if `VIX > vix_threshold`; fails open |
| 6 | SPY macro gate | `bot/filters/regime.py` | blocks non-SPY BUYs if `SPY < SMA(spy_sma_period)`; fails open |
| 7 | Earnings blackout | `bot/earnings.py` | blocks BUYs within `earnings_blackout_days` of an earnings date; fails open |
| 8 | Correlation guard | `bot/correlation.py` | blocks BUY if correlation with an open position `>= max_correlation`; static matrix fallback |
| 9 | Weekly EMA confirmation | strategy | `weekly EMA(20) > weekly EMA(50)`; fails permissive |

**Exits** (strategy):
- **Exit 1 — Trailing stop**: floor starts at `entry_price × (1 - stop_loss_pct/100)`,
  ratchets up via `highest_price × (1 - trailing_stop_pct/100)`, never moves down.
- **Exit 2 — Take-profit**: `entry_price + (entry_price × stop_loss_pct/100) × take_profit_rr`
  (set `take_profit_rr=0` to disable).

### Beyond the daily EMA job

Four subsystems run alongside or independently of the daily EMA job:

- **Shadow trading (`bot/shadow.py`)** — runs inside `_run` on every daily job.
  Evaluates the full 9-filter strategy against real Alpaca bars and records the
  *hypothetical* entries/exits/P&L to `bot/trade_journal/shadow_trades.csv` /
  `shadow_log.jsonl` **without placing any order**, closing the backtest-to-live
  gap. `apply_execution_costs()` here models slippage/commission and is reused by
  `scripts/backtest_with_costs.py`.
- **Health monitor (`bot/health_monitor.py`)** — runs at the tail of `_run` when
  `health_monitoring` (default true). Computes rolling win-rate / Sharpe /
  drawdown from the journals for each strategy (EMA and GEM separately),
  compares them to hardcoded 2005-2025 backtest baselines, and emits Discord
  WARNING/CRITICAL alerts on degradation. `scripts/analyse_trades.py` recomputes
  the same health block for the dashboard.
- **GEM / Dual Momentum (`bot/strategies/dual_momentum.py`, `bot/rebalance.py`)**
  — a **monthly** portfolio strategy, completely separate from the per-bar EMA
  job. Holds one of SPY / VEU / AGG at 100% based on 12-month absolute + relative
  momentum (T-bill via `^IRX`); it deliberately does **not** use the 9-filter
  stack. Run via `python -m bot rebalance` on
  `.github/workflows/monthly_rebalance.yml` (last trading day of the month).
  GEM's current holding is read from `bot/trade_journal/rebalance_log.jsonl`, not
  inferred from the commingled paper account (the EMA bot may also hold SPY).
- **Walk-forward (`scripts/walk_forward.py`)** — **quarterly** parameter
  re-optimisation on `.github/workflows/quarterly_reoptimise.yml`. Grid-searches
  params over rolling 5y-train / 1y-validate windows. With `--promote` it may
  rewrite `config.yaml`, but only when (a) at least `MIN_PROMOTED_WINDOWS` (3)
  windows agree on the same param set **and** (b) the candidate's average
  validation Sharpe beats the current incumbent's over the same windows. A
  first-business-day-of-quarter self-guard makes the broad cron idempotent. See
  the walk-forward-params gotcha below.

### Key files and ownership

- `bot/config.py` — `Settings`, the only place that reads env/`.env`/`config.yaml`.
- `bot/job.py` — daily orchestration, signal audit log, order placement.
- `bot/strategies/ema_cross_filtered.py` — filters 1/2/4/9, both exits, `last_filter_snapshot`.
- `bot/strategies/registry.py` — `get_strategy(name, **kwargs)` / `REGISTRY`.
- `bot/filters/regime.py` — filters 5/6 (VIX, SPY macro).
- `bot/earnings.py` — filter 7 (earnings blackout).
- `bot/correlation.py` — filter 8 (correlation guard, static matrix).
- `bot/risk/manager.py` — `RiskManager` (positions, notional, drawdown halt, kill switch).
- `bot/risk/sizing.py` — `atr_position_size()`.
- `bot/execution/broker.py` — async wrapper over `alpaca-py`.
- `bot/notify.py` — Discord embed sender (no-op if webhook unset).
- `bot/weekly_summary.py` — Monday Discord performance summary.
- `bot/shadow.py` — shadow trader + execution-cost model (runs inside `_run`).
- `bot/health_monitor.py` — strategy-degradation monitor + Discord alerts.
- `bot/strategies/dual_momentum.py` — GEM monthly dual-momentum strategy.
- `bot/rebalance.py` — monthly GEM rebalance entry point (`python -m bot rebalance`).
- `scripts/walk_forward.py` — quarterly walk-forward param re-optimisation.
- `bot/data/historical.py` / `yfinance_historical.py` — bar fetching (Alpaca / yfinance).
- `scripts/analyse_trades.py` — builds `docs/analytics.json` for the dashboard.
- `docs/` — static dashboard, deployed via Vercel (`vercel.json`, `outputDirectory: docs`).

## Common tasks

```powershell
# Run all tests
python -m pytest

# Single test
python -m pytest tests/test_job.py::test_buy_signal_approved_places_order_and_notifies

# Backtest (yfinance only, zero Alpaca order calls)
python -m bot backtest --strategy ema_cross_filtered --tickers AAPL --from 2024-01-01 --to 2024-06-01
python -m bot backtest --strategy ema_cross_filtered --tickers AAPL,MSFT,SPY --from 2020-01-01 --to 2024-06-01

# Compare strategies side by side
python -m bot compare --tickers AAPL --from 2024-01-01 --to 2024-06-01

# Filter parameter sweep (2005-2025, no Alpaca creds, no config.yaml changes)
python scripts/tune_filters.py

# Filter isolation (which exit/entry filter drives a result change)
python scripts/filter_isolation.py

# Synthetic dashboard data test — generate, analyse, then delete the
# generated trade_journal files (never commit them)
python scripts/generate_synthetic_trades.py
python scripts/analyse_trades.py --out docs/ --json
rm bot/trade_journal/trades_live.csv

# Daily job (dry run — no orders placed)
python -m bot job --dry-run

# Weekly Discord summary (dry run — prints instead of posting)
python -m bot weekly --dry-run

# Monthly GEM rebalance (dry run — no orders placed)
python -m bot rebalance --dry-run

# Walk-forward re-optimisation (yfinance only; --promote is gated by 3-window
# consensus + incumbent Sharpe comparison + first-business-day-of-quarter guard)
python scripts/walk_forward.py --out results/

# Dashboard deploy — automatic via Vercel on push to main (vercel.json,
# outputDirectory: docs); no manual deploy step needed.
```

## Known gotchas

1. **Pydantic silent drop** — see Config section above. A new `config.yaml`
   key with no matching `Settings` field is silently ignored.

2. **Walk-forward output is advisory — the live params are `fast_period=20` /
   `stop_loss_pct=2.5`.** These are the validated local params. The quarterly
   walk-forward once emitted `fast_period=25` / `stop_loss_pct=2.0` and that
   output was **discarded**: its provenance was a single 6-trade 2018 window
   winning a 2-way tie, the incumbent was never compared, and the cron's
   OR-semantics re-ran the job daily on a frozen dataset. `scripts/walk_forward.py`
   now requires 3-window consensus (`MIN_PROMOTED_WINDOWS`) **and** an incumbent
   Sharpe comparison before it may rewrite `config.yaml`, and self-guards to the
   first business day of the quarter. Don't reinstate 25/2.0 without fresh,
   multi-window evidence.

3. **`bar_not_ready` is a normal, silent no-op.** Alpaca only returns
   completed bars; if today's bar isn't published yet when the cron fires,
   `_process_ticker` logs `bar_not_ready` and returns (exit 0) — the job
   simply re-runs tomorrow. Not a failure.

4. **`_pending_entry` / fill-price lifecycle.** A BUY sets `_in_position=True`,
   a *provisional* `_entry_price = close`, and `_pending_entry = True`. The
   real fill price (`bar.open` of the *next* bar) overwrites it, and stops/
   take-profit are anchored to that. If that next bar has `open <= 0` (bad
   data), the guard logs `pending_entry_skipped` and waits another bar. In
   live trading this never crosses job runs — `job.py` builds a fresh
   strategy instance and calls `on_start()` every run, so multi-bar survival
   only matters within one run's warm-up loop or a backtest.

5. **`force_exit_position()` resyncs strategy state with the broker.** If a
   BUY order raises or comes back `status == "rejected"`, `job.py` calls
   `strategy.force_exit_position()` to reset `_in_position`/`_pending_entry`/
   `_entry_price` to flat. Without this the strategy would believe it holds a
   position the broker never opened and would stop emitting signals for that
   ticker.

6. **Filters 5–9 all "fail open" / "fail permissive" on data errors.**
   `RegimeFilter`, `EarningsFilter`, `CorrelationGuard`, and the weekly EMA
   check all return `evaluated: False, passed: True` if their yfinance call
   fails — a data outage never silently halts trading, but it also means a
   *persistent* outage silently disables those filters with only a
   `*_fetch_failed` / `*_data_unavailable` warning log, no alert.

7. **Correlation guard's static matrix is the real fallback today.**
   `bot/correlation.py::_STATIC_CORRELATIONS` is a hardcoded 60-day matrix
   from a point-in-time backtest. `fetch_dynamic()` only replaces it when
   `dynamic_correlation` is true *and* the yfinance call succeeds — and
   `config.yaml` currently sets `dynamic_correlation: false`, so every run uses
   the static table.

8. **`highest_correlation()` return type vs. `_build_filter_record()`
   check.** `CorrelationGuard.highest_correlation()` returns a
   `(ticker, correlation)` tuple, but `bot/job.py::_build_filter_record()`
   tests `isinstance(corr_value, (int, float))` — a tuple never matches, so
   the `"correlation"` entry in `signal_log.jsonl` / `last_filter_snapshot`
   always reports `evaluated: False`. The actual BUY gate
   (`corr_guard.is_blocked()`) is unaffected — only the audit log
   under-reports.

9. **Three different trade-journal CSV schemas coexist.**
   `bot/weekly_summary.py::load_trades()` reads `trades_live.csv` (repo root)
   with columns `ticker, side, entry_price, exit_price, pnl, entry_ts,
   exit_ts, qty`. `scripts/analyse_trades.py` reads
   `bot/trade_journal/trades_live.csv` with columns `id, timestamp, ticker,
   side, qty, entry_price, exit_price, pnl_usd, reason, strategy`.
   `bot/logging/logger.py::TradeJournal._COLUMNS` defines a third schema
   again. Nothing in the live job currently writes any of these files — if
   you add a writer, pick one schema/path and update all readers, or the
   dashboard/weekly summary will keep reporting "zero trades".

10. **`analyse_trades.py`'s plain-text summary crashes on Windows.**
    `_print_summary()` prints Unicode box-drawing characters (`─`) that raise
    `UnicodeEncodeError` on the default Windows cp1252 console. Always run it
    with `--json` and/or `--out docs/` on Windows.

11. **`git rebase` inverts `--ours`/`--theirs`.** The Actions bot commits
    `docs/analytics.json` frequently, so local work usually has to be rebased
    onto `origin/main`. During a **rebase**, `--ours` = the branch you're
    rebasing *onto* (the remote upstream) and `--theirs` = your local commits
    being replayed — the **opposite** of the merge convention. To keep a local
    file wholesale through a rebase conflict, use `git checkout <local-ref> --
    <file>` (or `--theirs`), never `--ours` (which takes the remote version).

## What not to build

- **No ML-based signals.** The backtest framework, sweep scripts, and tests
  are all built around deterministic rule-based filters; there's no
  training/inference infrastructure and adding one is out of scope.
- **No IWM or TLT.** Both were removed from the ticker universe after the
  2005-2025 backtest (IWM: 25% win rate, 0.22x R:R, negative Sharpe; TLT: 0%
  win rate — see `config.yaml` comments). Don't re-add them without new
  backtest evidence.
- **No intraday bars.** `timeframe` is `1Day` everywhere and the daily job
  runs once after market close; switching to `1Min`/`1Hour` would require
  reworking the scheduler, warm-up windows, and every filter's lookback math.
- **No more than 9 filters without removing one first.** The numbering is
  load-bearing for `scripts/tune_filters.py` / `scripts/filter_isolation.py`
  sweep results and for `last_filter_snapshot`'s fixed key set
  (`FILTER_KEYS` in `bot/job.py` / `FILTER_BLOCK_KEYS` in
  `scripts/analyse_trades.py`). Retire a filter's number (like Filter 3)
  rather than reusing or appending past 9.

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
- `.github/workflows/monthly_rebalance.yml` — runs `python -m bot rebalance`
  on the last-trading-day window each month (GEM / dual momentum).
- `.github/workflows/quarterly_reoptimise.yml` — runs
  `scripts/walk_forward.py --promote` quarterly. The cron is intentionally broad
  (days 1–7 / Mondays of quarter months); the script self-guards to the first
  business day so extra firings are no-ops.
