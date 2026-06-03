# Trading Bot

Personal algorithmic paper-trading bot built in Python as a learning project.
**Paper trading only — no real money.**

> **Important:** The backtest does **NOT** place any orders on Alpaca.
> Your paper account will show **zero positions** after running a backtest — this is correct.
> Alpaca is only used to fetch historical price data.

---

## Architecture

```
Config (.env + config.yaml)
        │
        ▼
   Data Feed  (WebSocket live / Alpaca historical REST)
        │
        ▼
  Strategy Engine  ──→  Signal
        │
        ▼
   Risk Manager  ──→  approve / reject
        │
        ▼
  Order Execution (Alpaca paper trading)
        │
        ▼
 Logging & Trade Journal (trades.csv)
```

---

## Setup (Windows)

### Step 1 — Navigate to the project folder

```powershell
cd C:\Users\Sam\Documents\TradingBot
```

> **Always run this first.** Every command below assumes your terminal is in this folder.
> Running Python from a different folder causes "No module named bot" errors.

### Step 2 — Install dependencies

```powershell
C:\Python312\python.exe -m pip install -r requirements.txt
```

### Step 3 — Configure credentials

```powershell
copy .env.example .env
```

Then open `.env` in a text editor and replace the placeholder values with your real
Alpaca paper trading API key and secret. Get them from:
**https://app.alpaca.markets → Paper Trading → API Keys**

### Step 4 — Verify the setup

```powershell
C:\Python312\python.exe -c "import bot; print('OK')"
C:\Python312\python.exe -m bot --help
```

---

## Usage

> If `python` is not recognised, use the full path:
> `C:\Python312\python.exe -m bot backtest ...`

### Run a backtest (baseline)

```powershell
C:\Python312\python.exe -m bot backtest --strategy ema_cross --ticker AAPL --from 2024-01-01 --to 2024-06-01
```

### Run a backtest (filtered — trend + RSI + stop loss)

```powershell
C:\Python312\python.exe -m bot backtest --strategy ema_cross_filtered --ticker AAPL --from 2024-01-01 --to 2024-06-01
```

### Compare baseline vs filtered side by side

```powershell
C:\Python312\python.exe -m bot compare --ticker AAPL --from 2024-01-01 --to 2024-06-01
```

### Run live paper trading

```powershell
C:\Python312\python.exe -m bot live
```

Press **Ctrl+C** to trigger the kill switch — all open positions will be closed before exit.

### Run tests

```powershell
C:\Python312\python.exe -m pytest
```

---

## Project structure

```
TradingBot/
├── .github/
│   └── workflows/
│       └── daily_job.yml   ← GitHub Actions cron (4:15 pm ET Mon-Fri)
├── bot/
│   ├── __main__.py         ← enables python -m bot
│   ├── main.py             ← live trading loop (continuous WebSocket)
│   ├── job.py              ← daily scheduled job (run-once, <60 s)
│   ├── backtest.py         ← backtester (zero broker API calls)
│   ├── notify.py           ← Discord webhook helper (optional)
│   ├── config.py           ← pydantic-settings (reads .env + config.yaml)
│   ├── data/
│   │   ├── feed.py         ← live WebSocket data feed
│   │   └── historical.py   ← historical OHLCV fetcher + SQLite cache
│   ├── strategies/
│   │   ├── base.py         ← Strategy protocol and Signal types
│   │   ├── ema_cross.py    ← EMA crossover (baseline, no filters)
│   │   ├── ema_cross_filtered.py ← EMA + trend filter + RSI + stop loss
│   │   ├── rsi_strategy.py ← RSI oversold/overbought strategy
│   │   └── registry.py     ← name → class lookup
│   ├── risk/
│   │   └── manager.py      ← position limits, drawdown halt, kill switch
│   ├── execution/
│   │   └── broker.py       ← Alpaca REST wrapper (+ calendar check)
│   └── logging/
│       └── logger.py       ← structlog setup + TradeJournal → trades.csv
├── tests/
│   ├── test_ema_cross.py   ← strategy unit tests (no API calls)
│   ├── test_risk_manager.py
│   ├── test_rsi_strategy.py
│   └── test_job.py         ← daily job tests (all broker calls mocked)
├── cache/                  ← SQLite bar cache (auto-created; safe to delete)
├── logs/                   ← rolling log files (auto-created)
├── charts/                 ← equity curve PNGs (auto-created)
├── .env.example            ← copy to .env and fill in credentials
├── .gitignore
├── config.yaml             ← tickers, strategy, risk params
├── requirements.txt
└── README.md
```

---

## Adding a new strategy

1. Create `bot/strategies/<name>.py` implementing the `Strategy` protocol
2. Add it to `REGISTRY` in `bot/strategies/registry.py`
3. Set `strategy: <name>` in `config.yaml`

No changes to the core engine required.

---

## Config reference (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `tickers` | `[AAPL]` | Tickers to watch |
| `strategy` | `ema_cross` | Strategy name |
| `timeframe` | `1Day` | Bar timeframe (`1Min`, `1Hour`, `1Day`) |
| `fast_period` | `20` | EMA fast period |
| `slow_period` | `50` | EMA slow period |
| `trend_sma_period` | `200` | Trend filter SMA period (filtered strategy) |
| `rsi_period` | `14` | RSI period (filtered strategy) |
| `rsi_overbought` | `70` | RSI overbought threshold (filtered strategy) |
| `stop_loss_pct` | `1.5` | Stop loss % below entry (filtered strategy) |
| `max_positions` | `3` | Max concurrent open positions |
| `max_notional_per_trade` | `500` | USD notional cap per trade |
| `drawdown_halt_pct` | `5` | Halt if equity drops this % from session start |

---

## Deploying to GitHub Actions

> **Before you push:** confirm `.gitignore` includes `.env`.
> The `.env` file is for local development only — GitHub Actions uses secrets.
> Never commit `.env` to git.

### Step 1 — Push to a private GitHub repo

```powershell
# In C:\Users\Sam\Documents\TradingBot
git init
git add .
git commit -m "Initial commit"
```

Create a **private** repo on GitHub (keep API keys off public repos), then:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### Step 2 — Add GitHub Secrets

Go to your repo on GitHub:
**Settings > Secrets and variables > Actions > New repository secret**

Add these four secrets:

| Secret name           | Value                                   |
|-----------------------|-----------------------------------------|
| `APCA_API_KEY_ID`     | Your Alpaca paper trading key           |
| `APCA_API_SECRET_KEY` | Your Alpaca paper trading secret        |
| `APCA_BASE_URL`       | `https://paper-api.alpaca.markets`      |
| `DISCORD_WEBHOOK_URL` | Your Discord webhook URL (optional)     |

### Step 3 — Automatic schedule

The job runs automatically at **8:15 pm UTC (4:15 pm ET, 9:15 pm UK) Mon–Fri** via the
workflow in `.github/workflows/daily_job.yml`.

No further setup is needed — GitHub Actions picks it up on the next push.

### Step 4 — Manual trigger (for testing)

Go to **Actions tab > Daily Trading Job > Run workflow > Run workflow**.

Use this to verify the job works before relying on the schedule.
Check the run logs in the Actions tab to see which ticker was processed,
whether a signal fired, and whether an order was placed.

### Step 5 — Discord notifications (optional)

1. Create a Discord server (or use an existing one).
2. In a channel: **Edit Channel > Integrations > Webhooks > New Webhook**.
3. Copy the webhook URL and add it as the `DISCORD_WEBHOOK_URL` secret above.

You will then receive embeds for: trades opened, trades closed, blocked
signals, daily heartbeat (no signal), and job failures.

### Step 6 — Email alerts on failure (no extra setup needed)

GitHub sends an email automatically when a scheduled workflow run fails.
Ensure email notifications are enabled in **GitHub Settings > Notifications**.

---

## Running the daily job locally

Test the job before pushing to GitHub:

```powershell
# Dry run — no orders placed, full log output
C:\Python312\python.exe -m bot job --dry-run

# Live mode — places real paper orders if a signal fires
C:\Python312\python.exe -m bot job
```

---

## What good backtest results look like

**Baseline (`ema_cross`)** — daily bars, EMA 20/50:
- 5–20 trades over 6 months
- Win rate 40–60%
- Equity curve moves in steps, not a noisy saw

**Filtered (`ema_cross_filtered`)** — same period:
- Fewer trades (some BUYs blocked by filters — correct behaviour)
- Higher win rate (filters remove low-quality entries)
- Lower max drawdown (stop loss caps individual losses)

If either strategy shows 100+ trades, `timeframe` in `config.yaml` is not being respected.
If filtered trade count drops to 0–2, relax `rsi_overbought` to `75`.
