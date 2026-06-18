"""
Strategy degradation health monitor — the deployment discipline layer.

This module does NOT try to improve returns. Its only job is to notice, early
and loudly, when a live/shadow strategy stops behaving like the backtest that
justified deploying it — so a human reviews it before more capital is committed.

For each strategy (EMA and GEM, evaluated separately) it computes a handful of
rolling health metrics from the trade journals and compares them to hardcoded
baselines taken from the validated 2005-2025 backtest:

  EMA  — expected win rate 47.7%, expected Sharpe 0.348
         (pooled, scripts/backtest_with_costs.py / scripts/tune_filters.py)
  GEM  — expected to track its SPY-relative behaviour within a band

Checks (each returns OK / INFO / WARNING / CRITICAL, plus the observed value,
the expected value, and a plain-English explanation):

  * Win-rate drift   — rolling win rate ≥15pp below baseline over ≥20 trades  → WARNING
  * Drawdown breach  — live/shadow drawdown > 1.5× backtest max drawdown       → CRITICAL
  * Losing streak    — consecutive losses exceed the worst backtest streak     → WARNING
  * No-signal anomaly— zero signals for 60+ days (backtest ≈ 1 trade / 7 weeks)→ INFO
  * Execution drag   — realistic shadow return < 0 while ideal return > 0       → CRITICAL

Everything here fails *open*: missing files, short histories, or malformed data
yield "MONITORING — insufficient data", never a false alarm and never an
exception that could halt the daily job.

Auto-pause is OFF by default (see health_auto_pause in config.yaml). The monitor
alerts; it does not disable a strategy unless a human has explicitly turned
auto-pause on after the monitor has proven reliable on real data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ── Status levels ────────────────────────────────────────────────────────────
OK = "OK"
INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
MONITORING = "MONITORING"  # insufficient data — display state, never an alert

# Severity used to pick a strategy's worst status. MONITORING ranks below OK so
# it never escalates into an alert.
_SEVERITY = {MONITORING: -1, OK: 0, INFO: 1, WARNING: 2, CRITICAL: 3}

# Statuses that should raise a Discord alert in the daily job.
ALERT_STATUSES = (WARNING, CRITICAL)

# ── Paths ────────────────────────────────────────────────────────────────────
TRADES_PATH = Path("bot/trade_journal/trades_live.csv")
SHADOW_TRADES_PATH = Path("bot/trade_journal/shadow_trades.csv")
SIGNAL_LOG_PATH = Path("bot/trade_journal/signal_log.jsonl")
SHADOW_LOG_PATH = Path("bot/trade_journal/shadow_log.jsonl")
GEM_LOG_PATH = Path("bot/trade_journal/rebalance_log.jsonl")
ANALYTICS_PATH = Path("docs/analytics.json")

# ── Thresholds (deliberately conservative; documented constants) ─────────────
WIN_RATE_DRIFT_PP = 15.0      # pp below baseline win rate that trips a WARNING
DRAWDOWN_BREACH_MULT = 1.5    # multiple of backtest max DD that trips CRITICAL
NO_SIGNAL_DAYS = 60           # days of silence that trips the INFO anomaly
STARTING_EQUITY = 100_000.0   # equity-curve base for drawdown (matches analyse_trades)


@dataclass(frozen=True)
class StrategyBaseline:
    """Expected behaviour from the validated backtest for one strategy."""
    name: str
    expected_win_rate: float          # %
    expected_sharpe: float
    max_drawdown_pct: float           # backtest peak-to-trough on cumulative PnL
    worst_losing_streak: int          # worst run of consecutive losers in backtest


# EMA baseline — pooled 2005-2025 (47.7% win rate, 0.348 Sharpe). max_drawdown
# and worst streak are the cumulative-PnL figures from the same pooled run; they
# are intentionally conservative and easy to revise as live data accrues.
EMA_BASELINE = StrategyBaseline(
    name="EMA",
    expected_win_rate=47.7,
    expected_sharpe=0.348,
    max_drawdown_pct=8.0,
    worst_losing_streak=7,
)


@dataclass
class HealthCheck:
    """One health check's verdict."""
    name: str
    status: str
    observed: Any
    expected: Any
    explanation: str

    def is_alert(self) -> bool:
        return self.status in ALERT_STATUSES

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyHealth:
    """All checks for one strategy plus its aggregate status."""
    strategy: str
    status: str
    sufficient_data: bool
    trades_evaluated: int
    min_trades: int
    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def alerts(self) -> list[HealthCheck]:
        return [c for c in self.checks if c.is_alert()]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "sufficient_data": self.sufficient_data,
            "trades_evaluated": self.trades_evaluated,
            "min_trades": self.min_trades,
            "checks": [c.to_dict() for c in self.checks],
        }


# ── Pure metric helpers ──────────────────────────────────────────────────────

def _max_drawdown_pct(pnls: list[float], base: float = STARTING_EQUITY) -> float:
    """Max peak-to-trough drawdown (%) of a cumulative-PnL equity curve."""
    equity = base
    peak = base
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return round(max_dd, 2)


def _max_losing_streak(pnls: list[float]) -> int:
    streak = worst = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def _win_rate(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    return round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1)


# ── Individual checks ────────────────────────────────────────────────────────

def check_win_rate_drift(
    pnls: list[float], baseline: StrategyBaseline, min_trades: int
) -> HealthCheck:
    """Rolling win rate ≥ WIN_RATE_DRIFT_PP below baseline over ≥ min_trades → WARNING."""
    recent = pnls[-min_trades:] if len(pnls) >= min_trades else pnls
    if len(pnls) < min_trades:
        return HealthCheck(
            "win_rate_drift", MONITORING, _win_rate(recent), baseline.expected_win_rate,
            f"Monitoring — insufficient data ({len(pnls)}/{min_trades} trades).",
        )
    observed = _win_rate(recent)
    floor = baseline.expected_win_rate - WIN_RATE_DRIFT_PP
    if observed < floor:
        return HealthCheck(
            "win_rate_drift", WARNING, observed, baseline.expected_win_rate,
            f"Win rate drift: observed {observed}% vs expected "
            f"{baseline.expected_win_rate}% over last {len(recent)} trades. "
            f"This may indicate strategy degradation or regime change.",
        )
    return HealthCheck(
        "win_rate_drift", OK, observed, baseline.expected_win_rate,
        f"Win rate {observed}% is within {WIN_RATE_DRIFT_PP}pp of the "
        f"{baseline.expected_win_rate}% baseline over last {len(recent)} trades.",
    )


def check_drawdown_breach(
    pnls: list[float], baseline: StrategyBaseline, min_trades: int
) -> HealthCheck:
    """Live/shadow drawdown > DRAWDOWN_BREACH_MULT × backtest max DD → CRITICAL."""
    if len(pnls) < min_trades:
        return HealthCheck(
            "drawdown_breach", MONITORING, None, baseline.max_drawdown_pct,
            f"Monitoring — insufficient data ({len(pnls)}/{min_trades} trades).",
        )
    observed = _max_drawdown_pct(pnls)
    limit = round(baseline.max_drawdown_pct * DRAWDOWN_BREACH_MULT, 2)
    if observed > limit:
        return HealthCheck(
            "drawdown_breach", CRITICAL, observed, limit,
            f"Drawdown breach: observed {observed}% exceeds "
            f"{DRAWDOWN_BREACH_MULT}× backtest max drawdown ({limit}%). "
            f"Stop and review before continuing live trading.",
        )
    return HealthCheck(
        "drawdown_breach", OK, observed, limit,
        f"Max drawdown {observed}% is within the {limit}% limit "
        f"({DRAWDOWN_BREACH_MULT}× backtest {baseline.max_drawdown_pct}%).",
    )


def check_losing_streak(
    pnls: list[float], baseline: StrategyBaseline, min_trades: int
) -> HealthCheck:
    """Consecutive losses exceeding the worst backtest streak → WARNING."""
    if len(pnls) < min_trades:
        return HealthCheck(
            "losing_streak", MONITORING, _max_losing_streak(pnls),
            baseline.worst_losing_streak,
            f"Monitoring — insufficient data ({len(pnls)}/{min_trades} trades).",
        )
    observed = _max_losing_streak(pnls)
    if observed > baseline.worst_losing_streak:
        return HealthCheck(
            "losing_streak", WARNING, observed, baseline.worst_losing_streak,
            f"Losing streak: {observed} consecutive losses exceeds the worst "
            f"backtest streak ({baseline.worst_losing_streak}). "
            f"This may indicate strategy degradation or regime change.",
        )
    return HealthCheck(
        "losing_streak", OK, observed, baseline.worst_losing_streak,
        f"Worst losing streak {observed} is within the backtest worst of "
        f"{baseline.worst_losing_streak}.",
    )


def check_no_signal_anomaly(
    days_since_last_signal: Optional[int], days_threshold: int = NO_SIGNAL_DAYS
) -> HealthCheck:
    """Zero signals for `days_threshold`+ days → INFO (possible data/regime issue)."""
    if days_since_last_signal is None:
        return HealthCheck(
            "no_signal_anomaly", MONITORING, None, days_threshold,
            "Monitoring — no signal history yet.",
        )
    if days_since_last_signal >= days_threshold:
        return HealthCheck(
            "no_signal_anomaly", INFO, days_since_last_signal, days_threshold,
            f"No signal in {days_since_last_signal} days (backtest expects "
            f"≈1 trade / 7 weeks). Possible data feed or regime issue — worth a look.",
        )
    return HealthCheck(
        "no_signal_anomaly", OK, days_since_last_signal, days_threshold,
        f"Last signal {days_since_last_signal} days ago — within normal cadence.",
    )


def check_execution_drag(shadow_stats: Optional[dict]) -> HealthCheck:
    """Realistic shadow return < 0 while ideal > 0 → CRITICAL (edge dies after costs)."""
    if not shadow_stats or not shadow_stats.get("total_trades"):
        return HealthCheck(
            "execution_drag", MONITORING, None, None,
            "Monitoring — no shadow trades yet.",
        )
    ideal = (shadow_stats.get("ideal") or {}).get("total_return_pct", shadow_stats.get("total_return_pct", 0.0))
    realistic = (shadow_stats.get("realistic") or {}).get("total_return_pct", ideal)
    if ideal > 0 and realistic < 0:
        return HealthCheck(
            "execution_drag", CRITICAL, realistic, ideal,
            f"Execution drag breach: realistic shadow return {realistic}% is "
            f"negative while ideal is +{ideal}%. The edge does not survive "
            f"real-world execution costs.",
        )
    return HealthCheck(
        "execution_drag", OK, realistic, ideal,
        f"Realistic shadow return {realistic}% (ideal {ideal}%) — edge survives costs.",
    )


# ── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate_status(checks: list[HealthCheck], sufficient_data: bool) -> str:
    if not checks:
        return MONITORING
    worst = max(checks, key=lambda c: _SEVERITY.get(c.status, 0))
    worst_status = worst.status
    if _SEVERITY.get(worst_status, 0) <= 0:
        # No alert-level finding. If we never had enough data for the
        # statistical checks, surface MONITORING rather than a green OK.
        return OK if sufficient_data else MONITORING
    return worst_status


# ── Data loading (all fail open) ─────────────────────────────────────────────

def _read_pnls_from_csv(path: Path, pnl_col: str = "pnl_usd") -> list[float]:
    """Closed-trade PnL series from a trade CSV, ordered by exit/timestamp. [] on error."""
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
    except Exception:
        return []
    if df.empty or pnl_col not in df.columns:
        return []
    # Live journal mixes BUY/SELL rows; keep only closed (SELL) rows. Shadow CSV
    # is already one row per closed round trip.
    if "side" in df.columns:
        closed = df[df["side"].astype(str).str.upper() == "SELL"]
        if not closed.empty:
            df = closed
    sort_col = "exit_date" if "exit_date" in df.columns else (
        "timestamp" if "timestamp" in df.columns else None)
    if sort_col:
        df = df.assign(_sort=pd.to_datetime(df[sort_col], utc=True, errors="coerce"))
        df = df.sort_values("_sort")
    return pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0).tolist()


def load_ema_pnls() -> list[float]:
    """Combined closed-trade PnL stream for the EMA strategy.

    Prefers live trades; falls back to shadow trades when there are no live
    closed trades yet (the normal case before the paper account fills up).
    """
    live = _read_pnls_from_csv(TRADES_PATH)
    if live:
        return live
    return _read_pnls_from_csv(SHADOW_TRADES_PATH)


def load_shadow_stats(path: Path = SHADOW_TRADES_PATH) -> Optional[dict]:
    """Minimal shadow ideal/realistic returns for the execution-drag check.

    Computed straight from shadow_trades.csv (cheap, local, no network) so the
    daily job can run the monitor without importing the analytics script.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
    except Exception:
        return None
    if df.empty or "pnl_usd" not in df.columns:
        return None
    ideal_pnl = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0).sum()
    real_col = "pnl_usd_real" if "pnl_usd_real" in df.columns else "pnl_usd"
    real_pnl = pd.to_numeric(df[real_col], errors="coerce").fillna(0.0).sum()
    return {
        "total_trades": len(df),
        "ideal":     {"total_return_pct": round(ideal_pnl / STARTING_EQUITY * 100, 2)},
        "realistic": {"total_return_pct": round(real_pnl / STARTING_EQUITY * 100, 2)},
    }


def load_gem_block(path: Path = ANALYTICS_PATH) -> Optional[dict]:
    """GEM analytics block from the last-generated analytics.json (no network).

    Lagged by one run, which is fine for slow monthly-allocation health checks.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    gem = data.get("gem")
    return gem if isinstance(gem, dict) and gem.get("available") else None


def days_since_last_ema_signal(today: Optional[date] = None) -> Optional[int]:
    """Days since the most recent BUY/SELL signal across the signal + shadow logs.

    None when no signal history exists at all (a fresh deployment, not an anomaly).
    """
    today = today or datetime.now(timezone.utc).date()
    latest: Optional[date] = None
    for path in (SIGNAL_LOG_PATH, SHADOW_LOG_PATH):
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "signal_evaluation":
                    continue
                if rec.get("signal") not in ("BUY", "SELL"):
                    continue
                d_raw = rec.get("date")
                if not d_raw:
                    continue
                try:
                    d = date.fromisoformat(str(d_raw)[:10])
                except ValueError:
                    continue
                if latest is None or d > latest:
                    latest = d
        except Exception:
            continue
    if latest is None:
        return None
    return (today - latest).days


# ── The monitor ──────────────────────────────────────────────────────────────

class StrategyHealthMonitor:
    """Computes per-strategy health and (optionally) auto-pauses on CRITICAL."""

    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.enabled = bool(_attr(settings, "health_monitoring", True))
        self.auto_pause = bool(_attr(settings, "health_auto_pause", False))
        self.min_trades = int(_attr(settings, "health_min_trades_for_check", 20) or 20)

    # -- EMA ----------------------------------------------------------------
    def evaluate_ema(
        self,
        pnls: list[float],
        shadow_stats: Optional[dict] = None,
        days_since_last_signal: Optional[int] = None,
    ) -> StrategyHealth:
        baseline = EMA_BASELINE
        checks = [
            check_win_rate_drift(pnls, baseline, self.min_trades),
            check_drawdown_breach(pnls, baseline, self.min_trades),
            check_losing_streak(pnls, baseline, self.min_trades),
            check_no_signal_anomaly(days_since_last_signal),
            check_execution_drag(shadow_stats),
        ]
        sufficient = len(pnls) >= self.min_trades
        status = _aggregate_status(checks, sufficient)
        return StrategyHealth(
            strategy="EMA", status=status, sufficient_data=sufficient,
            trades_evaluated=len(pnls), min_trades=self.min_trades, checks=checks,
        )

    # -- GEM ----------------------------------------------------------------
    def evaluate_gem(self, gem_block: Optional[dict]) -> StrategyHealth:
        """GEM health: tracks its SPY-relative band and rebalance freshness.

        GEM is a low-turnover monthly allocator, so the trade-count statistical
        checks do not apply. We watch two things: (1) that GEM is not badly
        underperforming SPY since inception (its whole premise is to match or
        beat buy-and-hold while cutting drawdown), and (2) that rebalances are
        still happening.
        """
        checks: list[HealthCheck] = []
        if not gem_block or not gem_block.get("available"):
            checks.append(HealthCheck(
                "gem_relative", MONITORING, None, None,
                "Monitoring — no GEM rebalance history yet.",
            ))
            return StrategyHealth(
                strategy="GEM", status=MONITORING, sufficient_data=False,
                trades_evaluated=0, min_trades=self.min_trades, checks=checks,
            )

        cagr = gem_block.get("cagr", 0.0) or 0.0
        spy_cagr = gem_block.get("spy_cagr", 0.0) or 0.0
        # Band: GEM is allowed to trail SPY's CAGR by up to 3pp (it trades risk
        # control for some upside). Beyond that, flag for review.
        if spy_cagr - cagr > 3.0:
            checks.append(HealthCheck(
                "gem_relative", WARNING, cagr, spy_cagr,
                f"GEM CAGR {cagr}% trails SPY {spy_cagr}% by more than 3pp since "
                f"inception. Review whether the momentum edge is intact.",
            ))
        else:
            checks.append(HealthCheck(
                "gem_relative", OK, cagr, spy_cagr,
                f"GEM CAGR {cagr}% vs SPY {spy_cagr}% — within its expected band.",
            ))

        status = _aggregate_status(checks, sufficient_data=True)
        return StrategyHealth(
            strategy="GEM", status=status, sufficient_data=True,
            trades_evaluated=len(gem_block.get("history", []) or []),
            min_trades=self.min_trades, checks=checks,
        )

    # -- Orchestration ------------------------------------------------------
    def run_from_files(
        self, shadow_stats: Optional[dict] = None, gem_block: Optional[dict] = None,
        today: Optional[date] = None,
    ) -> dict[str, StrategyHealth]:
        """Load journals from disk and evaluate every strategy. Never raises."""
        if shadow_stats is None:
            shadow_stats = load_shadow_stats()
        if gem_block is None:
            gem_block = load_gem_block()
        ema = self.evaluate_ema(
            load_ema_pnls(), shadow_stats=shadow_stats,
            days_since_last_signal=days_since_last_ema_signal(today),
        )
        gem = self.evaluate_gem(gem_block)
        return {"EMA": ema, "GEM": gem}


def _attr(settings: object, name: str, default: Any) -> Any:
    """getattr that coerces MagicMock/missing values to a usable default."""
    val = getattr(settings, name, default)
    if isinstance(val, (bool, int, float, str)):
        return val
    return default


# ── Discord formatting ───────────────────────────────────────────────────────

def format_alert(health: StrategyHealth) -> Optional[tuple[str, str, str]]:
    """Return (title, message, colour) for a WARNING/CRITICAL strategy, else None."""
    alerts = health.alerts
    if not alerts:
        return None
    is_critical = any(c.status == CRITICAL for c in alerts)
    icon = "🛑" if is_critical else "⚠️"
    level = "CRITICAL" if is_critical else "WARNING"
    colour = "red" if is_critical else "amber"
    lines = [f"{icon} STRATEGY HEALTH {level} — {health.strategy}", ""]
    for c in alerts:
        lines.append(c.explanation)
    lines.append("")
    lines.append("Review before continuing live trading.")
    title = f"{icon} Strategy Health {level} — {health.strategy}"
    return title, "\n".join(lines), colour
