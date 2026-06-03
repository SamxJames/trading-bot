"""
Configuration layer.

Loads all settings from config.yaml and environment variables using
pydantic-settings. This is the single source of truth for every tunable
parameter in the system — tickers, strategy selection, risk thresholds,
and broker credentials.

Nothing in the rest of the codebase should read os.environ or open
config.yaml directly; import get_settings() from here instead.

Priority (highest to lowest):
  init kwargs → env vars → .env file → config.yaml → field defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import yaml
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlConfigSource(PydanticBaseSettingsSource):
    """Load non-secret settings from a YAML file."""

    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        yaml_path: str = "config.yaml",
    ) -> None:
        super().__init__(settings_cls)
        self._data: Dict[str, Any] = {}
        try:
            with open(yaml_path, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass

    def get_field_value(
        self, field_name: str, field_info: FieldInfo
    ) -> Tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, self.field_is_complex(field_info)

    def __call__(self) -> Dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Broker credentials (from .env only, never config.yaml) ---
    apca_api_key_id: str = Field(..., description="Alpaca API key ID")
    apca_api_secret_key: str = Field(..., description="Alpaca API secret key")
    apca_base_url: str = Field(
        "https://paper-api.alpaca.markets",
        description="Alpaca base URL; paper trading by default",
    )

    # --- Universe ---
    tickers: List[str] = ["AAPL"]

    # --- Data ---
    timeframe: str = "1Day"          # 1Min | 1Hour | 1Day

    # --- Strategy selection ---
    strategy: str = "ema_cross"
    fast_period: int = 20            # EMA fast period (suits daily bars)
    slow_period: int = 50            # EMA slow period (suits daily bars)

    # --- Filtered strategy params ---
    trend_sma_period: int = 200      # trend filter: only buy if close > SMA(200)
    rsi_period: int = 14             # RSI confirmation period
    rsi_oversold: float = 30.0       # RSIStrategy: buy when RSI crosses UP through this
    rsi_overbought: float = 70.0     # block BUY if RSI >= this value
    stop_loss_pct: float = 1.5       # per-trade stop loss (% below entry)

    # --- Risk parameters ---
    max_positions: int = 3
    max_notional_per_trade: float = 500.0
    drawdown_halt_pct: float = 5.0

    # --- Notifications (optional) ---
    discord_webhook_url: str = ""   # leave empty to disable Discord notifications

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSource(settings_cls),
            file_secret_settings,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance (lazily initialized)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
