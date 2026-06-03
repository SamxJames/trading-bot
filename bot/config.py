"""
Configuration layer.

Loads all settings from config.yaml and environment variables using
pydantic-settings.  This is the single source of truth for every tunable
parameter in the system — tickers, strategy selection, risk thresholds,
and broker credentials.

Nothing in the rest of the codebase should read os.environ or open
config.yaml directly; import get_settings() from here instead.

Priority (highest to lowest):
  init kwargs → env vars → .env file → config.yaml → field defaults

Credential fields use explicit AliasChoices so the mapping from
environment variable to Python attribute is unambiguous:

  APCA_API_KEY_ID     →  settings.apca_api_key_id
  APCA_API_SECRET_KEY →  settings.apca_api_secret_key
  APCA_BASE_URL       →  settings.apca_base_url
  DISCORD_WEBHOOK_URL →  settings.discord_webhook_url

env_ignore_empty=True means an empty string is treated as "not set".
This causes a clear ValidationError ("field required") if a GitHub
Actions secret is undefined, instead of silently passing "" to the
Alpaca SDK and receiving the opaque "must supply authentication" error.

Locally, copy .env.example → .env and fill in credentials.
On GitHub Actions, set APCA_API_KEY_ID / APCA_API_SECRET_KEY /
APCA_BASE_URL as repository secrets.  No .env file is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import yaml
from pydantic import AliasChoices, Field
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
        # Filter out None so YAML keys with no value don't mask env-var defaults.
        # Empty strings are kept (e.g. discord_webhook_url: "").
        return {k: v for k, v in self._data.items() if v is not None}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env for local dev; silently ignored when the file doesn't exist
        # (e.g. GitHub Actions, CI).  Credentials are read from OS env vars
        # in both cases — dotenv is just an extra convenience source.
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Treat empty strings as "not provided".  Prevents GitHub Actions
        # from silently passing "" when a secret is not configured.
        env_ignore_empty=True,
        # Allow fields to be populated by their Python name as well as alias
        # (needed so Settings(apca_api_key_id=...) works in tests).
        populate_by_name=True,
    )

    # -------------------------------------------------------------------------
    # Broker credentials
    # Explicit AliasChoices: env var name first (exact match), then field name
    # as fallback so both APCA_API_KEY_ID=… and apca_api_key_id=… work.
    # -------------------------------------------------------------------------
    apca_api_key_id: str = Field(
        ...,
        validation_alias=AliasChoices("APCA_API_KEY_ID", "apca_api_key_id"),
        description="Alpaca API key ID — set via APCA_API_KEY_ID env var",
    )
    apca_api_secret_key: str = Field(
        ...,
        validation_alias=AliasChoices("APCA_API_SECRET_KEY", "apca_api_secret_key"),
        description="Alpaca API secret key — set via APCA_API_SECRET_KEY env var",
    )
    apca_base_url: str = Field(
        "https://paper-api.alpaca.markets",
        validation_alias=AliasChoices("APCA_BASE_URL", "apca_base_url"),
        description="Alpaca base URL (paper trading by default)",
    )

    # -------------------------------------------------------------------------
    # Universe
    # -------------------------------------------------------------------------
    tickers: List[str] = ["AAPL"]

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    timeframe: str = "1Day"           # 1Min | 1Hour | 1Day

    # -------------------------------------------------------------------------
    # Strategy selection
    # -------------------------------------------------------------------------
    strategy: str = "ema_cross"
    fast_period: int = 20             # EMA fast period (suits daily bars)
    slow_period: int = 50             # EMA slow period (suits daily bars)

    # -------------------------------------------------------------------------
    # Filtered strategy params
    # -------------------------------------------------------------------------
    trend_sma_period: int = 200       # trend filter: only buy if close > SMA(200)
    rsi_period: int = 14              # RSI confirmation period
    rsi_oversold: float = 30.0        # RSIStrategy: buy when RSI crosses UP through this
    rsi_overbought: float = 70.0      # block BUY if RSI >= this value
    stop_loss_pct: float = 1.5        # per-trade stop loss (% below entry)

    # -------------------------------------------------------------------------
    # Risk parameters
    # -------------------------------------------------------------------------
    max_positions: int = 3
    max_notional_per_trade: float = 500.0
    drawdown_halt_pct: float = 5.0

    # -------------------------------------------------------------------------
    # Notifications (optional)
    # -------------------------------------------------------------------------
    discord_webhook_url: str = Field(
        "",
        validation_alias=AliasChoices("DISCORD_WEBHOOK_URL", "discord_webhook_url"),
        description="Discord webhook URL — leave blank to disable notifications",
    )

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
            init_settings,       # 1st — constructor kwargs (tests, CLI overrides)
            env_settings,        # 2nd — OS environment variables (GitHub Actions)
            dotenv_settings,     # 3rd — .env file (local dev)
            YamlConfigSource(settings_cls),   # 4th — config.yaml (non-secrets)
            file_secret_settings,             # 5th — /run/secrets etc.
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance (lazily initialized)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
