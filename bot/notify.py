"""
Discord notification helper.

Posts a structured embed to a Discord webhook URL.
Completely optional — if DISCORD_WEBHOOK_URL is not set (or empty) every
call is a silent no-op so the rest of the pipeline is unaffected.

Colour names  → Discord integer colours
  "green"  → 0x1D9E75   (teal-green)
  "red"    → 0xE24B4A   (red)
  "amber"  → 0xBA7517   (amber)
  "grey"   → 0x888780   (grey, default)

Usage:
    from bot import notify
    await notify.send("Trade Opened", "BUY 3 AAPL @ $175.00", colour="green")
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from bot.logging.logger import get_logger

log = get_logger(__name__)

_COLOURS: dict[str, int] = {
    "green": 0x1D9E75,
    "red":   0xE24B4A,
    "amber": 0xBA7517,
    "grey":  0x888780,
}


async def send(title: str, message: str, colour: str = "grey") -> None:
    """
    Post a Discord embed to the configured webhook.

    Parameters
    ----------
    title:   Embed title (bold, shown at the top of the card).
    message: Embed description (body text, supports Markdown).
    colour:  One of "green", "red", "amber", "grey".  Defaults to "grey".

    If DISCORD_WEBHOOK_URL is empty the call returns immediately without
    making any network request.  Notification failures are logged as
    warnings but never propagated — the caller must not depend on delivery.
    """
    # Lazy import keeps settings out of module-level init
    from bot.config import get_settings
    url = get_settings().discord_webhook_url
    if not url:
        log.warning("notify_skipped", reason="DISCORD_WEBHOOK_URL not configured")
        return

    colour_int = _COLOURS.get(colour, _COLOURS["grey"])
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "embeds": [
            {
                "title": title,
                "description": message,
                "color": colour_int,
                "footer": {"text": f"TradingBot | {now_str} UTC"},
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        log.info("notify_sent", title=title)
    except Exception as exc:
        log.warning("notify_failed", title=title, error=str(exc))
