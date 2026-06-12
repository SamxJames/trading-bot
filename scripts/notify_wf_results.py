"""
scripts/notify_wf_results.py

Posts walk-forward reoptimisation results to Discord.
Runs as the final step of quarterly_reoptimise.yml.

Reads results/wf_results.json and posts a summary embed showing:
  - Windows run and promoted
  - Consensus params (if any were promoted)
  - Whether config.yaml was updated
"""

import json
import os
import sys
from pathlib import Path

import httpx

RESULTS_PATH = Path("results/wf_results.json")
WEBHOOK_URL  = os.environ.get("DISCORD_WEBHOOK_URL", "")


def _colour(promoted: int, total: int) -> int:
    """Green if majority promoted, amber if some, red if none."""
    if promoted == 0:
        return 0xFF4B4B   # red
    if promoted / total >= 0.5:
        return 0x00E5A0   # green
    return 0xF5A623       # amber


def build_embed(data: dict) -> dict:
    promoted = data.get("promoted_count", 0)
    total    = data.get("total_windows", 0)
    params   = data.get("consensus_params") or {}
    gen_date = data.get("generated_at", "unknown")

    if params:
        param_lines = "\n".join(f"  `{k}`: **{v}**" for k, v in params.items())
        params_field = {
            "name": "✅ Promoted params",
            "value": param_lines,
            "inline": False,
        }
        description = (
            f"**{promoted}/{total}** windows passed validation.\n"
            f"config.yaml has been updated with consensus params."
        )
    else:
        params_field = {
            "name": "⚠️ No params promoted",
            "value": "Current config.yaml params retained unchanged.",
            "inline": False,
        }
        description = (
            f"**{promoted}/{total}** windows passed validation.\n"
            f"No parameter set generalised well enough — current params retained."
        )

    return {
        "embeds": [{
            "title": "📊 Quarterly Walk-Forward Reoptimisation",
            "description": description,
            "color": _colour(promoted, total),
            "fields": [params_field],
            "footer": {"text": f"Generated {gen_date}"},
        }]
    }


def main() -> None:
    if not WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set — skipping notification")
        return

    if not RESULTS_PATH.exists():
        print(f"No results file found at {RESULTS_PATH} — skipping notification")
        return

    data  = json.loads(RESULTS_PATH.read_text())
    embed = build_embed(data)

    resp = httpx.post(WEBHOOK_URL, json=embed, timeout=10)
    if resp.status_code not in (200, 204):
        print(f"Discord post failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    print("Walk-forward results posted to Discord")


if __name__ == "__main__":
    main()
