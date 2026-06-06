"""
Overwrites the trades_live.csv file in Google Drive with the local copy.
Runs as the final step of daily_job.yml after each trading session.

Required env vars:
  GDRIVE_SERVICE_ACCOUNT_JSON  — full JSON content of the service account key
  GDRIVE_FILE_ID               — Drive file ID of trades_live.csv
"""

import json
import os
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
LOCAL_PATH = Path("bot/trade_journal/trades_live.csv")
MIME_TYPE = "text/csv"


def main() -> None:
    sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    file_id = os.environ.get("GDRIVE_FILE_ID")

    if not sa_json:
        print("ERROR: GDRIVE_SERVICE_ACCOUNT_JSON not set", file=sys.stderr)
        sys.exit(1)
    if not file_id:
        print("ERROR: GDRIVE_FILE_ID not set", file=sys.stderr)
        sys.exit(1)

    if not LOCAL_PATH.exists():
        print(f"WARNING: {LOCAL_PATH} does not exist — nothing to sync")
        return  # exit 0, not a failure (no trades yet is valid)

    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json), scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)

    media = MediaFileUpload(str(LOCAL_PATH), mimetype=MIME_TYPE, resumable=False)
    service.files().update(fileId=file_id, media_body=media).execute()

    row_count = sum(1 for _ in LOCAL_PATH.open()) - 1  # subtract header
    print(f"Synced {LOCAL_PATH} → Drive ({max(row_count, 0)} trades)")


if __name__ == "__main__":
    main()
