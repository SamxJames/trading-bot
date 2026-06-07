"""
Overwrites analytics.json in Google Drive with the locally generated copy.
Runs after analyse_trades.py in daily_job.yml.

Required env vars:
  GDRIVE_SERVICE_ACCOUNT_JSON  — full JSON content of the service account key
  GDRIVE_ANALYTICS_FILE_ID     — Drive file ID of analytics.json
"""

import json
import os
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
LOCAL_PATH = Path("results/analytics.json")
MIME_TYPE = "application/json"


def main() -> None:
    sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    file_id = os.environ.get("GDRIVE_ANALYTICS_FILE_ID")

    if not sa_json:
        print("ERROR: GDRIVE_SERVICE_ACCOUNT_JSON not set", file=sys.stderr)
        sys.exit(1)
    if not file_id:
        print("ERROR: GDRIVE_ANALYTICS_FILE_ID not set", file=sys.stderr)
        sys.exit(1)

    if not LOCAL_PATH.exists():
        print(f"WARNING: {LOCAL_PATH} does not exist — nothing to sync")
        return  # exit 0, valid if no trades yet

    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json), scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)

    media = MediaFileUpload(str(LOCAL_PATH), mimetype=MIME_TYPE, resumable=False)
    service.files().update(fileId=file_id, media_body=media).execute()

    print(f"Synced {LOCAL_PATH} → Drive (file ID: {file_id})")


if __name__ == "__main__":
    main()
