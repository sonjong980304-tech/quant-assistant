#!/usr/bin/env bash
# launchd(com.darttext.web)가 실행하는 FastAPI 웹서버 기동 스크립트.
# exec로 프로세스를 교체해 launchd가 uvicorn PID를 직접 감시(KeepAlive)하게 한다.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/Users/gyuyeong/projects/.venv/bin/python"
exec "$PYTHON" -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --log-level warning
