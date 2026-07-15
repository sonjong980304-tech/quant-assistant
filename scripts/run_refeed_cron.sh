#!/bin/bash
# launchd 매일 실행용 래퍼: 복사본 재수집 이어받기 → 전체 완료 시 원본 교체(swap) + 자동 종료.
# DART 일일 한도까지 진행되면 refeed_full.py가 스스로 중단하고, 다음 날 이어받는다.
set -u
export HOME="/Users/gyuyeong"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
PROJ="/Users/gyuyeong/projects/dart-text2sql-wiki"
PY="/usr/bin/python3"
DB="$PROJ/data/market_refeed.db"
cd "$PROJ" || exit 1

echo "===== $(date '+%Y-%m-%d %H:%M') 재수집 이어받기 ====="
REFEED_DB="$DB" "$PY" scripts/refeed_full.py

TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM company" 2>/dev/null)
DONE=$(sqlite3 "$DB" "SELECT value FROM ingest_meta WHERE key='refeed_done'" 2>/dev/null | tr ',' '\n' | grep -c .)
echo "진행: ${DONE:-0} / ${TOTAL:-?}"

if [ -n "${TOTAL:-}" ] && [ "${TOTAL:-0}" -gt 0 ] && [ "${DONE:-0}" -ge "${TOTAL}" ]; then
  echo "전체 완료 → 원본(market.db)에 반영(swap)"
  "$PY" scripts/swap_refeed.py
  echo "swap 완료 → launchd 자동 중지"
  launchctl bootout "gui/$(id -u)/com.dart.refeed" 2>/dev/null \
    || launchctl unload "$HOME/Library/LaunchAgents/com.dart.refeed.plist" 2>/dev/null
fi
echo "===== 종료 ====="
