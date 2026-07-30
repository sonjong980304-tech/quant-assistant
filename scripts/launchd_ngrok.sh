#!/usr/bin/env bash
# launchd(com.darttext.ngrok)가 실행하는 ngrok 터널 기동 스크립트.
# 인증정보는 .env(NGROK_USERNAME/PASSWORD/DOMAIN)에서 로드한다 — plist에 평문 저장 금지.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 웹서버(com.darttext.web)가 launchd에 의해 거의 동시에 뜨므로 준비될 때까지 잠깐 대기
for _ in $(seq 1 30); do
  curl -s -o /dev/null http://127.0.0.1:8000/ 2>/dev/null && break
  sleep 1
done

NGROK_BIN="/opt/homebrew/bin/ngrok"
if [ -n "${NGROK_DOMAIN:-}" ]; then
  exec "$NGROK_BIN" http 8000 --url="https://${NGROK_DOMAIN}" \
    --basic-auth "${NGROK_USERNAME}:${NGROK_PASSWORD}"
else
  exec "$NGROK_BIN" http 8000 --basic-auth "${NGROK_USERNAME}:${NGROK_PASSWORD}"
fi
