#!/usr/bin/env bash
# 현재 ngrok 외부 접속 URL을 출력 (서버가 켜져 있을 때).
URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
  | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(t[0]['public_url'] if t else '')" 2>/dev/null)
if [ -n "$URL" ]; then
  echo "현재 외부 접속 URL: $URL"
  echo "  (아이디: ${NGROK_USERNAME:-sonjong} / 비밀번호: .env의 NGROK_PASSWORD)"
else
  echo "ngrok 터널이 없습니다. ./start_server.sh 로 먼저 실행하세요."
fi
