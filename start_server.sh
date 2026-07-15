#!/usr/bin/env bash
# FastAPI(포트 8000) + ngrok 외부 접속 실행.
#   - FastAPI를 백그라운드로 띄우고
#   - ngrok http 8000 --basic-auth "아이디:비밀번호" 로 외부 터널 생성
#   - 외부 접속 URL을 출력한다.
# 인증 정보는 .env의 NGROK_USERNAME / NGROK_PASSWORD 를 사용.
set -euo pipefail
cd "$(dirname "$0")"

# ---- .env 로드 (NGROK_USERNAME/PASSWORD 등) ----
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
NGROK_USERNAME="${NGROK_USERNAME:-}"
NGROK_PASSWORD="${NGROK_PASSWORD:-}"

# ---- ngrok 설치 확인 ----
if ! command -v ngrok >/dev/null 2>&1; then
  cat <<'EOF'
❌ ngrok 이 설치되어 있지 않습니다.

설치 방법 (둘 중 하나):
  1) Homebrew:   brew install ngrok/ngrok/ngrok
  2) 직접 다운로드: https://ngrok.com/download

설치 후 1회 인증 토큰 등록이 필요합니다(무료 가입):
  ngrok config add-authtoken <YOUR_TOKEN>
  토큰 확인: https://dashboard.ngrok.com/get-started/your-authtoken

등록 후 다시 ./start_server.sh 를 실행하세요.
EOF
  exit 1
fi

if [ -z "$NGROK_USERNAME" ] || [ -z "$NGROK_PASSWORD" ]; then
  echo "❌ .env 에 NGROK_USERNAME / NGROK_PASSWORD 가 설정돼 있지 않습니다."
  exit 1
fi

# ---- Python 인터프리터 선택 (projects 공용 uv venv 우선) ----
# 이 폴더에서 그냥 `python3` 를 쓰면 pyenv 가 잡혀 fastapi 등 의존성이 없을 수 있으므로,
# 상위 projects/.venv 의 python 을 직접 지정한다. (SERVER_PYTHON 환경변수로 재정의 가능)
PROJECT_VENV_PY="$(cd .. 2>/dev/null && pwd)/.venv/bin/python"
if [ -n "${SERVER_PYTHON:-}" ] && [ -x "${SERVER_PYTHON}" ]; then
  PYTHON="${SERVER_PYTHON}"
elif [ -x "$PROJECT_VENV_PY" ]; then
  PYTHON="$PROJECT_VENV_PY"
else
  PYTHON="python3"
  echo "⚠️  projects/.venv 를 찾지 못해 시스템 python3 로 실행합니다 (fastapi 미설치 시 실패할 수 있음)."
fi
echo "▶ Python 인터프리터: $PYTHON"

# ---- FastAPI 백그라운드 실행 ----
pkill -f "uvicorn web.app" 2>/dev/null || true
sleep 1
nohup "$PYTHON" -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --log-level warning \
  > /tmp/uvicorn.log 2>&1 &
echo "▶ FastAPI 서버 시작 (포트 8000, 로그: /tmp/uvicorn.log)"

# 서버 준비 대기
for _ in $(seq 1 25); do
  if curl -s -o /dev/null http://127.0.0.1:8000/ 2>/dev/null; then break; fi
  sleep 1
done

# ---- ngrok 터널 실행 ----
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
pkill -f "ngrok http" 2>/dev/null || true
sleep 1
if [ -n "$NGROK_DOMAIN" ]; then
  nohup ngrok http 8000 --url="https://${NGROK_DOMAIN}" \
    --basic-auth "${NGROK_USERNAME}:${NGROK_PASSWORD}" > /tmp/ngrok.log 2>&1 &
  echo "▶ ngrok 터널 시작 (고정 도메인: ${NGROK_DOMAIN}, 사용자: ${NGROK_USERNAME})"
else
  nohup ngrok http 8000 --basic-auth "${NGROK_USERNAME}:${NGROK_PASSWORD}" \
    > /tmp/ngrok.log 2>&1 &
  echo "▶ ngrok 터널 시작 (임시 도메인, 사용자: ${NGROK_USERNAME})"
fi

# ---- 외부 접속 URL 추출 (ngrok 로컬 API) ----
URL=""
for _ in $(seq 1 15); do
  URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
    | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(t[0]['public_url'] if t else '')" 2>/dev/null || true)
  [ -n "$URL" ] && break
  sleep 1
done

echo ""
if [ -n "$URL" ]; then
  echo "============================================================"
  echo "✅ 외부 접속 URL : $URL"
  echo "   아이디        : ${NGROK_USERNAME}"
  echo "   비밀번호      : (.env의 NGROK_PASSWORD)"
  echo "   ngrok 대시보드: http://localhost:4040"
  echo "============================================================"
  echo "종료하려면: ./stop_server.sh"
else
  echo "⚠️  URL 추출 실패. 다음을 확인하세요:"
  echo "    - /tmp/ngrok.log (authtoken 미등록 등)"
  echo "    - http://localhost:4040 (ngrok 대시보드)"
fi
