#!/usr/bin/env bash
# FastAPI + ngrok 동시 종료.
echo "■ 서버/터널 종료 중..."
if pkill -f "ngrok http" 2>/dev/null; then echo "  - ngrok 종료됨"; else echo "  - ngrok 미실행"; fi
if pkill -f "uvicorn web.app" 2>/dev/null; then echo "  - FastAPI 종료됨"; else echo "  - FastAPI 미실행"; fi
echo "완료."
