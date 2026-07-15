# launchd 자동 갱신 (Phase 3)

Mac launchd로 주가/재무를 자동 증분 갱신한다. 두 작업 모두 **누락일 메꿈**
방식이라, Mac이 꺼져 있거나 잠들어 실행을 건너뛰어도 다음 실행 때 빠진
구간을 다시 채운다.

| plist | 실행 시각 | 하는 일 | LLM 비용 |
|---|---|---|---|
| `com.darttext.updateprices.plist` | 평일(월~금) 18:30 | 장마감 후 거래일 종가 증분 적재 | 0 (미사용) |
| `com.darttext.updatefinancials.plist` | 매일 07:30 | 새로 공시된 분기 재무만 증분 적재 | 0 (미사용) |

수집 단계에서는 LLM을 전혀 호출하지 않으므로 **갱신 비용은 0**이다.
(LLM은 질의 시 Text-to-SQL에만 쓰인다.)

## 설치

```sh
# 1) 로그 디렉토리 준비
mkdir -p /Users/gyuyeong/dart-text2sql-wiki/data/logs

# 2) LaunchAgents 로 복사 (또는 심볼릭 링크)
cp /Users/gyuyeong/dart-text2sql-wiki/scripts/launchd/com.darttext.updateprices.plist \
   ~/Library/LaunchAgents/
cp /Users/gyuyeong/dart-text2sql-wiki/scripts/launchd/com.darttext.updatefinancials.plist \
   ~/Library/LaunchAgents/

# 3) 등록 (load)
launchctl load ~/Library/LaunchAgents/com.darttext.updateprices.plist
launchctl load ~/Library/LaunchAgents/com.darttext.updatefinancials.plist
```

## 해제 / 재적용

```sh
# 해제 (unload)
launchctl unload ~/Library/LaunchAgents/com.darttext.updateprices.plist
launchctl unload ~/Library/LaunchAgents/com.darttext.updatefinancials.plist

# plist 를 수정했다면 unload → (재복사) → load 순으로 다시 적용
```

## 즉시 1회 실행 / 상태 확인

```sh
# 등록된 작업을 즉시 한 번 실행 (스케줄과 무관)
launchctl start com.darttext.updateprices
launchctl start com.darttext.updatefinancials

# 등록 여부 확인
launchctl list | grep darttext

# 로그 확인
tail -f /Users/gyuyeong/dart-text2sql-wiki/data/logs/update_prices.log
tail -f /Users/gyuyeong/dart-text2sql-wiki/data/logs/update_financials.log
```

## 누락일 메꿈 동작

- **주가(`update_prices.py`)**: `prices` 테이블의 마지막 적재일 다음날부터
  오늘까지를 매 실행 시 다시 훑는다. Mac이 며칠 꺼져 있었어도 그 사이의 빠진
  거래일이 한 번에 채워진다. 주말/공휴일은 pykrx 응답에 데이터가 없어 자연히
  건너뛴다. 적재 후 `ingest_meta('latest_price_date')`를 갱신한다.
- **재무(`update_financials.py`)**: DART에 **실제로 공시된** 분기만 적재한다.
  날짜로 분기를 추정하지 않고, `fetchAllAccounts` 응답이 비어 있지 않은(=공시
  존재) 분기만 받는다. 이미 `financials`에 있는 `(종목, 분기)`는 API 호출 전에
  스킵하므로, 공시가 없는 날엔 새 데이터 0건으로 빠르게 종료한다.

## 수동 실행 (디버그)

```sh
cd /Users/gyuyeong/dart-text2sql-wiki
python3 scripts/update_prices.py
python3 scripts/update_financials.py
```

> 주의: 두 스크립트는 외부 API(pykrx / OpenDART)를 호출한다. 다른 수집
> 프로세스가 동시에 돌고 있을 땐 중복 호출을 피하기 위해 수동 실행을 자제할 것.
