# 매크로 지표 에이전트 — 임계값/데이터소스 factcheck (2026-07-14)

브레인스토밍 인터뷰 Round 2에서 사용자 요청으로 구현 전 1차 출처 검증. `.omc/plans/redesign-multiagent-architecture.md` §8.2 초안 임계값을 검증한 결과.

## 검증 결과

| 항목 | 확정값 | 출처(URL) | 확신도 |
|---|---|---|---|
| FRED 장단기금리차 시리즈 ID | `T10Y2Y` (10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity). 단위=%, 일간 갱신, 비계절조정, 1976-06-01부터 | [fred.stlouisfed.org/series/T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y) | **확정** |
| 금리차 역전(inversion) 정의 | 스프레드 < 0%가 표준 컨벤션(단기금리>장기금리). 1955년 이후 거의 모든 침체를 선행. 역전→침체 시차 평균 12~24개월(SF Fed) | Chicago Fed, SF Fed, CNBC 등 교차확인 | **확정** |
| ±0.5%p "정상/평탄화" 2차 경계값 | 학계·기관의 표준 컨벤션으로 확인되지 않음 — 설계자 재량의 완충구간으로 보임 | 검색 결과 중 이 특정 숫자(0.5%p)를 표준으로 명시한 1차 출처 없음 | **불확실** — 사용자 결정 필요 |
| CNN Fear & Greed Index 스케일 | 0~100. 공식 구간: Extreme Fear 0-24 / Fear ~25-44 / Neutral 45-55 / Greed ~56-75 / Extreme Greed 76-100 (7개 하위지표 동일가중 합성) | [cnn.com/markets/fear-and-greed](https://www.cnn.com/markets/fear-and-greed), [CNN Business 설명](https://www.cnn.com/2025/04/07/business/what-is-cnn-fear-and-greed-index) | **확정** |
| FRED VIX 시리즈 ID | `VIXCLS` (CBOE Volatility Index, 공식, CBOE 출처, 일간, 1990-01-02부터) | [fred.stlouisfed.org/series/VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | **확정** |
| VIX 레짐 구간 <15/15-20/20-30/≥30 | 업계에서 실제 쓰이는 컨벤션과 일치(<15 안정·복만, 15-20 정상, 20-30 공포 형성, ≥30 패닉) | Volatility Box, 복수 실무 가이드 교차확인 | **확정 (VIX 기준으로는)** |

## 핵심 발견 — 설계문서 §8.2의 지표 혼동

`.omc/plans/redesign-multiagent-architecture.md` §8.2는 데이터 소스를 **"CNN Fear & Greed Index"(0~100 스케일)**로 명시했지만, 판정 임계값(`<15 안정/15~20 보통/20~30 경계/≥30 공포`)은 **VIX(변동성지수, 10~80대 스케일)의 실무 컨벤션과 정확히 일치**한다. CNN Fear & Greed Index의 실제 공식 구간(0-24/25-44/45-55/56-75/76-100)과는 스케일 자체가 다르다.

→ 설계문서가 두 개의 서로 다른 지표를 뒤섞어 적었다. 아래 둘 중 하나를 사용자가 선택해야 함:

1. **CNN Fear & Greed Index 유지**: Selenium 필요(JS 동적 렌더링, 비공식 엔드포인트, 페이지 구조 변경에 취약). 임계값은 CNN 공식 구간(0-24/25-44/45-55/56-75/76-100)으로 다시 써야 함.
2. **VIX(FRED VIXCLS)로 전환**: 장단기금리차(T10Y2Y)와 완전히 동일한 `pandas_datareader` FRED 경로로 통일 가능 — **Selenium 의존성 자체가 이번 기능에서 사라짐**. 임계값은 설계문서 초안 그대로(<15/15-20/20-30/≥30) 사용 가능(이미 검증됨).

## 참고: ±0.5%p 경계값

표준 컨벤션 미확인. 구현 시 이 값은 "역전(0% 미만)"이라는 확정된 임계값과 별개로, 사용자가 직접 정하거나 임의값임을 명시하고 채택해야 함.
