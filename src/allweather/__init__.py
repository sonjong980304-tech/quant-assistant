"""올웨더 포트폴리오 모니터링 (표시+알림 전용).

.omc/specs/brainstorming-all-weather-portfolio.md 참고.

quant_trader가 실제 운용 중인 올웨더 포트폴리오(QQQ/삼성전자/TLT/ACE KRX금현물 + IEF/TIP/BIL
7종목, 낙폭(MDD) -20% 이내 제약을 건 샤프비율 최적화 몬테카를로 리밸런싱)를 이 프로젝트 웹
화면에서 모니터링한다. 매달 1일 배치가 실제 가격으로 walk-forward 백테스트를 돌려 비중·MDD·
CAGR·누적수익률·샤프·소르티노를 계산해 DB(all_weather_snapshot)에 이력으로 쌓고, 화면은 그
저장값을 읽기만 한다. 계산이 끝나면 quant_trader와 동일한 텔레그램 채널로 직전 달 대비 비중
변경분을 포함한 알림을 보낸다.

2026-07 사용자 최종 결정: BIL 추가 후 실측 비교에서 MDD -20% 제약 버전(run_monte_carlo_mdd_
constrained, montecarlo.py)이 무제약 샤프 극대화보다 샤프비율·MDD 둘 다 더 나아 배치 기본값
으로 채택했다(무제약 run_monte_carlo도 여전히 남아있어 필요시 pipeline의 monte_carlo_fn
인자로 주입 가능).

이 기능은 어디까지나 "표시+알림"이며, 실제 매매 주문 실행은 quant_trader의 영역으로 남긴다
(quant_trader는 read-only 참고만 했고 import·수정하지 않는다 — Approach B로 계산 로직만 복제).
"""
