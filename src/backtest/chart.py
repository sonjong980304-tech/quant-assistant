"""백테스트 NAV 차트 PNG 생성.

스펙: 차트가 필요한 응답은 matplotlib으로 PNG 파일을 만들고 경로만 출력한다
(터미널에 그림을 직접 그리지 않음). 헤드리스 환경을 위해 Agg 백엔드를 강제한다.
"""
from __future__ import annotations


def save_nav_chart(dates, navs, path: str, benchmark=None, title: str = "백테스트") -> str:
    """NAV(순자산가치) 시계열을 PNG로 저장하고 파일 경로를 반환한다.

    dates: x축 라벨(리밸런싱 날짜), navs: 전략 NAV, benchmark: (선택) 벤치마크 NAV.
    """
    import matplotlib  # 지연 import — 무거운 라이브러리는 필요 시점에만 로드한다

    matplotlib.use("Agg")  # 헤드리스(파일 저장 전용) 백엔드
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(navs)))
    ax.plot(x, navs, label="Strategy", linewidth=1.8)
    if benchmark is not None and len(benchmark) == len(navs):
        ax.plot(x, benchmark, label="Benchmark", linewidth=1.2, linestyle="--")
    # x축 라벨은 과밀 방지를 위해 최대 12개만 표시
    if dates:
        step = max(1, len(dates) // 12)
        ticks = list(range(0, len(dates), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("NAV (start=1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
