"""시계열 라인차트 렌더링 — 데이터를 base64 PNG 문자열로 만든다(계층형 응답용).

레거시 전용 src/backtest/chart.py(save_nav_chart)와 **독립**이다. 그 파일은 구버전 6노드
파이프라인에서만 쓰이므로 재사용하지 않고, 코딩 패턴(지연 import + Agg 백엔드 + x축 라벨
최대 12개 + plt.close)만 참고해 새로 작성했다.

두 가지 이유로 아래 규칙을 지킨다:
1. matplotlib은 함수 내부에서 **지연 import**하고 ``matplotlib.use("Agg")`` (헤드리스 백엔드)로
   강제한다 — 서버 시작 속도/의존성 격리(save_nav_chart와 동일 이유).
2. 한글 제목이 깨지는 버그(save_nav_chart 실측: ``Glyph ... missing from font(s) DejaVu Sans``)를
   피하려고, 시스템에 설치된 한글 지원 폰트를 찾아 ``rcParams['font.family']``에 설정한다.
   폰트를 못 찾아도 예외를 삼켜 기본 폰트로 안전하게 폴백한다(그림이 깨져 보이는 건 허용,
   서버 에러는 불허).

파일로 저장하지 않고 ``io.BytesIO``에 PNG로 써서 base64 문자열(데이터 URI 접두사 없이)로
반환하므로, 총괄 응답 dict에 그대로 실어 프론트가 ``<img src="data:image/png;base64,...">``로
표시할 수 있다.
"""
from __future__ import annotations


# asof(캘린더)와 무관하게, 리눅스 서버 배포까지 고려한 한글 지원 폰트 후보(우선순위 순).
# macOS는 "AppleGothic", 리눅스 서버는 "NanumGothic"/"Noto Sans CJK KR" 등이 흔하다.
_KOREAN_FONT_CANDIDATES: tuple[str, ...] = (
    "AppleGothic",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "NanumBarunGothic",
    "Malgun Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
)


def _configure_korean_font(matplotlib) -> None:
    """시스템에 설치된 한글 지원 폰트를 찾아 rcParams에 설정한다(못 찾으면 조용히 폴백).

    폰트 탐색/설정이 어떤 이유로든 실패해도 예외를 밖으로 던지지 않는다 — 차트가 한글
    없이 깨져 보이는 것은 허용하되, 폰트 문제로 서버가 에러를 내는 것은 막는다.
    """
    try:
        import matplotlib.font_manager as fm

        available = {f.name for f in fm.fontManager.ttflist}
        for name in _KOREAN_FONT_CANDIDATES:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break
        # 한글 폰트 사용 시 마이너스 기호(−)가 네모로 깨지는 것 방지.
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001 — 폰트 문제로 절대 크래시하지 않는다
        pass


def render_line_chart_base64(
    dates: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str = "",
) -> str:
    """시계열 라인차트를 그려 순수 base64 PNG 문자열(데이터 URI 접두사 없이)로 반환한다.

    Args:
        dates: x축 라벨(시간 순, 과거→최신). 과밀 방지를 위해 최대 12개만 표시한다.
        series: {"라벨": [값들], ...} — 여러 라인(예: 전략 vs 벤치마크)을 동시 지원하며,
            단일 라인도 가능하다. 값에 None이 섞여 있으면 그 지점은 결측(NaN)으로 처리해
            선이 끊길 뿐 예외를 내지 않는다.
        title: 차트 제목(한글 가능 — _configure_korean_font로 폰트 폴백).
        ylabel: y축 라벨(선택).

    Returns:
        base64.b64encode(png_bytes).decode() 결과 문자열.
    """
    import base64
    import io
    import math

    import matplotlib  # 지연 import — 모듈 최상단 금지(서버 시작 속도/의존성 격리)

    matplotlib.use("Agg")  # 헤드리스(파일/버퍼 저장 전용) 백엔드
    _configure_korean_font(matplotlib)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(dates)))
    for label, values in series.items():
        y = [float(v) if v is not None else math.nan for v in values]
        ax.plot(x[: len(y)], y, label=str(label), linewidth=1.8)

    # x축 라벨은 과밀 방지를 위해 최대 12개만 표시(save_nav_chart와 동일 패턴).
    if dates:
        step = max(1, len(dates) // 12)
        ticks = list(range(0, len(dates), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(dates[i]) for i in ticks], rotation=45, ha="right", fontsize=7)

    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:  # 라인이 하나면 범례가 불필요(단일 종가 추이 등)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)  # 메모리 누수 방지(save_nav_chart와 동일)
    return base64.b64encode(buf.getvalue()).decode()


# 산점도 점 라벨(종목명)을 실제로 찍는 최대 점 수 — 이보다 많으면 라벨이 겹쳐 뭉개지므로
# 점만 찍는다(라인차트 x축 라벨을 최대 12개로 제한하는 것과 같은 과밀 방지 관례).
_SCATTER_LABEL_MAX = 30


def render_scatter_chart_base64(
    x: list[float],
    y: list[float],
    labels: list[str] | None,
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    """산점도를 그려 순수 base64 PNG 문자열(데이터 URI 접두사 없이)로 반환한다.

    render_line_chart_base64와 동일한 한글폰트 폴백/Agg 백엔드/지연 import/plt.close 관례를
    그대로 재사용한다(공용 폰트 설정은 _configure_korean_font). 두 팩터(예: 이익수익률 vs
    투하자본수익률)의 관계를 점으로 뿌린다.

    Args:
        x, y: 각 점의 x/y 좌표(길이 동일). 값에 None이 없어야 한다(호출부 scatter_data에서
            이미 두 좌표가 모두 있는 점만 걸러 넘긴다).
        labels: 각 점의 라벨(종목명 등). 주어지고 점이 과밀하지 않으면(_SCATTER_LABEL_MAX 이하)
            점 옆에 라벨을 표기한다. None이면 라벨 없이 점만 찍는다.
        x_label / y_label: 축 라벨(한글 가능). title: 제목(한글 가능).

    Returns:
        base64.b64encode(png_bytes).decode() 결과 문자열.
    """
    import base64
    import io

    import matplotlib  # 지연 import — 라인차트와 동일(서버 시작 속도/의존성 격리)

    matplotlib.use("Agg")  # 헤드리스 백엔드
    _configure_korean_font(matplotlib)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(x, y, s=18, alpha=0.6, edgecolors="none")

    # 점이 적을 때만 라벨을 붙인다(많으면 겹쳐 뭉개지므로 생략 — 과밀 방지).
    if labels and len(labels) <= _SCATTER_LABEL_MAX:
        for xi, yi, name in zip(x, y, labels):
            ax.annotate(str(name), (xi, yi), fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)  # 메모리 누수 방지(라인차트와 동일)
    return base64.b64encode(buf.getvalue()).decode()


def render_bar_chart_base64(
    labels: list[str],
    values: list[float],
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    """막대그래프를 그려 순수 base64 PNG 문자열(데이터 URI 접두사 없이)로 반환한다.

    render_line_chart_base64/render_scatter_chart_base64와 동일한 한글폰트 폴백/Agg
    백엔드/지연 import/plt.close 관례를 그대로 재사용한다. quantile_bucket_means(분위별
    평균) 같은 "그룹별 단일 값 비교" 결과를 시각화할 때 쓴다.

    Args:
        labels: 각 막대의 x축 라벨(예: "1분위".."5분위"). values와 길이 동일.
        values: 각 막대의 높이(음수 가능 — 적자 구간 등).
        x_label / y_label: 축 라벨(한글 가능). title: 제목(한글 가능).

    Returns:
        base64.b64encode(png_bytes).decode() 결과 문자열.
    """
    import base64
    import io

    import matplotlib  # 지연 import — 다른 차트 함수와 동일(서버 시작 속도/의존성 격리)

    matplotlib.use("Agg")  # 헤드리스 백엔드
    _configure_korean_font(matplotlib)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(labels)))
    ax.bar(x, values, color="#4C72B0")
    ax.set_xticks(x)
    # 라벨이 많으면(과밀 방지, 라인차트 x축 관례와 동일) 회전+작은 글씨로 표시.
    ax.set_xticklabels([str(v) for v in labels], rotation=45 if len(labels) > 8 else 0, ha="right" if len(labels) > 8 else "center", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axhline(0, color="black", linewidth=0.8)  # 음수 막대가 섞여도 기준선이 보이게
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)  # 메모리 누수 방지(다른 차트 함수와 동일)
    return base64.b64encode(buf.getvalue()).decode()
