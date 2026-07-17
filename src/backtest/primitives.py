"""SQL/Python 조합형 분석 파이프라인의 사전검증 프리미티브(v1 6종).

.omc/wiki/dart-text2sql-wiki-sql-python.md 아키텍처 결정 참고.
**절대 원칙**: LLM은 파이썬 소스코드를 절대 생성하지 않는다. LLM은 이 6개 프리미티브를
JSON으로 "어떤 순서로 조립할지"만 지시하고(pipeline_exec.py), 결정론적 실행기가 고정 dict
디스패치로 그대로 실행한다(eval/exec 없음).

- get_cross_section / zscore / neutralize / combine: 기존 metrics_at()/select_stocks()를
  얇게 래핑(새 로직 최소화).
- regress: 신규 순수 OLS(K-ratio용 기울기/표준오차).
- optimize_weights: Riskfolio-Lib의 3가지 최적화(max_sharpe/min_variance/risk_parity)만
  감싼 래퍼 — 라이브러리 임의 호출은 노출하지 않는다.

모두 순수 함수: DB 접속은 인자로 받은 conn을 통해서만 하고, 프리미티브 내부에서 자체적으로
접속을 새로 열지 않는다. 네트워크/LLM/외부라이브러리 호출은 파마프렌치(fama_french.py)의
DI 관례대로 주입 가능한 함수(metrics_fn/solve_fn)로 분리해 mock 단위테스트가 가능하다.
"""
from __future__ import annotations

from typing import Callable

from .data_access import metrics_at, price_history_batch
from .selection import select_stocks


# --------------------------------------------------------------------------
# 1. get_cross_section — metrics_at() 래핑 (횡단면 스냅샷)
# --------------------------------------------------------------------------
def get_cross_section(
    conn, asof, fields=None, markets=None, metrics_fn: Callable | None = None
) -> list[dict]:
    """시점 asof의 유효 지표 행 목록. 기존 metrics_at()를 얇게 래핑한다
    (look-ahead 방지/생존편향 제거/이상치 가드를 그대로 재사용).

    fields가 주어지면 그 컬럼만 남긴다(식별자 stock_code/name/sector/market/quarter는 항상 유지).
    markets가 주어지면 그 시장(예: ["KOSPI"])만 남긴다(run_backtest/select_stocks의 markets
    파라미터와 동일 이름·의미 — "코스피 전 종목"류 질문에서 시장 필터가 필요한 correlation/
    quantile_bucket_means/scatter_data 파이프라인이 이 단계에서 바로 걸러낼 수 있게 한다).
    metrics_fn은 테스트 주입용(기본=metrics_at) — SQL은 metrics_at의 바인딩 쿼리를 재사용한다.
    """
    metrics_fn = metrics_fn or metrics_at
    rows = metrics_fn(conn, asof)
    if markets:
        rows = [r for r in rows if r.get("market") in markets]
    if fields:
        keep = set(fields) | {"stock_code", "name", "sector", "market", "quarter"}
        rows = [{k: v for k, v in r.items() if k in keep} for r in rows]
    return rows


# --------------------------------------------------------------------------
# 2. zscore — select_stocks(단일 기준 zscore 조합) 래핑
# --------------------------------------------------------------------------
def zscore(rows: list[dict], field: str, direction: str = "high", n: int | None = None) -> list[dict]:
    """단일 팩터 횡단면 z-score 랭킹. 기존 select_stocks의 zscore 조합을 1개 기준으로 래핑한다.

    반환은 우수순으로 정렬된 행 목록이며 각 행에 select_stocks의 '_score'가 붙는다
    (select_stocks 관례: 점수가 낮을수록 우수). direction='high'면 큰 값이 우수.
    """
    size = n if n is not None else (len(rows) or 1)
    return select_stocks(
        rows,
        [{"key": field, "direction": direction, "weight": 1.0}],
        combine="zscore",
        n=size,
    )


# --------------------------------------------------------------------------
# 3. neutralize — 그룹(섹터) 내 평균 제거 (섹터중립화)
# --------------------------------------------------------------------------
def neutralize(rows: list[dict], field: str, by: str = "sector", method: str = "demean") -> list[dict]:
    """섹터(기본) 중립화: 같은 그룹(by) 안에서 field의 그룹간 수준차를 제거한다.

    method='demean'(기본, 회귀 유지): 그룹 평균만 뺀다.
    method='zscore': 그룹 평균을 빼고 그룹 표준편차(모집단, ddof=0)로 나눠 정규화한다 —
    섹터마다 변동성이 다르면 순수 디민만으로는 섹터 간 비교가 왜곡되므로(변동성 큰 섹터가
    과대 반영), 실제 섹터중립 포트폴리오 구성에는 이 표준화가 필요하다. 그룹 표본이 1개뿐이면
    표준편차가 0이라 정규화가 수학적으로 정의되지 않으므로 해당 행은 None을 반환한다
    (correlation()의 분산=0 처리와 동일 원칙 — 억지로 0을 주지 않음).

    각 행에 '{field}_neutral'을 부여해 반환(원 순서/원 필드 유지). select_stocks에는
    중립화 기능이 없어 여기서만 최소한의 그룹 통계 로직을 더한다.
    """
    if method not in ("demean", "zscore"):
        raise ValueError(f"지원하지 않는 method: {method} (demean|zscore만 가능)")
    groups: dict = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        groups.setdefault(r.get(by), []).append(v)
    means = {g: sum(vs) / len(vs) for g, vs in groups.items()}
    stds: dict = {}
    if method == "zscore":
        import numpy as np

        stds = {g: float(np.asarray(vs, dtype=float).std()) for g, vs in groups.items()}
    out = []
    for r in rows:
        v = r.get(field)
        nr = dict(r)
        g = r.get(by)
        if v is None or g not in means:
            nr[f"{field}_neutral"] = None
        elif method == "demean":
            nr[f"{field}_neutral"] = v - means[g]
        else:  # zscore
            sd = stds[g]
            nr[f"{field}_neutral"] = ((v - means[g]) / sd) if sd else None
        out.append(nr)
    return out


# --------------------------------------------------------------------------
# 3b. winsorize — IQR(사분위범위) 기반 이상치 완화
# --------------------------------------------------------------------------
def winsorize(rows: list[dict], field: str, k: float = 1.5) -> list[dict]:
    """field 값을 [Q1-k·IQR, Q3+k·IQR] 범위 밖이면 경계로 눌러 붙인다(clip, 행 삭제 없음).

    zscore/combine의 z-score 조합은 극단치 하나가 평균·표준편차 자체를 흔들 수 있고,
    combine의 rank_sum은 순위만 매겨 극단치 크기에 영향받지 않는 대신 "얼마나 튀었는지"
    정보를 버린다. winsorize는 그 중간 방식으로, 극단치의 크기 정보는 유지하되 일정
    범위 밖의 값만 경계로 눌러 붙인다. neutralize와 같은 관례로 원본 field는 보존하고
    '{field}_winsorized'를 새로 추가한다 — 뒤 단계(zscore/combine 등)에 그 필드를
    넘기면 눌린 값 기준으로 계산된다. None은 그대로 None 유지, 사분위 계산에서 제외.
    표본이 4개 미만이면 사분위 경계를 안정적으로 낼 수 없어 원본값을 그대로 통과시킨다.
    k=1.5는 박스플롯 whisker 기준으로 흔히 쓰는 이상치 경계 배수.
    """
    import numpy as np

    vals = [r[field] for r in rows if r.get(field) is not None]
    if len(vals) < 4:
        return [dict(r, **{f"{field}_winsorized": r.get(field)}) for r in rows]

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    out = []
    for r in rows:
        v = r.get(field)
        nr = dict(r)
        nr[f"{field}_winsorized"] = None if v is None else float(min(max(v, lower), upper))
        out.append(nr)
    return out


# --------------------------------------------------------------------------
# 3c. remove_outliers — IQR 기반 이상치 "행 제거"(winsorize=값 누르기와 목적이 다름)
# --------------------------------------------------------------------------
def remove_outliers(rows: list[dict], field: str, method: str = "iqr", k: float = 1.5) -> list[dict]:
    """field 값이 [Q1-k·IQR, Q3+k·IQR] 범위를 벗어나는 row를 통째로 제거해 나머지를 반환한다.

    winsorize(극단치를 경계로 눌러 값 자체를 바꿈, 행 보존)와 목적이 다르다 — 이건 산점도에서
    "이상치를 빼고 그려줘" 같은 요청처럼 이상치 행을 실제로 걷어낼 때 쓴다. field 값이 None인
    row는 이 필드로 이상치 판단이 불가하므로 조용히 유지한다(데이터 자체를 지우는 게 아니라 이
    필드 기준으로 튄 행만 제거하는 것). 유효 표본이 4개 미만이면 사분위 경계를 안정적으로 낼 수
    없어 원본을 그대로 통과시킨다(winsorize와 동일한 표본부족 관례). 원본 rows는 변경하지 않고
    새 리스트를 반환한다. k=1.5는 박스플롯 whisker 기준의 흔한 이상치 경계 배수.
    """
    if method != "iqr":
        raise ValueError(f"지원하지 않는 method: {method} (iqr만 가능)")
    import numpy as np

    vals = [r[field] for r in rows if r.get(field) is not None]
    if len(vals) < 4:
        return list(rows)

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    out = []
    for r in rows:
        v = r.get(field)
        if v is None or lower <= v <= upper:
            out.append(r)
    return out


# --------------------------------------------------------------------------
# 3d. scatter_data — rows에서 두 필드를 산점도용 (x, y, labels) dict로 변환
# --------------------------------------------------------------------------
def scatter_data(rows: list[dict], x_field: str, y_field: str, label_field: str = "name") -> dict:
    """rows(예: get_cross_section→remove_outliers 결과)에서 두 필드를 뽑아 산점도용 dict로 만든다.

    반환 {"x":[...], "y":[...], "labels":[...], "x_field":.., "y_field":..} — 총괄
    에이전트(_extract_scatter_data)가 이 형태를 인식해 산점도를 렌더링한다. 산점도의 한 점은
    x·y 두 좌표가 모두 있어야 찍히므로, 둘 중 하나라도 None인 row는 제외한다(라벨은 label_field,
    없으면 stock_code로 폴백). "이익수익률과 투하자본수익률 산점도" 같은 파이프라인의 마지막
    단계로 쓴다.
    """
    xs: list = []
    ys: list = []
    labels: list = []
    for r in rows:
        x, y = r.get(x_field), r.get(y_field)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        labels.append(r.get(label_field) or r.get("stock_code"))
    return {"x": xs, "y": ys, "labels": labels, "x_field": x_field, "y_field": y_field}


# --------------------------------------------------------------------------
# 4. combine — select_stocks(멀티팩터 가중조합) 래핑
# --------------------------------------------------------------------------
def combine(
    rows: list[dict],
    criteria: list[dict],
    method: str = "zscore",
    n: int = 20,
    sectors=None,
    markets=None,
    sector_neutral: bool = False,
) -> list[dict]:
    """멀티팩터 가중조합 선정. 기존 select_stocks를 그대로 감싼다(사실상 이 함수가 combine).

    criteria 예: [{"key":"per","direction":"low","weight":0.5},
                  {"key":"roe","direction":"high","weight":0.5}]
    method: 'and' | 'rank_sum' | 'zscore'.
    sector_neutral: method="zscore"일 때만 유효 — 섹터별로 따로 z-score를 구해(섹터 내부
    상대순위) 전체 비교한다(select_stocks에 그대로 전달). 기본 False면 기존 동작 그대로.
    """
    return select_stocks(
        rows, criteria, combine=method, n=n, sectors=sectors, markets=markets,
        sector_neutral=sector_neutral,
    )


# --------------------------------------------------------------------------
# 5. regress — 신규 순수 OLS (K-ratio 계산용)
# --------------------------------------------------------------------------
def regress(y, x=None) -> dict:
    """단순선형회귀(OLS) y = intercept + slope·x. K-ratio에 필요한 기울기/표준오차를 반환한다.

    x가 없으면 0,1,2,... 시간 인덱스를 사용한다(누적수익률 시계열의 추세 회귀 = K-ratio 용도).
    반환 dict: {slope, intercept, se_slope, se_intercept, r_squared, t_stat, n, k_ratio}
      · k_ratio = slope / se_slope (Kestner K-ratio의 기본형: 추세의 기울기를 그 표준오차로 나눔).
    """
    import numpy as np

    yv = np.asarray(list(y), dtype=float)
    n = len(yv)
    if n < 3:
        raise ValueError("회귀에는 관측치가 3개 이상 필요합니다")
    xv = np.arange(n, dtype=float) if x is None else np.asarray(list(x), dtype=float)
    if len(xv) != n:
        raise ValueError("x와 y의 길이가 다릅니다")

    xbar, ybar = xv.mean(), yv.mean()
    sxx = float(((xv - xbar) ** 2).sum())
    if sxx == 0:
        raise ValueError("x의 분산이 0이라 회귀 불가")
    slope = float(((xv - xbar) * (yv - ybar)).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    resid = yv - (intercept + slope * xv)
    dof = n - 2
    sigma2 = float((resid ** 2).sum() / dof)  # 잔차분산(불편추정)
    se_slope = float((sigma2 / sxx) ** 0.5)
    se_intercept = float((sigma2 * (1.0 / n + xbar ** 2 / sxx)) ** 0.5)
    ss_tot = float(((yv - ybar) ** 2).sum())
    r_squared = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
    k_ratio = slope / se_slope if se_slope > 0 else 0.0
    t_stat = k_ratio
    return {
        "slope": slope,
        "intercept": intercept,
        "se_slope": se_slope,
        "se_intercept": se_intercept,
        "r_squared": r_squared,
        "t_stat": t_stat,
        "n": n,
        "k_ratio": k_ratio,
    }


# --------------------------------------------------------------------------
# 5b. correlation — 두 크로스섹션 필드 간 피어슨 상관계수
# --------------------------------------------------------------------------
def correlation(rows: list[dict], field_x: str, field_y: str) -> dict:
    """rows에서 field_x/field_y 두 값을 뽑아 피어슨 상관계수를 구한다(PBR-GPA 같은 팩터간 비교용).

    compute_ic(팩터 vs 미래 실현수익률의 순위상관)와는 목적이 다르다 — 이건 두 "동시점"
    필드끼리의 선형 상관관계를 본다. 값이 None인 행은 조용히 제외한다(None 섞인 필드가
    많으므로 에러 대신 흡수). 유효 표본이 2개 미만이면 표준편차 자체를 정의할 수 없어
    ValueError. 표본이 있어도 한쪽 분산이 0(전부 같은 값)이면 상관계수가 수학적으로
    정의되지 않으므로 correlation=None으로 그 사실을 그대로 반환한다(억지로 0을 주지 않음).
    """
    import numpy as np

    xs, ys = [], []
    for r in rows:
        x, y = r.get(field_x), r.get(field_y)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    n = len(xs)
    if n < 2:
        raise ValueError(f"상관계수 계산에는 유효 표본이 2개 이상 필요합니다(현재 {n}개)")
    xarr, yarr = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if xarr.std() == 0 or yarr.std() == 0:
        return {"correlation": None, "n": n}
    return {"correlation": float(np.corrcoef(xarr, yarr)[0, 1]), "n": n}


# --------------------------------------------------------------------------
# 5c. quantile_bucket_means — bucket_field 기준 N분위로 나눠 분위별 value_field 평균
# --------------------------------------------------------------------------
def quantile_bucket_means(rows: list[dict], bucket_field: str, value_field: str, n: int = 5) -> list[dict]:
    """bucket_field(예: PBR) 오름차순으로 정렬해 동일 개수로 n분위 나누고, 분위별 value_field
    (예: GPA)의 평균을 구한다("PBR 분위수별 평균 GPA" 같은 팩터 분석용).

    분위 경계는 값 기준이 아니라 "정렬 후 동일 개수로 분할"이다 — 값이 한쪽에 몰려 있어도
    분위별 표본 수가 고르게 유지된다(퀀트 팩터 분석에서 흔히 쓰는 방식). 두 필드 중 하나라도
    None인 행은 제외한다. bucket=1이 bucket_field가 가장 낮은 그룹, bucket=n이 가장 높은
    그룹이다. 유효 표본이 n개 미만이면 분위를 나눌 수 없어 ValueError.
    """
    valid = [r for r in rows if r.get(bucket_field) is not None and r.get(value_field) is not None]
    if len(valid) < n:
        raise ValueError(
            f"유효 표본 {len(valid)}개가 분위 수 {n}개보다 적어 분위를 나눌 수 없습니다"
        )
    ordered = sorted(valid, key=lambda r: r[bucket_field])
    size = len(ordered)
    out = []
    for i in range(n):
        lo = size * i // n
        hi = size * (i + 1) // n
        group = ordered[lo:hi]
        bucket_vals = [r[bucket_field] for r in group]
        value_vals = [r[value_field] for r in group]
        out.append({
            "bucket": i + 1,
            "count": len(group),
            "bucket_range": [bucket_vals[0], bucket_vals[-1]],
            "mean_value": sum(value_vals) / len(value_vals),
        })
    return out


# --------------------------------------------------------------------------
# 5d. histogram_buckets — field 값을 균등폭 구간으로 나눠 구간별 개수(진짜 히스토그램)
# --------------------------------------------------------------------------
def histogram_buckets(rows: list[dict], field: str, num_buckets: int = 10) -> dict:
    """rows에서 field 값을 뽑아 num_buckets개의 균등폭 구간으로 나눠 구간별 개수를 센다.

    quantile_bucket_means(동일 "개수"로 나누는 분위수)와 달리 이건 값의 범위를 동일 "폭"으로
    나눈다(구간별 표본수는 다를 수 있음 — 진짜 히스토그램 정의). None 값은 조용히 제외한다
    (다른 프리미티브와 동일 관례).

    반환: {"field": field, "num_buckets": 실제사용된구간수, "bucket_edges": [n+1개, 오름차순],
           "counts": [n개], "n": 유효표본수}
    각 구간은 [edges[i], edges[i+1]) 반열림이며, 마지막 구간만 최댓값을 포함하는 닫힌구간이다.
    """
    if num_buckets < 1:
        raise ValueError(f"num_buckets는 1 이상이어야 합니다(현재 {num_buckets})")
    vals = [r[field] for r in rows if r.get(field) is not None]
    n = len(vals)
    if n == 0:
        raise ValueError(f"유효 표본이 0개라 히스토그램을 만들 수 없습니다(field={field})")

    lo, hi = min(vals), max(vals)
    if lo == hi:
        # 전부 동일값 → 폭이 0이라 구간을 나눌 수 없다(0으로 나누기 방지). 단일 구간으로 반환.
        return {"field": field, "num_buckets": 1, "bucket_edges": [lo, hi], "counts": [n], "n": n}

    width = (hi - lo) / num_buckets
    counts = [0] * num_buckets
    for v in vals:
        idx = int((v - lo) / width)
        if idx == num_buckets:  # v == hi일 때 경계 넘침 → 마지막 구간으로 보정
            idx = num_buckets - 1
        counts[idx] += 1
    bucket_edges = [lo + i * width for i in range(num_buckets)] + [hi]
    return {"field": field, "num_buckets": num_buckets, "bucket_edges": bucket_edges,
            "counts": counts, "n": n}


# --------------------------------------------------------------------------
# 6. optimize_weights — Riskfolio-Lib 3종 최적화만 감싼 래퍼
# --------------------------------------------------------------------------
_OPTIMIZE_METHODS = ("max_sharpe", "min_variance", "risk_parity")


def _normalize_returns(returns, assets):
    """returns를 (자산이름 list, 행=기간·열=자산 2D matrix)로 정규화한다.

    - dict{asset: [수익률...]} : 키=자산, 값=시계열 → 전치해 행=기간으로 만든다.
    - list[list] (행=기간, 열=자산) : assets 이름 목록이 있으면 그대로, 없으면 asset_0..N.
    """
    if isinstance(returns, dict):
        cols = list(returns.keys())
        series = [list(returns[c]) for c in cols]
        length = len(series[0]) if series else 0
        if any(len(s) != length for s in series):
            raise ValueError("자산별 수익률 시계열 길이가 서로 다릅니다")
        matrix = [[series[j][i] for j in range(len(cols))] for i in range(length)]
        return cols, matrix
    matrix = [list(row) for row in returns]
    width = len(matrix[0]) if matrix else 0
    cols = list(assets) if assets else [f"asset_{i}" for i in range(width)]
    if len(cols) != width:
        raise ValueError("assets 개수가 수익률 열 수와 다릅니다")
    return cols, matrix


def _riskfolio_solve(matrix, method: str, rf: float) -> list[float]:
    """Riskfolio-Lib 실제 호출(v7.3.0 검증). 자유 호출 금지 — 여기서만 3종 메서드로 제한한다.

    검증된 시그니처(2026-07-12 실측):
      rp.Portfolio(returns=df) → assets_stats(method_mu='hist', method_cov='hist')
      · max_sharpe   : optimization(model='Classic', rm='MV', obj='Sharpe',  rf, l=0, hist=True)
      · min_variance : optimization(model='Classic', rm='MV', obj='MinRisk', rf, l=0, hist=True)
      · risk_parity  : rp_optimization(model='Classic', rm='MV', rf, b=None, hist=True)
    반환 w는 index=자산, 컬럼 'weights'인 DataFrame(합=1, 롱온리).
    """
    import numpy as np  # 지연 import — fama_french.py의 pandas_datareader 패턴과 동일
    import pandas as pd
    import riskfolio as rp

    df = pd.DataFrame(np.asarray(matrix, dtype=float))
    port = rp.Portfolio(returns=df)
    port.assets_stats(method_mu="hist", method_cov="hist")
    if method == "max_sharpe":
        w = port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=rf, l=0, hist=True)
    elif method == "min_variance":
        w = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=rf, l=0, hist=True)
    else:  # risk_parity
        w = port.rp_optimization(model="Classic", rm="MV", rf=rf, b=None, hist=True)
    if w is None or len(w) == 0:
        raise ValueError("포트폴리오 최적화 해를 찾지 못했습니다")
    return [float(w.iloc[i, 0]) for i in range(w.shape[0])]


def optimize_weights(
    returns,
    method: str = "max_sharpe",
    assets=None,
    rf: float = 0.0,
    solve_fn: Callable | None = None,
) -> dict:
    """포트폴리오 최적화 비중을 반환한다(Riskfolio-Lib 3종만 노출).

    returns: dict{asset: [수익률...]} 또는 list[list](행=기간, 열=자산; assets 이름 필요).
    method: 'max_sharpe' | 'min_variance' | 'risk_parity' (그 외는 거부).
    반환: {asset: weight} (Riskfolio 기본 롱온리, 합=1).
    solve_fn은 테스트 주입용(기본=_riskfolio_solve). 라이브러리 임의 API는 노출하지 않는다.
    """
    if method not in _OPTIMIZE_METHODS:
        raise ValueError(
            f"지원하지 않는 method: {method} (max_sharpe/min_variance/risk_parity만 가능)"
        )
    solve_fn = solve_fn or _riskfolio_solve
    cols, matrix = _normalize_returns(returns, assets)
    weights = solve_fn(matrix, method, rf)
    return {c: float(w) for c, w in zip(cols, weights)}


# --------------------------------------------------------------------------
# 7. run_backtest_primitive — engine.run_backtest()를 파이프라인에서 호출 가능하게 래핑
# --------------------------------------------------------------------------
# pipeline_exec.MAX_SIZE와 같은 값이지만, primitives ← pipeline_exec 순환 import를 피하려고
# 여기 별도 상수로 둔다(architect 재검수 지적: start_year/end_year는 정수라 실행기의
# _check_size가 이 프리미티브가 내부에서 만드는 window를 못 봄 — 반드시 연산 시작 '전'에
# 프리미티브 자체가 상한을 강제해야 한다. 값을 바꾸면 pipeline_exec.MAX_SIZE도 함께 바꿀 것).
_MAX_REBALANCE_STEPS = 4000
_MIN_START_YEAR = 1990  # 극단적 과거 연도로 거대한 날짜 리스트가 애초에 생성되지 않도록 하한
_VALID_REBALANCE_FREQS = {"monthly", "quarterly", "semiannual", "annual"}


def _resolve_rebalance_dates(conn, start_year, end_year, rebalance, dates_fn, max_date_fn) -> list[str]:
    """리밸런싱 시점 목록을 안전상한 검사 후 해석한다(run_backtest_primitive/compute_ic_primitive 공용).

    무거운 연산(전종목 metrics_at 반복)을 시작하기 '전'에 거부하는 순서를 지킨다
    (같은 머신에 실거래봇이 상주 — 자원 남용 방지는 사전 검사여야 사후 검사보다 의미 있음).
    """
    from .data_access import rebalance_dates as _rebalance_dates

    if start_year < _MIN_START_YEAR:
        raise ValueError(f"start_year {start_year} < 하한 {_MIN_START_YEAR}(자원 남용 방지)")
    if rebalance not in _VALID_REBALANCE_FREQS:
        raise ValueError(f"지원하지 않는 rebalance: {rebalance} ({sorted(_VALID_REBALANCE_FREQS)}만 가능)")
    # end_year가 병적으로 크면(예: 10억) dates_fn이 리스트를 만드는 과정 자체가 메모리를
    # 소진할 수 있다 — 리스트 생성 '전'에 연도 범위부터 거부한다(LOW, architect 3차 권고).
    if end_year - start_year > _MAX_REBALANCE_STEPS:
        raise ValueError(
            f"end_year-start_year {end_year - start_year} > 상한 {_MAX_REBALANCE_STEPS}(자원 남용 방지)"
        )

    dates_fn = dates_fn or _rebalance_dates
    all_dates = dates_fn(start_year, end_year, rebalance)
    if len(all_dates) > _MAX_REBALANCE_STEPS:
        raise ValueError(
            f"리밸런싱 시점 {len(all_dates)}개 > 상한 {_MAX_REBALANCE_STEPS}개(자원 남용 방지)"
        )
    maxd = max_date_fn(conn) if max_date_fn else conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    dates = [d for d in all_dates if not maxd or d <= maxd]
    if len(dates) < 2:
        raise ValueError("선택 기간에 데이터가 부족합니다(주가 시계열 범위 확인).")
    return dates


def _resolve_engine_inputs(
    conn, start_year, end_year, rebalance, market, dates_fn, max_date_fn,
    callbacks_fn, benchmark_fn_factory, backtest_fn,
):
    """리밸런싱 날짜 해석 + market별 콜백/벤치마크/백테스트 함수 기본값 선택.

    run_backtest_primitive와 search_strategy가 공유하는 준비 단계(둘 다 "무거운 연산(전종목
    metrics_at 반복) 전 사전상한 검사 → market별 콜백 선택" 순서가 동일) — 중복 제거용 공용 헬퍼.
    """
    from .data_access import build_benchmark_fn as _build_benchmark_fn
    from .data_access import build_callbacks as _build_callbacks
    from .engine import run_backtest as _run_backtest

    dates = _resolve_rebalance_dates(conn, start_year, end_year, rebalance, dates_fn, max_date_fn)
    if market == "US":
        from .data_access_us import build_callbacks_us as _build_callbacks_us
        callbacks_fn = callbacks_fn or _build_callbacks_us
    else:
        callbacks_fn = callbacks_fn or _build_callbacks
    benchmark_fn_factory = benchmark_fn_factory or _build_benchmark_fn
    backtest_fn = backtest_fn or _run_backtest
    return dates, callbacks_fn, benchmark_fn_factory, backtest_fn


def run_backtest_primitive(
    conn,
    start_year: int,
    end_year: int,
    weights: dict | None = None,
    criteria: list | None = None,
    combine: str = "zscore",
    n: int = 20,
    sectors=None,
    markets=None,
    rebalance: str = "quarterly",
    with_benchmark: bool = True,
    market: str = "KR",
    callbacks_fn: Callable | None = None,
    benchmark_fn_factory: Callable | None = None,
    sp500_fn_factory: Callable | None = None,
    dates_fn: Callable | None = None,
    max_date_fn: Callable | None = None,
    backtest_fn: Callable | None = None,
) -> dict:
    """리밸런싱 백테스트를 실행해 NAV 시계열/성과지표를 반환한다.

    weights가 주어지면(예: optimize_weights의 출력을 $ref로 연결) 그 비중으로 buy&hold
    시뮬레이션, 아니면 criteria로 종목을 선정해 동일가중 리밸런싱한다(web/app.py의 기존
    /api/backtest 호출 관례와 동일 — build_callbacks/build_benchmark_fn/rebalance_dates 재사용).
    *_fn 인자는 테스트 주입용(기본은 실제 DB 기반 구현).

    market: "KR"(기본)이면 기존 한국 콜백/단일 벤치마크로 100% 기존 동작.
    "US"이면 data_access_us 콜백을 쓰고, 벤치마크를 이중 계산한다 — S&P500 실제지수(^GSPC,
    메인, 성과의 benchmark_return/excess_return/beta)와 동일가중 유니버스(보조, universe_*).
    무거운 NAV 시뮬레이션은 1번만 하고 유니버스 벤치마크 성과만 추가 계산해 병합한다.

    안전: rebalance 시점 수(window)가 상한을 넘거나 start_year가 너무 이르면, 무거운 연산
    (전종목 metrics_at 반복인 callbacks_fn/benchmark_fn_factory)을 시작하기 '전'에 거부한다
    (같은 머신에 실거래봇이 상주 — 자원 남용 방지는 사전 검사여야 사후 검사보다 의미 있음).
    """
    dates, callbacks_fn, benchmark_fn_factory, backtest_fn = _resolve_engine_inputs(
        conn, start_year, end_year, rebalance, market, dates_fn, max_date_fn,
        callbacks_fn, benchmark_fn_factory, backtest_fn,
    )

    # 여기부터 무거운 연산(전종목 metrics_at 반복) — 위 상한 검사를 반드시 통과한 뒤에만 도달
    metrics_fn, price_fn = callbacks_fn(conn)
    params = {"n": n, "criteria": criteria or [], "combine": combine, "sectors": sectors,
              "markets": markets, "rebalance": rebalance}

    if market == "US":
        return _run_us_backtest(
            dates, metrics_fn, price_fn, params, weights, with_benchmark, rebalance,
            benchmark_fn_factory, sp500_fn_factory, backtest_fn,
        )

    # KR(기본) — 기존 동작 100% 그대로(하위호환)
    bench_fn = benchmark_fn_factory(dates, metrics_fn, price_fn) if with_benchmark else None
    return backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=bench_fn, weights=weights)


def _run_us_backtest(dates, metrics_fn, price_fn, params, weights, with_benchmark, rebalance,
                     benchmark_fn_factory, sp500_fn_factory, backtest_fn) -> dict:
    """미국 백테스트: S&P500(메인)+동일가중 유니버스(보조) 이중 벤치마크로 실행/병합한다.

    무거운 NAV 시뮬레이션(backtest_fn)은 1번만 호출하고 메인 벤치마크(S&P500)를 엔진에 넣어
    성과를 계산한다. 유니버스 벤치마크는 그 결과 navs에 performance()만 다시 적용(순수·경량)해
    universe_* 키로 병합한다 — 전종목 순회 같은 무거운 연산은 중복 실행하지 않는다.
    """
    from .data_access_us import build_sp500_benchmark_fn as _build_sp500_benchmark_fn

    sp500_fn_factory = sp500_fn_factory or _build_sp500_benchmark_fn
    sp500_fn = sp500_fn_factory(dates) if with_benchmark else None
    universe_fn = benchmark_fn_factory(dates, metrics_fn, price_fn) if with_benchmark else None

    result = backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=sp500_fn, weights=weights)

    if with_benchmark and isinstance(result, dict) and "navs" in result:
        from .engine import PERIODS_PER_YEAR
        from .performance import performance as _performance

        ppy = PERIODS_PER_YEAR.get(rebalance, 4)
        out_dates = result.get("dates") or dates
        universe_levels = [universe_fn(d) for d in out_dates]
        perf = result.get("performance") or {}
        if len(universe_levels) == len(result["navs"]) and all(x is not None for x in universe_levels):
            perf_uni = _performance(result["navs"], ppy, benchmark=universe_levels)
            perf["universe_return"] = perf_uni.get("benchmark_return")
            perf["universe_excess_return"] = perf_uni.get("excess_return")
            perf["universe_beta"] = perf_uni.get("beta")
        result["performance"] = perf
        result["benchmark_sp500"] = result.get("benchmark")  # 메인 벤치마크(S&P500) 레벨 별칭
        result["benchmark_universe"] = universe_levels        # 보조 벤치마크(동일가중 유니버스) 레벨
    return result


# --------------------------------------------------------------------------
# 8. compute_ic_primitive — 팩터 정보계수(IC): 팩터값 순위 vs 다음구간 수익률 순위의 Spearman 상관
# --------------------------------------------------------------------------
_MIN_IC_SAMPLES = 3  # 순위상관 계산에 필요한 최소 종목 수(1~2개는 상관계수가 무의미)


def _spearman_ic(factor_vals, fwd_rets) -> float:
    """두 시계열의 순위상관(Spearman). 동순위는 평균순위(pandas.rank 기본 method='average')로 처리."""
    import numpy as np
    import pandas as pd

    fr = pd.Series(factor_vals).rank()
    rr = pd.Series(fwd_rets).rank()
    if fr.std() == 0 or rr.std() == 0:
        return 0.0  # 한쪽이 전부 동순위면 상관계수 정의 불가 → 0(무상관)으로 처리
    return float(np.corrcoef(fr.to_numpy(), rr.to_numpy())[0, 1])


def compute_ic_primitive(
    conn,
    start_year: int,
    end_year: int,
    field: str,
    rebalance: str = "quarterly",
    dates_fn: Callable | None = None,
    max_date_fn: Callable | None = None,
    callbacks_fn: Callable | None = None,
) -> dict:
    """팩터(field)의 정보계수(IC) 시계열과 요약통계를 계산한다.

    매 리밸런싱 시점마다 그 시점의 팩터값 순위와, 다음 리밸런싱 시점까지의 실현수익률
    순위 간 Spearman 순위상관(IC)을 구한다. run_backtest_primitive와 동일한 rebalance_dates/
    build_callbacks/안전상한(_MIN_START_YEAR/_MAX_REBALANCE_STEPS)을 재사용한다 — 이 프리미티브도
    전종목 metrics_at을 반복 호출하는 무거운 연산이라 같은 사전검사가 필요하다.

    반환: {dates, ic_series, mean_ic, ic_std, ir, hit_rate, n}
      · ic_series: 각 구간의 IC(리밸런싱 시점 순서)
      · mean_ic/ic_std: IC 평균/표준편차(표본표준편차, n<2면 0)
      · ir: mean_ic/ic_std (정보비율 — IC가 얼마나 일관됐는지)
      · hit_rate: 부호가 mean_ic와 같은 구간의 비율(예측 방향 적중률)
    표본이 _MIN_IC_SAMPLES 미만인 구간은 상관계수가 무의미해 건너뛴다. 유효 구간이 하나도
    없으면 ValueError.
    """
    from .data_access import build_callbacks as _build_callbacks

    dates = _resolve_rebalance_dates(conn, start_year, end_year, rebalance, dates_fn, max_date_fn)
    callbacks_fn = callbacks_fn or _build_callbacks

    # 여기부터 무거운 연산(전종목 metrics_at 반복) — 위 상한 검사를 반드시 통과한 뒤에만 도달
    metrics_fn, price_fn = callbacks_fn(conn)

    ic_series: list[float] = []
    used_dates: list[str] = []
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        factor_vals, fwd_rets = [], []
        for row in metrics_fn(t):
            v = row.get(field)
            if v is None:
                continue
            code = row.get("stock_code")
            p0, p1 = price_fn(t, code), price_fn(t1, code)
            if p0 and p1 and p0 > 0:
                factor_vals.append(v)
                fwd_rets.append(p1 / p0 - 1)
        if len(factor_vals) < _MIN_IC_SAMPLES:
            continue
        ic_series.append(_spearman_ic(factor_vals, fwd_rets))
        used_dates.append(t)

    if not ic_series:
        raise ValueError("IC를 계산할 수 있는 유효 구간이 없습니다(팩터/가격 데이터 부족).")

    import numpy as np

    arr = np.asarray(ic_series, dtype=float)
    mean_ic = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ir = mean_ic / ic_std if ic_std > 0 else 0.0
    positive_mean = mean_ic >= 0
    hit_rate = float(sum(1 for x in ic_series if (x >= 0) == positive_mean) / len(ic_series))

    return {
        "dates": used_dates,
        "ic_series": ic_series,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ir": ir,
        "hit_rate": hit_rate,
        "n": len(ic_series),
    }


# --------------------------------------------------------------------------
# 9. compute_technical_indicator — TA-Lib 기술지표(SMA/EMA/RSI/MACD/볼린저밴드)
# --------------------------------------------------------------------------
# RSI/볼린저밴드는 기간 고정(brainstorming-talib-reverse-backtest.md 확정), SMA/EMA/MACD는
# 파라미터로 기간 조정 가능. TA-Lib 자체의 수치 정확성은 검증 대상이 아니다(라이브러리 신뢰) —
# compute_fn을 DI 주입해 실제 talib 없이도 배선(파라미터 해석/필드명/기존 필드 보존)을
# 단위테스트할 수 있다(optimize_weights의 solve_fn과 동일한 관례).
_RSI_PERIOD = 14
_BBANDS_PERIOD = 20
_BBANDS_NBDEV = 2
_DEFAULT_MA_PERIOD = 20
_DEFAULT_MACD = {"fast": 12, "slow": 26, "signal": 9}
_SUPPORTED_INDICATORS = {"sma", "ema", "rsi", "macd", "bollinger"}
_INDICATOR_LOOKBACK_BUFFER_DAYS = 30  # 거래일→캘린더일 환산(주말·공휴일) 버퍼


def _resolve_indicator_spec(indicator: dict) -> dict:
    """지표 요청을 실제 계산에 쓸 고정 파라미터로 해석한다.

    RSI/볼린저밴드는 사용자가 period를 지정해도 무시하고 항상 고정값(14일 / 20일·표준편차2)을
    쓴다. SMA/EMA/MACD는 파라미터로 조정 가능하며 미지정시 기본값을 쓴다.
    """
    name = indicator.get("name")
    if name not in _SUPPORTED_INDICATORS:
        raise ValueError(f"지원하지 않는 지표: {name} ({sorted(_SUPPORTED_INDICATORS)}만 가능)")
    if name in ("sma", "ema"):
        return {"name": name, "params": {"period": indicator.get("period") or _DEFAULT_MA_PERIOD}}
    if name == "rsi":
        return {"name": "rsi", "params": {"period": _RSI_PERIOD}}
    if name == "macd":
        return {"name": "macd", "params": {
            "fast": indicator.get("fast", _DEFAULT_MACD["fast"]),
            "slow": indicator.get("slow", _DEFAULT_MACD["slow"]),
            "signal": indicator.get("signal", _DEFAULT_MACD["signal"]),
        }}
    return {"name": "bollinger", "params": {"period": _BBANDS_PERIOD, "nbdev": _BBANDS_NBDEV}}


def _lookback_days_for(resolved_indicators: list[dict]) -> int:
    """지표 계산에 필요한 최대 기간을 캘린더일 lookback으로 넉넉히 환산한다."""
    periods = []
    for ind in resolved_indicators:
        p = ind["params"]
        if ind["name"] == "macd":
            periods.append(p["slow"] + p["signal"])
        else:
            periods.append(p["period"])
    max_period = max(periods) if periods else _DEFAULT_MA_PERIOD
    return max_period * 2 + _INDICATOR_LOOKBACK_BUFFER_DAYS


def _talib_compute(name: str, closes, params: dict) -> dict:
    """실제 TA-Lib 계산(지연 import — fama_french/riskfolio와 동일 관례). compute_fn 기본값."""
    import numpy as np
    import talib

    def _f(val) -> float | None:
        return None if np.isnan(val) else float(val)

    arr = np.asarray(list(closes), dtype=float)
    if name == "sma":
        period = params["period"]
        field = f"sma_{period}"
        if len(arr) < period:
            return {field: None}
        return {field: _f(talib.SMA(arr, timeperiod=period)[-1])}
    if name == "ema":
        period = params["period"]
        field = f"ema_{period}"
        if len(arr) < period:
            return {field: None}
        return {field: _f(talib.EMA(arr, timeperiod=period)[-1])}
    if name == "rsi":
        period = params["period"]
        if len(arr) < period + 1:
            return {"rsi_14": None}
        return {"rsi_14": _f(talib.RSI(arr, timeperiod=period)[-1])}
    if name == "macd":
        fast, slow, signal = params["fast"], params["slow"], params["signal"]
        if len(arr) < slow + signal:
            return {"macd": None, "macd_signal": None, "macd_hist": None}
        macd, macd_signal, macd_hist = talib.MACD(
            arr, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return {
            "macd": _f(macd[-1]), "macd_signal": _f(macd_signal[-1]), "macd_hist": _f(macd_hist[-1]),
        }
    # bollinger
    period, nbdev = params["period"], params["nbdev"]
    if len(arr) < period:
        return {"bollinger_upper": None, "bollinger_middle": None, "bollinger_lower": None}
    upper, middle, lower = talib.BBANDS(arr, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
    return {
        "bollinger_upper": _f(upper[-1]), "bollinger_middle": _f(middle[-1]), "bollinger_lower": _f(lower[-1]),
    }


# 신호 백테스트(run_signal_backtest)가 참조하는 SERIES(kind=indicator)의 이름들.
# compute_technical_indicator는 마지막 값([-1])만 쓰지만, 신호 백테스트는 매일의 교차를
# 판정하려면 전 구간 시계열이 필요하다 → 아래 compute_indicator_series로 전체 배열을 돌려준다.
_SIGNAL_INDICATOR_NAMES = {
    "sma", "ema", "rsi", "macd", "macd_signal", "bollinger_upper", "bollinger_lower",
}


def compute_indicator_series(name: str, closes, period: int | None = None):
    """지표(name)의 **전 구간 시계열**(numpy array)을 반환한다(신호 백테스트용).

    _talib_compute가 마지막 값만 [-1]로 반환하는 것과 달리, 골든/데드크로스처럼 "매일의
    교차"를 봐야 하는 신호 백테스트를 위해 배열 전체를 그대로 돌려준다(TA-Lib 자체의 수치
    정확성은 검증 대상이 아니다 — 라이브러리 신뢰. run_signal_backtest가 DI로 주입 가능).
    period 미지정 시 각 지표의 기본기간(_talib_compute와 동일 상수)을 쓴다. macd/macd_signal은
    표준 12/26/9, 볼린저는 20·표준편차2 고정(compute_technical_indicator와 동일 규약).
    """
    import numpy as np
    import talib

    arr = np.asarray(list(closes), dtype=float)
    if name == "sma":
        return talib.SMA(arr, timeperiod=period or _DEFAULT_MA_PERIOD)
    if name == "ema":
        return talib.EMA(arr, timeperiod=period or _DEFAULT_MA_PERIOD)
    if name == "rsi":
        return talib.RSI(arr, timeperiod=period or _RSI_PERIOD)
    if name in ("macd", "macd_signal"):
        macd, macd_signal, _hist = talib.MACD(
            arr, fastperiod=_DEFAULT_MACD["fast"], slowperiod=_DEFAULT_MACD["slow"],
            signalperiod=_DEFAULT_MACD["signal"],
        )
        return macd if name == "macd" else macd_signal
    if name in ("bollinger_upper", "bollinger_lower"):
        upper, _middle, lower = talib.BBANDS(
            arr, timeperiod=period or _BBANDS_PERIOD, nbdevup=_BBANDS_NBDEV, nbdevdn=_BBANDS_NBDEV,
        )
        return upper if name == "bollinger_upper" else lower
    raise ValueError(f"지원하지 않는 지표: {name} ({sorted(_SIGNAL_INDICATOR_NAMES)}만 가능)")


def compute_technical_indicator(
    conn,
    rows: list[dict],
    asof: str,
    indicators: list[dict],
    history_fn: Callable | None = None,
    compute_fn: Callable | None = None,
) -> list[dict]:
    """rows(예: get_cross_section 결과)의 각 종목에 기술지표 필드를 추가해 반환한다.

    반환은 rows와 동일하게 stock_code를 키로 갖는 list[dict]이며, 원 필드는 그대로 유지한 채
    지표 필드만 추가된다 — combine()의 rows 인자로 그대로 넘겨 기존 필드(PER/PBR 등)와 함께
    criteria에서 참조할 수 있다. 가격 시계열은 price_history_batch로 전종목 한 번에(배치)
    조회한다(개별 종목마다 쿼리하지 않음).
    """
    history_fn = history_fn or price_history_batch
    compute_fn = compute_fn or _talib_compute

    resolved = [_resolve_indicator_spec(ind) for ind in indicators]
    codes = [r["stock_code"] for r in rows]
    lookback_days = _lookback_days_for(resolved)
    history = history_fn(conn, codes, asof=asof, lookback_days=lookback_days)

    out = []
    for r in rows:
        code = r["stock_code"]
        closes = [p["close"] for p in history.get(code, [])]
        nr = dict(r)
        for ind in resolved:
            nr.update(compute_fn(ind["name"], closes, ind["params"]))
        out.append(nr)
    return out


# --------------------------------------------------------------------------
# 10. search_strategy — 성과지표 제약을 만족하는 종목선정 전략 탐색(역백테스트)
# --------------------------------------------------------------------------
# 탐색 대상은 candidates(criteria 조합)뿐이다(brainstorming-talib-reverse-backtest.md 확정).
# n/rebalance/sectors/markets/market/start_year/end_year는 호출 시 고정값으로 모든 후보에
# 동일 적용된다(Non-Goal: 이 파라미터들을 탐색범위로 넓히지 않음). run_backtest_primitive와
# 동일한 "무거운 연산 전 사전상한 검사" 관례를 따른다(같은 머신에 실거래봇 상주).
_MAX_SEARCH_CANDIDATES = 20

_CONSTRAINT_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _satisfies_constraints(performance: dict | None, constraints: list[dict]) -> bool:
    """performance가 constraints(전부 AND)를 만족하는지 판정한다. 지표 없으면 탈락."""
    if performance is None:
        return False
    for c in constraints:
        metric, op, value = c["metric"], c["op"], c["value"]
        if op not in _CONSTRAINT_OPS:
            raise ValueError(f"지원하지 않는 연산자: {op} ({sorted(_CONSTRAINT_OPS)}만 가능)")
        actual = performance.get(metric)
        if actual is None or not _CONSTRAINT_OPS[op](actual, value):
            return False
    return True


def search_strategy(
    conn,
    candidates: list[list[dict]],
    start_year: int,
    end_year: int,
    n: int = 20,
    combine: str = "zscore",
    sectors=None,
    markets=None,
    rebalance: str = "quarterly",
    market: str = "KR",
    with_benchmark: bool = False,
    constraints: list[dict] | None = None,
    rank_by: str = "sharpe",
    callbacks_fn: Callable | None = None,
    benchmark_fn_factory: Callable | None = None,
    dates_fn: Callable | None = None,
    max_date_fn: Callable | None = None,
    backtest_fn: Callable | None = None,
) -> list[dict]:
    """후보 criteria 조합마다 백테스트를 실행해 {criteria, performance, holdings}를 반환한다.

    candidates 길이가 상한(20)을 넘으면 실행 전에 거부한다(자원 남용 방지). 종목별 지표
    계산 콜백(callbacks_fn)은 후보 반복 전체에서 **1회만** 생성해 공유한다 — run_backtest_
    primitive를 후보마다 그대로 재호출하면 콜백이 매번 새로 만들어져 캐시가 무효화되고
    최대 20배 느려지므로, 여기서는 engine.run_backtest를 직접 호출해 콜백을 재사용한다.
    *_fn 인자는 테스트 주입용(run_backtest_primitive와 동일한 DI 관례).
    """
    if not candidates:
        raise ValueError("candidates가 비어 있습니다(탐색할 후보 전략이 없습니다)")
    if len(candidates) > _MAX_SEARCH_CANDIDATES:
        raise ValueError(
            f"후보 전략 {len(candidates)}개 > 상한 {_MAX_SEARCH_CANDIDATES}개(자원 남용 방지)"
        )
    # 제약조건 연산자는 무거운 연산(콜백 생성/run_backtest 반복) '전'에 미리 검증한다 —
    # 오타 연산자 하나로 후보 20개를 전부 돌린 뒤에야 에러가 나면 안 된다(사전검사 원칙).
    for c in constraints or []:
        if c.get("op") not in _CONSTRAINT_OPS:
            raise ValueError(
                f"지원하지 않는 연산자: {c.get('op')} ({sorted(_CONSTRAINT_OPS)}만 가능)"
            )

    dates, callbacks_fn, benchmark_fn_factory, backtest_fn = _resolve_engine_inputs(
        conn, start_year, end_year, rebalance, market, dates_fn, max_date_fn,
        callbacks_fn, benchmark_fn_factory, backtest_fn,
    )

    # 캐시 공유 핵심: 콜백을 후보 반복 진입 '전'에 단 한 번만 생성해 모든 후보에 재사용
    metrics_fn, price_fn = callbacks_fn(conn)
    bench_fn = benchmark_fn_factory(dates, metrics_fn, price_fn) if with_benchmark else None

    results = []
    for criteria in candidates:
        params = {"n": n, "criteria": criteria, "combine": combine, "sectors": sectors,
                  "markets": markets, "rebalance": rebalance}
        res = backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=bench_fn, weights=None)
        results.append({
            "criteria": criteria,
            "performance": res.get("performance"),
            "holdings": res.get("holdings"),
        })

    if constraints:
        results = [r for r in results if _satisfies_constraints(r["performance"], constraints)]
    results.sort(
        key=lambda r: (r["performance"] or {}).get(rank_by, float("-inf")),
        reverse=True,
    )
    return results


# --------------------------------------------------------------------------
# 11. QVM(Quality-Value-Momentum) 멀티팩터 — 저수준 5종 + 조립 compute_qvm_scores
# --------------------------------------------------------------------------
# 사용자 확정 파이프라인: invert(가치 역수) → winsorize_pct(1%/99%) → 섹터 z-score(<5표본
# 전체폴백) → 카테고리 합성(등가중, 결측 제외) → 결측필터(raw 7개 중 3개 이상 결측 제외)
# → 2차 z-score → 최종점수 (z_Q+z_V+z_M)/3. 저수준 함수는 모두 순수 함수(원본 rows 불변,
# 새 dict 반환)이며 기존 winsorize(IQR)/neutralize는 절대 건드리지 않고 별도로 추가한다.
_VALUE_INVERT_NAMES = {"per": "ep", "pbr": "bp", "psr": "sp"}


def invert_field(rows: list[dict], field: str, out_field: str) -> list[dict]:
    """가치 팩터 역수 변환: row[field]가 None이거나 <=0이면 out_field=None, 아니면 1/field.

    E/P=1/PER, B/P=1/PBR, S/P=1/PSR처럼 "낮을수록 좋은" 가치지표를 "높을수록 좋은" 방향으로
    뒤집는다. PER은 SoT상 이미 적자면 None이므로 그 None이 자연히 전파된다. 원본 field는 보존.
    """
    out = []
    for r in rows:
        v = r.get(field)
        nr = dict(r)
        nr[out_field] = (1.0 / v) if (v is not None and v > 0) else None
        out.append(nr)
    return out


def winsorize_pct(rows: list[dict], field: str, lower_pct: float = 0.01, upper_pct: float = 0.99) -> list[dict]:
    """유효값의 lower_pct/upper_pct 분위수(numpy.percentile) 밖을 경계로 눌러 붙인다(행 삭제 없음).

    기존 winsorize(IQR·k 방식)와 별개의 새 함수 — QVM 1%/99% 윈저라이즈 전용(기존 winsorize는
    다른 용도로 계속 쓰이므로 절대 수정하지 않는다). None은 그대로 유지하고 분위 계산에서
    제외한다. '{field}_winsorized' 신규 필드를 추가한다.
    """
    import numpy as np

    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return [dict(r, **{f"{field}_winsorized": r.get(field)}) for r in rows]
    lo, hi = np.percentile(vals, [lower_pct * 100.0, upper_pct * 100.0])
    out = []
    for r in rows:
        v = r.get(field)
        nr = dict(r)
        nr[f"{field}_winsorized"] = None if v is None else float(min(max(v, lo), hi))
        out.append(nr)
    return out


def sector_zscore_with_fallback(
    rows: list[dict], field: str, sector_field: str = "sector", min_sector_n: int = 5
) -> list[dict]:
    """섹터 내 z-score. 섹터 표본이 min_sector_n 미만이면 전체 유니버스 z-score로 폴백한다.

    neutralize(method='zscore')에는 표본부족 폴백이 없어(1표본이면 None) 별도로 추가한다
    (neutralize는 다른 곳에서 쓰여 하위호환을 깨면 안 되므로 건드리지 않는다). 섹터 표본수가
    충분하면 그 섹터 평균/표준편차로, 부족하면 전체 유니버스 평균/표준편차로 각 종목 값을
    표준화한다. 표준편차 0(전부 동일값)이면 0으로 나눗셈을 피해 0.0으로 둔다. None은 None 유지.
    '{field}_zscore' 신규 필드를 추가한다.
    """
    import numpy as np

    by_sector: dict = {}
    universe_vals: list = []
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        universe_vals.append(v)
        by_sector.setdefault(r.get(sector_field), []).append(v)

    def _stats(vals):
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(arr.std())

    uni_mean, uni_std = _stats(universe_vals) if universe_vals else (0.0, 0.0)
    sector_stats = {s: _stats(vs) for s, vs in by_sector.items() if len(vs) >= min_sector_n}

    out = []
    for r in rows:
        v = r.get(field)
        nr = dict(r)
        if v is None:
            nr[f"{field}_zscore"] = None
        else:
            mean, std = sector_stats.get(r.get(sector_field), (uni_mean, uni_std))
            nr[f"{field}_zscore"] = ((v - mean) / std) if std else 0.0
        out.append(nr)
    return out


def composite_score(
    rows: list[dict], fields: list[str], out_field: str, weights: list[float] | None = None
) -> list[dict]:
    """fields의 z-score 값들을 가중평균해 out_field에 담는다(결측 제외 재정규화).

    각 row에서 값이 있는(None 아닌) 필드들만, 그 필드의 가중치 비율로 재정규화한 가중평균을
    낸다("존재하는 필드들의 가중치 합으로 나눈 가중평균"). weights 미지정시 등가중. 나열된
    필드가 전부 결측이면 out_field=None. 카테고리 합성(Q/V/M)과 최종 카테고리 결합에 공용으로 쓴다.
    """
    ws = list(weights) if weights is not None else [1.0] * len(fields)
    if len(ws) != len(fields):
        raise ValueError("weights 길이가 fields 길이와 다릅니다")
    out = []
    for r in rows:
        num = 0.0
        den = 0.0
        for f, w in zip(fields, ws):
            v = r.get(f)
            if v is not None:
                num += v * w
                den += w
        nr = dict(r)
        nr[out_field] = (num / den) if den else None
        out.append(nr)
    return out


def drop_missing_factors(rows: list[dict], fields: list[str], max_missing: int) -> list[dict]:
    """fields 중 None인 개수가 max_missing을 초과하는 row를 제외하고 나머지를 반환한다."""
    return [r for r in rows if sum(1 for f in fields if r.get(f) is None) <= max_missing]


def _universe_zscore(rows: list[dict], in_field: str, out_field: str) -> list[dict]:
    """유니버스 전체 평균/표준편차로 in_field를 z-score 표준화해 out_field에 담는다(2차 표준화).

    combine()/zscore()는 '선정+정렬'까지 하는 함수라 '전체 row에 값만 얹기' 용도로는 의미가
    어긋나므로, 여기서 numpy로 직접 z-score만 계산한다. None은 None 유지, 표준편차 0이면 0.0.
    """
    import numpy as np

    vals = [r[in_field] for r in rows if r.get(in_field) is not None]
    if not vals:
        return [dict(r, **{out_field: None}) for r in rows]
    arr = np.asarray(vals, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    out = []
    for r in rows:
        v = r.get(in_field)
        nr = dict(r)
        nr[out_field] = None if v is None else (((v - mean) / std) if std else 0.0)
        out.append(nr)
    return out


def _normalize_category_weights(category_weights) -> list[float]:
    """category_weights를 [quality, value, momentum] 순서의 숫자 리스트로 정규화한다.

    LLM이 파이프라인 JSON을 생성할 때 리스트 대신 {"quality":..,"value":..,"momentum":..}
    같은 딕셔너리로 주는 사례가 실서버에서 재현됐다 — list(dict)는 값이 아니라 키를
    반환하므로 그 문자열과 z-score를 곱하려다 "can't multiply sequence by non-int of
    type 'float'"로 크래시했다. 리스트/튜플은 그대로 통과시키고, 딕셔너리는 quality/
    value/momentum 키(대소문자 무관)로 순서를 맞춰 추출한다. 필요한 키가 없으면 계산을
    계속 진행하는 대신 원인을 바로 알 수 있는 ValueError로 즉시 실패시킨다.
    """
    if isinstance(category_weights, dict):
        lowered = {str(k).lower(): v for k, v in category_weights.items()}
        missing = [k for k in ("quality", "value", "momentum") if k not in lowered]
        if missing:
            raise ValueError(
                "category_weights 딕셔너리에는 quality/value/momentum 키가 모두 "
                f"있어야 합니다(누락: {missing}): {category_weights}"
            )
        return [float(lowered["quality"]), float(lowered["value"]), float(lowered["momentum"])]
    return [float(w) for w in category_weights]


def compute_qvm_scores(
    rows: list[dict],
    quality_fields=("roe", "gp_a", "cfo_ratio"),
    value_source_fields=("per", "pbr", "psr"),
    momentum_field: str = "momentum_12_1",
    sector_field: str = "sector",
    min_sector_n: int = 5,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
    max_missing: int = 3,
    category_weights=(1 / 3, 1 / 3, 1 / 3),
) -> list[dict]:
    """사용자 확정 QVM 파이프라인을 그대로 조립해 각 종목의 최종 qvm_score를 계산한다.

    순서: ① 가치 역수(per→ep 등) ② 각 raw 팩터 1%/99% winsorize ③ 섹터 z-score(<5표본
    전체폴백) ④ 카테고리 합성(Q/V/M 각 등가중, 결측 제외) ⑤ 결측필터(raw 7개 중 결측이
    max_missing개 이상이면 제외 — 사용자 rule6: 3개 이상 결측 제외) ⑥ 카테고리 합성점수 2차
    z-score ⑦ 최종 = 카테고리 z-score 가중평균(category_weights, 결측 카테고리는 제외 재정규화).

    반환 row에는 투명성을 위해 원본 필드 + 역수(ep/bp/sp) + 각 팩터의 _winsorized/_zscore
    중간값 + quality_z/value_z/momentum_z(2차 표준화 카테고리 점수) + qvm_score(최종)를 모두 담는다.
    각 행에는 결측필터로 제외된 종목 수를 담은 내부 마커 '_qvm_excluded_count'도 동일하게
    붙는다(요약 정보 노출용 — src/agents/domain_backtest.py의 qvm_summary가 이 마커로
    excluded_count를 복원한다. combine() 등 후속 선정 단계도 원본 dict 참조를 그대로
    넘기므로 top_n으로 걸러진 뒤에도 이 마커는 살아남는다).

    max_missing은 '결측이 이 개수 이상이면 제외'(사용자 rule6 표현)로 해석한다 — 내부적으로는
    drop_missing_factors(초과 시 제외)에 max_missing-1을 넘긴다(3 이상 제외 = 2 초과 제외).
    """
    quality_fields = list(quality_fields)
    value_source_fields = list(value_source_fields)

    # ① 가치 역수 변환(per→ep, pbr→bp, psr→sp)
    value_fields = [_VALUE_INVERT_NAMES.get(f, f"{f}_inv") for f in value_source_fields]
    for src, inv in zip(value_source_fields, value_fields):
        rows = invert_field(rows, src, inv)

    raw_factors = quality_fields + value_fields + [momentum_field]

    # ② 각 raw 팩터 1%/99% winsorize → ③ 섹터 z-score(<5표본 전체폴백)
    z_fields = []
    for f in raw_factors:
        rows = winsorize_pct(rows, f, winsorize_lower, winsorize_upper)
        wf = f"{f}_winsorized"
        rows = sector_zscore_with_fallback(rows, wf, sector_field, min_sector_n)
        z_fields.append(f"{wf}_zscore")

    nq, nv = len(quality_fields), len(value_fields)
    quality_z_fields = z_fields[:nq]
    value_z_fields = z_fields[nq:nq + nv]
    momentum_z_fields = z_fields[nq + nv:]

    # ④ 카테고리 합성(내부 팩터 z-score 등가중, 결측 제외)
    rows = composite_score(rows, quality_z_fields, "quality_composite")
    rows = composite_score(rows, value_z_fields, "value_composite")
    rows = composite_score(rows, momentum_z_fields, "momentum_composite")

    # ⑤ 결측필터(raw 팩터 중 결측이 max_missing개 이상이면 제외)
    n_before_missing_filter = len(rows)
    rows = drop_missing_factors(rows, raw_factors, max_missing - 1)
    excluded_count = n_before_missing_filter - len(rows)

    # ⑥ 카테고리 합성점수 2차 z-score(유니버스 전체)
    rows = _universe_zscore(rows, "quality_composite", "quality_z")
    rows = _universe_zscore(rows, "value_composite", "value_z")
    rows = _universe_zscore(rows, "momentum_composite", "momentum_z")

    # ⑦ 최종 점수 = 카테고리 z-score 가중평균(결측 카테고리 제외 재정규화)
    rows = composite_score(
        rows, ["quality_z", "value_z", "momentum_z"], "qvm_score",
        weights=_normalize_category_weights(category_weights),
    )
    # 결측필터 제외 종목 수를 각 행에 마커로 남긴다(반환 '형태'는 여전히 list[dict] 그대로).
    for r in rows:
        r["_qvm_excluded_count"] = excluded_count
    return rows


def get_cross_section_qvm(
    conn, asof, markets=None, metrics_fn: Callable | None = None, momentum_fn: Callable | None = None
) -> list[dict]:
    """QVM용 크로스섹션: get_cross_section 결과에 12-1 모멘텀(momentum_12_1)을 배치로 병합한다.

    momentum_12_1_batch로 전종목 모멘텀을 상수 회수 배치 SQL로 계산해 각 row에 얹는다
    (종목별 반복 SQL 금지 — kr 스크리닝 지연 문제를 악화시키지 않는다). metrics_fn/momentum_fn은
    테스트 주입용(기본은 실제 DB 기반). compute_qvm_scores의 입력으로 그대로 넘긴다.
    """
    from .data_access import momentum_12_1_batch as _momentum_batch

    rows = get_cross_section(conn, asof, markets=markets, metrics_fn=metrics_fn)
    momentum_fn = momentum_fn or _momentum_batch
    codes = [r["stock_code"] for r in rows]
    mom = momentum_fn(conn, codes, asof)
    out = []
    for r in rows:
        nr = dict(r)
        nr["momentum_12_1"] = mom.get(r["stock_code"])
        out.append(nr)
    return out


def run_qvm_backtest(
    conn,
    start_year: int,
    end_year: int,
    quality_fields=None,
    value_source_fields=None,
    momentum_field: str = "momentum_12_1",
    min_sector_n: int = 5,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
    max_missing: int = 3,
    category_weights=(1 / 3, 1 / 3, 1 / 3),
    n: int = 20,
    rebalance: str = "quarterly",
    markets=None,
    with_benchmark: bool = True,
    callbacks_fn: Callable | None = None,
    benchmark_fn_factory: Callable | None = None,
    dates_fn: Callable | None = None,
    max_date_fn: Callable | None = None,
    backtest_fn: Callable | None = None,
    cross_section_fn: Callable | None = None,
    compute_fn: Callable | None = None,
) -> dict:
    """QVM 전략 리밸런싱 백테스트. 엔진/선정/성과 코드는 건드리지 않고 metrics_fn만 감싼다.

    각 리밸런싱 시점 t마다 get_cross_section_qvm(모멘텀 포함) → compute_qvm_scores로 qvm_score를
    미리 계산해 row에 얹은 뒤, 그 점수를 criteria=[{"key":"qvm_score","direction":"high"}] 단일
    기준으로 run_backtest_primitive(=engine.run_backtest)에 넘긴다. 반환은 기존 백테스트와 동일
    스키마({dates, navs, benchmark, performance, holdings}). *_fn 인자는 테스트 주입용(DI).
    """
    from .data_access import build_callbacks as _build_callbacks

    base_callbacks_fn = callbacks_fn or _build_callbacks
    _base_metrics_fn, price_fn = base_callbacks_fn(conn)
    cross_section_fn = cross_section_fn or get_cross_section_qvm
    compute_fn = compute_fn or compute_qvm_scores

    qkwargs = {
        "momentum_field": momentum_field, "min_sector_n": min_sector_n,
        "winsorize_lower": winsorize_lower, "winsorize_upper": winsorize_upper,
        "max_missing": max_missing, "category_weights": category_weights,
    }
    if quality_fields is not None:
        qkwargs["quality_fields"] = quality_fields
    if value_source_fields is not None:
        qkwargs["value_source_fields"] = value_source_fields

    # 시점별 캐시: 엔진과 벤치마크가 같은 시점 metrics_fn을 각각 호출하므로, QVM 크로스섹션
    # 재계산(전종목 metrics_at + 모멘텀)을 시점당 1번으로 줄인다(build_callbacks의 캐시 관례와 동일).
    _qvm_cache: dict = {}

    def qvm_metrics_fn(t):
        if t not in _qvm_cache:
            rows = cross_section_fn(conn, t, markets=markets)
            _qvm_cache[t] = compute_fn(rows, **qkwargs)
        return _qvm_cache[t]

    return run_backtest_primitive(
        conn, start_year, end_year,
        criteria=[{"key": "qvm_score", "direction": "high", "weight": 1.0}],
        combine="zscore", n=n, sectors=None, markets=markets, rebalance=rebalance,
        with_benchmark=with_benchmark,
        callbacks_fn=lambda _conn: (qvm_metrics_fn, price_fn),
        benchmark_fn_factory=benchmark_fn_factory, dates_fn=dates_fn,
        max_date_fn=max_date_fn, backtest_fn=backtest_fn,
    )
