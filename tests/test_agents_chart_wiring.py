"""총괄 차트 배선(src/agents/supervisor.py) 단위 테스트 (TDD — 로직 파트).

검증 대상:
- wants_chart(question): 명시적 차트 키워드가 있을 때만 True(오탐 방지). 재시도 피드백이
  덧붙기 전 '원본 question'으로 판단해야 한다.
- _extract_chart_data(domain_results, conn): 그릴 시계열 데이터를 우선순위
  (backtest > kr/us 가격 > macro)로 하나만 고른다. 그릴 게 없으면 None.
- answer_with_verification: wants_chart가 True이고 그릴 데이터가 있을 때만 성공 응답에
  chart_base64/chart_title를 덧붙인다. 키워드 없으면/불확실 응답이면 차트 필드 없음.
"""
from __future__ import annotations

import base64

from src.agents.supervisor import answer_with_verification, wants_chart

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── wants_chart — 결정론적 키워드 감지(오탐 방지) ────────────────────────────

def test_wants_chart_true_for_graph_keyword():
    assert wants_chart("삼성전자 최근 1년 주가 그래프 그려줘") is True


def test_wants_chart_true_for_chart_keyword():
    assert wants_chart("골든크로스 전략 백테스트 차트로 보여줘") is True


def test_wants_chart_true_for_english_plot():
    assert wants_chart("plot the nav curve") is True
    assert wants_chart("show me a chart") is True


def test_wants_chart_true_for_visualize_and_trend():
    assert wants_chart("스프레드 시각화 해줘") is True
    assert wants_chart("금리차 추이 보여줘") is True


def test_wants_chart_false_for_plain_question():
    """일반 질문(키워드 없음)은 False — 불필요한 차트 생성을 막는다(오탐 방지)."""
    assert wants_chart("삼성전자 PER 알려줘") is False
    assert wants_chart("지금 매크로 신호 어때?") is False
    assert wants_chart("PER 낮은 5개 회사") is False


def test_wants_chart_empty_is_false():
    assert wants_chart("") is False
    assert wants_chart(None) is False


# ── _extract_chart_data — 우선순위/데이터 선택 ────────────────────────────────

def test_extract_chart_data_backtest_uses_dates_navs_and_benchmark():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {
            "blocked": False,
            "result": {
                "dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "navs": [1.0, 1.1, 1.2],
                "benchmark": [1.0, 1.05, 1.08],
            },
        }
    }
    out = sup._extract_chart_data(domain_results, conn=None)
    assert out is not None
    dates, series, title = out
    assert dates == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert series["전략"] == [1.0, 1.1, 1.2]
    assert series["벤치마크"] == [1.0, 1.05, 1.08]
    assert "백테스트" in title


def test_extract_chart_data_backtest_without_benchmark():
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert list(series) == ["전략"]
    assert series["전략"] == [1.0, 1.1]


def test_extract_chart_data_prefers_backtest_over_kr(monkeypatch):
    """여러 조건이 동시 해당하면 backtest가 우선(질문당 차트 1개)."""
    import src.agents.supervisor as sup

    # kr 히스토리 함수가 호출되면 실패로 표시 — backtest 우선이면 호출되면 안 된다.
    monkeypatch.setattr(sup, "get_price_history_kr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("kr 우선순위 위반")))
    domain_results = {
        "backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}},
        "kr": {"stock_code": "005930"},
    }
    _, _, title = sup._extract_chart_data(domain_results, conn=None)
    assert "백테스트" in title


def test_extract_chart_data_kr_price_history(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "get_price_history_kr",
        lambda conn, code, **k: [
            {"stock_code": "005930", "date": "2026-01-01", "close": 70000.0},
            {"stock_code": "005930", "date": "2026-01-02", "close": 71000.0},
        ],
    )
    domain_results = {"kr": {"stock_code": "005930", "price": [{"close": 71000.0}]}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert dates == ["2026-01-01", "2026-01-02"]
    assert list(series.values())[0] == [70000.0, 71000.0]
    assert "005930" in title


def test_extract_chart_data_macro_spread_only(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "get_macro_history",
        lambda conn, **k: [
            {"as_of": "2026-01-01", "spread": 0.6},
            {"as_of": "2026-01-02", "spread": 0.4},
        ],
    )
    domain_results = {"macro": {"available": True, "spread": {"value": 0.4}}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert dates == ["2026-01-01", "2026-01-02"]
    assert list(series.values())[0] == [0.6, 0.4]
    assert "금리차" in title


def test_extract_chart_data_returns_none_for_screening_list():
    """스크리닝 결과(여러 종목 리스트, 단일 시계열 아님)는 조용히 None — 에러 아님."""
    import src.agents.supervisor as sup

    domain_results = {"kr": {"intent": "screening", "result": [{"name": "A"}, {"name": "B"}]}}
    assert sup._extract_chart_data(domain_results, conn=None) is None


def test_extract_chart_data_returns_none_when_kr_history_empty(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(sup, "get_price_history_kr", lambda *a, **k: [])
    domain_results = {"kr": {"stock_code": "005930"}}
    assert sup._extract_chart_data(domain_results, conn=None) is None


def test_extract_chart_data_returns_none_for_empty_results():
    import src.agents.supervisor as sup

    assert sup._extract_chart_data({}, conn=None) is None


# ── _extract_scatter_data — 백테스트 결과가 scatter_data(dict)면 산점도로 인식 ──────────

def test_extract_scatter_data_recognizes_backtest_scatter_result():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {
            "blocked": False,
            "result": {
                "x": [5.0, 8.0, 3.0], "y": [12.0, 20.0, 9.0],
                "labels": ["가", "나", "다"],
                "x_field": "earnings_yield", "y_field": "roc",
            },
        }
    }
    out = sup._extract_scatter_data(domain_results)
    assert out is not None
    x, y, labels, x_label, y_label, title = out
    assert x == [5.0, 8.0, 3.0]
    assert y == [12.0, 20.0, 9.0]
    assert labels == ["가", "나", "다"]
    assert x_label == "earnings_yield" and y_label == "roc"
    assert "산점도" in title and "earnings_yield" in title


def test_extract_scatter_data_none_for_line_backtest_result():
    """기존 시계열(dates/navs) 백테스트 결과는 산점도가 아니다 → None(라인차트 경로로 감)."""
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}
    assert sup._extract_scatter_data(domain_results) is None


def test_extract_scatter_data_none_when_blocked():
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"blocked": True, "result": {
        "x": [1.0], "y": [2.0], "labels": ["가"], "x_field": "a", "y_field": "b"}}}
    assert sup._extract_scatter_data(domain_results) is None


def test_build_chart_renders_scatter_png_when_backtest_is_scatter():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"result": {
            "x": [5.0, 8.0, 3.0], "y": [12.0, 20.0, 9.0],
            "labels": ["가", "나", "다"],
            "x_field": "earnings_yield", "y_field": "roc",
        }}
    }
    out = sup._build_chart(domain_results, conn=None)
    assert out is not None
    chart_base64, title = out
    assert base64.b64decode(chart_base64)[:8] == _PNG_MAGIC
    assert "산점도" in title


def test_build_chart_scatter_takes_priority_over_line(monkeypatch):
    """산점도 데이터가 있으면 라인차트보다 우선(질문당 차트 1개)."""
    import src.agents.supervisor as sup

    # 라인 경로가 호출되면 실패 표시 — 산점도 우선이면 호출되면 안 된다.
    monkeypatch.setattr(
        sup, "_extract_chart_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("산점도 우선순위 위반")),
    )
    domain_results = {"backtest": {"result": {
        "x": [1.0, 2.0], "y": [3.0, 4.0], "labels": ["가", "나"],
        "x_field": "earnings_yield", "y_field": "roc"}}}
    out = sup._build_chart(domain_results, conn=None)
    assert out is not None
    assert "산점도" in out[1]


def test_answer_with_verification_adds_scatter_chart_when_requested():
    """'산점도 그려줘' + 백테스트 scatter 결과 → 성공 응답에 chart_base64(PNG) 배선."""
    from src.agents.supervisor import answer_with_verification

    bt = {"result": {
        "x": [5.0, 8.0, 3.0], "y": [12.0, 20.0, 9.0], "labels": ["가", "나", "다"],
        "x_field": "earnings_yield", "y_field": "roc"}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "이익수익률과 투하자본수익률 산점도 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64")
    assert base64.b64decode(res["chart_base64"])[:8] == _PNG_MAGIC
    assert "산점도" in res["chart_title"]


# ── answer_with_verification — 성공 경로에 chart_base64/chart_title 배선 ──────

def _valid_verify(question, domain_results, llm_fn):
    return {"valid": True, "reason": "일치"}


def test_answer_with_verification_adds_chart_when_requested_backtest():
    bt = {"result": {"dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
                     "navs": [1.0, 1.1, 1.2], "benchmark": [1.0, 1.05, 1.08]}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "골든크로스 전략 백테스트 그래프로 보여줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64")
    raw = base64.b64decode(res["chart_base64"])
    assert raw[:8] == _PNG_MAGIC          # 실제 PNG인지까지 확인
    assert res.get("chart_title") and "백테스트" in res["chart_title"]


def test_answer_with_verification_no_chart_without_keyword():
    """차트 키워드가 없으면 데이터가 있어도 차트를 만들지 않는다(오탐 방지)."""
    bt = {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "이 전략 수익률 알려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res.get("chart_base64") is None


def test_answer_with_verification_uses_original_question_for_wants_chart():
    """재시도 피드백이 붙은 dispatch_question이 아니라 원본 question으로 차트 여부 판단."""
    bt = {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}
    verdicts = iter([{"valid": False, "reason": "1차 실패"}, {"valid": True, "reason": "통과"}])

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    def flaky_verify(question, domain_results, llm_fn):
        return next(verdicts)

    res = answer_with_verification(
        "전략 그래프 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=flaky_verify,
    )
    assert res["attempts"] == 2
    assert res.get("chart_base64")  # 원본 질문의 '그려줘'로 차트 생성


def test_answer_with_verification_uncertain_has_no_chart():
    """3회 실패(불확실) 응답에는 차트 필드가 없다(또는 None)."""
    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}

    def always_invalid(question, domain_results, llm_fn):
        return {"valid": False, "reason": "실패"}

    res = answer_with_verification(
        "전략 그래프 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=always_invalid,
    )
    assert res["uncertain"] is True
    assert res.get("chart_base64") is None


# ── 다중 산출물 파이프라인(pipeline_exec 수정 후 backtest.result가 {out이름: 값} dict인
#    경우) 안에 중첩된 산점도/막대그래프 데이터도 인식해야 한다. 실서버 재현 버그:
#    "PBR-GPA 상관관계 구하고 산점도로, 분위별 평균은 막대그래프로" 같은 질문이
#    correlation+quantile_bucket_means+scatter_data 3개를 한 파이프라인으로 뽑는데,
#    예전엔(pipeline_exec 버그로) 마지막 산출물 하나만 남았고, 설령 다 남아도 이 함수들이
#    result를 "그 자체가 scatter/bucket 모양"인지만 봐서 중첩된 경우를 못 찾았다. ──────

def test_extract_scatter_data_finds_scatter_nested_in_multi_output_dict():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {
            "blocked": False,
            "result": {
                "corr": {"correlation": 0.42, "n": 655},
                "buckets": [{"bucket": 1, "count": 10, "bucket_range": [0.0, 1.0], "mean_value": 5.0}],
                "scatter": {
                    "x": [5.0, 8.0], "y": [12.0, 20.0], "labels": ["가", "나"],
                    "x_field": "pbr", "y_field": "gp_a",
                },
            },
        }
    }
    out = sup._extract_scatter_data(domain_results)
    assert out is not None
    x, y, labels, x_label, y_label, title = out
    assert x == [5.0, 8.0] and y == [12.0, 20.0]
    assert x_label == "pbr" and y_label == "gp_a"


# ── _extract_bar_data — quantile_bucket_means 결과(분위별 평균)를 막대그래프로 인식 ──

def test_extract_bar_data_recognizes_quantile_buckets_directly():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"blocked": False, "result": [
            {"bucket": 1, "count": 131, "bucket_range": [0.02, 0.37], "mean_value": 12.06},
            {"bucket": 2, "count": 131, "bucket_range": [0.37, 0.55], "mean_value": 15.82},
        ]}
    }
    out = sup._extract_bar_data(domain_results)
    assert out is not None
    labels, values, x_label, y_label, title = out
    assert labels == ["1분위", "2분위"]
    assert values == [12.06, 15.82]


def test_extract_bar_data_finds_buckets_nested_in_multi_output_dict():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"blocked": False, "result": {
            "corr": {"correlation": 0.42, "n": 655},
            "buckets": [
                {"bucket": 1, "count": 10, "bucket_range": [0.0, 1.0], "mean_value": 5.0},
                {"bucket": 2, "count": 10, "bucket_range": [1.0, 2.0], "mean_value": 8.0},
            ],
        }}
    }
    out = sup._extract_bar_data(domain_results)
    assert out is not None
    labels, values, x_label, y_label, title = out
    assert labels == ["1분위", "2분위"]
    assert values == [5.0, 8.0]


def test_extract_bar_data_does_not_false_positive_on_screening_rows():
    """일반 스크리닝 결과(종목 리스트)는 분위 버킷 모양이 아니므로 None(오탐 방지)."""
    import src.agents.supervisor as sup

    domain_results = {"kr": {"intent": "screening", "result": [
        {"stock_code": "005930", "name": "삼성전자", "pbr": 1.2},
    ]}}
    assert sup._extract_bar_data(domain_results) is None


def test_extract_bar_data_none_when_blocked():
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"blocked": True, "result": [
        {"bucket": 1, "count": 1, "bucket_range": [0.0, 1.0], "mean_value": 1.0},
    ]}}
    assert sup._extract_bar_data(domain_results) is None


# ── _build_charts — 산점도+막대그래프가 동시에 있으면 둘 다 렌더(질문당 차트 여러 개 가능) ──

def test_build_charts_returns_both_scatter_and_bar_when_both_present():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"result": {
            "corr": {"correlation": 0.9, "n": 5},
            "buckets": [
                {"bucket": 1, "count": 1, "bucket_range": [0.0, 1.0], "mean_value": 5.0},
                {"bucket": 2, "count": 1, "bucket_range": [1.0, 2.0], "mean_value": 8.0},
            ],
            "scatter": {
                "x": [1.0, 2.0], "y": [3.0, 4.0], "labels": None,
                "x_field": "pbr", "y_field": "gp_a",
            },
        }}
    }
    charts = sup._build_charts(domain_results, conn=None)
    assert len(charts) == 2
    for chart_base64, title in charts:
        assert base64.b64decode(chart_base64)[:8] == _PNG_MAGIC
    titles = [t for _, t in charts]
    assert any("산점도" in t for t in titles)
    assert any("분위" in t or "막대" in t for t in titles)


def test_build_charts_falls_back_to_single_line_chart():
    """산점도/막대 데이터가 없으면 기존처럼 라인차트 1개만(회귀 없음)."""
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}
    charts = sup._build_charts(domain_results, conn=None)
    assert len(charts) == 1
    assert "백테스트" in charts[0][1]


def test_build_charts_empty_when_nothing_to_draw():
    import src.agents.supervisor as sup

    assert sup._build_charts({}, conn=None) == []


def test_answer_with_verification_exposes_charts_list_for_multi_output():
    """scatter+bar 둘 다 있으면 res['charts']에 둘 다, res['chart_base64']는 첫번째(하위호환)."""
    from src.agents.supervisor import answer_with_verification

    bt = {"result": {
        "corr": {"correlation": 0.9, "n": 5},
        "buckets": [
            {"bucket": 1, "count": 1, "bucket_range": [0.0, 1.0], "mean_value": 5.0},
            {"bucket": 2, "count": 1, "bucket_range": [1.0, 2.0], "mean_value": 8.0},
        ],
        "scatter": {"x": [1.0, 2.0], "y": [3.0, 4.0], "labels": None, "x_field": "pbr", "y_field": "gp_a"},
    }}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "PBR과 GPA 상관관계 산점도로 그려주고 분위별 평균도 막대그래프로 보여줘",
        conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res.get("chart_base64")  # 하위호환 — 기존 소비자는 여전히 단일 차트로 읽을 수 있음
    assert isinstance(res.get("charts"), list)
    assert len(res["charts"]) == 2
    assert res["charts"][0]["chart_base64"] == res["chart_base64"]


def test_answer_with_verification_no_chart_when_no_series_data():
    """차트 키워드는 있으나 그릴 시계열이 없으면(스크리닝 등) 차트 없이 텍스트 응답만."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"intent": "screening", "result": [{"name": "A"}]}}

    res = answer_with_verification(
        "저PER 종목 그래프로 보여줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64") is None


# ── _is_histogram_shape / _extract_histogram_data — histogram_buckets 결과(균등폭 구간별
#    빈도)를 히스토그램으로 인식. quantile_bucket_means(list, 분위별 평균)와 모양이 다르다. ──

def test_is_histogram_shape_true_for_histogram_result():
    import src.agents.supervisor as sup

    res = {"field": "pbr", "num_buckets": 10,
           "bucket_edges": [0.0, 1.0, 2.0], "counts": [3, 5], "n": 8}
    assert sup._is_histogram_shape(res) is True


def test_is_histogram_shape_false_for_screening_and_quantile_lists():
    import src.agents.supervisor as sup

    # 스크리닝 결과(종목 리스트)
    assert sup._is_histogram_shape([{"stock_code": "005930", "pbr": 1.2}]) is False
    # quantile_bucket_means 개별 원소(dict, bucket 키 있음)
    assert sup._is_histogram_shape(
        {"bucket": 1, "count": 10, "bucket_range": [0.0, 1.0], "mean_value": 5.0}
    ) is False
    # quantile_bucket_means 전체(list)
    assert sup._is_histogram_shape(
        [{"bucket": 1, "count": 10, "bucket_range": [0.0, 1.0], "mean_value": 5.0}]
    ) is False


def test_extract_histogram_data_recognizes_backtest_histogram_result():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"blocked": False, "result": {
            "field": "pbr", "num_buckets": 3,
            "bucket_edges": [0.0, 1.0, 2.0, 3.0], "counts": [10, 25, 7], "n": 42,
        }}
    }
    out = sup._extract_histogram_data(domain_results)
    assert out is not None
    edges, counts, x_label, title = out
    assert edges == [0.0, 1.0, 2.0, 3.0]
    assert counts == [10, 25, 7]
    assert x_label == "pbr"
    assert "히스토그램" in title and "pbr" in title


def test_extract_histogram_data_finds_histogram_nested_in_multi_output_dict():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"blocked": False, "result": {
            "corr": {"correlation": 0.42, "n": 655},
            "hist": {"field": "pbr", "num_buckets": 2,
                     "bucket_edges": [0.0, 1.0, 2.0], "counts": [3, 4], "n": 7},
        }}
    }
    out = sup._extract_histogram_data(domain_results)
    assert out is not None
    edges, counts, x_label, title = out
    assert edges == [0.0, 1.0, 2.0]
    assert counts == [3, 4]
    assert x_label == "pbr"


def test_extract_histogram_data_none_when_blocked():
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"blocked": True, "result": {
        "field": "pbr", "bucket_edges": [0.0, 1.0], "counts": [1], "n": 1}}}
    assert sup._extract_histogram_data(domain_results) is None


def test_build_charts_renders_histogram_png_when_present():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {"result": {
            "field": "pbr", "num_buckets": 3,
            "bucket_edges": [0.0, 1.0, 2.0, 3.0], "counts": [10, 25, 7], "n": 42,
        }}
    }
    charts = sup._build_charts(domain_results, conn=None)
    assert len(charts) == 1
    chart_base64, title = charts[0]
    assert base64.b64decode(chart_base64)[:8] == _PNG_MAGIC
    assert "히스토그램" in title


def test_build_charts_histogram_calls_render_with_correct_args(monkeypatch):
    """render_histogram_chart_base64가 edges/counts/x_label/title 인자로 호출되는지 확인."""
    import src.agents.supervisor as sup

    captured = {}

    def fake_render(edges, counts, x_label, title):
        captured["args"] = (edges, counts, x_label, title)
        return "ZmFrZQ=="  # 임의 base64

    monkeypatch.setattr(sup, "render_histogram_chart_base64", fake_render)
    domain_results = {
        "backtest": {"result": {
            "field": "pbr", "num_buckets": 2,
            "bucket_edges": [0.0, 1.0, 2.0], "counts": [3, 4], "n": 7,
        }}
    }
    charts = sup._build_charts(domain_results, conn=None)
    assert len(charts) == 1
    edges, counts, x_label, title = captured["args"]
    assert edges == [0.0, 1.0, 2.0]
    assert counts == [3, 4]
    assert x_label == "pbr"
    assert "히스토그램" in title


def test_answer_with_verification_adds_histogram_chart_when_requested():
    """'히스토그램 그려줘' + 백테스트 histogram 결과 → 성공 응답에 chart_base64(PNG) 배선."""
    from src.agents.supervisor import answer_with_verification

    bt = {"result": {"field": "pbr", "num_buckets": 3,
                     "bucket_edges": [0.0, 1.0, 2.0, 3.0], "counts": [10, 25, 7], "n": 42}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "코스피 PBR을 20구간으로 나눠 히스토그램 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64")
    assert base64.b64decode(res["chart_base64"])[:8] == _PNG_MAGIC
    assert "히스토그램" in res["chart_title"]


# ── chart_agent 폴백 배선 — _build_charts(결정론적 4케이스)가 아무것도 못 찾았을 때만
#    build_chart_freeform(LLM 자유선택 서브에이전트)을 보강으로 시도한다. 실사용 재현 버그:
#    "종목 20개 + 각각 수익률 하나씩"인 스크리닝 리스트는 산점도/분위수막대/히스토그램/
#    시계열 3케이스 중 어디에도 안 맞아 _build_charts가 빈 리스트를 내고, 그래서 진짜
#    이미지가 전혀 안 만들어졌다(최종답변 LLM이 텍스트 ASCII 막대로 대신 지어낸 부작용). ──

def test_answer_with_verification_uses_chart_fallback_when_build_charts_empty():
    """_build_charts가 못 찾는 스크리닝 리스트 모양 → chart_fallback_fn이 보강으로 호출된다."""
    from src.agents.supervisor import answer_with_verification

    screening = {"result": [
        {"stock_code": "005930", "name": "삼성전자", "return_12m": 12.3},
        {"stock_code": "000660", "name": "SK하이닉스", "return_12m": 45.6},
    ]}

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": screening}

    captured = {}

    def fake_chart_fallback(question, domain_results, llm_fn):
        captured["question"] = question
        captured["domain_results"] = domain_results
        return {"chart_base64": "PNG", "chart_title": "수익률 막대그래프"}

    res = answer_with_verification(
        "코스피 상위 종목 수익률 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["uncertain"] is False
    assert res["chart_base64"] == "PNG"
    assert res["chart_title"] == "수익률 막대그래프"
    assert captured["question"] == "코스피 상위 종목 수익률 그래프로 그려줘"
    # 폴백에는 도메인키 래퍼({"kr": {...}})가 아니라 **flat한 종목 리스트**(그릴 실제 데이터)를
    # 넘겨야 한다. 래퍼를 넘기면 build_chart_freeform이 받는 data가 dict가 돼 요약이 'dict,
    # 최상위 키: [kr]'로만 나오고 LLM이 정작 그릴 리스트(domain_results["kr"]["result"])를 못
    # 찾아 실행에 실패했다(실측). 멀티턴 경로가 flat한 result를 넘기는 것과 대칭.
    assert captured["domain_results"] == screening["result"]


def test_chart_fallback_receives_unwrapped_flat_list_for_single_domain():
    """단일 도메인 스크리닝: 폴백은 domain_results["kr"]["result"](flat 리스트)를 그대로 받는다."""
    from src.agents.supervisor import answer_with_verification

    stocks = [
        {"stock_code": "005930", "name": "삼성전자", "return_12m": 12.3},
        {"stock_code": "000660", "name": "SK하이닉스", "return_12m": 45.6},
    ]
    screening = {"intent": "screening", "result": stocks}

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": screening}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["data"] = data
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "코스피 상위 10개 수익률 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["chart_base64"] == "PNG"
    # 폴백이 받은 data는 flat 리스트여야 한다(래퍼 dict가 아니라).
    assert captured["data"] == stocks
    assert isinstance(captured["data"], list)


def test_chart_fallback_receives_per_domain_payloads_for_multi_domain():
    """복합 도메인(kr+macro): 각 도메인의 result만 뽑아 {도메인: payload} dict로 넘긴다
    (도메인별 껍데기 dict가 아니라 그릴 데이터만). 미국 도메인 비활성화와 무관하게 이
    '도메인별 payload 추출' 로직은 도메인 종류를 가리지 않으므로 여전히 활성인 kr+macro로 검증한다."""
    from src.agents.supervisor import answer_with_verification

    kr_rows = [{"name": "삼성전자", "return_12m": 12.3}]
    macro_rows = [{"as_of": "2026-07-01", "overall": "GREEN"}]

    def stub_route(question, llm_fn):
        return ["kr", "macro"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": kr_rows}, "macro": {"result": macro_rows}}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["data"] = data
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "한국 상위 종목 수익률이랑 매크로 신호 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["chart_base64"] == "PNG"
    # 도메인이 여러 개면 {도메인: 그 도메인의 result} — 껍데기 dict가 아니라 그릴 리스트만.
    assert captured["data"] == {"kr": kr_rows, "macro": macro_rows}


def test_answer_with_verification_does_not_call_chart_fallback_when_build_charts_already_found():
    """_build_charts가 이미 그렸으면(기존 3케이스) 폴백을 아예 호출하지 않는다(회귀 방지)."""
    from src.agents.supervisor import answer_with_verification

    bt = {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    def fake_chart_fallback(question, domain_results, llm_fn):
        raise AssertionError("_build_charts가 이미 찾았으면 폴백을 호출하면 안 된다")

    res = answer_with_verification(
        "이 전략 백테스트 그래프로 보여줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res.get("chart_base64")
    assert base64.b64decode(res["chart_base64"])[:8] == _PNG_MAGIC


def test_answer_with_verification_no_chart_fallback_without_keyword():
    """wants_chart(question)이 False면 _build_charts도 폴백도 아예 시도하지 않는다."""
    from src.agents.supervisor import answer_with_verification

    screening = {"result": [{"stock_code": "005930", "name": "삼성전자", "return_12m": 12.3}]}

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": screening}

    def fake_chart_fallback(question, domain_results, llm_fn):
        raise AssertionError("차트 키워드가 없으면 폴백을 시도하면 안 된다")

    res = answer_with_verification(
        "코스피 상위 종목 수익률 알려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res.get("chart_base64") is None


def test_answer_with_verification_chart_fallback_returning_none_leaves_no_chart():
    """폴백도 실패(None)하면 차트 없이 텍스트 응답만(에러 아님)."""
    from src.agents.supervisor import answer_with_verification

    screening = {"result": [{"stock_code": "005930", "name": "삼성전자", "return_12m": 12.3}]}

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": screening}

    def fake_chart_fallback(question, domain_results, llm_fn):
        return None

    res = answer_with_verification(
        "코스피 상위 종목 수익률 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64") is None
