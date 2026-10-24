"""자체 점검. 실행: python -m pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import engine

EXP001 = Path(__file__).parent.parent / "second-projects" / "experiments" / "results_001"


def fake_prices(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99,
                         "Close": c, "Volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


@pytest.mark.parametrize("rules,mode", [(["200일선"], "다수결"), (["RSI", "MACD", "볼린저"], "다수결"),
                                        (["골든크로스", "OBV(임의)"], "전부"), ([engine.VB], "다수결")])
def test_no_lookahead(rules, mode):
    """중간에서 자른 데이터로 계산해도 그 이전 결과가 같아야 한다 (미래 엿보기 없음)."""
    d = fake_prices()
    cut = d.index[len(d) // 2]
    full, _, _ = engine.strategy_returns(d, rules, mode, 0.001)
    part, _, _ = engine.strategy_returns(d.loc[:cut], rules, mode, 0.001)
    assert np.allclose(full.loc[part.index], part)


def test_hold_is_first_to_last_open():
    d = fake_prices()
    _, _, oo = engine.strategy_returns(d, ["200일선"], "다수결", 0.0)
    a, b = d.index[10], d.index[500]
    total = engine.metrics(engine.hold_returns(oo, a, b, 0.0))[0]
    # b일 값은 b→다음날 시가 수익이므로 끝은 b 다음 날 시가
    assert np.isclose(total, d["Open"].iloc[501] / d["Open"].iloc[10] - 1)


def test_all_mode_with_one_rule_equals_single():
    d = fake_prices()
    a = engine.strategy_returns(d, ["MACD"], "전부", 0.001)[0]
    b = engine.strategy_returns(d, ["MACD"], "다수결", 0.001)[0]
    assert a.equals(b)


@pytest.mark.skipif(not EXP001.exists(), reason="실험 001 저장 데이터가 있는 PC에서만")
@pytest.mark.parametrize("ticker,cost,start,end,rule,expected", [
    ("MSFT", 0.0035, "2016-01-01", "2026-06-30", "200일선", (3.867, 0.163, -0.292)),
    ("MSFT", 0.0035, "2016-01-01", "2026-06-30", None, (7.015, 0.220, -0.365)),
    ("BTC-USD", 0.0010, "2021-01-01", "2026-06-30", "200일선", (1.289, 0.163, -0.642)),
    ("BTC-USD", 0.0010, "2014-09-01", "2020-12-31", None, (61.174, 0.927, -0.834)),
])
def test_matches_experiment_001(ticker, cost, start, end, rule, expected):
    """실험 001 results.md(소수 첫째 자리 %)와 같은 숫자가 나와야 한다."""
    d = pd.read_csv(EXP001 / f"data_{ticker}.csv", index_col=0, parse_dates=True).loc[:"2026-10-22"]
    ret, _, oo = engine.strategy_returns(d, [rule or "200일선"], "다수결", cost)
    r = engine.hold_returns(oo, start, end, cost) if rule is None else ret.loc[start:end]
    assert np.allclose(engine.metrics(r), expected, atol=0.0006)


def fake_portfolio(n=900):
    data = {f"T{k}": fake_prices(n, seed=k) for k in range(6)}
    bench = fake_prices(n, seed=99)
    return data, bench


@pytest.mark.parametrize("weekly", [False, True])
def test_portfolio_no_lookahead(weekly):
    data, bench = fake_portfolio()
    cut = bench.index[700]
    args = (["200일선", "MACD", "RSI"], "다수결")
    full = engine.simulate(*engine.build_portfolio(data, bench.index, *args), bench["Open"], 3, 0.001, weekly,
                           bench.index[250])[0]
    short_data = {t: d.loc[:cut] for t, d in data.items()}
    short_idx = bench.index[bench.index <= cut]
    part = engine.simulate(*engine.build_portfolio(short_data, short_idx, *args), bench["Open"].loc[:cut], 3, 0.001,
                           weekly, bench.index[250])[0]
    assert np.allclose(full.loc[part.index], part)


def test_portfolio_without_signal_equals_index():
    """조건이 한 번도 안 맞으면 처음부터 끝까지 지수만 들고 있어야 한다."""
    data, bench = fake_portfolio()
    opn, rsi, cond = engine.build_portfolio(data, bench.index, ["200일선"], "다수결")
    eq, held, buys = engine.simulate(opn, rsi, cond & False, bench["Open"], 3, 0.002, False, bench.index[250])
    o = bench["Open"].loc[bench.index[250]:]
    assert np.allclose(eq, o / o.iloc[0] * (1 - 0.002)) and held.max() == 0 and buys.sum() == 0


EXP003 = Path(__file__).parent.parent / "second-projects" / "experiments" / "results_003"


@pytest.mark.skipif(not EXP003.exists(), reason="실험 003 저장 데이터가 있는 PC에서만")
@pytest.mark.parametrize("mode,weekly,period,expected,n_buys", [
    ("전부", True, ("2010-01-01", "2017-12-31"), (0.501, 0.052, -0.186), 433),
    ("전부", True, ("2018-01-01", "2026-06-30"), (0.270, 0.029, -0.388), 442),
    ("다수결", True, ("2010-01-01", "2017-12-31"), (-0.366, -0.055, -0.433), 976),
    ("전부", False, ("2010-01-01", "2017-12-31"), (0.026, 0.003, -0.274), 709),
])
def test_matches_experiment_004(mode, weekly, period, expected, n_buys):
    """실험 004 results.md의 '빈칸=DIA' 줄과 같은 숫자 (다우 29종목, 200일선·MACD·RSI, 10칸, 비용 0.35%)."""
    load = lambda f: pd.read_csv(f, index_col=0, parse_dates=True).loc[:"2026-10-22"]
    skip = {"DIA", "SPY", "^IRX", "HWM"}
    data = {f.stem[5:]: load(f) for f in EXP003.glob("data_*.csv") if f.stem[5:] not in skip}
    dia = load(EXP003 / "data_DIA.csv")
    assert len(data) == 29
    built = engine.build_portfolio(data, dia.index, ["200일선", "MACD", "RSI"], mode)
    eq, _, buys = engine.simulate(*built, dia["Open"], 10, 0.0035, weekly, "2010-01-01")
    a, b = period
    assert np.allclose(engine.metrics(eq.loc[a:b].pct_change().dropna()), expected, atol=0.0006)
    assert buys.loc[a:b].sum() == n_buys


@pytest.mark.parametrize("rules", [["200일선"], ["MACD"], ["RSI", "볼린저", "OBV(임의)"]])
def test_diagnose_adds_up(rules):
    """비용 0일 때: (보유한 날 수익) × (안 보유한 날 수익) = 그냥 들고 있기 (비용 0)."""
    d = fake_prices()
    a, b = d.index[220], d.index[700]
    x = engine.diagnose(d, rules, "다수결", 0.003, a, b)
    _, entries, oo = engine.strategy_returns(d, rules, "다수결", 0.0)
    gross = engine.metrics(engine.strategy_returns(d, rules, "다수결", 0.0)[0].loc[a:b])[0]
    hold = engine.metrics(oo.loc[a:b])[0]
    assert np.isclose((1 + gross) * (1 + x["out_move"]), 1 + hold)
    assert x["cagr0"] >= engine.metrics(engine.strategy_returns(d, rules, "다수결", 0.003)[0].loc[a:b])[1]
    # 구간 첫날부터 들고 있던 매매 1건을 빼면 매매 수 = 매수 횟수
    assert x["trades"] - entries.loc[a:b].sum() in (0, 1)


def test_diagnose_breakout():
    d = fake_prices()
    x = engine.diagnose(d, [engine.VB], "다수결", 0.001, d.index[10], d.index[500])
    _, entries, _ = engine.strategy_returns(d, [engine.VB], "다수결", 0.001)
    assert x["trades"] == entries.loc[d.index[10]:d.index[500]].sum() and 0 <= x["losing"] <= 1
