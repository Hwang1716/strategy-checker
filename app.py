"""화면 담당. 실행: python -m streamlit run app.py"""
import datetime as dt
import json
import urllib.request

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import engine

st.set_page_config(page_title="전략 검증기", page_icon="📊", layout="wide")
MAX_CANDIDATES = 30  # 야후 요청 제한 때문에 후보 수를 묶는다


@st.cache_data(ttl=12 * 3600, max_entries=300, show_spinner="시세 받는 중…")  # 개수 제한: 서버 메모리 보호
def load(ticker):
    """야후 일봉(분할·배당 반영). 같은 종목은 12시간 동안 다시 받지 않는다."""
    d = yf.download(ticker, period="max", auto_adjust=True, progress=False, multi_level_index=False)
    if d.empty:  # 없는 코드거나 요청 제한. 예외로 던져야 빈 결과가 12시간 캐시되지 않는다
        raise LookupError(ticker)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    # shortcut: 오늘 날짜(PC/서버 시간 기준) 캔들은 아직 안 끝났다고 보고 뺀다. 시간대 차이로 하루 어긋날 수 있음
    return d[d.index < pd.Timestamp.today().normalize()]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def upbit_coins():
    """업비트 원화 마켓 목록 (한글 이름 검색용, 키 불필요)."""
    with urllib.request.urlopen("https://api.upbit.com/v1/market/all", timeout=10) as r:
        return [m for m in json.load(r) if m["market"].startswith("KRW-")]


@st.cache_data(ttl=24 * 3600, max_entries=500, show_spinner="검색 중…")
def search(q):
    """이름·코드로 종목 찾기 → [(보여줄 이름, 야후 코드)]. 한글이면 업비트 코인 목록, 아니면 야후 검색."""
    if any("가" <= ch <= "힣" for ch in q):
        hits = sorted((m for m in upbit_coins() if q in m["korean_name"]), key=lambda m: m["korean_name"] != q)
        out = []
        for m in hits[:5]:
            base = m["market"][4:]
            # 야후는 이름이 겹치는 코인에 숫자를 붙인다 (예: SUI → SUI20947-USD)
            for x in yf.Search(m["english_name"], max_results=10).quotes:
                s = x["symbol"]
                if x.get("quoteType") == "CRYPTOCURRENCY" and s.endswith("-USD") and s[:-4].rstrip("0123456789") == base:
                    out.append((f"{m['korean_name']} ({s})", s))
                    break
        return out
    quotes = yf.Search(q, max_results=10).quotes
    return [(f"{x.get('shortname') or x.get('longname') or x['symbol']} ({x['symbol']})", x["symbol"])
            for x in quotes if x.get("quoteType") in ("EQUITY", "ETF", "CRYPTOCURRENCY")]


def pick(label, key, default=""):
    """이름이나 코드를 받아 검색 결과에서 고르게 한다. 검색이 안 되면 입력을 코드로 본다."""
    q = st.text_input(label, default, key=key + "_q",
                      help="영어 이름(apple, bitcoin), 코인 한글 이름(비트코인, 수이), 코드(MSFT, BTC-USD, BTC-KRW)").strip()
    if not q:
        return ""
    try:
        found = search(q)
    except Exception:
        found = []
    names = dict((s, n) for n, s in found)
    code = q.upper()
    options = list(names) + ([] if code in names else [code])
    if code in names:
        options.insert(0, options.pop(options.index(code)))
    names.setdefault(code, f"{code} (입력한 코드 그대로)")
    return st.selectbox("검색 결과", options, format_func=names.get, key=key + "_s", label_visibility="collapsed")


def is_coin(t):
    return t.endswith(("-USD", "-KRW"))


def bench_for(t):
    """비교 기준: 주식은 다우 지수 ETF(DIA), 코인은 같은 통화의 비트코인."""
    return ("BTC-" + t[-3:], f"비트코인(BTC-{t[-3:]})") if is_coin(t) else ("DIA", "다우 지수(DIA)")


def pct(v):
    return f"{v * 100:,.1f}%"


def periods_ui(coin, key):
    """시작일·검증 시작일·종료일·비용 입력. 종목 종류가 바뀌면 기본값이 다시 잡히도록 key를 바꾼다."""
    st.header("구간·비용")
    start = st.date_input("시작일", dt.date(2014, 9, 1) if coin else dt.date(2000, 1, 1), key=f"s{key}")
    split = st.date_input("검증 시작일 (이 날부터 검증 구간)", dt.date(2021, 1, 1) if coin else dt.date(2016, 1, 1),
                          key=f"m{key}")
    end = st.date_input("종료일", dt.date.today() - dt.timedelta(days=1), key=f"e{key}")
    cost = st.number_input("1회 거래 비용 (%, 수수료+슬리피지)", 0.0, 5.0, 0.10 if coin else 0.35, 0.05,
                           key=f"c{key}") / 100
    return start, split, end, cost


def rule_help():
    with st.expander("지표 설명"):
        for name, text in engine.RULE_HELP.items():
            st.markdown(f"- **{name}**: {text}")


def check_dates(start, split, end):
    if not start < split < end:
        st.warning("날짜는 시작일 < 검증 시작일 < 종료일 순서여야 합니다.")
        st.stop()
    start, split, end = (pd.Timestamp(x) for x in (start, split, end))
    return start, split, end, {"학습": (start, split - pd.Timedelta(days=1)), "검증": (split, end)}


def load_or_stop(tickers):
    out = {}
    for t in tickers:
        try:
            out[t] = load(t)
        except Exception:
            st.error(f"'{t}' 시세를 받지 못했습니다. 코드가 맞는지 확인하세요 (코인은 BTC-USD·BTC-KRW처럼 -USD나 -KRW를 "
                     "붙입니다). 코드가 맞다면 야후 서버 요청 제한일 수 있으니 잠시 후 다시 시도해 주세요.")
            st.stop()
    return out


def open_to_open(d):
    return (d["Open"].shift(-1) / d["Open"] - 1).iloc[:-1]


def verdict(verdicts, target, basis):
    if all(verdicts.values()):
        st.success(f"**후보**: 학습·검증 구간 모두에서 '{target}'보다 나았습니다 ({basis}). "
                   "그래도 우연일 수 있으니 다른 종목·기간에서도 확인하세요.")
    else:
        failed = ", ".join(k for k, v in verdicts.items() if not v)
        st.error(f"**탈락**: {failed} 구간에서 '{target}'를 이기지 못했습니다 ({basis}).")
    st.caption("판정 기준: 구간마다 CAGR이 비교 대상보다 높거나, 차이가 연 2%p 이내이면서 MDD(최대 낙폭)가 절반 이하. "
               "학습·검증 둘 다 통과해야 후보.")


def show_table(rows, verdicts, label, extra_cols=()):
    cols = ["구간", "대상", "총수익률", "CAGR(연 환산)", "MDD(최대 낙폭)", "매수 횟수", *extra_cols]
    table = pd.DataFrame(rows, columns=cols)
    for col in ["총수익률", "CAGR(연 환산)", "MDD(최대 낙폭)"]:
        table[col] = table[col].map(pct)
    table["판정"] = [("통과" if verdicts[p] else "탈락") if t.startswith("전략") else ""
                   for p, t in zip(table["구간"], table["대상"])]
    st.markdown(f"**전략** = {label}")
    st.dataframe(table, hide_index=True)


def show_chart(curves, split):
    fig = go.Figure()
    for name, eq in curves.items():
        fig.add_scatter(x=eq.index, y=eq, name=name, mode="lines", line=dict(width=3 if name.startswith("전략") else 1.5))
    fig.add_vline(x=split, line_dash="dash", line_color="gray")
    fig.add_annotation(x=split, y=1, yref="paper", text=" 검증 시작", showarrow=False, xanchor="left")
    fig.update_layout(title="1로 시작했을 때 자산 (로그 축, 비용 반영)", yaxis_type="log", yaxis_title="자산 (시작 = 1)",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.15), height=520)
    st.plotly_chart(fig)


def mode_radio():
    return st.radio("조합 방식 (지표 2개 이상일 때)", ["다수결", "전부"], horizontal=True,
                    help="다수결: 고른 지표의 과반수가 '보유'면 보유 / 전부: 모두 '보유'일 때만 보유")


def run_with(key, submitted, params):
    """[검증하기]를 누른 때의 설정만 결과에 쓴다. 설정을 바꾸는 중에는 다시 계산하지 않는다."""
    if submitted:
        st.session_state[key] = params
    if key not in st.session_state:
        st.info("왼쪽에서 설정을 고른 뒤 **[검증하기]** 버튼을 누르세요.")
        st.stop()
    if st.session_state[key] != params:
        st.caption("⚠️ 왼쪽 설정이 바뀌었습니다. 아래는 이전 설정의 결과이니 **[검증하기]**를 다시 누르세요.")
    return st.session_state[key]


def single():
    with st.sidebar:
        st.header("종목")
        ticker = pick("이름 또는 코드로 검색", "one", "MSFT")
        with st.form("one_form"):
            st.header("규칙")
            rules = st.multiselect("지표 (여러 개 고르면 조합)", engine.RULES, ["200일선"])
            mode = mode_radio()
            rule_help()
            start, split, end, cost = periods_ui(is_coin(ticker), ticker)
            go_ = st.form_submit_button("검증하기", type="primary")
    ticker, rules, mode, start, split, end, cost = run_with(
        "one_params", go_, (ticker, rules, mode, start, split, end, cost))

    if not ticker or not rules:
        st.warning("종목과 지표를 하나 이상 고르세요.")
        st.stop()
    if engine.VB in rules and len(rules) > 1:
        st.warning("변동성돌파는 하루 안에 사고파는 방식이라 다른 지표와 묶을 수 없습니다. 단독으로 골라 주세요.")
        st.stop()
    start, split, end, periods = check_dates(start, split, end)
    bench_ticker, bench_name = bench_for(ticker)
    data = load_or_stop([ticker, bench_ticker])
    d, bench_oo = data[ticker], open_to_open(data[bench_ticker])

    ret, entries, oo = engine.strategy_returns(d, rules, mode, cost)
    for pname, (a, b) in periods.items():
        if len(oo.loc[a:b]) < 60:
            st.warning(f"{pname} 구간에 데이터가 {len(oo.loc[a:b])}일뿐입니다 (최소 60일). "
                       f"'{ticker}' 데이터는 {d.index[0].date()}부터 있으니 시작일·검증 시작일을 뒤로 옮겨 보세요.")
            st.stop()
    if (start - d.index[0]).days < 300:
        st.info(f"데이터가 {d.index[0].date()}부터라 처음 몇 달은 200일 평균 같은 지표가 아직 계산되지 않아 "
                "'보유 안 함'으로 처리됩니다.")

    label = " + ".join(rules) + (f" ({mode})" if len(rules) > 1 else "")
    hold_name = f"그냥 들고 있기: {ticker}"
    rows, verdicts, stats = [], {}, {}
    for pname, (a, b) in periods.items():
        m_rule, m_hold = engine.metrics(ret.loc[a:b]), engine.metrics(engine.hold_returns(oo, a, b, cost))
        verdicts[pname] = engine.passes(m_rule, m_hold)
        stats[pname] = m_rule, m_hold
        rows.append([pname, "전략", *m_rule, int(entries.loc[a:b].sum())])
        rows.append([pname, hold_name, *m_hold, 1])
        if bench_ticker != ticker and len(bench_oo.loc[a:b]):
            rows.append([pname, f"비교 기준: {bench_name}", *engine.metrics(engine.hold_returns(bench_oo, a, b, cost)), 1])

    verdict(verdicts, hold_name, "실험 001 기준")
    show_table(rows, verdicts, label)
    curves = {f"전략: {label}": ret.loc[start:end], hold_name: engine.hold_returns(oo, start, end, cost)}
    if bench_ticker != ticker:
        curves[f"비교 기준: {bench_name}"] = engine.hold_returns(bench_oo, start, end, cost)
    show_chart({k: (1 + r).cumprod() for k, r in curves.items()}, split)

    def lines(pname, a, b):
        (_, cagr, mdd), (_, h_cagr, h_mdd) = stats[pname]
        x = engine.diagnose(d, rules, mode, cost, a, b)
        drag = x["cagr0"] - cagr
        if cagr < h_cagr and x["cagr0"] >= h_cagr:
            key = "비용만 없었다면 들고 있기를 이겼습니다 → **탈락의 주원인은 잦은 매매 비용**입니다."
        elif cagr < h_cagr and x["out_move"] > 0:
            key = "쉬는 동안 종목이 올랐습니다 → **주원인은 놓친 상승**입니다."
        elif cagr < h_cagr:
            key = "쉬는 동안엔 하락을 피했지만 보유 중 수익이 더 나빴습니다 → **주원인은 사고판 타이밍**입니다."
        elif abs(mdd) < abs(h_mdd):
            key = "들고 있기보다 덜 잃었습니다 → **주원인은 하락장 방어**입니다."
        else:
            key = "보유한 기간의 수익이 들고 있기보다 좋았습니다."
        out = [f"들고 있기보다 CAGR이 연 {pp(cagr - h_cagr)}. {key}",
               f"**거래 비용**: 비용이 없었다면 CAGR {pct(x['cagr0'])} → 실제 {pct(cagr)}. "
               f"매매 {x['trades']}번, 평균 {x['avg_days']:.0f}일 보유해 비용으로 연 {abs(drag) * 100:.1f}%p를 잃었습니다."]
        if x["held"] < 0.999:
            out.append(f"**쉬는 동안**: 기간의 {pct(1 - x['held'])}는 보유하지 않았고, 그동안 {ticker}는 "
                       + (f"+{pct(x['out_move'])} 올라 **상승을 놓쳤습니다**." if x["out_move"] > 0
                          else f"{pct(x['out_move'])} 내려 **하락을 피했습니다**."))
        out.append(f"**하락 방어**: 최대 낙폭 {pct(mdd)} vs 들고 있기 {pct(h_mdd)} → "
                   + ("덜 떨어졌습니다." if abs(mdd) < abs(h_mdd) else "오히려 더 떨어졌습니다."))
        if x["trades"]:
            out.append(f"**매매 성적**: {x['trades']}번 중 {pct(x['losing'])}가 손해 본 매매였습니다."
                       + (" 짧게 사고팔다 손해 보는 **헛매매(휩쏘)**가 많았습니다." if x["avg_days"] < 20 and x["losing"] > 0.5 else ""))
        return out

    explain(periods, verdicts, lines)


def pp(v):
    """CAGR 차이를 '+1.2%p 높았습니다'처럼."""
    return f"{abs(v) * 100:.1f}%p " + ("높았습니다" if v >= 0 else "낮았습니다")


def explain(periods, verdicts, lines):
    """'왜 이런 결과가 나왔나' 칸. lines(구간 이름, 시작, 끝) → 문장 목록. 원인 설명만, 투자 조언은 하지 않는다."""
    st.subheader("🔍 왜 이런 결과가 나왔나")
    if len(set(verdicts.values())) > 1:
        ok = [k for k, v in verdicts.items() if v][0]
        st.markdown(f"⚠️ **구간마다 결과가 다릅니다**: {ok} 구간만 통과했습니다. 시기를 타는 결과라 믿기 어렵습니다.")
    for col, (pname, (a, b)) in zip(st.columns(len(periods)), periods.items()):
        col.markdown(f"**{pname} 구간**\n" + "\n".join(f"- {s}" for s in lines(pname, a, b)))
    st.caption("숫자로 계산한 원인 설명이며, 어떤 전략을 쓰라는 권유가 아닙니다.")


def add_candidates(codes):
    have = st.session_state.cands
    st.session_state.cands = have + [c for c in codes if c and c not in have][:MAX_CANDIDATES - len(have)]


def portfolio():
    st.session_state.setdefault("cands", [])
    with st.sidebar:
        st.header("후보 종목")
        found = pick("이름 또는 코드로 검색", "many")
        st.button("후보에 추가", on_click=add_candidates, args=([found],), disabled=not found)
        bulk = st.text_input("코드 여러 개 한 번에 (쉼표로 구분)", placeholder="AAPL, MSFT, KO, JNJ")
        st.button("한꺼번에 추가", on_click=add_candidates,
                  args=([c.strip().upper() for c in bulk.split(",")],), disabled=not bulk)
        cands = st.multiselect(f"후보 (최대 {MAX_CANDIDATES}개, ×로 빼기)", st.session_state.cands, key="cands")
        with st.form("many_form"):
            st.header("규칙")
            choices = [r for r in engine.RULES if r != engine.VB]  # 변동성돌파는 하루짜리 매매라 칸 방식과 안 맞는다
            rules = st.multiselect("지표 (여러 개 고르면 조합)", choices, ["200일선", "MACD", "RSI"])
            mode = mode_radio()
            rule_help()
            st.header("칸 나누기")
            slots = st.number_input("칸 수 (한 칸 = 총자산 ÷ 칸 수)", 1, MAX_CANDIDATES, 10)
            weekly = st.radio("사고파는 날", ["매일", "주 1회 (주 첫 거래일)"], horizontal=True) != "매일"
            coin = bool(cands) and is_coin(cands[0])
            start, split, end, cost = periods_ui(coin, "coin" if coin else "stock")
            go_ = st.form_submit_button("검증하기", type="primary")
    cands, rules, mode, slots, weekly, start, split, end, cost = run_with(
        "many_params", go_, (list(cands), rules, mode, slots, weekly, start, split, end, cost))

    if len(cands) < 2 or not rules:
        st.info("왼쪽에서 후보 종목을 2개 이상 넣고 지표를 고르세요. 이름으로 검색해 하나씩 넣거나, 코드를 쉼표로 한꺼번에 넣을 수 있습니다.")
        st.stop()
    if len({t[-4:] if is_coin(t) else "stock" for t in cands}) > 1:
        st.warning("후보는 주식끼리, 또는 같은 통화의 코인끼리만 넣을 수 있습니다 (비교 기준과 비용을 하나로 정하기 위해).")
        st.stop()
    start, split, end, periods = check_dates(start, split, end)
    bench_ticker, bench_name = bench_for(cands[0])
    data = load_or_stop([*cands, bench_ticker])
    bench = data.pop(bench_ticker) if bench_ticker not in cands else data[bench_ticker]
    data = {t: data[t] for t in cands}

    dates = bench.index[bench.index <= end]
    if (dates >= start).sum() < 120:
        st.warning(f"비교 기준 데이터가 {bench.index[0].date()}부터라 시작일 이후 기간이 너무 짧습니다.")
        st.stop()
    opn, rsi, cond = engine.build_portfolio(data, dates, rules, mode)
    eq, held, buys = engine.simulate(opn, rsi, cond, bench["Open"].loc[:end], slots, cost, weekly, start)
    first = opn.loc[eq.index[0]]
    late = [t for t in cands if pd.isna(first[t])]
    o = opn.loc[eq.index[0]:, first.notna()]
    equal = (o.ffill() / o.iloc[0]).mean(axis=1) * (1 - cost) if len(o.columns) else None  # 시작일에 있던 후보만 똑같이
    bench_oo = open_to_open(bench)

    label = " + ".join(rules) + f" ({mode}, {slots}칸, {'주 1회' if weekly else '매일'})"
    bench_label = f"비교 기준: {bench_name}"
    grow = lambda s, a, b: engine.metrics(s.loc[a:b].pct_change().dropna())
    rows, verdicts, stats = [], {}, {}
    for pname, (a, b) in periods.items():
        m_rule = grow(eq, a, b)
        m_bench = engine.metrics(engine.hold_returns(bench_oo, a, b, cost))
        verdicts[pname] = engine.passes(m_rule, m_bench)
        stats[pname] = m_rule, m_bench, (grow(equal, a, b) if equal is not None else None)
        rows.append([pname, "전략", *m_rule, int(buys.loc[a:b].sum()), f"{held.loc[a:b].mean():.1f}"])
        rows.append([pname, bench_label, *m_bench, 1, ""])
        if equal is not None:
            rows.append([pname, "참고: 후보 균등 보유", *engine.metrics(equal.loc[a:b].pct_change().dropna()),
                         len(o.columns), ""])

    verdict(verdicts, bench_label[7:], "실험 003·004 기준")
    st.warning("**생존 편향 주의**: 후보를 '지금 아는 종목'으로 고르면, 그동안 망해서 사라진 종목은 빠지게 됩니다. "
               "그래서 실제보다 좋게 나오기 쉽습니다. 실험 003처럼 **과거 그 시점의 목록**(예: 2010년 다우 30종목)으로 고르면 덜 속습니다.")
    show_table(rows, verdicts, label, extra_cols=("평균 보유 종목 수",))
    if late:
        st.caption(f"시작일에 데이터가 없던 후보 {', '.join(late)}는 상장(데이터 시작) 200거래일 뒤부터 살 수 있고, "
                   "'똑같이 나눠 들고 있기'에서는 빠집니다.")
    curves = {f"전략: {label}": eq, bench_label: (1 + engine.hold_returns(bench_oo, start, end, cost)).cumprod()}
    if equal is not None:
        curves["참고: 후보 균등 보유"] = equal
    show_chart(curves, split)
    st.caption("방식: 돈을 칸 수만큼 나눠, 그날 종가 기준 조건을 충족한 후보를 다음 시가에 RSI 낮은 순으로 빈 칸에 삽니다. "
               "조건이 풀리면 팝니다. 빈 칸의 돈은 비교 기준 지수에 넣어 두고, 종목을 살 때 지수를 팔아 마련합니다"
               "(사고팔 때마다 비용). 비중 재조정은 하지 않습니다.")

    eq0 = engine.simulate(opn, rsi, cond, bench["Open"].loc[:end], slots, 0.0, weekly, start)[0]

    def lines(pname, a, b):
        (_, cagr, mdd), (_, b_cagr, b_mdd), m_eq = stats[pname]
        cagr0, n, bought = grow(eq0, a, b)[1], held.loc[a:b].mean(), int(buys.loc[a:b].sum())
        days = held.loc[a:b].sum() / bought if bought else 0
        if cagr < b_cagr and cagr0 >= b_cagr:
            key = "비용만 없었다면 지수를 이겼습니다 → **탈락의 주원인은 잦은 매매 비용**입니다."
        elif m_eq is not None and m_eq[1] > b_cagr and cagr < m_eq[1]:
            key = "후보 종목들은 지수보다 좋았는데 전략은 그걸 다 못 따라갔습니다 → **규칙대로 사고판 게 손해**였습니다."
        elif cagr < b_cagr:
            key = "후보 종목 자체도, 규칙도 지수를 넘지 못했습니다."
        else:
            key = "지수보다 나았습니다. 후보를 '지금 아는 종목'으로 골랐다면 생존 편향 때문일 수 있습니다."
        out = [f"지수보다 CAGR이 연 {pp(cagr - b_cagr)}. {key}",
               f"**거래 비용**: 비용이 없었다면 CAGR {pct(cagr0)} → 실제 {pct(cagr)}. 매수 {bought}번, 한 종목을 평균 "
               f"{days:.0f}거래일 들고 있어 비용으로 연 {abs(cagr0 - cagr) * 100:.1f}%p를 잃었습니다.",
               f"**칸 활용**: {slots}칸 중 평균 {n:.1f}칸만 찼습니다 → 자산의 약 {pct(1 - n / slots)}는 지수에 있었습니다"
               + (" (그래서 결과가 지수와 비슷해집니다)." if n / slots < 0.5 else ".")]
        if m_eq is not None:
            out.append(f"**종목 고르기 효과**: 후보를 똑같이 나눠 들고만 있었어도 지수보다 연 {pp(m_eq[1] - b_cagr)}.")
            out.append(f"**규칙 효과**: 같은 후보로 규칙대로 사고판 결과는 그냥 들고 있기보다 연 {pp(cagr - m_eq[1])}.")
        out.append(f"**하락 방어**: 최대 낙폭 {pct(mdd)} vs 지수 {pct(b_mdd)} → "
                   + ("덜 떨어졌습니다." if abs(mdd) < abs(b_mdd) else "오히려 더 떨어졌습니다."))
        return out

    explain(periods, verdicts, lines)


st.title("📊 전략 검증기")
st.caption("유튜브·블로그에서 본 매매 규칙이 '그냥 사서 들고 있기'보다 정말 나았는지, 과거 데이터로 정직하게 비교합니다.")
with st.sidebar:
    view = st.radio("검증 방식", ["한 종목", "여러 종목 (후보군)"], horizontal=True)

with st.container(border=True):
    st.markdown(
        "**읽기 전에** · 투자 조언이 아닙니다. 과거에 나았다고 앞으로도 낫다는 보장은 없습니다.\n"
        "- **생존 편향**: 지금 이름을 아는 종목은 살아남은 종목입니다. 사라진 종목에서는 같은 규칙이 크게 잃었을 수 있습니다.\n"
        "- **과최적화**: 여러 규칙·기간을 바꿔 가며 시도하면 우연히 좋은 결과가 하나쯤은 나옵니다. "
        "그래서 학습 구간과 검증 구간 **둘 다** 통과해야 '후보'로 봅니다.\n"
        "- 세금·환율은 빠져 있고, 데이터는 야후 파이낸스 무료 일봉입니다(지연·오류 가능).")

if view == "한 종목":
    single()
else:
    portfolio()
st.caption("매매 방식: 그날 종가로 신호 계산 → 다음 날 시가에 체결, 사고팔 때마다 비용 차감, 사기만 함(공매도 없음). "
           "변동성돌파만 당일 매수 → 다음 날 시가 매도. 데이터: 야후 파이낸스(yfinance), 분할·배당 반영 가격.")
