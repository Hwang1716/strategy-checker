"""계산 담당(화면 없음). 실험 001·002(second-projects/experiments)의 규칙·매매 방식을 그대로 옮겼다.

매매 방식: 그날 종가로 신호 계산 → 다음 날 시가에 체결, 사고팔 때마다 비용 차감, 사기만 함(공매도 없음).
"""
import numpy as np
import pandas as pd

VB = "변동성돌파"  # 당일 사고 다음 날 시가에 파는 방식이라 다른 지표와 묶을 수 없다
RULES = ["200일선", "골든크로스", "RSI", "MACD", "볼린저", "OBV(임의)", VB]
RULE_HELP = {
    "200일선": "종가가 200일 평균보다 위면 보유",
    "골든크로스": "50일 평균이 200일 평균보다 위면 보유",
    "RSI": "14일 RSI가 30 아래면 매수, 70 위면 매도",
    "MACD": "MACD(12,26)가 시그널(9)보다 위면 보유",
    "볼린저": "종가가 20일 평균−2×표준편차 아래면 매수, 20일 평균 위면 매도",
    "OBV(임의)": "OBV가 20일 평균보다 위면 보유 (교과서 규칙이 없어 임의로 정함)",
    VB: "시가 + 전날 (고가−저가)×0.5를 넘으면 매수, 다음 날 시가에 매도",
}


def rsi_wilder(close, n=14):
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def hold_state(enter, exit_):
    """매수 조건이 뜨면 1, 매도 조건이 뜨면 0, 그 사이는 앞 상태 유지."""
    s = pd.Series(np.where(enter, 1.0, np.where(exit_, 0.0, np.nan)), index=enter.index)
    return s.ffill().fillna(0.0)


def signals(d):
    """그날 종가까지의 데이터로 계산한 '보유하고 싶은 상태'(1/0). 실제 매매는 다음 날 시가."""
    c = d["Close"]
    sma = lambda s, n: s.rolling(n).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    mid, sd = sma(c, 20), c.rolling(20).std()
    obv = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()
    rsi = rsi_wilder(c)
    return pd.DataFrame({
        "200일선": (c > sma(c, 200)).astype(float),
        "골든크로스": (sma(c, 50) > sma(c, 200)).astype(float),
        "RSI": hold_state(rsi < 30, rsi > 70),
        "MACD": (macd > macd.ewm(span=9, adjust=False).mean()).astype(float),
        "볼린저": hold_state(c < mid - 2 * sd, c > mid),
        "OBV(임의)": (obv > sma(obv, 20)).astype(float),
    })


def want(d, rules, mode):
    """그날 종가 기준으로 보유하고 싶은가(True/False). 다수결 = 과반수 충족, 전부 = 모두 충족."""
    score = signals(d)[rules].sum(axis=1)
    return score == len(rules) if mode == "전부" else score > len(rules) / 2


def strategy_returns(d, rules, mode, cost):
    """하루 수익률·매수 여부·시가→시가 수익률. 날짜 t의 값 = t일 시가에 잡은 포지션의 t→t+1 시가 수익.

    mode: "다수결"(과반수 충족) 또는 "전부"(모두 충족). 규칙이 하나면 무시된다.
    """
    o = d["Open"]
    oo = o.shift(-1) / o - 1  # 오늘 시가 → 내일 시가
    if VB in rules:
        if len(rules) > 1:
            raise ValueError("변동성돌파는 다른 지표와 묶을 수 없습니다.")
        target = o + 0.5 * (d["High"] - d["Low"]).shift(1)
        entry = np.maximum(target, o)  # 갭 상승이면 시가에 체결
        hit = d["High"] >= target
        ret = pd.Series(np.where(hit, o.shift(-1) / entry - 1 - 2 * cost, 0.0), index=d.index)
        entries = hit.astype(int)
    else:
        pos = want(d, rules, mode).astype(float).shift(1).fillna(0)  # 어제 종가 신호 → 오늘 시가에 실행
        trade = pos.diff().abs().fillna(pos)
        ret = pos * oo - trade * cost
        entries = (pos.diff() == 1).astype(int)
    return ret.iloc[:-1], entries.iloc[:-1], oo.iloc[:-1]  # 마지막 날은 내일 시가가 없음


def hold_returns(oo, start, end, cost):
    """그냥 들고 있기: 구간 첫 시가에 사서 끝까지 (매수 비용 1회)."""
    r = oo.loc[start:end].copy()
    r.iloc[0] -= cost
    return r


def metrics(ret):
    """총수익률, CAGR(연 환산 수익률), MDD(최대 낙폭). 주말 거래가 있으면(코인) 1년=365일."""
    eq = (1 + ret).cumprod()
    years = len(ret) / (365 if ret.index.day_of_week.max() >= 5 else 252)
    total = eq.iloc[-1] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    mdd = (eq / eq.cummax() - 1).min()
    return total, cagr, mdd


def passes(rule, hold):
    """실험 001 기준: CAGR이 높거나, 차이 2%p 이내면서 MDD가 절반 이하. rule/hold = (총수익률, CAGR, MDD)."""
    return rule[1] > hold[1] or (hold[1] - rule[1] <= 0.02 and abs(rule[2]) <= abs(hold[2]) / 2)


# ── 여러 종목: 실험 003·004의 칸 나누기 방식 (빈 칸은 비교 기준 지수에 넣어 둔다) ──

def build_portfolio(data, dates, rules, mode, warmup=200):
    """종목별 시가·RSI·보유 조건(그날 종가 기준)을 날짜×종목 표로. data = {종목: 일봉}."""
    opn = pd.DataFrame({t: d["Open"] for t, d in data.items()}).reindex(dates)
    rsi = pd.DataFrame({t: rsi_wilder(d["Close"]) for t, d in data.items()}).reindex(dates)
    cond = {}
    for t, d in data.items():
        s = want(d, rules, mode)
        s.iloc[:warmup] = False  # 200일 평균 계산 전(상장 직후)에는 후보가 아님
        cond[t] = s
    return opn, rsi, pd.DataFrame(cond).reindex(dates, fill_value=False).astype(bool)


def simulate(opn, rsi, cond, bench_open, slots, cost, weekly, start):
    """매일(weekly면 매주 첫 거래일) 시가에, 전날 종가 기준 조건으로 팔고/사기.

    - 보유 종목의 조건이 풀리면 팔고, 판 돈은 지수로
    - 빈 칸이 있으면 조건 맞는 미보유 종목을 RSI 낮은 순으로 산다 (한 칸 = 그때 총자산/칸 수, 지수를 팔아 마련)
    반환: 시가 기준 자산 곡선, 보유 종목 수, 매수 수
    """
    week = opn.index.to_period("W")
    trade_day = np.asarray(week != np.roll(week, 1)) if weekly else np.ones(len(opn), bool)
    pos0 = int(np.searchsorted(opn.index, pd.Timestamp(start)))
    O = opn.ffill().fillna(0).to_numpy()  # 상장 전 빈 가격은 0 (0주 × 빈 값 = 빈 값으로 총자산이 깨지지 않게)
    listed = opn.notna().to_numpy()
    C, R, D = cond.to_numpy(), rsi.to_numpy(), bench_open.to_numpy()
    shares = np.zeros(opn.shape[1])
    idx = (1 - cost) / D[pos0]  # 지수 보유 수량
    eq, held_n, buys = [], [], []
    for i in range(pos0, len(O)):
        bought = 0
        if trade_day[i] and i > 0:
            for j in np.where((shares > 0) & ~C[i - 1])[0]:
                idx += shares[j] * O[i, j] * (1 - cost) * (1 - cost) / D[i]  # 종목 팔고 지수 사기
                shares[j] = 0
            value = shares @ O[i] + idx * D[i]
            free = slots - int((shares > 0).sum())
            cand = np.where(C[i - 1] & (shares == 0) & listed[i])[0]
            for j in cand[np.argsort(R[i - 1, cand])][:max(free, 0)]:
                amount = min(value / slots, idx * D[i])
                if amount <= 1e-12:
                    break
                idx -= amount / D[i]  # 지수 팔고 종목 사기
                shares[j] = amount * (1 - cost) * (1 - cost) / O[i, j]
                bought += 1
        total = shares @ O[i] + idx * D[i]
        assert (shares > 0).sum() <= slots and total > 0
        eq.append(total)
        held_n.append(int((shares > 0).sum()))
        buys.append(bought)
    ix = opn.index[pos0:]
    return pd.Series(eq, ix), pd.Series(held_n, ix), pd.Series(buys, ix)
