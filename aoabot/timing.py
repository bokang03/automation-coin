"""워뇨띠의 5분봉 매수 타이밍으로 매매하는 전략과 그 백테스트.

분석(scripts/12, 14, 15)에서 나온 것:
  - 워뇨띠의 매수는 거래량이 터지며 급락한 봉에 몰려 있고, 그 뒤 1~4시간 동안 가격이 반등한다.
  - 매도(숏)에는 단기 타이밍 우위가 없다. 워뇨띠의 숏은 보유 BTC의 헤지였다.
그래서 이 전략은 롱만 한다: 매수 신호 봉이 끝나면 사고, 정해진 시간이 지나면 판다.

신호는 models/aoa_timing.json (scripts/14_learn_timing.py). 백테스트와 실행 루프가 TimingParams 와 signals() 를 같이 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aoabot import risk
from aoabot.features5 import FEATS_5M, make_features_5m
from aoabot.features_tf import TF_FEATS, make_tf_features

KST = pd.Timedelta(hours=9)


@dataclass
class TimingParams:
    size: float = 0.42          # 한 번 진입 크기(자산 대비 배수)
    max_lev: float = 0.42       # 포지션 상한. size 보다 크면 신호가 이어질 때 추가 진입한다
    hold_bars: int = 48         # 마지막 진입 후 이만큼 지나면 청산 (5분봉 48개 = 4시간)
    entry: str = "taker"        # taker: 신호 봉 종가에 시장가 / maker: 신호 봉 종가에 지정가를 걸어 둔다
    maker_wait: int = 3         # 지정가가 이만큼(봉) 안 채워지면 취소
    exit_on_sell: bool = False  # 매도 신호가 나오면 보유 시간 전이라도 청산
    stop: float = 0.0           # 진입 평균가 대비 이만큼 떨어지면 손절(0 = 없음)
    min_prior: int = 1          # 직전 prior_window 봉 안에 매수 신호가 이만큼 먼저 있었어야 진입 (폭락의 첫 신호는 건너뛴다)
    prior_window: int = 48
    min_prev_month: float | None = None   # 직전 월봉 수익률이 이 값 이하이면 진입하지 않는다 (None = 조건 없음)
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    slip: float = 0.0002


def build_features(c: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """번들(학습된 신호)이 쓰는 입력을 만든다. 1시간·일·주·월봉 위치를 쓰는 번들이면 그것도 붙인다."""
    f = make_features_5m(c)
    if any(k in TF_FEATS for k in bundle.get("feats", FEATS_5M)):
        f = f.join(make_tf_features(c))
    return f


def signals(bundle: dict, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(매수 신호, 매도 신호) 불리언 배열. f: build_features 결과."""
    p = bundle["model"].predict_proba(f[bundle.get("feats", FEATS_5M)].to_numpy(dtype=float))
    ok = f["vol_regime"].notna().to_numpy()
    k = bundle.get("thr_scale", 1.0)
    return ok & (p[:, 2] >= bundle["thr_buy"] * k), ok & (p[:, 0] >= bundle["thr_sell"] * k)


def simulate(c: pd.DataFrame, bundle: dict, p: TimingParams, start=None, end=None, equity0: float = 10_000.0,
             lim: risk.Limits | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(봉별 자산·포지션 표, 거래 목록)을 돌려준다."""
    lim = lim or risk.Limits()
    f = build_features(c, bundle)
    hl = pd.DataFrame({"high": c.high.to_numpy(), "low": c.low.to_numpy()}, index=f.index)
    f = f.join(hl).loc[start:end]
    buy, sell = signals(bundle, f)
    PX, HI, LO = f.px.to_numpy(), f.high.to_numpy(), f.low.to_numpy()
    prior = pd.Series(buy.astype(float)).rolling(p.prior_window, min_periods=1).sum().to_numpy() - buy   # 이번 봉 이전의 신호 수
    if p.min_prev_month is not None:              # 월봉 조건: 직전 달이 크게 내린 뒤에는 사지 않는다 (자료가 없으면 사지 않는다)
        pm = f["1M_prev_ret"] if "1M_prev_ret" in f else make_tf_features(c)["1M_prev_ret"].reindex(f.index)
        buy = buy & (pm.to_numpy() > p.min_prev_month)
    T = f.index
    kst_day = (T + KST).strftime("%Y-%m-%d")

    cash, qty, avg, last_entry, first_entry = equity0, 0.0, 0.0, -1, -1
    pend = None                                   # 걸어 둔 지정가 매수: (가격, 수량, 만료 봉)
    rs = risk.RiskState()
    eq_out, lev_out, trades = np.zeros(len(f)), np.zeros(len(f)), []

    def close_pos(i, px, reason, fee):
        nonlocal cash, qty, avg, last_entry, first_entry
        cash += qty * (px - avg) - qty * px * fee
        trades.append(dict(opened=T[first_entry], closed=T[i], entry=avg, exit=px, qty=qty, ret=px / avg - 1, reason=reason,
                           bars=i - first_entry))
        qty, avg, last_entry, first_entry = 0.0, 0.0, -1, -1

    for i in range(len(f)):
        px = PX[i]
        if pend is not None:                      # 지정가 체결 확인: 가격이 지정가 아래로 내려와야 채워진 것으로 본다
            lpx, lq, exp = pend
            if LO[i] < lpx:
                cash -= lq * lpx * p.maker_fee
                avg = (avg * qty + lpx * lq) / (qty + lq)
                qty += lq
                last_entry = i
                first_entry = i if first_entry < 0 else first_entry
                pend = None
            elif i >= exp:
                pend = None
        if qty and p.stop and LO[i] <= avg * (1 - p.stop):
            close_pos(i, min(avg * (1 - p.stop), HI[i]) * (1 - p.slip), "손절", p.taker_fee)
        equity = cash + qty * (px - avg)
        risk.update(rs, kst_day[i], equity, lim)
        if rs.halted:
            if qty:
                close_pos(i, px * (1 - p.slip), "최대 낙폭 정지", p.taker_fee)
            pend = None
        else:
            if qty and (i - last_entry >= p.hold_bars or (p.exit_on_sell and sell[i])):
                close_pos(i, px * (1 - p.slip), "매도 신호" if (p.exit_on_sell and sell[i] and i - last_entry < p.hold_bars) else "시간 경과", p.taker_fee)
            if buy[i] and prior[i] >= p.min_prior and pend is None and risk.can_open(rs, equity, lim)[0]:
                add = min(p.size, p.max_lev - qty * px / equity)
                if add > 1e-9:
                    q = add * equity / px
                    if p.entry == "maker":
                        pend = (px, q, i + p.maker_wait)
                    else:
                        fp = px * (1 + p.slip)
                        cash -= q * fp * p.taker_fee
                        avg = (avg * qty + fp * q) / (qty + q)
                        qty += q
                        last_entry = i
                        first_entry = i if first_entry < 0 else first_entry
        equity = cash + qty * (px - avg)
        eq_out[i], lev_out[i] = equity, qty * px / equity if equity > 0 else 0.0
    sim = pd.DataFrame({"equity": eq_out, "lev": lev_out, "px": PX, "buy": buy, "sell": sell}, index=T)
    return sim, pd.DataFrame(trades)


def stats(sim: pd.DataFrame, tr: pd.DataFrame) -> dict:
    eq = sim.equity
    r = eq.pct_change().dropna()
    days = max((sim.index[-1] - sim.index[0]).days, 1)
    return dict(수익=eq.iloc[-1] / eq.iloc[0] - 1, 샤프=r.mean() / r.std() * np.sqrt(288 * 365) if r.std() > 0 else 0.0,
                최대낙폭=(eq / eq.cummax() - 1).min(), 단순보유=sim.px.iloc[-1] / sim.px.iloc[0] - 1,
                거래수=len(tr), 거래_일=len(tr) / days, 승률=float((tr.ret > 0).mean()) if len(tr) else np.nan,
                거래당_수익bp=float(tr.ret.mean() * 1e4) if len(tr) else np.nan, 시장노출시간=float((sim.lev > 0).mean()))
