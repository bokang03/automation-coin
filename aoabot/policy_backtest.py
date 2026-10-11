"""학습한 정책(aoabot.policy)을 1시간봉 차트에 돌리는 백테스트. 실행 루프와 같은 decide/apply_fill 을 쓴다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aoabot import risk
from aoabot.features import make_features
from aoabot.policy import MARKET_FEATS, NONE, OwnState, apply_fill, decide_x

KST = pd.Timedelta(hours=9)


def simulate(candles: pd.DataFrame, bundle: dict, start=None, end=None, equity0: float = 10_000.0,
             fee: float = 0.0005, slip: float = 0.0002, lim: risk.Limits | None = None) -> pd.DataFrame:
    """돌려주는 표: 인덱스 = 봉 마감 시각, 열 = equity(자산), lev(자산 대비 포지션 배수), act(행동), px."""
    lim = lim or risk.Limits()
    f = make_features(candles).loc[start:end]
    M, PX = f[MARKET_FEATS].to_numpy(dtype=float), f.px.to_numpy()
    st, cash, rs = OwnState(), equity0, risk.RiskState()
    out = np.zeros((len(f), 3))

    def trade(dqty: float, px: float, t) -> None:
        nonlocal cash
        fp = px * (1 + np.sign(dqty) * slip)
        cash -= abs(dqty) * fp * fee
        cash += apply_fill(st, dqty, fp, t)

    for i, t in enumerate(f.index):
        px = PX[i]
        equity = cash + st.qty * (px - st.avg_px)
        risk.update(rs, (t + KST).strftime("%Y-%m-%d"), equity, lim)
        act = NONE
        if rs.halted:
            if st.qty:
                trade(-st.qty, px, t)
        else:
            dqty, act = decide_x(bundle, M[i], px, st, equity, t)
            if act != NONE and dqty:
                if abs(st.qty + dqty) > abs(st.qty) and not risk.can_open(rs, equity, lim)[0]:
                    act = NONE                      # 하루 손실 한도에 걸리면 포지션을 키우는 주문만 막는다
                else:
                    trade(dqty, px, t)
        equity = cash + st.qty * (px - st.avg_px)
        out[i] = equity, st.qty * px / equity if equity > 0 else 0.0, act
    return pd.DataFrame(out, index=f.index, columns=["equity", "lev", "act"]).assign(px=PX)


def stats(sim: pd.DataFrame) -> dict:
    eq, px = sim.equity, sim.px
    r = eq.pct_change().dropna()
    hold = px.iloc[-1] / px.iloc[0] - 1
    # 타이밍 없이 평균 포지션만큼만 들고 있었을 때의 수익: 매매 판단이 보탠 몫을 가려내는 기준
    passive = float((sim.lev.mean() * px.pct_change().fillna(0)).add(1).prod() - 1)
    return dict(수익=eq.iloc[-1] / eq.iloc[0] - 1, 샤프=r.mean() / r.std() * np.sqrt(8760) if r.std() > 0 else 0.0,
                최대낙폭=(eq / eq.cummax() - 1).min(), 단순보유=hold, 평균포지션=sim.lev.mean(),
                평균포지션만_보유=passive, 매매횟수_일=(sim.act != 0).mean() * 24,
                회전율_일=sim.lev.diff().abs().mean() * 24)
