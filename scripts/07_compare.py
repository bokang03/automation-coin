"""진단(06_diagnose.py)의 문제점을 하나씩 고친 변형들을 같은 조건으로 비교한다.

변형과 파라미터는 결과를 보기 전에 진단 결과만 보고 정했다. 결과를 보고 고치지 않는다.
"""
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.backtest import load_candles, run
from aoabot.strategy import Params

V1 = Params()
LONG_HOLD = dict(max_hold_h=72, stop_atr=6.0)   # 진단 3: 워뇨띠의 수익은 1~3일 보유에서, 손절 ATR×3은 수익 거래도 잘랐다
VARIANTS = {
    "V1 현재 봇": (V1, 0.0005, 0.0002, False),
    "A 비용만 워뇨띠의 수준(지정가 전부 체결 가정)": (V1, 0.0002, 0.0, False),        # 진단 4
    "B 청산 완화(72시간, ATR×6)": (replace(V1, **LONG_HOLD), 0.0005, 0.0002, False),  # 진단 3
    "C 추세추종 + 청산 완화": (replace(V1, follow_trend=True, **LONG_HOLD), 0.0005, 0.0002, False),  # 진단 1
    "D 코인 1배 보유 + C": (replace(V1, follow_trend=True, **LONG_HOLD), 0.0005, 0.0002, True),  # 진단 6
}
PERIODS = {"2020-01~2026-09": ("2020-01-01", None), "2020~2021 (워뇨띠의 기간)": ("2020-01-01", "2021-12-31"),
           "2022~2026 (이후)": ("2022-01-01", None)}


def stats(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    return dict(총수익=eq.iloc[-1] / eq.iloc[0] - 1, 샤프=r.mean() / r.std() * np.sqrt(8760) if r.std() > 0 else 0.0,
                최대낙폭=(eq / eq.cummax() - 1).min())


def main():
    sets = {"BTC": ["data/candles_XBTUSD_1h.parquet", "data/candles_XBTUSD_1h_2021_2026.parquet"],
            "ETH": ["data/candles_ETHUSD_1h.parquet", "data/candles_ETHUSD_1h_2021_2026.parquet"]}
    rows = []
    for coin, paths in sets.items():
        c = load_candles(paths)
        px = c.set_index(c.timestamp.dt.tz_localize(None)).close
        for pname, (s, e) in PERIODS.items():
            hold = px.loc[s:e]
            rows.append(dict(코인=coin, 기간=pname, 전략="단순 보유(1배)", 거래=0, **stats(hold)))
            for vname, (p, fee, slip, core) in VARIANTS.items():
                eq, tr, _ = run(c, p, risk.Limits(), s, e, fee=fee, slip=slip)
                if core:  # 보유 1배 수익률 + 오버레이 수익률을 한 계좌에서 합친 것으로 근사
                    r = hold.reindex(eq.index).pct_change().fillna(0) + eq.pct_change().fillna(0)
                    eq = (1 + r).cumprod() * eq.iloc[0]
                rows.append(dict(코인=coin, 기간=pname, 전략=vname, 거래=len(tr), **stats(eq)))
    d = pd.DataFrame(rows)
    for (coin, pname), g in d.groupby(["코인", "기간"], sort=False):
        print(f"\n### {coin} {pname}")
        g = g.drop(columns=["코인", "기간"]).copy()
        g["총수익"] = g.총수익.map("{:+.1%}".format)
        g["최대낙폭"] = g.최대낙폭.map("{:.1%}".format)
        g["샤프"] = g.샤프.map("{:.2f}".format)
        print(g.to_string(index=False))
    d.to_csv("data/compare_variants.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
