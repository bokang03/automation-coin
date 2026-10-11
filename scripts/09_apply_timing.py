"""워뇨띠의 매매 빈도·시간대를 봇에 반영하고 검증한다.

규칙을 정할 때는 워뇨띠의 2018~2021 자료와 봇의 '진입 횟수'만 본다. 봇의 수익률은 보지 않는다.
  - 시간대: 워뇨띠의 라운드트립 진입 손익이 2018·2019·2020·2021 네 해 모두 플러스였던 4시간 구간(한국 시간)
  - 빈도: 2018~2021 구간에서 봇의 하루 진입 횟수가 워뇨띠의 2020~2021 하루 라운드트립 수와 가장 가까운 진입 기준
  - 하루 진입 상한: 워뇨띠가 거래한 날 중 2020~2021 하루 라운드트립 수의 90백분위
그다음 봇 성과를 2020~2026, 2022~2026(워뇨띠의 자료 이후)으로 본다.
"""
import sys
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.backtest import load_candles, run
from aoabot.strategy import Params

KST = pd.Timedelta(hours=9)
rt = pd.read_parquet("data/roundtrips.parquet")
rt = rt[rt.symbol == "XBTUSD"].copy()
rt["kst"] = rt.open + KST
rt["btc"] = np.where(rt.dir == "long", 1, -1) * rt.max_pos * (1 / rt.entry_vwap - 1 / rt.exit_vwap)

# 1) 시간대
rt["bucket"] = rt.kst.dt.hour // 4
pv = rt.pivot_table(index="bucket", columns=rt.kst.dt.year, values="btc", aggfunc="sum")
good = [b for b in pv.index if (pv.loc[b] > 0).all()]
hours = sorted(h for b in good for h in range(4 * b, 4 * b + 4))
print("4년 연속 이익 구간:", [f"{4 * b}-{4 * b + 3}시" for b in good], "→ 진입 허용 시각", hours)

# 2) 빈도 목표와 하루 상한
m = rt[(rt.kst >= "2020-01-01") & (rt.kst < "2022-01-01")]
target = len(m) / 731
per_day = m.groupby(m.kst.dt.normalize()).size()
cap = int(np.ceil(per_day.quantile(0.9)))
print(f"워뇨띠의 2020~2021 하루 라운드트립 {target:.2f}회, 거래한 날 90백분위 {cap}회 → 하루 진입 상한 {cap}")

BTC = load_candles(["data/candles_XBTUSD_1h.parquet", "data/candles_XBTUSD_1h_2021_2026.parquet"])
ETH = load_candles(["data/candles_ETHUSD_1h.parquet", "data/candles_ETHUSD_1h_2021_2026.parquet"])


def calibrate(base: Params, lim: risk.Limits) -> Params:
    best = None
    for th, vg in product([0.005, 0.01, 0.015, 0.02, 0.025, 0.03], [0.8, 1.0, 1.2, 1.4, 1.6]):
        p = replace(base, move_th=th, vol_gate=vg)
        eq, tr, _ = run(BTC, p, lim, "2018-04-01", "2021-12-31")
        rate = len(tr) / ((eq.index[-1] - eq.index[0]).days + 1)
        if best is None or abs(rate - target) < best[0]:
            best = (abs(rate - target), rate, p)
    print(f"  보정: 4시간 ±{best[2].move_th:.1%}, 변동성 레짐 ≥ {best[2].vol_gate} → 2018~2021 봇 진입 {best[1]:.2f}회/일")
    return best[2]


V1, L0 = Params(), risk.Limits()
L_cap = replace(L0, max_entries_per_day=cap)
print("빈도 보정 (시간대 필터 없음)")
p_f = calibrate(V1, L_cap)
print("빈도 보정 (시간대 필터 있음)")
p_tf = calibrate(replace(V1, entry_hours_kst=hours), L_cap)
variants = {
    "V1 현재 봇": (V1, L0),
    "T 시간대만 반영": (replace(V1, entry_hours_kst=hours), L0),
    "F 빈도만 반영": (p_f, L_cap),
    "TF 시간대+빈도": (p_tf, L_cap),
}

rows = []
for coin, c in (("BTC", BTC), ("ETH", ETH)):
    px = c.set_index(c.timestamp.dt.tz_localize(None)).close
    for pname, (s, e) in {"2020-01~2026-09": ("2020-01-01", None), "2022~2026 (이후)": ("2022-01-01", None)}.items():
        hold = px.loc[s:e]
        rows.append(dict(코인=coin, 기간=pname, 전략="단순 보유", 최종자산=10_000 * hold.iloc[-1] / hold.iloc[0],
                         샤프=np.nan, 최대낙폭=(hold / hold.cummax() - 1).min(), 진입_일=0.0, 승률=np.nan))
        for vname, (p, lim) in variants.items():
            eq, tr, _ = run(c, p, lim, s, e)
            r = eq.pct_change().dropna()
            rows.append(dict(코인=coin, 기간=pname, 전략=vname, 최종자산=eq.iloc[-1],
                             샤프=r.mean() / r.std() * np.sqrt(8760), 최대낙폭=(eq / eq.cummax() - 1).min(),
                             진입_일=len(tr) / ((eq.index[-1] - eq.index[0]).days + 1),
                             승률=(tr.pnl > 0).mean() if len(tr) else np.nan))
d = pd.DataFrame(rows)
d.to_csv("data/timing_variants.csv", index=False, encoding="utf-8-sig")
for (coin, pname), g in d.groupby(["코인", "기간"], sort=False):
    print(f"\n### {coin} {pname} (초기 10,000 USDT)")
    g = g.drop(columns=["코인", "기간"]).copy()
    g["수익"] = (g.최종자산 / 10_000 - 1).map("{:+.1%}".format)
    g["최종자산"] = g.최종자산.map("{:,.0f}".format)
    g["최대낙폭"] = g.최대낙폭.map("{:.1%}".format)
    print(g[["전략", "최종자산", "수익", "샤프", "최대낙폭", "진입_일", "승률"]].round(2).to_string(index=False))
