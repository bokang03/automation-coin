"""타이밍 전략의 변형을 보완용 구간(2022~2023)에서만 비교한다. 2024년 이후와 무작위 차트는 최종 시험용으로 남겨 둔다."""
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.policy import ROOT, load_bundle
from aoabot.timing import TimingParams, simulate, stats

pd.set_option("display.width", 250)
DEV = ("2022-01-01", "2023-12-31")
NOL = risk.Limits(daily_loss=9.9, max_drawdown=9.9)
b = load_bundle(ROOT / "models" / "aoa_timing.json")
b_all = {**b, "thr_buy": b["thr_buy_all"]}                                   # 1차: 워뇨띠의 모든 매수 빈도에 맞춘 기준선
charts = {"BTC": pd.read_parquet("data/binance_BTC_5m_202110_202610.parquet"), "ETH": pd.read_parquet("data/binance_ETH_5m_202110_202610.parquet")}
charts = {k: v[v.timestamp < "2024-01-02"] for k, v in charts.items()}      # 최종 시험 구간은 읽지도 않는다

base = TimingParams(size=round(b["meta"]["big_buy_size"], 2), max_lev=round(b["meta"]["big_buy_size"], 2), min_prior=0)
variants = {
    "1차: 모든 신호, 첫 신호에 진입": (b_all, base),
    "2차: 강한 신호만": (b, base),
    "3차: 강한 신호 + 폭락이 이어진 뒤(직전 4시간에 신호 1회 이상)": (b, replace(base, min_prior=1)),
    "3차 + 지정가 진입": (b, replace(base, min_prior=1, entry="maker")),
    "3차, 수수료 0 (신호 자체의 힘)": (b, replace(base, min_prior=1, taker_fee=0, slip=0)),
    "3차, 2시간 보유": (b, replace(base, min_prior=1, hold_bars=24)),
    "3차, 8시간 보유": (b, replace(base, min_prior=1, hold_bars=96)),
    "3차, 24시간 보유": (b, replace(base, min_prior=1, hold_bars=288)),
    "3차 + 매도 신호에 청산": (b, replace(base, min_prior=1, exit_on_sell=True)),
}
rows = []
for name, (bundle, p) in variants.items():
    for coin, c in charts.items():
        sim, tr = simulate(c, bundle, p, *DEV, lim=NOL)
        rows.append(dict(변형=name, 코인=coin, **stats(sim, tr)))
d = pd.DataFrame(rows)
for k in ("수익", "최대낙폭", "단순보유", "승률", "시장노출시간"):
    d[k] = d[k].map("{:+.1%}".format if k in ("수익", "단순보유") else "{:.1%}".format)
print(d.round(2).to_string(index=False))
