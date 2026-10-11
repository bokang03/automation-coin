"""월봉 조건(직전 달이 크게 내렸으면 사지 않는다)의 효과를 이미 받아 둔 차트로 본다. 사후 확인용이며, 채택 여부는 새 차트로 따로 검증한다.

조건의 기준값은 워뇨띠의 차트(2020~2021)에서 나온 것이다: 직전 월봉 수익률 하위 20% 경계(-7.13%) 이하에서만 봇 신호 뒤 수익이 음수였다
(scripts/20_tf_analysis.py)."""
import glob
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot import risk
from aoabot.policy import ROOT, load_bundle
from aoabot.timing import TimingParams, simulate, stats

pd.set_option("display.width", 250)
THR = -0.0713
b = load_bundle(ROOT / "models" / "aoa_timing.json")
LIM, NOL = risk.Limits(daily_loss=0.03, max_drawdown=0.25), risk.Limits(daily_loss=9.9, max_drawdown=9.9)
base, filt = TimingParams(), TimingParams(min_prev_month=THR)
LENGTH = pd.Timedelta(days=365)

jobs = []       # (묶음, 이름, 캔들, 시작, 끝)
for coin in ("BTC", "ETH"):
    c = pd.read_parquet(f"data/binance_{coin}_5m_202110_202610.parquet")
    jobs.append(("BTC·ETH 2022~2023 (보완용)", coin, c[c.timestamp < "2024-01-02"], "2022-01-01", "2023-12-31"))
    jobs.append(("BTC·ETH 2024~ (최종 시험에 쓴 구간)", coin, c[c.timestamp >= "2023-10-01"], "2024-01-01", None))
groups = {"무작위 알트 seed 7": "data/final_test_seed7.csv", "무작위 알트 seed 11": "data/final_test_seed11.csv",
          "워뇨띠가 거래한 코인": "data/his_coins_test.csv", "거래대금 큰 알트 seed 23": "data/liquid_alts_seed23.csv"}
files = {os.path.basename(f)[:-8]: f for f in glob.glob("data/random5m/*.parquet")}
for gname, path in groups.items():
    d = pd.read_csv(path)
    for chart in d.차트[~d.차트.str.startswith(("BTC ", "ETH "))].unique():
        sym = chart.split(" ")[0]
        cands = [k for k in files if k.rsplit("_", 1)[0] == sym]
        if not cands:
            continue
        m = pd.Series(chart).str.extract(r"(\d{4}-\d{2}-\d{2})")[0].iloc[0]
        if isinstance(m, str):                      # 무작위 차트: 시작일부터 1년
            start = pd.Timestamp(m)
            key = min(cands, key=lambda k: abs((pd.Timestamp(k.rsplit("_", 1)[1]) - start).days))
            jobs.append((gname, sym, pd.read_parquet(files[key]), start, start + LENGTH))
        else:                                       # 워뇨띠 코인: 2024-01 부터 끝까지
            key = min(cands, key=lambda k: pd.Timestamp(k.rsplit("_", 1)[1]))
            jobs.append((gname, sym, pd.read_parquet(files[key]), "2024-01-01", None))

rows = []
for gname, sym, c, s, e in jobs:
    for name, p in (("조건 없음", base), ("월봉 조건", filt)):
        for lname, lim in (("안전장치 있음", LIM), ("안전장치 없음", NOL)):
            sim, tr = simulate(c, b, p, s, e, lim=lim)
            rows.append(dict(묶음=gname, 코인=sym, 조건=name, 안전장치=lname, **stats(sim, tr)))
d = pd.DataFrame(rows)
d.to_csv("data/month_filter_check.csv", index=False, encoding="utf-8-sig")
g = d.groupby(["묶음", "안전장치", "조건"], sort=False).agg(차트수=("수익", "size"), 평균=("수익", "mean"), 중앙값=("수익", "median"),
                                                  수익난차트=("수익", lambda s_: f"{(s_ > 0).sum()}/{len(s_)}"), 평균낙폭=("최대낙폭", "mean"), 평균거래수=("거래수", "mean"))
for k in ("평균", "중앙값"):
    g[k] = g[k].map("{:+.1%}".format)
g["평균낙폭"] = g.평균낙폭.map("{:.1%}".format)
print(g.round(0).to_string())
