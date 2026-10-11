"""1시간·일·주·월봉에서 워뇨띠가 어느 지점에 사고 팔았나, 그리고 그 지점에서 산 뒤 가격이 어떻게 됐나. XBTUSD 2020~2021.

  ① 봉 안의 위치별로 매수·매도가 평소의 몇 배였나
  ② 그 위치에서 워뇨띠가 크게 산 뒤, 그리고 봇 신호 뒤 4시간 수익률 (시장 평균 흐름 제외)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot.features_tf import PARTS, TF_NAMES, TFS, make_tf_features
from aoabot.policy import ROOT, load_bundle

pd.set_option("display.width", 250)
PART_NAMES = {"ret": "진행 중인 봉의 수익률", "pos": "진행 중인 봉 범위 안 위치(0=저가,1=고가)", "prev_ret": "직전 봉 수익률",
              "prev_pos": "직전 봉 범위 대비 위치(0=전 저가,1=전 고가)", "elapsed": "봉 진행 정도(0=시작,1=마감)"}

c = pd.read_parquet("data/candles_XBTUSD_5m.parquet")
c = c[(c.timestamp >= "2019-10-01") & (c.timestamp < "2022-01-01")]
tf = make_tf_features(c)
o = pd.read_parquet("data/timing_oof_2020_2021.parquet").join(tf)
ev = pd.read_parquet("data/trade_events_5m_2020_2021.parquet")
bs = ev[ev.side == "Buy"].groupby("bar")["size"].sum().reindex(o.index).fillna(0).to_numpy()
buy, sell = (o.act == 1).to_numpy(), (o.act == -1).to_numpy()
big = buy & (bs >= np.median(bs[buy]))
b = load_bundle(ROOT / "models" / "aoa_timing.json")
strong = (o.p_buy >= b["thr_buy"]).to_numpy()
prior = pd.Series(strong.astype(float)).rolling(48, min_periods=1).sum().to_numpy() - strong
bot = strong & (prior >= 1)                                      # 현재 봇이 실제로 진입하는 신호
lc = np.log(o.px.to_numpy())
n, K = len(o), 48
fwd = np.full(n, np.nan)
fwd[:-K] = (lc[K:] - lc[:-K] - np.mean(lc[K:] - lc[:-K])) * 1e4   # 4시간 뒤 수익률(bp), 평균 흐름 제외
print(f"5분봉 {n:,}개. 워뇨띠 매수 봉 {buy.sum():,} (그중 크게 산 봉 {big.sum():,}), 매도 봉 {sell.sum():,}, 봇 진입 신호 {bot.sum():,}")

rows = []
for t in TFS:
    print("\n" + "=" * 100 + f"\n{TF_NAMES[t]}")
    for part in PARTS:
        col = o[f"{t}_{part}"]
        q = pd.qcut(col, 5, duplicates="drop")
        g = pd.DataFrame({"구간": q, "buy": buy, "big": big, "sell": sell, "bot": bot, "fwd": fwd})
        tab = g.groupby("구간", observed=True).agg(봉수=("buy", "size"), 매수배수=("buy", "mean"), 큰매수배수=("big", "mean"), 매도배수=("sell", "mean"))
        tab[["매수배수", "큰매수배수", "매도배수"]] /= [buy.mean(), big.mean(), sell.mean()]
        tab["큰매수 뒤 4시간(bp)"] = g[g.big].groupby("구간", observed=True).fwd.mean()
        tab["봇신호 뒤 4시간(bp)"] = g[g.bot].groupby("구간", observed=True).fwd.mean()
        tab["봇신호수"] = g[g.bot].groupby("구간", observed=True).fwd.size()
        print(f"\n[{PART_NAMES[part]}]")
        print(tab.round(2).to_string())
        for i, (iv, r) in enumerate(tab.iterrows()):
            rows.append(dict(봉=TF_NAMES[t], 항목=PART_NAMES[part], 구간=str(iv), 순위=i, **r.to_dict()))
res = pd.DataFrame(rows)
res.to_csv("data/tf_analysis.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("요약: 워뇨띠의 큰 매수가 가장 몰린 지점 (평소 대비 배수 상위)")
print(res.nlargest(10, "큰매수배수")[["봉", "항목", "구간", "큰매수배수", "매도배수", "큰매수 뒤 4시간(bp)"]].round(2).to_string(index=False))
print("\n요약: 워뇨띠의 매도가 가장 몰린 지점")
print(res.nlargest(8, "매도배수")[["봉", "항목", "구간", "매도배수", "큰매수배수"]].round(2).to_string(index=False))
print("\n요약: 봇 신호 뒤 수익이 구간에 따라 가장 크게 갈리는 항목 (가장 좋은 구간 − 가장 나쁜 구간, 신호 50개 이상인 구간만)")
ok = res[res.봇신호수 >= 50]
sp = ok.groupby(["봉", "항목"])["봇신호 뒤 4시간(bp)"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
print(sp.head(8).round(1).to_string())
