"""봇의 타이밍과 워뇨띠의 실제 매수가 어디서 다른지 찾는다 (워뇨띠의 차트 2020~2021, 학습에 안 쓴 분기 예측 사용)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot.policy import ROOT, load_bundle

pd.set_option("display.width", 250)
b = load_bundle(ROOT / "models" / "aoa_timing.json")
o = pd.read_parquet("data/timing_oof_2020_2021.parquet")
ev = pd.read_parquet("data/trade_events_5m_2020_2021.parquet")
buy_ev = ev[ev.side == "Buy"].groupby("bar").agg(size=("size", "sum"), maker=("maker", "mean"), kind=("kind", "first"))
o = o.join(buy_ev)
lc = np.log(o.px.to_numpy())
n = len(o)
HZ = {"+1시간": 12, "+4시간": 48, "+24시간": 288}


def fwd(mask, weights=None):
    idx = np.flatnonzero(mask)
    out = {"봉수": len(idx)}
    for name, k in HZ.items():
        j = idx[idx + k < n]
        v = (lc[j + k] - lc[j] - np.mean(lc[k:] - lc[:-k])) * 1e4
        w = None if weights is None else weights[j]
        out[name] = round(float(np.average(v, weights=w)), 1)
    return out


his = (o.act == 1).to_numpy()
sig = (o.p_buy >= b["thr_buy"]).to_numpy()
size = o["size"].fillna(0).to_numpy()
print("① 워뇨띠의 매수를 크기별로: 크게 산 곳이 더 잘 올랐나 (이후 수익률 bp, 시장 평균 흐름 제외)")
q = pd.qcut(o["size"][his], 4, labels=["작게(하위 25%)", "중하", "중상", "크게(상위 25%)"])
rows = {}
for lab in q.cat.categories:
    m = np.zeros(n, bool); m[np.flatnonzero(his)[(q == lab).to_numpy()]] = True
    rows[f"{lab} 중앙 {o['size'][m].median():.2f}배"] = fwd(m)
rows["전체(같은 비중)"] = fwd(his); rows["전체(크기 가중)"] = fwd(his, size)
print(pd.DataFrame(rows).T.to_string())

print("\n② 봇 매수 신호를 '워뇨띠도 ±30분 안에 산 곳'과 '워뇨띠는 안 산 곳'으로 나누면")
near = pd.Series(his.astype(float), index=o.index).rolling(13, center=True, min_periods=1).max().to_numpy() > 0
print(pd.DataFrame({"봇 신호 + 워뇨띠도 삼": fwd(sig & near), "봇 신호, 워뇨띠는 안 삼": fwd(sig & ~near),
                    "워뇨띠만 삼(봇 신호 없음)": fwd(his & ~pd.Series(sig.astype(float), index=o.index).rolling(13, center=True, min_periods=1).max().to_numpy().astype(bool))}).T.to_string())

print("\n③ 연속 신호 안에서의 순서: 직전 4시간 안에 신호가 몇 번 있었나")
prev = pd.Series(sig.astype(float), index=o.index).rolling(48, min_periods=1).sum().to_numpy() - sig
rows = {"첫 신호(직전 4시간에 없음)": fwd(sig & (prev == 0)), "2~3번째": fwd(sig & (prev >= 1) & (prev <= 2)), "4번째 이후": fwd(sig & (prev >= 3))}
print(pd.DataFrame(rows).T.to_string())
hp = pd.Series(his.astype(float), index=o.index).rolling(48, min_periods=1).sum().to_numpy() - his
print("워뇨띠의 매수 중 '직전 4시간에 이미 산 적 있는' 비율:", f"{(hp[his] > 0).mean():.0%}", " / 봇 신호 중 '직전 4시간에 이미 신호가 있던' 비율:", f"{(prev[sig] > 0).mean():.0%}")
big = his & (size >= np.quantile(size[his], 0.75))
ps = pd.Series(sig.astype(float), index=o.index).rolling(48, min_periods=1).sum().to_numpy()
print(f"워뇨띠가 크게 산 봉 직전 4시간의 봇 신호 수: 중앙값 {np.median(ps[big]):.0f}, 평균 {ps[big].mean():.1f}  (작게 산 봉: 평균 {ps[his & ~big].mean():.1f})")

print("\n④ 신호 강도(확률)별: 확신이 높을수록 잘 오르나")
ps_ = o.p_buy.to_numpy()
edges = np.quantile(ps_[sig], [0, .5, .8, .95, 1.0])
rows = {}
for a, c_, lab in zip(edges[:-1], edges[1:], ["하위 50%", "50~80%", "80~95%", "상위 5%"]):
    m = sig & (ps_ >= a) & (ps_ <= c_)
    rows[f"신호 강도 {lab} (확률 {a:.2f}~{c_:.2f})"] = {**fwd(m), "워뇨띠도 산 비율": f"{near[m].mean():.0%}", "그때 워뇨띠 매수 크기": round(float(np.median(size[m & his])) if (m & his).any() else 0, 2)}
print(pd.DataFrame(rows).T.to_string())

print("\n⑤ 워뇨띠는 급락 매수 뒤 얼마나 들고 있었나: 크게 산 봉 이후 포지션 변화")
tr = pd.read_parquet("data/trades_pos.parquet")
x = tr[(tr.symbol == "XBTUSD") & (tr.ts >= "2020-01-01")].set_index("ts").pos.resample("5min", label="right", closed="right").last().reindex(o.index).ffill()
w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"]); w["date"] = pd.to_datetime(w.date)
eq = (w.groupby("date").walletbalance.last() / 1e8).reindex(o.index.normalize()).ffill().to_numpy() * o.px.to_numpy()
lev = (x / eq).to_numpy()
idx = np.flatnonzero(big)
for name, k in {"+1시간": 12, "+4시간": 48, "+8시간": 96, "+24시간": 288, "+72시간": 864}.items():
    j = idx[idx + k < n]
    d = lev[j + k] - lev[j]
    print(f"  {name}: 포지션 변화 중앙값 {np.median(d):+.2f}배, 산 만큼 이상 되판 비율 {(d <= -size[j] * 0.9).mean():.0%}, 더 산 비율 {(d >= size[j] * 0.5).mean():.0%}")
