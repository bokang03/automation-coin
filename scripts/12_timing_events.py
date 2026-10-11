"""워뇨띠의 매수·매도 체결을 5분봉 차트 위에 올려 타이밍을 분석한다. XBTUSD 2020-01 ~ 2021-12.

  ① 체결이 봉의 어느 위치에서 일어났나 (저점 근처에서 샀나, 종가보다 싸게 샀나)
  ② 체결 직전·직후 가격 경로 (무엇을 보고 들어갔고, 들어간 뒤 얼마나 빨리 유리해졌나)
  ③ 진입(포지션 확대)과 청산(포지션 축소)을 나눠서 같은 분석
  ④ 지정가(메이커)와 시장가(테이커) 차이
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
START, END = "2020-01-01", "2021-12-31"

c = pd.read_parquet("data/candles_XBTUSD_5m.parquet")
c = c.set_index(c.timestamp.dt.tz_localize(None)).sort_index().loc["2019-12-01":"2022-01-01"]
lc = np.log(c.close)

tr = pd.read_parquet("data/trades_pos.parquet")
x = tr[(tr.symbol == "XBTUSD") & (tr.ts >= START) & (tr.ts <= END + " 23:59")].copy()
x["prev_pos"] = x.pos - x.sq
x["kind"] = np.where((x.prev_pos == 0) | (np.sign(x.prev_pos) == np.sign(x.sq)), "진입·확대", "축소·청산")
x["maker"] = x.lastliquidityind == "AddedLiquidity"
x["bar"] = x.ts.dt.ceil("5min")
w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"])
w["date"] = pd.to_datetime(w.date)
bal = w.groupby("date").walletbalance.last() / 1e8
x["eq_usd"] = bal.reindex(x.ts.dt.normalize()).ffill().to_numpy() * x.lastpx
x["size"] = x.lastqty / x.eq_usd                     # 체결 규모(자산 대비 배수)

# 5분봉 × 방향 × 진입/청산 단위로 묶은 '매매 사건'
g = x.groupby(["bar", "side", "kind"]).agg(usd=("lastqty", "sum"), size=("size", "sum"),
                                           px=("lastpx", lambda s: np.average(s, weights=x.loc[s.index, "lastqty"])),
                                           maker=("maker", lambda s: np.average(s, weights=x.loc[s.index, "lastqty"]))).reset_index()
g = g[g.bar.isin(c.index)]
g["sgn"] = np.where(g.side == "Buy", 1, -1)
print(f"체결 {len(x):,}건 → 5분봉 매매 사건 {len(g):,}건 (5분봉 {g.bar.nunique():,}개, 전체 5분봉의 {g.bar.nunique() / len(c.loc[START:END]):.1%})")
print(g.groupby(["side", "kind"]).agg(사건수=("usd", "size"), 거래대금_억달러=("usd", lambda s: s.sum() / 1e8), 지정가비중=("maker", "mean")).round(2))

pos_idx = c.index.get_indexer(g.bar)
hi, lo, cl, op = c.high.to_numpy(), c.low.to_numpy(), c.close.to_numpy(), c.open.to_numpy()


def wavg(v, wt):
    m = ~np.isnan(v)
    return np.average(v[m], weights=wt[m])


print("\n" + "=" * 110)
print("① 체결 위치: 그 5분봉·1시간 범위 안에서 어디서 체결됐나 (0 = 최저가, 1 = 최고가), 그리고 종가 대비 유리했나")
hour_end = g.bar.dt.ceil("h")
h1 = c.resample("1h", label="right", closed="right").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
hh, hl_, hc = (h1[k].reindex(hour_end).to_numpy() for k in ("high", "low", "close"))
g["pos5"] = (g.px - lo[pos_idx]) / np.where(hi[pos_idx] > lo[pos_idx], hi[pos_idx] - lo[pos_idx], np.nan)
g["pos60"] = (g.px - hl_) / np.where(hh > hl_, hh - hl_, np.nan)
g["edge_vs_5m_close"] = g.sgn * (cl[pos_idx] / g.px - 1)          # + = 그 봉 종가보다 유리하게 체결
g["edge_vs_1h_close"] = g.sgn * (hc / g.px - 1)                    # + = 그 시간 종가보다 유리하게 체결
t = g.groupby(["side", "kind"]).apply(lambda d: pd.Series({
    "5분봉 내 위치": wavg(d.pos5.to_numpy(), d.usd.to_numpy()), "1시간 범위 내 위치": wavg(d.pos60.to_numpy(), d.usd.to_numpy()),
    "5분봉 종가 대비 유리(bp)": wavg(d.edge_vs_5m_close.to_numpy(), d.usd.to_numpy()) * 1e4,
    "1시간 종가 대비 유리(bp)": wavg(d.edge_vs_1h_close.to_numpy(), d.usd.to_numpy()) * 1e4}), include_groups=False)
print(t.round(2))
mk = g.assign(유형=np.where(g.maker > 0.5, "지정가 위주", "시장가 위주")).groupby("유형").apply(lambda d: pd.Series({
    "사건수": len(d), "1시간 범위 내 위치(매수)": wavg(d.pos60[d.sgn > 0].to_numpy(), d.usd[d.sgn > 0].to_numpy()),
    "1시간 범위 내 위치(매도)": wavg(d.pos60[d.sgn < 0].to_numpy(), d.usd[d.sgn < 0].to_numpy()),
    "1시간 종가 대비 유리(bp)": wavg(d.edge_vs_1h_close.to_numpy(), d.usd.to_numpy()) * 1e4}), include_groups=False)
print(mk.round(2))

print("\n" + "=" * 110)
print("② 체결 전후 가격 경로: 체결가 대비 누적 수익률(bp), 워뇨띠의 방향 기준 (+ = 워뇨띠에게 유리). 거래대금 가중 평균")
offs = {"-6시간": -72, "-2시간": -24, "-1시간": -12, "-30분": -6, "-15분": -3, "-5분": -1, "체결봉 종가": 0, "+5분": 1, "+15분": 3,
        "+30분": 6, "+1시간": 12, "+2시간": 24, "+4시간": 48, "+8시간": 96, "+24시간": 288, "+72시간": 864}
lcv = lc.to_numpy()
n = len(lcv)


def path(d):
    idx = c.index.get_indexer(d.bar)
    out = {}
    for name, k in offs.items():
        j = np.clip(idx + k, 0, n - 1)
        out[name] = wavg(d.sgn.to_numpy() * (lcv[j] - np.log(d.px.to_numpy())), d.usd.to_numpy()) * 1e4
    return pd.Series(out)


print(g.groupby("kind").apply(path, include_groups=False).round(1).to_string())
print(g.groupby(["side", "kind"]).apply(path, include_groups=False).round(1).to_string())
print("\n(지정가/시장가별)")
print(g.assign(유형=np.where(g.maker > 0.5, "지정가 위주", "시장가 위주")).groupby(["kind", "유형"]).apply(path, include_groups=False).round(1).to_string())

print("\n" + "=" * 110)
print("③ 진입(확대) 직전 차트 상태: 직전 수익률과 최근 고점·저점 대비 위치 (거래대금 가중 중앙값에 가까운 평균)")
ent = g[g.kind == "진입·확대"].copy()
idx = c.index.get_indexer(ent.bar)
for name, k in {"15분": 3, "1시간": 12, "4시간": 48, "24시간": 288}.items():
    ent[f"직전 {name} 수익률(방향 기준, bp)"] = ent.sgn * (lcv[idx - 1] - lcv[np.clip(idx - 1 - k, 0, n - 1)]) * 1e4
for name, k in {"1시간": 12, "4시간": 48, "24시간": 288}.items():
    rh = c.high.rolling(k).max().to_numpy()[idx]
    rl = c.low.rolling(k).min().to_numpy()[idx]
    ent[f"{name} 범위 내 체결 위치(매수 0=저점, 매도 0=고점)"] = np.where(ent.sgn > 0, (ent.px - rl) / (rh - rl), (rh - ent.px) / (rh - rl))
cols = [k for k in ent.columns if "직전" in k or "범위 내 체결" in k]
print(ent.groupby("side").apply(lambda d: pd.Series({k: wavg(d[k].to_numpy(), d.usd.to_numpy()) for k in cols}), include_groups=False).round(2).T.to_string())
q = ent["4시간 범위 내 체결 위치(매수 0=저점, 매도 0=고점)"]
print("진입 거래대금의 4시간 범위 내 위치 분포:", {f"{a:.1f}~{b:.1f}": round(float(ent.usd[(q >= a) & (q < b + 1e-9)].sum() / ent.usd.sum()), 3)
                                    for a, b in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0))})

print("\n" + "=" * 110)
print("④ 한 번 들어갈 때 얼마나 나눠서, 얼마 동안 들어가나: 같은 방향 진입이 30분 이상 끊기지 않고 이어진 묶음")
e2 = ent.sort_values("bar")
grp = ((e2.sgn != e2.sgn.shift()) | (e2.bar.diff() > pd.Timedelta(minutes=30))).cumsum()
cl_ = e2.groupby(grp).agg(start=("bar", "first"), end=("bar", "last"), n=("usd", "size"), size=("size", "sum"), sgn=("sgn", "first"),
                          first_px=("px", "first"), vwap=("px", lambda s: np.average(s, weights=e2.loc[s.index, "usd"])))
cl_["dur_min"] = (cl_.end - cl_.start).dt.total_seconds() / 60 + 5
cl_["avg_vs_first_bp"] = cl_.sgn * (cl_.first_px / cl_.vwap - 1) * 1e4     # + = 처음보다 유리한 가격에 더 실음(물타기), - = 불리한 가격에 더 실음(불타기)
print(f"진입 묶음 {len(cl_):,}개 (하루 {len(cl_) / 731:.1f}개)")
print(cl_[["n", "dur_min", "size", "avg_vs_first_bp"]].describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
big = cl_[cl_["size"] >= cl_["size"].quantile(0.75)]
print(f"큰 묶음(상위 25%, 자산의 {big['size'].min():.2f}배 이상): 지속 중앙값 {big.dur_min.median():.0f}분, 5분봉 {big.n.median():.0f}개에 나눠 진입, "
      f"처음 가격 대비 평균가 {big.avg_vs_first_bp.median():+.1f}bp")
cl_.to_parquet("data/entry_clusters_2020_2021.parquet")
g.to_parquet("data/trade_events_5m_2020_2021.parquet")
