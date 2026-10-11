"""워뇨띠의 하루 매매 빈도와 매매 시각(한국 시간)을 조사한다. XBTUSD 기준."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
KST = pd.Timedelta(hours=9)

tr = pd.read_parquet("data/trades_pos.parquet")
x = tr[tr.symbol == "XBTUSD"].copy()
x["kst"] = x.ts + KST
x["day"] = x.kst.dt.normalize()
x["btc"] = x.lastqty / x.lastpx                    # 체결 규모(BTC)
x["flip"] = np.sign(x.pos) != np.sign(x.pos.shift().fillna(0))  # 롱↔숏↔무포 전환

# 라운드트립(03_positions.py 결과) + BTC 손익 추정
rt = pd.read_parquet("data/roundtrips.parquet")
rt = rt[rt.symbol == "XBTUSD"].copy()
rt["kst_open"] = rt.open + KST
rt["btc"] = np.where(rt.dir == "long", 1, -1) * rt.max_pos * (1 / rt.entry_vwap - 1 / rt.exit_vwap)

print("=" * 100)
print("① 하루 매매 빈도 (거래한 날 기준, 중앙값)")
d = x.groupby("day").agg(주문=("orderid", "nunique"), 체결=("lastqty", "size"), 전환=("flip", "sum"), 거래BTC=("btc", "sum"))
d["year"] = d.index.year
days_total = x.groupby(x.day.dt.year).day.agg(lambda s: (s.max() - s.min()).days + 1)
g = d.groupby("year").agg(거래한날=("주문", "size"), 주문=("주문", "median"), 체결=("체결", "median"), 방향전환=("전환", "median"), 하루거래BTC=("거래BTC", "median"))
g["거래일비율"] = g.거래한날 / days_total.values
rtd = rt.groupby(rt.kst_open.dt.year).size() / days_total.values
g["라운드트립/일"] = rtd.values
print(g.round(2))

print("=" * 100)
print("② 시간대 (한국 시간): 주문 비중, 거래량 비중, 라운드트립 진입 비중과 손익")
x["hour"] = x.kst.dt.hour
orders = x.drop_duplicates("orderid")
h = pd.DataFrame({
    "주문비중": orders.hour.value_counts(normalize=True).sort_index(),
    "거래량비중": x.groupby("hour").btc.sum() / x.btc.sum(),
    "진입건수": rt.groupby(rt.kst_open.dt.hour).size(),
    "진입손익BTC": rt.groupby(rt.kst_open.dt.hour).btc.sum(),
    "승률": rt.groupby(rt.kst_open.dt.hour).btc.apply(lambda s: (s > 0).mean()),
})
h["손익비중"] = h.진입손익BTC / rt.btc.sum()
print(h.round(3).to_string())
q = h.거래량비중
print(f"가장 활발한 4시간: {q.nlargest(4).index.tolist()}시  / 가장 조용한 4시간: {q.nsmallest(4).index.tolist()}시")

# 연도별로 같은 시간대 패턴이 유지되는지 (진입 손익을 6개 구간으로)
rt["hb"] = pd.cut(rt.kst_open.dt.hour, [-1, 3, 7, 11, 15, 19, 23], labels=["0-3시", "4-7시", "8-11시", "12-15시", "16-19시", "20-23시"])
print("\n연도별 시간대 진입 손익(BTC):")
print(rt.pivot_table(index="hb", columns=rt.kst_open.dt.year, values="btc", aggfunc="sum", observed=True).round(0))
print("\n연도별 시간대 진입 건수:")
print(rt.pivot_table(index="hb", columns=rt.kst_open.dt.year, values="btc", aggfunc="size", observed=True))

print("=" * 100)
print("③ 요일 (한국 시간)")
names = "월화수목금토일"
wd = pd.DataFrame({
    "거래량비중": x.groupby(x.kst.dt.dayofweek).btc.sum() / x.btc.sum(),
    "진입건수": rt.groupby(rt.kst_open.dt.dayofweek).size(),
    "진입손익BTC": rt.groupby(rt.kst_open.dt.dayofweek).btc.sum(),
})
wd.index = [names[i] for i in wd.index]
print(wd.round(3).to_string())

print("=" * 100)
print("④ 펀딩비 정산 시각(한국 13·21·05시) 전후 30분의 거래량")
m = x.kst.dt.minute + 60 * x.kst.dt.hour
funding = np.array([5, 13, 21]) * 60
dist = np.min(np.abs(((m.values[:, None] - funding[None, :]) + 720) % 1440 - 720), axis=1)
near = dist <= 30
print(f"정산 ±30분(하루의 {3 * 60 / 1440:.1%} 시간)에 거래량 {x.btc[near].sum() / x.btc.sum():.1%}")

print("=" * 100)
print("⑤ 매매 사이 간격: 라운드트립 종료 → 다음 진입까지")
gap = (rt.open.shift(-1) - rt.close).dt.total_seconds() / 3600
print(rt.assign(gap=gap).groupby(rt.open.dt.year).gap.describe(percentiles=[.25, .5, .75]).round(2))

print("=" * 100)
print("⑥ 몇 시에 포지션을 들고 있어서 돈을 벌었나 (시간봉 보유 손익, 한국 시간)")
import sys
sys.path.insert(0, ".")
c = pd.read_parquet("data/candles_XBTUSD_1h.parquet")
px = c.set_index(c.timestamp.dt.tz_localize(None)).close
pos = x.set_index("ts").pos.resample("1h", label="right", closed="right").last().reindex(px.index).ffill().fillna(0)
hp = (pos * (1 / px - 1 / px.shift(-1))).loc["2018-04":"2021-12"]    # 봉 t~t+1 동안의 보유 손익(BTC)
hk = ((hp.index + KST).hour)
s = hp.groupby(hk).sum()
print(pd.DataFrame({"보유손익BTC": s.round(0), "비중": (s / s.sum()).round(3)}).T.to_string())
