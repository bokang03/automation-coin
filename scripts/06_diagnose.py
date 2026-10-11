"""봇 백테스트가 왜 손실인지, 봇 규칙과 워뇨띠의 실제 XBTUSD 매매를 대조한다."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aoabot.features import make_features

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# ---------- 준비: 체결, 시간봉, 시간별 포지션·순노출 ----------
tr = pd.read_parquet("data/trades_pos.parquet")
x = tr[tr.symbol == "XBTUSD"].reset_index(drop=True)
c = pd.read_parquet("data/candles_XBTUSD_1h.parquet")
f = make_features(c)
hl = c.set_index(c.timestamp.dt.tz_localize(None))[["high", "low"]]
f = f.join(hl)
w = pd.read_parquet("data/wallet.parquet").dropna(subset=["date"])
w["date"] = pd.to_datetime(w.date)
bal = (w.groupby("date").walletbalance.last() / 1e8)

pos_h = x.set_index("ts").pos.resample("1h", label="right", closed="right").last().reindex(f.index).ffill().fillna(0)
bal_h = bal.reindex(f.index.normalize()).ffill().values
f["pos"] = pos_h.values
f["net"] = 1 + f.pos / (bal_h * f.px)          # 순 달러 노출(배수): BTC 증거금 + 인버스 계약
f = f.loc["2018-04":"2021-12"]

# ---------- 라운드트립 재구성 (BTC 손익 포함) ----------
sq, px, ts, pos = x.sq.values, x.lastpx.values, x.ts.values, x.pos.values
rts, cur, prev = [], None, 0.0
for i in range(len(x)):
    p = pos[i]
    if prev == 0 and p != 0:
        cur = dict(open=ts[i], dir=np.sign(p), btc=0.0, qty_in=0.0, cost_in=0.0, max_pos=0.0)
    if cur is not None:
        flip = prev != 0 and p != 0 and np.sign(p) != np.sign(prev)
        q_close = -prev if flip else sq[i]                   # 반전 체결은 청산분만 이번 거래로
        cur["btc"] += q_close / px[i]                         # 인버스 계약 손익 = Σ 계약수/가격
        if np.sign(q_close) == cur["dir"]:
            cur["qty_in"] += abs(q_close); cur["cost_in"] += abs(q_close) / px[i]
        cur["max_pos"] = max(cur["max_pos"], abs(prev if flip else p))
        if p == 0 or flip:
            cur["close"] = ts[i]; rts.append(cur)
            cur = dict(open=ts[i], dir=np.sign(p), btc=p / px[i], qty_in=abs(p), cost_in=abs(p) / px[i], max_pos=abs(p)) if flip else None
    prev = p
rt = pd.DataFrame(rts)
rt["open"], rt["close"] = pd.to_datetime(rt.open), pd.to_datetime(rt.close)
rt["hold_h"] = (rt.close - rt.open).dt.total_seconds() / 3600
rt["entry"] = rt.qty_in / rt.cost_in
rt["ret"] = rt.btc / rt.cost_in                                   # 증거금(BTC) 대비 아닌, 투입 명목 대비 수익률
rt["year"] = rt.open.dt.year
rt = rt.join(f[["r_4h", "vol_regime", "atr24", "net"]], on=rt.open.dt.floor("h"))
total_btc = rt.btc.sum()

print("=" * 100)
print("① 돈을 번 거래 vs 개수가 많은 거래 — 통계를 '횟수'로 뽑으면 무엇을 놓치나")
print(f"XBTUSD 라운드트립 {len(rt)}건, 계약 손익 합계 {total_btc:,.0f} BTC")
g = rt.groupby("year").agg(건수=("btc", "size"), BTC손익=("btc", "sum"), 보유중앙값h=("hold_h", "median"), 규모중앙값=("max_pos", "median"))
g["건수비중"] = g.건수 / len(rt); g["손익비중"] = g.BTC손익 / total_btc
print(g.round(3))
rt["hb"] = pd.cut(rt.hold_h, [0, 1, 4, 12, 24, 72, 1e5], labels=["<1h", "1-4h", "4-12h", "12-24h", "1-3d", ">3d"])
g = rt.groupby("hb", observed=True).agg(건수=("btc", "size"), BTC손익=("btc", "sum"), 승률=("btc", lambda s: (s > 0).mean()))
g["손익비중"] = g.BTC손익 / total_btc
print(g.round(3))
top = rt.nlargest(20, "btc")
print(f"상위 20건 손익 {top.btc.sum():,.0f} BTC (전체의 {top.btc.sum() / total_btc:.0%}), 이 중 12시간 넘게 보유 {int((top.hold_h > 12).sum())}건, 중앙 보유 {top.hold_h.median():.0f}시간")

print("=" * 100)
print("② 봇의 진입 조건(변동성≥1.6, 4시간 ±3% 역방향)에서 워뇨띠가 실제로 매매했나")
cond = (rt.vol_regime >= 1.6) & (rt.r_4h.abs() >= 0.03)
counter = cond & (np.sign(rt.r_4h) == -rt.dir)
follow = cond & (np.sign(rt.r_4h) == rt.dir)
print(f"봇 조건에 해당하는 워뇨띠의 진입: {cond.sum()}건 ({cond.mean():.1%}), 손익 {rt.btc[cond].sum():,.0f} BTC ({rt.btc[cond].sum() / total_btc:.1%})")
print(f"  그중 역추세(봇과 같은 방향) {counter.sum()}건 손익 {rt.btc[counter].sum():+,.0f} BTC, 승률 {(rt.btc[counter] > 0).mean():.0%}")
print(f"  그중 추세추종(봇과 반대)   {follow.sum()}건 손익 {rt.btc[follow].sum():+,.0f} BTC, 승률 {(rt.btc[follow] > 0).mean():.0%}")
hours_long = (f.vol_regime >= 1.6) & (f.r_4h <= -0.03)
hours_short = (f.vol_regime >= 1.6) & (f.r_4h >= 0.03)
dnet = f.net.shift(-4) - f.net
print(f"봇 조건 발생 시간: 롱 신호 {hours_long.sum()}시간, 숏 신호 {hours_short.sum()}시간 (전체 {len(f)}시간)")
print(f"  롱 신호 후 4시간 워뇨띠의 순노출 변화 평균 {dnet[hours_long].mean():+.3f}배 (늘린 비율 {(dnet[hours_long] > 0.05).mean():.0%}, 줄인 비율 {(dnet[hours_long] < -0.05).mean():.0%})")
print(f"  숏 신호 후 4시간 워뇨띠의 순노출 변화 평균 {dnet[hours_short].mean():+.3f}배 (줄인 비율 {(dnet[hours_short] < -0.05).mean():.0%}, 늘린 비율 {(dnet[hours_short] > 0.05).mean():.0%})")
print(f"  (평소 4시간 순노출 변화 평균 {dnet.mean():+.3f}배)")

print("=" * 100)
print("③ 수수료: 워뇨띠는 수수료를 냈나, 받았나")
t = x.copy(); t["year"] = t.ts.dt.year
t["notional_btc"] = t.lastqty / t.lastpx
fee = t.groupby("year").agg(수수료BTC=("execcomm", lambda s: s.sum() / 1e8), 거래대금BTC=("notional_btc", "sum"),
                            메이커비중=("lastliquidityind", lambda s: (s == "AddedLiquidity").mean()))
fee["실효수수료율"] = fee.수수료BTC / fee.거래대금BTC
fu = tr[(tr.symbol == "XBTUSD")]
print(fee.round(5))
print(f"수수료율 분포: {t.commission.value_counts().head(4).to_dict()}  (음수 = 리베이트를 받음)")
print(f"워뇨띠의 라운드트립 수익률 중앙값 {rt.ret.median():+.3%}  vs  봇 왕복 비용 0.14% (수수료 0.05%×2 + 슬리피지 0.02%×2)")
print(f"수익률이 0.14% 미만인 워뇨띠의 이익 거래: {((rt.ret > 0) & (rt.ret < 0.0014)).sum()}건 / 이익 거래 {(rt.ret > 0).sum()}건")

print("=" * 100)
print("④ 봇의 손절(ATR×3)·12시간 청산이 워뇨띠의 거래에 적용됐다면")
def mae(r):
    bars = f.loc[r.open.floor("h") + pd.Timedelta(hours=1): r.close.ceil("h")]
    if bars.empty or np.isnan(r.atr24):
        return np.nan
    worst = bars.low.min() if r.dir > 0 else bars.high.max()
    return r.dir * (worst / r.entry - 1) / r.atr24
rt["mae_atr"] = rt.apply(mae, axis=1)
stopped = rt.mae_atr <= -3
timed = rt.hold_h > 12
win = rt.btc > 0
print(f"도중에 ATR×3 이상 불리하게 갔던 거래 {stopped.sum()}건, 그중 결국 이익으로 끝난 것 {(stopped & win).sum()}건 → 손익 {rt.btc[stopped & win].sum():+,.0f} BTC")
print(f"12시간 넘게 보유한 거래 {timed.sum()}건, 손익 {rt.btc[timed].sum():+,.0f} BTC ({rt.btc[timed].sum() / total_btc:.0%})")
print(f"손절 또는 12시간 규칙에 걸렸을 거래의 손익 합 {rt.btc[stopped | timed].sum():+,.0f} BTC ({rt.btc[stopped | timed].sum() / total_btc:.0%})")

print("=" * 100)
print("⑤ 계약상 숏은 하락 베팅이었나, 보유 BTC 헤지였나")
sh = rt[rt.dir < 0].copy()
sh["short_lev"] = sh.max_pos / (bal.reindex(sh.open.dt.normalize()).ffill().values * sh.entry)
hedge = sh.short_lev <= 1.0
print(f"숏 라운드트립 {len(sh)}건 중 숏 규모가 보유 BTC 가치 이하(순노출이 0 이상 유지 = 헤지) {hedge.sum()}건 ({hedge.mean():.0%})")
print(f"  헤지성 숏 손익 {sh.btc[hedge].sum():+,.0f} BTC, 실제 순숏(하락 베팅) 손익 {sh.btc[~hedge].sum():+,.0f} BTC")
lb = f.net > 0.25; sb = f.net < -0.25
print(f"시간 기준 순노출: 순롱 {lb.mean():.0%}, 중립 {(~lb & ~sb).mean():.0%}, 순숏 {sb.mean():.0%}  → 봇은 롱/숏을 대칭으로 다룸")

print("=" * 100)
print("⑥ 같은 시기(2020~2021) 봇의 거래와 워뇨띠의 실제 포지션 비교")
bt = pd.read_csv("data/bt_btc_2020_2026.csv", parse_dates=["opened", "closed"])
bt = bt[bt.opened < "2022-01-01"].copy()
fn = f.net
bt["his_net"] = fn.reindex(bt.opened).values
bt["his_dnet"] = (fn.shift(-4) - fn).reindex(bt.opened).values
agree = np.sign(bt.his_dnet.where(bt.his_dnet.abs() > 0.05, 0)) == bt.side
print(f"봇 거래 {len(bt)}건(2020~2021), 봇 손익 {bt.pnl.sum():+,.0f} USDT")
print(f"  진입 후 4시간 동안 워뇨띠도 같은 방향으로 순노출을 늘린 경우 {agree.mean():.0%}, 반대로 움직인 경우 {(np.sign(bt.his_dnet.where(bt.his_dnet.abs() > 0.05, 0)) == -bt.side).mean():.0%}, 변화 없음 {(bt.his_dnet.abs() <= 0.05).mean():.0%}")
print(f"  봇 숏 진입 시점 워뇨띠의 순노출 평균 {bt.his_net[bt.side < 0].mean():+.2f}배, 봇 롱 진입 시점 {bt.his_net[bt.side > 0].mean():+.2f}배")

print("=" * 100)
print("⑦ 워뇨띠의 우위는 얼마나 빨리 사라지나: 워뇨띠의 순노출 방향을 N시간 늦게 따라 할 때")
fwd = np.log(f.px).shift(-1) - np.log(f.px)
y = pd.Series(np.select([f.net > 0.25, f.net < -0.25], [1, -1], 0), f.index)
for lag in (0, 1, 2, 4, 8, 24):
    pz = y.shift(lag).fillna(0)
    net_r = pz * fwd - pz.diff().abs().fillna(0) * 0.00075
    net_r = net_r.dropna()
    print(f"  {lag:>2}시간 지연: 샤프 {net_r.mean() / net_r.std() * np.sqrt(8760):5.2f}, 누적 {np.expm1(net_r.sum()):+8.0%}")
bh = fwd.dropna(); print(f"  (단순 보유 샤프 {bh.mean() / bh.std() * np.sqrt(8760):.2f})")
h_pnl = (f.pos * (1 / f.px - 1 / f.px.shift(-1))).groupby(f.index.year).sum()
real = w[(w.transacttype == "RealisedPNL") & (w.address == "XBTUSD")].groupby(w.date.dt.year).amount.sum() / 1e8
print("시간봉 보유만으로 설명되는 계약 손익 vs 실제 실현손익(XBTUSD, BTC):")
print(pd.DataFrame({"시간봉_보유손익": h_pnl, "실제_실현손익": real}).round(0))
