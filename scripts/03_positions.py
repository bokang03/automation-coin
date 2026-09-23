"""체결 내역으로 심볼별 포지션을 재구성하고, 포지션이 0에서 시작해 0으로 끝나는 구간을 '라운드트립'으로 묶는다."""
import pandas as pd, numpy as np
ex = pd.read_parquet("data/executions.parquet")
tr = ex[ex.exectype == "Trade"].copy()
tr["sq"] = np.where(tr.side == "Buy", 1, -1) * tr.lastqty
tr["pos"] = tr.groupby("symbol").sq.cumsum()
tr.to_parquet("data/trades_pos.parquet")

# 펀딩 레코드(lastqty=보유 수량)로 재구성 검증
fu = ex[(ex.exectype == "Funding") & (ex.symbol == "XBTUSD")]
x = tr[tr.symbol == "XBTUSD"][["ts", "pos"]]
m = pd.merge_asof(fu[["ts", "lastqty", "homenotional"]].sort_values("ts"), x, on="ts")
m["match"] = m.pos.abs() == m.lastqty
print("funding check XBTUSD match rate:", m.match.mean(), "n=", len(m))

rts = []
for sym, g in tr.groupby("symbol"):
    pos = g.pos.values; sq = g.sq.values; px = g.lastpx.values; ts = g.ts.values; comm = g.execcomm.values
    start = 0; prev = 0.0
    for i in range(len(g)):
        p = pos[i]
        crossed = prev != 0 and (p == 0 or np.sign(p) != np.sign(prev))
        if crossed:
            sl = slice(start, i + 1)
            d = np.sign(prev)
            qty = sq[sl]; prices = px[sl]
            entry_mask = np.sign(qty) == d
            e_qty = np.abs(qty[entry_mask]).sum()
            rts.append(dict(symbol=sym, dir="long" if d > 0 else "short", open=ts[start], close=ts[i],
                            max_pos=np.abs(pos[sl]).max(), entry_vwap=(np.abs(qty[entry_mask]) * prices[entry_mask]).sum() / e_qty,
                            exit_vwap=(np.abs(qty[~entry_mask]) * prices[~entry_mask]).sum() / max(np.abs(qty[~entry_mask]).sum(), 1),
                            n_fills=i + 1 - start, comm_sat=comm[sl].sum()))
            start = i + 1 if p == 0 else i
        elif prev == 0 and p != 0:
            start = i
        prev = p
rt = pd.DataFrame(rts)
rt["hold_h"] = (rt.close - rt.open).dt.total_seconds() / 3600
sgn = np.where(rt.dir == "long", 1, -1)
rt["ret"] = sgn * (rt.exit_vwap / rt.entry_vwap - 1)
rt.to_parquet("data/roundtrips.parquet")
print(rt.symbol.value_counts().head(8))
