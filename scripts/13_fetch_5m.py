"""바이낸스 USDT 선물 5분봉을 받는다. 사용: python scripts/13_fetch_5m.py BTC/USDT:USDT 2021-10-01 2026-10-10"""
import sys
from pathlib import Path

import ccxt
import pandas as pd

sym, start, end = sys.argv[1], pd.Timestamp(sys.argv[2], tz="UTC"), pd.Timestamp(sys.argv[3], tz="UTC")
out = Path("data") / f"binance_{sym.split('/')[0]}_5m_{start:%Y%m}_{end:%Y%m}.parquet"
ex = ccxt.binanceusdm({"enableRateLimit": True})
rows, since = [], int(start.timestamp() * 1000)
while since < int(end.timestamp() * 1000):
    batch = ex.fetch_ohlcv(sym, "5m", since=since, limit=1500)
    if not batch:
        break
    rows += batch
    since = batch[-1][0] + 300_000
    if len(rows) % 60000 < 1500:
        print(sym, pd.Timestamp(batch[-1][0], unit="ms"), len(rows), flush=True)
c = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
c["timestamp"] = pd.to_datetime(c.t, unit="ms", utc=True) + pd.Timedelta(minutes=5)      # 봉 마감 시각
c = c[c.timestamp <= end].drop(columns="t").reset_index(drop=True)
c.to_parquet(out)
print("saved", out, len(c), c.timestamp.iloc[0], "~", c.timestamp.iloc[-1])
