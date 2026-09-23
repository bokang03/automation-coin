"""BitMEX 공개 API에서 OHLCV 캔들을 받는다. 사용: python 02_fetch_candles.py XBTUSD 5m"""
import sys, time, os, requests, pandas as pd
sym, bin_ = sys.argv[1], sys.argv[2]
start = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2018-02-01", tz="UTC")
end = pd.Timestamp(sys.argv[4] if len(sys.argv) > 4 else "2022-01-01", tz="UTC")
out = os.path.join(os.path.dirname(__file__), "..", "data", f"candles_{sym}_{bin_}_{start:%Y}_{end:%Y}.parquet" if len(sys.argv) > 3 else f"candles_{sym}_{bin_}.parquet")
rows, cur = [], start
while cur < end:
    r = requests.get("https://www.bitmex.com/api/v1/trade/bucketed", params=dict(
        binSize=bin_, symbol=sym, count=1000, startTime=cur.isoformat(), partial="false"), timeout=30)
    if r.status_code == 429:
        time.sleep(int(r.headers.get("Retry-After", 10))); continue
    r.raise_for_status(); batch = r.json()
    if not batch: break
    rows += batch; cur = pd.Timestamp(batch[-1]["timestamp"]) + pd.Timedelta(seconds=1)
    print(sym, bin_, cur, len(rows), flush=True)
    time.sleep(2.1)
df = pd.DataFrame(rows)[["timestamp","open","high","low","close","volume","vwap","trades"]]
df["timestamp"] = pd.to_datetime(df.timestamp); df = df[df.timestamp < end].drop_duplicates("timestamp")
df.to_parquet(out); print("saved", out, len(df))
