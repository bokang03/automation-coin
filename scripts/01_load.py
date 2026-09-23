"""원본 CSV를 parquet으로 변환한다."""
import glob, os, pandas as pd
RAW = os.environ.get("AOA_RAW", os.path.join(os.path.dirname(__file__), "..", "raw"))  # zip을 풀어 둔 폴더
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
cols = ["execid","orderid","symbol","side","lastqty","lastpx","lastliquidityind","orderqty","price","stoppx",
        "exectype","ordtype","timeinforce","execinst","ordstatus","commission","text","execcomm","homenotional",
        "foreignnotional","transacttime"]
frames = []
for f in sorted(glob.glob(os.path.join(RAW, "aoa-execution-*.csv"))):
    df = pd.read_csv(f, usecols=cols, encoding="utf-8-sig", low_memory=False)
    frames.append(df); print(f, len(df))
ex = pd.concat(frames, ignore_index=True)
ex["ts"] = pd.to_datetime(ex["transacttime"], format="mixed")
ex = ex.drop(columns="transacttime").sort_values("ts").reset_index(drop=True)
ex.to_parquet(os.path.join(OUT, "executions.parquet"))
w = pd.read_csv(glob.glob(os.path.join(RAW, "aoa-wallet-*.csv"))[0], encoding="utf-8-sig")
w.to_parquet(os.path.join(OUT, "wallet.parquet"))
print(ex.shape, w.shape)
