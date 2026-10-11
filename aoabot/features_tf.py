"""1시간·일·주·월봉 안에서 지금 가격이 어느 지점인지. 5분봉에서 계산하고, 그 5분봉이 끝난 시점까지의 정보만 쓴다.

봉마다 다섯 가지를 본다 (이름 앞에 1h_, 1d_, 1w_, 1M_ 가 붙는다):
  ret       진행 중인 봉의 시가 대비 현재 수익률 (지금 양봉인가 음봉인가, 얼마나)
  pos       진행 중인 봉의 저가~고가 범위 안에서 현재 위치 (0 = 저가, 1 = 고가)
  prev_ret  직전에 끝난 봉의 수익률
  prev_pos  직전 봉의 저가~고가 대비 현재 위치 (0 미만 = 전 봉 저가 아래, 1 초과 = 전 봉 고가 위)
  elapsed   진행 중인 봉이 얼마나 지났나 (0 = 막 시작, 1 = 곧 마감)
봉의 기준 시각은 UTC다 (일봉은 한국 시각 09시 시작, 주봉은 월요일 시작, 월봉은 1일 시작).
"""
import numpy as np
import pandas as pd

TFS = ["1h", "1d", "1w", "1M"]
TF_NAMES = {"1h": "1시간봉", "1d": "일봉", "1w": "주봉", "1M": "월봉"}
PARTS = ["ret", "pos", "prev_ret", "prev_pos", "elapsed"]
TF_FEATS = [f"{tf}_{p}" for tf in TFS for p in PARTS]
BAR = pd.Timedelta(minutes=5)
TF_WARMUP = pd.Timedelta(days=63)     # 전월 봉 전체가 있어야 월봉 지표가 나온다


def _period(open_time: pd.DatetimeIndex, tf: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(그 봉이 속한 상위 봉의 시작 시각, 끝 시각)"""
    if tf == "1h":
        start = open_time.floor("h")
        return start, start + pd.Timedelta(hours=1)
    if tf == "1d":
        start = open_time.floor("D")
        return start, start + pd.Timedelta(days=1)
    per = open_time.to_period("W-SUN" if tf == "1w" else "M")
    return per.start_time, (per + 1).start_time


def make_tf_features(c: pd.DataFrame) -> pd.DataFrame:
    """c: timestamp(5분봉 마감 시각), open, high, low, close. 돌려주는 표의 인덱스 = 5분봉 마감 시각(UTC, tz 없음)."""
    ts = c.timestamp.dt.tz_localize(None) if c.timestamp.dt.tz is not None else c.timestamp
    d = pd.DataFrame({"open": c.open.to_numpy(), "high": c.high.to_numpy(), "low": c.low.to_numpy(), "close": c.close.to_numpy()},
                     index=pd.DatetimeIndex(ts)).sort_index()
    open_time = d.index - BAR
    f = pd.DataFrame(index=d.index)
    for tf in TFS:
        start, end = _period(open_time, tf)
        key = pd.Series(start, index=d.index)
        g = d.groupby(key.to_numpy())
        first_open_time = pd.Series(open_time, index=d.index).groupby(key.to_numpy()).transform("first")
        whole = (first_open_time - key) <= 2 * BAR                       # 상위 봉의 처음부터 자료가 있는가
        cur_open = g.open.transform("first")
        cur_high, cur_low = g.high.cummax(), g.low.cummin()
        rng = (cur_high - cur_low).replace(0, np.nan)
        f[f"{tf}_ret"] = (d.close / cur_open - 1).where(whole)
        f[f"{tf}_pos"] = ((d.close - cur_low) / rng).where(whole)
        agg = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        agg[~whole.groupby(key.to_numpy()).first().reindex(agg.index).to_numpy()] = np.nan
        consecutive = pd.Series(agg.index, index=agg.index).diff() <= pd.Timedelta(days=32)   # 자료가 끊긴 뒤의 '직전 봉'은 쓰지 않는다
        prev = agg.shift(1).where(consecutive, np.nan).reindex(key.to_numpy())
        prev.index = d.index
        f[f"{tf}_prev_ret"] = prev.close / prev.open - 1
        f[f"{tf}_prev_pos"] = (d.close - prev.low) / (prev.high - prev.low).replace(0, np.nan)
        f[f"{tf}_elapsed"] = (d.index - start) / (end - start)
    return f[TF_FEATS]
