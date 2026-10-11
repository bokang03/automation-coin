"""5분봉 차트 상태. 과거 참조는 최대 7일(2,016봉)이라 실시간 봇이 몇 번의 조회로 계산할 수 있다.

워뇨띠의 체결 타이밍 분석(scripts/12_timing_events.py)에서 중요했던 것들을 담는다:
직전 5분~3일 수익률, 최근 1시간~7일 범위 안의 위치, 변동성·거래량이 터졌는지, 방금 끝난 봉과 최근 1시간의 꼬리 모양.
"""
import numpy as np
import pandas as pd

HORIZONS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "12h": 144, "24h": 288, "72h": 864}
RANGES = {"1h": 12, "4h": 48, "24h": 288, "7d": 2016}
WARMUP_BARS = 2016 + 288


def make_features_5m(c: pd.DataFrame) -> pd.DataFrame:
    """c: timestamp(봉 마감 시각), open, high, low, close, volume. 돌려주는 표의 인덱스 = 봉 마감 시각(UTC, tz 없음)."""
    ts = c.timestamp.dt.tz_localize(None) if c.timestamp.dt.tz is not None else c.timestamp
    c = c.set_index(ts).sort_index()
    o, h, l, cl, v = c.open, c.high, c.low, c.close, c.volume
    f = pd.DataFrame(index=c.index)
    lc = np.log(cl)
    for name, k in HORIZONS.items():
        f[f"r_{name}"] = lc.diff(k)
    tr = pd.concat([h - l, (h - cl.shift()).abs(), (l - cl.shift()).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(288).mean() / cl
    f["atr24h"] = atr24
    f["vol_burst"] = (tr.rolling(12).mean() / cl) / atr24            # 최근 1시간 변동성 / 최근 24시간
    f["vol_regime"] = atr24 / atr24.rolling(2016).mean()            # 최근 24시간 변동성 / 최근 7일
    for name, k in RANGES.items():
        hi, lo = h.rolling(k).max(), l.rolling(k).min()
        f[f"range_pos_{name}"] = (cl - lo) / (hi - lo).replace(0, np.nan)
        f[f"dd_high_{name}"] = cl / hi - 1
        f[f"up_low_{name}"] = cl / lo - 1
    f["vol_ratio"] = v / v.rolling(288).mean()
    f["vol_ratio_1h"] = v.rolling(12).sum() / (v.rolling(2016).mean() * 12)
    rng = (h - l).replace(0, np.nan)
    f["body"] = (cl - o) / rng
    f["lower_wick"] = (np.minimum(o, cl) - l) / rng
    f["upper_wick"] = (h - np.maximum(o, cl)) / rng
    f["bar_range"] = (h - l) / cl / atr24                            # 방금 끝난 봉의 크기(평소 1봉 변동폭 대비)
    o1, h1, l1 = o.shift(11), h.rolling(12).max(), l.rolling(12).min()
    rng1 = (h1 - l1).replace(0, np.nan)
    f["lower_wick_1h"] = (np.minimum(o1, cl) - l1) / rng1
    f["upper_wick_1h"] = (h1 - np.maximum(o1, cl)) / rng1
    f["hour"] = f.index.hour
    f["px"] = cl
    return f


FEATS_5M = [f"r_{k}" for k in HORIZONS] + ["atr24h", "vol_burst", "vol_regime"] + \
           [f"{p}_{k}" for k in RANGES for p in ("range_pos", "dd_high", "up_low")] + \
           ["vol_ratio", "vol_ratio_1h", "body", "lower_wick", "upper_wick", "bar_range", "lower_wick_1h", "upper_wick_1h", "hour"]
