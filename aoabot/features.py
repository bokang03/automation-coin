"""시간봉 캔들에서 시장 상태 피처를 만든다. 분석과 봇에서 같은 함수를 쓴다."""
import numpy as np, pandas as pd

def make_features(c: pd.DataFrame) -> pd.DataFrame:
    c = c.set_index(c.timestamp.dt.tz_localize(None)).sort_index()
    f = pd.DataFrame(index=c.index)
    lc = np.log(c.close)
    for h in (1, 4, 12, 24, 72, 168):
        f[f"r_{h}h"] = lc.diff(h)
    tr = pd.concat([c.high - c.low, (c.high - c.close.shift()).abs(), (c.low - c.close.shift()).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24).mean() / c.close
    f["atr24"] = atr24
    f["vol_regime"] = atr24 / atr24.rolling(24 * 30).mean()
    for h in (24, 168):
        hi, lo = c.high.rolling(h).max(), c.low.rolling(h).min()
        f[f"range_pos_{h}h"] = (c.close - lo) / (hi - lo)
        f[f"dd_from_high_{h}h"] = c.close / hi - 1
    f["vol_ratio"] = c.volume / c.volume.rolling(24 * 7).mean()
    f["vol_ratio_4h"] = c.volume.rolling(4).sum() / (c.volume.rolling(24 * 7).mean() * 4)
    for n in (20, 50, 200):
        f[f"ma{n}_gap"] = c.close / c.close.rolling(n).mean() - 1
    f["ma_gap_1d_50"] = c.close / c.close.rolling(24 * 50).mean() - 1
    # 마지막 봉 모양
    body = (c.close - c.open) / (c.high - c.low).replace(0, np.nan)
    f["body"] = body
    f["lower_wick"] = (np.minimum(c.open, c.close) - c.low) / (c.high - c.low).replace(0, np.nan)
    f["upper_wick"] = (c.high - np.maximum(c.open, c.close)) / (c.high - c.low).replace(0, np.nan)
    f["hour"] = f.index.hour
    f["px"] = c.close
    return f
